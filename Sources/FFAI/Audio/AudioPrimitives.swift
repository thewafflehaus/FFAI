// AudioPrimitives — CPU-native building blocks for neural audio codecs.
//
// FFAI's `Ops.*` GPU wrappers are tuned for the LLM decode path (gemv,
// rope, rms-norm, single-step causal conv). Neural codecs instead lean
// on *general* 1-D convolution: arbitrary kernel sizes, strides,
// dilations, grouping, and transposed (fractionally-strided) variants.
// metaltile ships no kernel for that shape family, so — per the FFAI
// porting contract — these primitives use a documented CPU path.
//
// All math runs on the host over shared-storage `MTLBuffer`s (every
// FFAI tensor is CPU-addressable via `buffer.contents()`), so results
// stay MTLBuffer-native and can be handed straight to a GPU `Ops` call
// or to a TTS model without a copy. Codecs run once per utterance —
// not in a hot autoregressive loop — so the CPU cost is acceptable and
// avoids inventing un-benchmarked kernels.
//
// Tensor layout convention for audio: NCL — [batch, channels, length].

import Foundation
// Use the modern (ILP64-capable) Accelerate BLAS headers so `cblas_sgemm`
// is not flagged as deprecated.
#if canImport(Accelerate)
import Accelerate
#endif

/// CPU-native audio math. Mirrors the subset of `MLX`/`MLXNN` ops the
/// ported codecs need, operating directly on FFAI `Tensor`s (f32).
public enum AudioMath {

    // ─── Tensor <-> [Float] helpers ──────────────────────────────────

    /// Wrap a flat `[Float]` array into a freshly-allocated f32 Tensor.
    public static func tensor(_ values: [Float], shape: [Int],
                              device: Device = .shared) -> Tensor {
        precondition(values.count == shape.reduce(1, *),
                     "AudioMath.tensor: value count \(values.count) != shape \(shape)")
        let t = Tensor.empty(shape: shape, dtype: .f32, device: device)
        t.copyIn(from: values)
        return t
    }

    /// Read a tensor's contents as `[Float]`, converting from f16/bf16
    /// when necessary. Codec checkpoints frequently ship f16/bf16.
    public static func floats(_ t: Tensor) -> [Float] {
        switch t.dtype {
        case .f32:
            return t.toArray(as: Float.self)
        case .f16:
            let raw = t.toArray(as: UInt16.self)
            return raw.map { Float(Float16(bitPattern: $0)) }
        case .bf16:
            let raw = t.toArray(as: UInt16.self)
            return raw.map { bf16ToFloat($0) }
        case .i32:
            return t.toArray(as: Int32.self).map { Float($0) }
        case .u32:
            return t.toArray(as: UInt32.self).map { Float($0) }
        case .i8:
            return t.toArray(as: Int8.self).map { Float($0) }
        case .u8:
            return t.toArray(as: UInt8.self).map { Float($0) }
        }
    }

    private static func bf16ToFloat(_ bits: UInt16) -> Float {
        // bf16 is the top 16 bits of an IEEE-754 f32.
        let widened = UInt32(bits) << 16
        return Float(bitPattern: widened)
    }

    // ─── Elementwise activations ─────────────────────────────────────

    /// Snake activation: x + (1/(alpha+eps)) * sin(alpha*x)^2.
    /// `alpha` has one value per channel; `x` is NCL.
    public static func snake(_ x: [Float], shape: [Int], alpha: [Float],
                             eps: Float = 1e-9) -> [Float] {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        precondition(alpha.count == c, "snake: alpha count != channels")
        var out = [Float](repeating: 0, count: x.count)
        for b in 0..<n {
            for ch in 0..<c {
                let a = alpha[ch]
                let recip = 1.0 / (a + eps)
                let base = (b * c + ch) * l
                for i in 0..<l {
                    let v = x[base + i]
                    let s = sin(a * v)
                    out[base + i] = v + recip * s * s
                }
            }
        }
        return out
    }

    /// Hyperbolic tangent, elementwise.
    public static func tanhAll(_ x: [Float]) -> [Float] {
        x.map { tanhf($0) }
    }

    /// Sigmoid Linear Unit (SiLU / swish): x * sigmoid(x).
    public static func silu(_ x: [Float]) -> [Float] {
        x.map { $0 / (1.0 + expf(-$0)) }
    }

    /// GELU (tanh approximation), elementwise.
    public static func gelu(_ x: [Float]) -> [Float] {
        let c: Float = 0.7978845608028654 // sqrt(2/pi)
        return x.map { v in
            0.5 * v * (1.0 + tanhf(c * (v + 0.044715 * v * v * v)))
        }
    }

    /// ELU activation, elementwise.
    public static func elu(_ x: [Float], alpha: Float = 1.0) -> [Float] {
        x.map { $0 > 0 ? $0 : alpha * (expf($0) - 1.0) }
    }

    /// Leaky ReLU, elementwise.
    public static func leakyRelu(_ x: [Float], slope: Float = 0.01) -> [Float] {
        x.map { $0 > 0 ? $0 : slope * $0 }
    }

    // ─── Convolution ─────────────────────────────────────────────────

    /// Padded 1-D convolution, NCL in/out.
    ///
    /// - x:       input, shape `[N, Cin, L]`.
    /// - weight:  shape `[Cout, Cin/groups, K]` (PyTorch conv1d layout).
    /// - bias:    optional `[Cout]`.
    /// Returns output `[N, Cout, Lout]` where
    /// `Lout = (L + 2*pad - dilation*(K-1) - 1) / stride + 1`.
    public static func conv1d(
        x: [Float], xShape: [Int],
        weight: [Float], wShape: [Int],
        bias: [Float]?,
        stride: Int = 1, padding: Int = 0,
        dilation: Int = 1, groups: Int = 1
    ) -> (data: [Float], shape: [Int]) {
        let (n, cIn, l) = (xShape[0], xShape[1], xShape[2])
        let (cOut, cInPerGroup, k) = (wShape[0], wShape[1], wShape[2])
        precondition(cIn % groups == 0 && cOut % groups == 0,
                     "conv1d: channels not divisible by groups")
        precondition(cInPerGroup == cIn / groups,
                     "conv1d: weight Cin/groups mismatch \(wShape) vs in \(cIn) groups \(groups)")
        let effectiveK = dilation * (k - 1) + 1
        let lOut = (l + 2 * padding - effectiveK) / stride + 1
        precondition(lOut > 0, "conv1d: non-positive output length \(lOut)")

        var out = [Float](repeating: 0, count: n * cOut * lOut)
        let cOutPerGroup = cOut / groups

        for b in 0..<n {
            for g in 0..<groups {
                for ocLocal in 0..<cOutPerGroup {
                    let oc = g * cOutPerGroup + ocLocal
                    let outBase = (b * cOut + oc) * lOut
                    let biasVal = bias?[oc] ?? 0
                    for t in 0..<lOut {
                        var acc: Float = biasVal
                        let inStart = t * stride - padding
                        for icLocal in 0..<cInPerGroup {
                            let ic = g * cInPerGroup + icLocal
                            let xBase = (b * cIn + ic) * l
                            let wBase = (oc * cInPerGroup + icLocal) * k
                            for kk in 0..<k {
                                let inIdx = inStart + kk * dilation
                                if inIdx >= 0 && inIdx < l {
                                    acc += x[xBase + inIdx] * weight[wBase + kk]
                                }
                            }
                        }
                        out[outBase + t] = acc
                    }
                }
            }
        }
        return (out, [n, cOut, lOut])
    }

    /// Transposed (fractionally-strided) 1-D convolution, NCL in/out.
    ///
    /// - x:       input, shape `[N, Cin, L]`.
    /// - weight:  shape `[Cin, Cout/groups, K]` (PyTorch convTranspose1d layout).
    /// - bias:    optional `[Cout]`.
    /// `Lout = (L-1)*stride - 2*pad + dilation*(K-1) + outputPadding + 1`.
    public static func convTransposed1d(
        x: [Float], xShape: [Int],
        weight: [Float], wShape: [Int],
        bias: [Float]?,
        stride: Int = 1, padding: Int = 0,
        dilation: Int = 1, outputPadding: Int = 0, groups: Int = 1
    ) -> (data: [Float], shape: [Int]) {
        let (n, cIn, l) = (xShape[0], xShape[1], xShape[2])
        let (wCin, cOutPerGroup, k) = (wShape[0], wShape[1], wShape[2])
        precondition(wCin == cIn, "convT1d: weight Cin mismatch \(wShape) vs \(cIn)")
        precondition(cIn % groups == 0, "convT1d: Cin not divisible by groups")
        let cOut = cOutPerGroup * groups
        let cInPerGroup = cIn / groups
        let lOut = (l - 1) * stride - 2 * padding + dilation * (k - 1) + outputPadding + 1
        precondition(lOut > 0, "convT1d: non-positive output length \(lOut)")

        var out = [Float](repeating: 0, count: n * cOut * lOut)
        // Scatter: each input sample distributes into the output.
        for b in 0..<n {
            for g in 0..<groups {
                for icLocal in 0..<cInPerGroup {
                    let ic = g * cInPerGroup + icLocal
                    let xBase = (b * cIn + ic) * l
                    for t in 0..<l {
                        let xv = x[xBase + t]
                        if xv == 0 { continue }
                        let outStart = t * stride - padding
                        for ocLocal in 0..<cOutPerGroup {
                            let oc = g * cOutPerGroup + ocLocal
                            let outBase = (b * cOut + oc) * lOut
                            let wBase = (ic * cOutPerGroup + ocLocal) * k
                            for kk in 0..<k {
                                let outIdx = outStart + kk * dilation
                                if outIdx >= 0 && outIdx < lOut {
                                    out[outBase + outIdx] += xv * weight[wBase + kk]
                                }
                            }
                        }
                    }
                }
            }
        }
        // Add bias per output channel.
        if let bias = bias {
            for b in 0..<n {
                for oc in 0..<cOut {
                    let outBase = (b * cOut + oc) * lOut
                    let bv = bias[oc]
                    for t in 0..<lOut { out[outBase + t] += bv }
                }
            }
        }
        return (out, [n, cOut, lOut])
    }

    // ─── Normalization ───────────────────────────────────────────────

    /// LayerNorm over the last dimension. `x` is `[..., D]` flattened
    /// as rows of length `D`.
    public static func layerNorm(
        _ x: [Float], rows: Int, dim: Int,
        weight: [Float]?, bias: [Float]?, eps: Float = 1e-5
    ) -> [Float] {
        precondition(x.count == rows * dim, "layerNorm: size mismatch")
        var out = [Float](repeating: 0, count: x.count)
        for r in 0..<rows {
            let base = r * dim
            var mean: Float = 0
            for i in 0..<dim { mean += x[base + i] }
            mean /= Float(dim)
            var variance: Float = 0
            for i in 0..<dim {
                let d = x[base + i] - mean
                variance += d * d
            }
            variance /= Float(dim)
            let invStd = 1.0 / sqrtf(variance + eps)
            for i in 0..<dim {
                var v = (x[base + i] - mean) * invStd
                if let w = weight { v *= w[i] }
                if let bi = bias { v += bi[i] }
                out[base + i] = v
            }
        }
        return out
    }

    /// Group normalization over an NCL tensor.
    public static func groupNorm(
        _ x: [Float], shape: [Int], groups: Int,
        weight: [Float]?, bias: [Float]?, eps: Float = 1e-5
    ) -> [Float] {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        precondition(c % groups == 0, "groupNorm: channels not divisible by groups")
        let cPerGroup = c / groups
        var out = [Float](repeating: 0, count: x.count)
        for b in 0..<n {
            for g in 0..<groups {
                var mean: Float = 0
                let count = cPerGroup * l
                for cl in 0..<cPerGroup {
                    let ch = g * cPerGroup + cl
                    let base = (b * c + ch) * l
                    for i in 0..<l { mean += x[base + i] }
                }
                mean /= Float(count)
                var variance: Float = 0
                for cl in 0..<cPerGroup {
                    let ch = g * cPerGroup + cl
                    let base = (b * c + ch) * l
                    for i in 0..<l {
                        let d = x[base + i] - mean
                        variance += d * d
                    }
                }
                variance /= Float(count)
                let invStd = 1.0 / sqrtf(variance + eps)
                for cl in 0..<cPerGroup {
                    let ch = g * cPerGroup + cl
                    let base = (b * c + ch) * l
                    for i in 0..<l {
                        var v = (x[base + i] - mean) * invStd
                        if let w = weight { v *= w[ch] }
                        if let bi = bias { v += bi[ch] }
                        out[base + i] = v
                    }
                }
            }
        }
        return out
    }

    /// L2-normalize each row of a `[rows, dim]` matrix.
    public static func l2NormalizeRows(_ x: [Float], rows: Int, dim: Int,
                                       eps: Float = 1e-12) -> [Float] {
        var out = x
        for r in 0..<rows {
            let base = r * dim
            var ss: Float = 0
            for i in 0..<dim { ss += x[base + i] * x[base + i] }
            let inv = 1.0 / max(sqrtf(ss), eps)
            for i in 0..<dim { out[base + i] *= inv }
        }
        return out
    }

    // ─── Linear algebra ──────────────────────────────────────────────

    /// Dense matmul: `a [m,k] · b [k,n]` -> `[m,n]`. Uses Accelerate's
    /// `vDSP_mmul` (non-deprecated, single-precision).
    public static func matmul(_ a: [Float], _ b: [Float],
                              m: Int, k: Int, n: Int) -> [Float] {
        precondition(a.count == m * k && b.count == k * n,
                     "matmul: operand size mismatch")
        var out = [Float](repeating: 0, count: m * n)
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                out.withUnsafeMutableBufferPointer { op in
                    vDSP_mmul(ap.baseAddress!, 1, bp.baseAddress!, 1,
                              op.baseAddress!, 1,
                              vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
                }
            }
        }
        return out
    }

    /// Affine linear layer: `x [rows, inDim] · W^T + b`, where `weight`
    /// is the PyTorch `[outDim, inDim]` layout. Transposes `weight` once
    /// then defers to `matmul`.
    public static func linear(_ x: [Float], rows: Int, inDim: Int,
                              weight: [Float], outDim: Int,
                              bias: [Float]?) -> [Float] {
        precondition(weight.count == outDim * inDim, "linear: weight size mismatch")
        // weightᵀ : [inDim, outDim].
        var wt = [Float](repeating: 0, count: inDim * outDim)
        for o in 0..<outDim {
            for i in 0..<inDim {
                wt[i * outDim + o] = weight[o * inDim + i]
            }
        }
        var out = matmul(x, wt, m: rows, k: inDim, n: outDim)
        if let bias = bias {
            for r in 0..<rows {
                let base = r * outDim
                for o in 0..<outDim { out[base + o] += bias[o] }
            }
        }
        return out
    }

    /// Numerically-stable softmax over the last dimension.
    public static func softmaxRows(_ x: [Float], rows: Int, dim: Int) -> [Float] {
        var out = [Float](repeating: 0, count: x.count)
        for r in 0..<rows {
            let base = r * dim
            var mx = -Float.greatestFiniteMagnitude
            for i in 0..<dim { mx = max(mx, x[base + i]) }
            var sum: Float = 0
            for i in 0..<dim {
                let e = expf(x[base + i] - mx)
                out[base + i] = e
                sum += e
            }
            let inv = 1.0 / sum
            for i in 0..<dim { out[base + i] *= inv }
        }
        return out
    }

    // ─── Layout helpers ──────────────────────────────────────────────

    /// Transpose the last two dims of a 3-D `[d0, d1, d2]` tensor.
    public static func transpose12(_ x: [Float], shape: [Int]) -> [Float] {
        let (d0, d1, d2) = (shape[0], shape[1], shape[2])
        var out = [Float](repeating: 0, count: x.count)
        for a in 0..<d0 {
            for i in 0..<d1 {
                for j in 0..<d2 {
                    out[(a * d2 + j) * d1 + i] = x[(a * d1 + i) * d2 + j]
                }
            }
        }
        return out
    }

    /// Reflection-pad the last (length) axis of an NCL tensor.
    public static func reflectionPad1d(_ x: [Float], shape: [Int],
                                       left: Int, right: Int) -> (data: [Float], shape: [Int]) {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        let lOut = l + left + right
        precondition(left < l && right < l || (left == 0 && right == 0),
                     "reflectionPad1d: pad larger than input length")
        var out = [Float](repeating: 0, count: n * c * lOut)
        for b in 0..<n {
            for ch in 0..<c {
                let inBase = (b * c + ch) * l
                let outBase = (b * c + ch) * lOut
                for t in 0..<lOut {
                    var src = t - left
                    // Reflect without repeating the edge sample.
                    if src < 0 { src = -src }
                    if src >= l { src = 2 * (l - 1) - src }
                    src = min(max(src, 0), l - 1)
                    out[outBase + t] = x[inBase + src]
                }
            }
        }
        return (out, [n, c, lOut])
    }

    /// Constant-pad the last axis of an NCL tensor with zeros.
    public static func zeroPad1d(_ x: [Float], shape: [Int],
                                 left: Int, right: Int) -> (data: [Float], shape: [Int]) {
        let (n, c, l) = (shape[0], shape[1], shape[2])
        let lOut = l + left + right
        var out = [Float](repeating: 0, count: n * c * lOut)
        for b in 0..<n {
            for ch in 0..<c {
                let inBase = (b * c + ch) * l
                let outBase = (b * c + ch) * lOut
                for t in 0..<l { out[outBase + left + t] = x[inBase + t] }
            }
        }
        return (out, [n, c, lOut])
    }
}

/// Weight-normalized convolution helper. PyTorch `weight_norm` stores a
/// magnitude `weight_g` and direction `weight_v`; the effective weight
/// is `g * v / ||v||` reduced over every axis except dim 0.
public enum WeightNorm {
    /// Reconstruct an effective conv weight from `(g, v)`.
    /// `v` has shape `[d0, d1, d2]`; `g` broadcasts over dim 0.
    public static func effectiveWeight(g: [Float], v: [Float],
                                       shape: [Int], exceptDim: Int = 0) -> [Float] {
        let (d0, d1, d2) = (shape[0], shape[1], shape[2])
        var out = [Float](repeating: 0, count: v.count)
        let eps: Float = 1e-12
        if exceptDim == 0 {
            // norm reduced over dims 1,2 -> one scalar per d0 slice.
            for a in 0..<d0 {
                var ss: Float = 0
                let sliceBase = a * d1 * d2
                for i in 0..<(d1 * d2) {
                    let val = v[sliceBase + i]
                    ss += val * val
                }
                let norm = sqrtf(ss) + eps
                // g is stored with the same broadcast shape; it has one
                // value per d0 slice (others are 1-sized).
                let gVal = g[a * (g.count / d0)]
                let scale = gVal / norm
                for i in 0..<(d1 * d2) {
                    out[sliceBase + i] = v[sliceBase + i] * scale
                }
            }
        } else {
            fatalError("WeightNorm.effectiveWeight: exceptDim \(exceptDim) unsupported")
        }
        return out
    }
}
