// VADCompute — CPU numeric primitives shared by the VAD model families
// (SileroVAD, SmartTurn, Sortformer).
//
// FFAI's GPU `Ops` surface targets the causal-LM hot path: GEMV,
// elementwise, RMSNorm, SSM step, decode SDPA. The VAD models need a
// different primitive set — strided/padded conv1d, an LSTM cell,
// full-sequence multi-head attention, layer norm, GELU. Those kernels
// do not exist in metaltile today, and inventing them is out of scope
// (and explicitly disallowed by the porting contract).
//
// VAD models are tiny — SileroVAD ~1.5M params, SmartTurn ~8M,
// Sortformer ~30M — and run on short clips, so a straight CPU forward
// is correct and fast enough (sub-second for a few seconds of audio).
// This file is the documented CPU fallback path the porting contract
// sanctions. Every routine here is a plain `[Float]` numeric kernel; no
// MTLBuffer / GPU dispatch is involved. Weights are still loaded into
// `Tensor` (GPU-resident) via SafeTensors and copied to host once at
// load time through `Tensor.toFloatArray()`.

import Foundation
import Accelerate

// ─── Tensor host-readback ────────────────────────────────────────────

extension Tensor {
    /// Read the tensor's contents into a host `[Float]`, converting from
    /// whatever storage dtype it uses. VAD weight tensors are small;
    /// this runs once per weight at load time.
    func toFloatArray() -> [Float] {
        switch dtype {
        case .f32:
            return toArray(as: Float.self)
        case .f16:
            return toArray(as: Float16.self).map { Float($0) }
        case .bf16:
            let bits = toArray(as: UInt16.self)
            return bits.map { Float(bitPattern: UInt32($0) << 16) }
        case .i32:
            return toArray(as: Int32.self).map { Float($0) }
        case .u32:
            return toArray(as: UInt32.self).map { Float($0) }
        case .i8:
            return toArray(as: Int8.self).map { Float($0) }
        case .u8:
            return toArray(as: UInt8.self).map { Float($0) }
        }
    }
}

// ─── Activations ─────────────────────────────────────────────────────

enum VADMath {
    /// Numerically-stable logistic sigmoid.
    static func sigmoid(_ x: Float) -> Float {
        if x >= 0 {
            return 1 / (1 + Foundation.exp(-x))
        } else {
            let e = Foundation.exp(x)
            return e / (1 + e)
        }
    }

    static func sigmoid(_ xs: [Float]) -> [Float] { xs.map(sigmoid) }

    /// ReLU, in place.
    static func reluInPlace(_ xs: inout [Float]) {
        for i in xs.indices where xs[i] < 0 { xs[i] = 0 }
    }

    /// Exact GELU (erf form) — matches PyTorch's default `nn.GELU()`.
    static func gelu(_ x: Float) -> Float {
        0.5 * x * (1 + erff(x / Float(2.0).squareRoot()))
    }

    static func gelu(_ xs: [Float]) -> [Float] { xs.map(gelu) }

    static func tanhActivation(_ xs: [Float]) -> [Float] {
        xs.map { Foundation.tanh($0) }
    }

    /// Single-precision error function.
    private static func erff(_ x: Float) -> Float {
        // Foundation exposes the C `erff`; call through Double for
        // portability if the symbol is unavailable.
        Float(Foundation.erf(Double(x)))
    }

    /// In-place softmax over a contiguous slice.
    static func softmaxInPlace(_ xs: inout [Float], range: Range<Int>) {
        var maxV = -Float.greatestFiniteMagnitude
        for i in range where xs[i] > maxV { maxV = xs[i] }
        var sum: Float = 0
        for i in range {
            let e = Foundation.exp(xs[i] - maxV)
            xs[i] = e
            sum += e
        }
        if sum > 0 {
            for i in range { xs[i] /= sum }
        }
    }
}

// ─── Linear (matmul + optional bias) ─────────────────────────────────

/// Dense layer with row-major `weight` of shape `[outFeatures,
/// inFeatures]` and optional `bias` of length `outFeatures`. Computes
/// `y = x · Wᵀ + b` for a batch of `rows` input vectors.
struct VADLinear {
    let weight: [Float]      // [out, in]
    let bias: [Float]?       // [out] or nil
    let inFeatures: Int
    let outFeatures: Int

    init(weight: [Float], bias: [Float]?, inFeatures: Int, outFeatures: Int) {
        precondition(weight.count == inFeatures * outFeatures,
                     "VADLinear: weight count \(weight.count) != \(inFeatures)*\(outFeatures)")
        if let bias { precondition(bias.count == outFeatures, "VADLinear: bias length mismatch") }
        self.weight = weight
        self.bias = bias
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
    }

    /// Apply to a single `[inFeatures]` vector → `[outFeatures]`.
    func apply(_ x: [Float]) -> [Float] {
        precondition(x.count == inFeatures, "VADLinear: input length mismatch")
        var out = bias ?? [Float](repeating: 0, count: outFeatures)
        weight.withUnsafeBufferPointer { w in
            x.withUnsafeBufferPointer { xp in
                for o in 0..<outFeatures {
                    var acc: Float = 0
                    let base = o * inFeatures
                    vDSP_dotpr(w.baseAddress! + base, 1, xp.baseAddress!, 1,
                               &acc, vDSP_Length(inFeatures))
                    out[o] += acc
                }
            }
        }
        return out
    }

    /// Apply to `[rows, inFeatures]` (row-major) → `[rows, outFeatures]`.
    func applyRows(_ x: [Float], rows: Int) -> [Float] {
        precondition(x.count == rows * inFeatures, "VADLinear: rows input length mismatch")
        var out = [Float](repeating: 0, count: rows * outFeatures)
        x.withUnsafeBufferPointer { xp in
            weight.withUnsafeBufferPointer { w in
                for r in 0..<rows {
                    let xBase = r * inFeatures
                    let oBase = r * outFeatures
                    for o in 0..<outFeatures {
                        var acc: Float = 0
                        vDSP_dotpr(xp.baseAddress! + xBase, 1,
                                   w.baseAddress! + o * inFeatures, 1,
                                   &acc, vDSP_Length(inFeatures))
                        out[oBase + o] = acc + (bias?[o] ?? 0)
                    }
                }
            }
        }
        return out
    }
}

// ─── Conv1d ──────────────────────────────────────────────────────────

/// 1-D convolution matching PyTorch `nn.Conv1d`.
///
/// Input / output use the `[channels, length]` layout (channel-major,
/// the PyTorch convention). `weight` is `[outChannels, inChannels,
/// kernelSize]` row-major. `bias`, if present, is `[outChannels]`.
struct VADConv1d {
    let weight: [Float]      // [outC, inC, K]
    let bias: [Float]?       // [outC] or nil
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let stride: Int
    let padding: Int

    init(weight: [Float], bias: [Float]?,
         inChannels: Int, outChannels: Int, kernelSize: Int,
         stride: Int = 1, padding: Int = 0) {
        precondition(weight.count == outChannels * inChannels * kernelSize,
                     "VADConv1d: weight count mismatch")
        if let bias { precondition(bias.count == outChannels, "VADConv1d: bias length mismatch") }
        self.weight = weight
        self.bias = bias
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
    }

    /// Output length for an input of `inLength` samples.
    func outputLength(forInputLength inLength: Int) -> Int {
        (inLength + 2 * padding - kernelSize) / stride + 1
    }

    /// Apply to `[inChannels, inLength]` → `[outChannels, outLength]`.
    func apply(_ x: [Float], inLength: Int) -> (values: [Float], outLength: Int) {
        precondition(x.count == inChannels * inLength, "VADConv1d: input length mismatch")
        let outLen = outputLength(forInputLength: inLength)
        precondition(outLen > 0, "VADConv1d: non-positive output length")
        var out = [Float](repeating: 0, count: outChannels * outLen)

        x.withUnsafeBufferPointer { xp in
            weight.withUnsafeBufferPointer { w in
                for oc in 0..<outChannels {
                    let outBase = oc * outLen
                    let wOcBase = oc * inChannels * kernelSize
                    let b = bias?[oc] ?? 0
                    for t in 0..<outLen {
                        var acc: Float = b
                        let inStart = t * stride - padding
                        for ic in 0..<inChannels {
                            let xIcBase = ic * inLength
                            let wIcBase = wOcBase + ic * kernelSize
                            for k in 0..<kernelSize {
                                let idx = inStart + k
                                if idx >= 0 && idx < inLength {
                                    acc += w[wIcBase + k] * xp[xIcBase + idx]
                                }
                            }
                        }
                        out[outBase + t] = acc
                    }
                }
            }
        }
        return (out, outLen)
    }
}

// ─── LayerNorm ───────────────────────────────────────────────────────

/// Layer normalization over the last dimension, matching PyTorch
/// `nn.LayerNorm`.
struct VADLayerNorm {
    let weight: [Float]      // [dim]
    let bias: [Float]        // [dim]
    let dim: Int
    let eps: Float

    init(weight: [Float], bias: [Float], dim: Int, eps: Float = 1e-5) {
        precondition(weight.count == dim && bias.count == dim, "VADLayerNorm: param length mismatch")
        self.weight = weight
        self.bias = bias
        self.dim = dim
        self.eps = eps
    }

    /// Normalize each `[dim]`-length row of `[rows, dim]` input.
    func applyRows(_ x: [Float], rows: Int) -> [Float] {
        precondition(x.count == rows * dim, "VADLayerNorm: rows input length mismatch")
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
            let inv = 1 / (variance + eps).squareRoot()
            for i in 0..<dim {
                out[base + i] = (x[base + i] - mean) * inv * weight[i] + bias[i]
            }
        }
        return out
    }

    /// Normalize a single `[dim]` vector.
    func apply(_ x: [Float]) -> [Float] { applyRows(x, rows: 1) }
}

// ─── LSTM cell ───────────────────────────────────────────────────────

/// Single-layer unidirectional LSTM matching PyTorch `nn.LSTM`.
///
/// Weights follow PyTorch's packed layout: `weightIH` is `[4*hidden,
/// input]`, `weightHH` is `[4*hidden, hidden]`, biases are `[4*hidden]`.
/// Gate order within the `4*hidden` block is input, forget, cell,
/// output (the PyTorch `i, f, g, o` convention).
struct VADLSTM {
    let weightIH: [Float]    // [4H, input]
    let weightHH: [Float]    // [4H, hidden]
    let biasIH: [Float]?     // [4H]
    let biasHH: [Float]?     // [4H]
    let inputSize: Int
    let hiddenSize: Int

    init(weightIH: [Float], weightHH: [Float],
         biasIH: [Float]?, biasHH: [Float]?,
         inputSize: Int, hiddenSize: Int) {
        precondition(weightIH.count == 4 * hiddenSize * inputSize, "VADLSTM: weightIH count mismatch")
        precondition(weightHH.count == 4 * hiddenSize * hiddenSize, "VADLSTM: weightHH count mismatch")
        self.weightIH = weightIH
        self.weightHH = weightHH
        self.biasIH = biasIH
        self.biasHH = biasHH
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
    }

    /// Run the LSTM over a `[seqLen, inputSize]` sequence.
    ///
    /// Returns the full hidden-state sequence `[seqLen, hiddenSize]`
    /// plus the final `(hidden, cell)` state for streaming continuation.
    func run(_ x: [Float], seqLen: Int,
             initialHidden: [Float]? = nil,
             initialCell: [Float]? = nil)
        -> (hiddenSeq: [Float], finalHidden: [Float], finalCell: [Float])
    {
        precondition(x.count == seqLen * inputSize, "VADLSTM: input length mismatch")
        let H = hiddenSize
        var h = initialHidden ?? [Float](repeating: 0, count: H)
        var c = initialCell ?? [Float](repeating: 0, count: H)
        precondition(h.count == H && c.count == H, "VADLSTM: state length mismatch")

        var hiddenSeq = [Float](repeating: 0, count: seqLen * H)
        var gates = [Float](repeating: 0, count: 4 * H)

        x.withUnsafeBufferPointer { xp in
            weightIH.withUnsafeBufferPointer { wih in
                weightHH.withUnsafeBufferPointer { whh in
                    for t in 0..<seqLen {
                        let xBase = t * inputSize
                        // gates = Wih·x + Whh·h + bih + bhh
                        for g in 0..<(4 * H) {
                            var acc: Float = (biasIH?[g] ?? 0) + (biasHH?[g] ?? 0)
                            var partial: Float = 0
                            vDSP_dotpr(wih.baseAddress! + g * inputSize, 1,
                                       xp.baseAddress! + xBase, 1,
                                       &partial, vDSP_Length(inputSize))
                            acc += partial
                            h.withUnsafeBufferPointer { hp in
                                var hp2: Float = 0
                                vDSP_dotpr(whh.baseAddress! + g * H, 1,
                                           hp.baseAddress!, 1,
                                           &hp2, vDSP_Length(H))
                                acc += hp2
                            }
                            gates[g] = acc
                        }
                        // Split gates: i, f, g, o.
                        for j in 0..<H {
                            let i = VADMath.sigmoid(gates[j])
                            let f = VADMath.sigmoid(gates[H + j])
                            let gC = Foundation.tanh(gates[2 * H + j])
                            let o = VADMath.sigmoid(gates[3 * H + j])
                            let newC = f * c[j] + i * gC
                            c[j] = newC
                            h[j] = o * Foundation.tanh(newC)
                        }
                        for j in 0..<H { hiddenSeq[t * H + j] = h[j] }
                    }
                }
            }
        }
        return (hiddenSeq, h, c)
    }
}

// ─── STFT + mel spectrogram ──────────────────────────────────────────

/// CPU short-time Fourier transform + mel filterbank, sufficient for the
/// VAD audio front-ends (SmartTurn's Whisper-style log-mel, Sortformer's
/// NeMo-style log-mel).
enum VADAudioFrontend {

    /// Periodic Hann window of `size` samples (PyTorch / librosa default
    /// for STFT — `torch.hann_window(periodic=true)`).
    static func hannWindow(size: Int) -> [Float] {
        guard size > 1 else { return [Float](repeating: 1, count: max(size, 0)) }
        var w = [Float](repeating: 0, count: size)
        for n in 0..<size {
            w[n] = 0.5 - 0.5 * cosf(2 * Float.pi * Float(n) / Float(size))
        }
        return w
    }

    /// Naive real-input DFT magnitude-squared (power spectrum) for one
    /// frame. Returns `nFft/2 + 1` power-spectrum bins. O(nFft²) — fine
    /// for the small frame sizes (≤ 512) the VAD front-ends use.
    private static func framePowerSpectrum(_ frame: [Float], nFft: Int) -> [Float] {
        let nBins = nFft / 2 + 1
        var power = [Float](repeating: 0, count: nBins)
        for k in 0..<nBins {
            var re: Float = 0
            var im: Float = 0
            let w = -2 * Float.pi * Float(k) / Float(nFft)
            for n in 0..<frame.count {
                let angle = w * Float(n)
                re += frame[n] * cosf(angle)
                im += frame[n] * sinf(angle)
            }
            power[k] = re * re + im * im
        }
        return power
    }

    /// Compute the power spectrogram of `audio`: a `[numFrames,
    /// nFft/2+1]` row-major array.
    ///
    /// Framing matches `torch.stft(center=true)`: the signal is
    /// reflect-padded by `nFft/2` on each side, then windowed frames are
    /// taken every `hopLength` samples. `window` must be `nFft` samples
    /// long (center-pad shorter windows before calling).
    static func powerSpectrogram(_ audio: [Float], window: [Float],
                                 nFft: Int, hopLength: Int)
        -> (values: [Float], numFrames: Int, nBins: Int)
    {
        precondition(window.count == nFft, "powerSpectrogram: window length must equal nFft")
        let nBins = nFft / 2 + 1
        // Center reflect-pad.
        let pad = nFft / 2
        var padded: [Float]
        if pad > 0 && audio.count > pad {
            var left = [Float](repeating: 0, count: pad)
            for i in 0..<pad { left[i] = audio[pad - i] }
            var right = [Float](repeating: 0, count: pad)
            let n = audio.count
            for i in 0..<pad { right[i] = audio[n - 2 - i] }
            padded = left + audio + right
        } else {
            // Too short to reflect — zero-pad instead.
            padded = [Float](repeating: 0, count: pad) + audio + [Float](repeating: 0, count: pad)
        }

        let numFrames = padded.count >= nFft ? (padded.count - nFft) / hopLength + 1 : 0
        var spec = [Float](repeating: 0, count: numFrames * nBins)
        var frame = [Float](repeating: 0, count: nFft)
        for f in 0..<numFrames {
            let start = f * hopLength
            for n in 0..<nFft { frame[n] = padded[start + n] * window[n] }
            let power = framePowerSpectrum(frame, nFft: nFft)
            for k in 0..<nBins { spec[f * nBins + k] = power[k] }
        }
        return (spec, numFrames, nBins)
    }

    /// Convert a frequency in Hz to the Slaney mel scale.
    private static func hzToMelSlaney(_ hz: Float) -> Float {
        let fMin: Float = 0
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = (minLogHz - fMin) / fSp
        let logstep = Float(log(6.4)) / 27.0
        if hz >= minLogHz {
            return minLogMel + logf(hz / minLogHz) / logstep
        }
        return (hz - fMin) / fSp
    }

    /// Convert a Slaney-mel value back to Hz.
    private static func melToHzSlaney(_ mel: Float) -> Float {
        let fMin: Float = 0
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = (minLogHz - fMin) / fSp
        let logstep = Float(log(6.4)) / 27.0
        if mel >= minLogMel {
            return minLogHz * expf(logstep * (mel - minLogMel))
        }
        return fMin + fSp * mel
    }

    /// Build a `[nBins, nMels]` Slaney mel filterbank with Slaney
    /// normalization — matches `librosa.filters.mel(norm='slaney')` and
    /// the `melFilters` helper in mlx-audio-swift.
    static func melFilterbank(sampleRate: Int, nFft: Int, nMels: Int) -> [Float] {
        let nBins = nFft / 2 + 1
        let fMax = Float(sampleRate) / 2
        // Mel-spaced center frequencies (nMels + 2 points).
        let melMin = hzToMelSlaney(0)
        let melMax = hzToMelSlaney(fMax)
        var hzPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            let mel = melMin + (melMax - melMin) * Float(i) / Float(nMels + 1)
            hzPoints[i] = melToHzSlaney(mel)
        }
        // FFT bin center frequencies.
        var binHz = [Float](repeating: 0, count: nBins)
        for k in 0..<nBins { binHz[k] = Float(k) * Float(sampleRate) / Float(nFft) }

        var fb = [Float](repeating: 0, count: nBins * nMels)
        for m in 0..<nMels {
            let lower = hzPoints[m]
            let center = hzPoints[m + 1]
            let upper = hzPoints[m + 2]
            for k in 0..<nBins {
                let hz = binHz[k]
                var weight: Float = 0
                if hz >= lower && hz <= center && center > lower {
                    weight = (hz - lower) / (center - lower)
                } else if hz > center && hz <= upper && upper > center {
                    weight = (upper - hz) / (upper - center)
                }
                fb[k * nMels + m] = max(0, weight)
            }
            // Slaney normalization: scale by 2 / (upper - lower).
            let enorm = 2.0 / (upper - lower)
            for k in 0..<nBins { fb[k * nMels + m] *= enorm }
        }
        return fb
    }

    /// Apply a `[nBins, nMels]` filterbank to a `[numFrames, nBins]`
    /// power spectrogram → `[numFrames, nMels]` mel spectrogram.
    static func applyMelFilterbank(power: [Float], numFrames: Int, nBins: Int,
                                   filterbank: [Float], nMels: Int) -> [Float] {
        precondition(power.count == numFrames * nBins, "applyMelFilterbank: power shape mismatch")
        precondition(filterbank.count == nBins * nMels, "applyMelFilterbank: filterbank shape mismatch")
        var mel = [Float](repeating: 0, count: numFrames * nMels)
        for f in 0..<numFrames {
            let pBase = f * nBins
            let mBase = f * nMels
            for m in 0..<nMels {
                var acc: Float = 0
                for k in 0..<nBins {
                    acc += power[pBase + k] * filterbank[k * nMels + m]
                }
                mel[mBase + m] = acc
            }
        }
        return mel
    }
}

// ─── Full-sequence multi-head attention ──────────────────────────────

/// Compute scaled-dot-product multi-head self-attention for a single
/// batch element.
///
/// `q`, `k`, `v` are each `[seqLen, numHeads*headDim]` row-major (the
/// post-projection layout). Output is `[seqLen, numHeads*headDim]`.
/// `scale` divides the QKᵀ scores before softmax.
func vadMultiHeadAttention(
    q: [Float], k: [Float], v: [Float],
    seqLen: Int, numHeads: Int, headDim: Int, scale: Float
) -> [Float] {
    let dModel = numHeads * headDim
    precondition(q.count == seqLen * dModel, "vadMHA: q length mismatch")
    precondition(k.count == seqLen * dModel, "vadMHA: k length mismatch")
    precondition(v.count == seqLen * dModel, "vadMHA: v length mismatch")
    var out = [Float](repeating: 0, count: seqLen * dModel)
    var scores = [Float](repeating: 0, count: seqLen)

    q.withUnsafeBufferPointer { qp in
        k.withUnsafeBufferPointer { kp in
            v.withUnsafeBufferPointer { vp in
                for head in 0..<numHeads {
                    let hOff = head * headDim
                    for i in 0..<seqLen {
                        let qBase = i * dModel + hOff
                        // scores[j] = (q_i · k_j) / scale
                        for j in 0..<seqLen {
                            let kBase = j * dModel + hOff
                            var dot: Float = 0
                            vDSP_dotpr(qp.baseAddress! + qBase, 1,
                                       kp.baseAddress! + kBase, 1,
                                       &dot, vDSP_Length(headDim))
                            scores[j] = dot / scale
                        }
                        VADMath.softmaxInPlace(&scores, range: 0..<seqLen)
                        // out_i = Σ_j scores[j] · v_j
                        let outBase = i * dModel + hOff
                        for j in 0..<seqLen {
                            let w = scores[j]
                            if w == 0 { continue }
                            let vBase = j * dModel + hOff
                            for d in 0..<headDim {
                                out[outBase + d] += w * vp[vBase + d]
                            }
                        }
                    }
                }
            }
        }
    }
    return out
}
