// GPT-OSS MoE feed-forward — block-sparse experts + the MXFP4 codec.
//
// GPT-OSS-20B's feed-forward half is a mixture-of-experts block:
//
//   logits        = router(x) + router_bias        // [numExperts]
//   (idx, weight) = topK(logits) then softmax(idx)  // topK-then-softmax
//   y             = Σ_{e ∈ idx} weight_e · expert_e(x)
//
// Each expert is a *clipped* SwiGLU (the GPT-OSS gating form):
//
//   gate = gate_proj(x) + gate_bias
//   up   = up_proj(x)   + up_bias
//   gate = clip(gate, max:  limit)               // swiglu_limit, 7.0
//   up   = clip(up,   -limit … limit)
//   glu  = gate · sigmoid(1.702 · gate)          // α-swish
//   act  = glu · (up + 1)                        // the "+1" GPT-OSS form
//   y    = down_proj(act) + down_bias
//
// FFAI's element-wise GPU ops have no `clip`, and the per-expert
// activation vector is tiny (`intermediate` = 2880, topK = 4 experts).
// So the gate/up GEMVs run on the GPU, the clip + α-swish + `(up+1)`
// fold runs on the host (one small CPU pass per selected expert), and
// the down GEMV runs back on the GPU. The MoE layer already has a CPU
// sync point for the router top-K, so the extra readbacks are free.
//
// ─── The MXFP4 expert codec ──────────────────────────────────────────
//
// The published GPT-OSS-20B checkpoints ship the MoE expert weights
// MXFP4-quantized — a 4-bit microscaling float format:
//
//   weight  [E, outDim, inDim/8]   uint32  — 8 fp4 codes per word
//   scales  [E, outDim, inDim/32]  uint8   — one e8m0 exponent / 32 codes
//   bias    [E, outDim]            f16     — per-output-row bias
//
// Each fp4 code is a 4-bit index into a fixed 16-entry lookup table
// (`mxfp4LUT`); the dequantized value is `LUT[code] · 2^(e8m0 - 127)`.
//
// FFAI's `QuantizedLinear` speaks mlx *affine* int4, not MXFP4. Rather
// than add an MXFP4 GPU kernel, the experts are transcoded at load
// time: each 32-value MXFP4 group is dequantized to fp32 and re-packed
// as an affine int4 group (`w = q·scale + bias`, q ∈ 0…15). The
// re-packed form stays ~4-bit, so a 20B-param model still fits in
// ~10 GB — dequantizing all experts to fp16 instead would need ~38 GB.
// The transcode is lossy (MXFP4's non-uniform LUT vs affine's uniform
// grid) but the per-group error is small and the integration test's
// coherence checker tolerates it.

import Foundation
import Metal

// ─── MXFP4 constants ─────────────────────────────────────────────────

private enum MXFP4 {
    /// MXFP4 group size — one e8m0 scale byte per 32 fp4 codes.
    static let groupSize = 4 << 3
    /// fp4 codes packed per uint32 word.
    static let codesPerWord = 8
    /// The 16-entry MXFP4 value lookup table. Code → fp value; the
    /// top bit is sign. Matches mlx-lm's `gpt_oss` dequant table.
    static let lut: [Float] = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    /// e8m0 exponent bias.
    static let e8m0Bias = 127
}

/// The α coefficient of GPT-OSS's swish gate (`gate · sigmoid(α·gate)`).
private let gptOSSSwishAlpha: Float = 1.702

// ─── buildGPTOSSMoE — load + transcode the MoE block ─────────────────

/// Build one layer's GPT-OSS MoE feed-forward block. Reads the biased
/// router and the three MXFP4-packed expert projections, transcodes
/// each expert to affine int4, and wires the `GPTOSSExpert` list.
func buildGPTOSSMoE(
    prefix p: String, weights: SafeTensorsBundle, quantMap: GPTOSSQuantMap,
    hidden: Int, intermediate: Int,
    numExperts: Int, topK: Int, swigluLimit: Float,
    dtype: DType, device: Device
) throws -> GPTOSSMoELayer {
    // Router: hidden → numExperts logits. Carries a `.bias` projection
    // bias and is affine-quantized on the published checkpoints (the
    // experts are the only MXFP4 tensors).
    let router: GPTOSSBiasedLinear = {
        let inner: AnyLinear
        if let q = quantMap.config(for: "\(p).router"),
           weights.isQuantized("\(p).router"),
           let t = try? weights.quantizedTriplet("\(p).router") {
            inner = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: q.bits, groupSize: q.groupSize))
        } else {
            inner = AnyLinear(Linear(
                weight: (try? weights.tensor(named: "\(p).router.weight"))!))
        }
        let bias = weights.has("\(p).router.bias")
            ? castGPTOSSTensor(try! weights.tensor(named: "\(p).router.bias"),
                               to: dtype, device: device)
            : nil
        return GPTOSSBiasedLinear(linear: inner, bias: bias)
    }()

    // Expert projections. The checkpoint ships them stacked over the
    // expert axis: gate_proj / up_proj are [E, intermediate, hidden],
    // down_proj is [E, hidden, intermediate].
    let gate = try transcodeStackedExperts(
        base: "\(p).experts.gate_proj", in: weights,
        numExperts: numExperts, outDim: intermediate, inDim: hidden,
        dtype: dtype, device: device)
    let up = try transcodeStackedExperts(
        base: "\(p).experts.up_proj", in: weights,
        numExperts: numExperts, outDim: intermediate, inDim: hidden,
        dtype: dtype, device: device)
    let down = try transcodeStackedExperts(
        base: "\(p).experts.down_proj", in: weights,
        numExperts: numExperts, outDim: hidden, inDim: intermediate,
        dtype: dtype, device: device)

    var experts: [GPTOSSExpert] = []
    experts.reserveCapacity(numExperts)
    for e in 0..<numExperts {
        experts.append(GPTOSSExpert(
            gateProj: gate[e], upProj: up[e], downProj: down[e]))
    }

    return GPTOSSMoELayer(
        router: router, experts: experts,
        topK: topK, hidden: hidden, swigluLimit: swigluLimit,
        dtype: dtype)
}

// ─── transcodeStackedExperts — MXFP4 → affine int4 ───────────────────
//
// The transcode fits a PER-GROUP affine grid: for each 32-value MXFP4
// group it finds the group's min / max dequantized value and lays a
// 16-level uniform grid across exactly that range —
//
//   affine scale = (groupMax − groupMin) / 15
//   affine bias  = groupMin
//   affine code  = round((LUT[code]·mxScale − bias) / scale)
//
// A per-group fit is materially more accurate than a fixed full-range
// (±6) grid: most trained-weight groups occupy a small sub-range, and
// the fixed grid would waste most of its 16 levels on the unused
// extremes — coarse near-zero quantization that flattens the MoE
// output enough to push greedy decode into repetition loops.
//
// The per-group fit is kept fast (the transcode covers ~20B codes) by
// scanning each group's 16 packed bytes through the 16-entry MXFP4 LUT
// with `while` loops over raw pointers — a debug build does NOT inline
// `Range`'s `formIndex` iterator witness, which otherwise dominates.
// The e8m0 microscale is looked up from a 256-entry `2^(e8m0−127)`
// table so no `exp2` runs in the hot loop.

/// `2^(e8m0 − 127)` for every possible e8m0 byte — the MXFP4 group
/// microscale. Precomputed so the transcode never calls `exp2`.
private let mxfp4MicroScaleForByte: [Float] = {
    (0..<256).map { e8m0 in
        Float(Foundation.exp2(Double(e8m0 - MXFP4.e8m0Bias)))
    }
}()

/// A transcoded affine-int4 expert projection: the `QuantizedLinear`
/// triplet plus the per-output-row bias read straight from the
/// checkpoint.
struct GPTOSSExpertProjection {
    let linear: QuantizedLinear
    /// Per-output-row bias `[outDim]`, in the activation dtype.
    let bias: Tensor
}

/// Transcode a stacked `[E, outDim, inDim]` MXFP4 expert tensor into
/// `E` per-expert affine-int4 `QuantizedLinear`s (+ their biases).
private func transcodeStackedExperts(
    base: String, in weights: SafeTensorsBundle,
    numExperts: Int, outDim: Int, inDim: Int,
    dtype: DType, device: Device
) throws -> [GPTOSSExpertProjection] {
    let packed = try weights.tensor(named: "\(base).weight")    // u32
    let scales = try weights.tensor(named: "\(base).scales")    // u8
    let biases = try weights.tensor(named: "\(base).bias")      // f16

    precondition(packed.dtype == .u32,
                 "GPT-OSS MoE: \(base).weight expected u32, got \(packed.dtype)")
    precondition(scales.dtype == .u8,
                 "GPT-OSS MoE: \(base).scales expected u8, got \(scales.dtype)")
    precondition(inDim % MXFP4.groupSize == 0,
                 "GPT-OSS MoE: inDim \(inDim) must be a multiple of "
                 + "MXFP4 group size \(MXFP4.groupSize)")

    let wordsPerRow = inDim / MXFP4.codesPerWord     // packed u32 / output row
    let groupsPerRow = inDim / MXFP4.groupSize       // scale bytes / output row
    let bytesPerRow = wordsPerRow * 4                // packed bytes / output row
    precondition(
        packed.elementCount == numExperts * outDim * wordsPerRow,
        "GPT-OSS MoE: \(base).weight count mismatch — got \(packed.shape)")
    precondition(
        scales.elementCount == numExperts * outDim * groupsPerRow,
        "GPT-OSS MoE: \(base).scales count mismatch — got \(scales.shape)")

    let biasFloats = readGPTOSSFloats(biases)
    precondition(biasFloats.count == numExperts * outDim,
                 "GPT-OSS MoE: \(base).bias count mismatch")

    let rowWords = wordsPerRow
    let rowGroups = groupsPerRow
    // The affine group size equals the MXFP4 group size (32) and both
    // pack 8 codes per u32, so the dst word/byte layout matches the src
    // 1:1 — the transcode is a pure nibble remap, no repacking.

    // Pre-allocate every expert's device-resident affine triplet so the
    // transcode writes straight into GPU memory. Experts are
    // independent — the outer loop parallelizes across cores.
    let weightTs = (0..<numExperts).map { _ in
        Tensor.empty(shape: [outDim, rowWords], dtype: .u32, device: device)
    }
    let scaleTs = (0..<numExperts).map { _ in
        Tensor.empty(shape: [outDim, rowGroups], dtype: dtype, device: device)
    }
    let biasTs = (0..<numExperts).map { _ in
        Tensor.empty(shape: [outDim, rowGroups], dtype: dtype, device: device)
    }

    // Bytes per MXFP4 group: 32 codes / 2 codes-per-byte = 16.
    let bytesPerGroup = MXFP4.groupSize / 2

    DispatchQueue.concurrentPerform(iterations: numExperts) { e in
        // Pointer views derived inside the closure — `Tensor` is
        // `@unchecked Sendable`, raw pointers are not. The packed
        // source is read-only; each expert's destinations are disjoint.
        // The src/dst weight word layouts are identical (the remap is
        // nibble-for-nibble in place). The hot loops are `while` loops
        // over raw pointers: a debug build does NOT inline `Range`'s
        // `formIndex` iterator witness, which dominates a `for i in
        // 0..<n` over a ~20-element-billion transcode.
        let weightBytesPerExpert = outDim * bytesPerRow
        let groupsPerExpert = outDim * rowGroups
        let packedBytes = packed.buffer.contents()
            .advanced(by: packed.offset + e * weightBytesPerExpert)
            .assumingMemoryBound(to: UInt8.self)
        let scaleHost = scales.buffer.contents()
            .advanced(by: scales.offset + e * groupsPerExpert)
            .assumingMemoryBound(to: UInt8.self)
        let wDstBytes = weightTs[e].buffer.contents()
            .advanced(by: weightTs[e].offset)
            .assumingMemoryBound(to: UInt8.self)
        let sDst = scaleTs[e].buffer.contents().advanced(by: scaleTs[e].offset)
        let bDst = biasTs[e].buffer.contents().advanced(by: biasTs[e].offset)

        MXFP4.lut.withUnsafeBufferPointer { lut in
        mxfp4MicroScaleForByte.withUnsafeBufferPointer { microTbl in
            // Per-group affine fit. `g` indexes groups; each group is
            // `bytesPerGroup` packed bytes (= 32 codes) and one e8m0
            // scale byte. Output scale/bias dtype branches once.
            let dt = dtype
            var g = 0
            while g < groupsPerExpert {
                let mxScale = microTbl[Int(scaleHost[g])]
                let byteBase = g * bytesPerGroup

                // Pass 1: min / max of the group's LUT values.
                var lo: Float = 1e30
                var hi: Float = -1e30
                var b = 0
                while b < bytesPerGroup {
                    let byte = packedBytes[byteBase + b]
                    let v0 = lut[Int(byte & 0x0F)]
                    let v1 = lut[Int(byte >> 4)]
                    if v0 < lo { lo = v0 }
                    if v0 > hi { hi = v0 }
                    if v1 < lo { lo = v1 }
                    if v1 > hi { hi = v1 }
                    b &+= 1
                }
                // Affine grid over [lo, hi]·mxScale, 16 levels.
                let affBias = lo * mxScale
                let span = (hi - lo) * mxScale
                let affScale = span > 0 ? span / 15.0 : 0
                let invScale: Float = affScale > 0 ? 1.0 / affScale : 0

                // Write scale / bias in the activation dtype.
                switch dt {
                case .f16:
                    sDst.assumingMemoryBound(to: Float16.self)[g] =
                        Float16(affScale)
                    bDst.assumingMemoryBound(to: Float16.self)[g] =
                        Float16(affBias)
                case .bf16:
                    sDst.assumingMemoryBound(to: UInt16.self)[g] =
                        bf16Bits(affScale)
                    bDst.assumingMemoryBound(to: UInt16.self)[g] =
                        bf16Bits(affBias)
                case .f32:
                    sDst.assumingMemoryBound(to: Float.self)[g] = affScale
                    bDst.assumingMemoryBound(to: Float.self)[g] = affBias
                default:
                    fatalError("GPT-OSS MoE: unsupported dtype \(dt)")
                }

                // Pass 2: remap each code to its affine int4 quantum.
                // affine code  q = round((LUT[code]·mxScale − affBias)
                //                         / affScale)
                //                = round((LUT[code] − lo)·mxScale·invScale)
                let codeScale = mxScale * invScale
                b = 0
                while b < bytesPerGroup {
                    let byte = packedBytes[byteBase + b]
                    var q0 = Int(((lut[Int(byte & 0x0F)] - lo) * codeScale)
                        .rounded())
                    var q1 = Int(((lut[Int(byte >> 4)] - lo) * codeScale)
                        .rounded())
                    if q0 < 0 { q0 = 0 }; if q0 > 15 { q0 = 15 }
                    if q1 < 0 { q1 = 0 }; if q1 > 15 { q1 = 15 }
                    wDstBytes[byteBase + b] = UInt8(q0 | (q1 << 4))
                    b &+= 1
                }
                g &+= 1
            }
        }
        }
    }

    var out: [GPTOSSExpertProjection] = []
    out.reserveCapacity(numExperts)
    for e in 0..<numExperts {
        let qlinear = QuantizedLinear(
            weight: weightTs[e], scales: scaleTs[e], biases: biasTs[e],
            bits: 4, groupSize: MXFP4.groupSize)

        // Per-output-row expert bias, in the activation dtype.
        let rowBias = Tensor.empty(shape: [outDim], dtype: dtype, device: device)
        writeGPTOSSFloats(Array(biasFloats[(e * outDim)..<((e + 1) * outDim)]),
                          into: rowBias)

        out.append(GPTOSSExpertProjection(linear: qlinear, bias: rowBias))
    }
    return out
}

/// Round a `Float` to its bf16 bit pattern (round-to-nearest-even on
/// the truncated low 16 bits). Used by the transcode's direct-to-tensor
/// scale/bias writes.
@inline(__always)
private func bf16Bits(_ v: Float) -> UInt16 {
    let bits = v.bitPattern
    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
    return UInt16(rounded >> 16)
}

// ─── GPTOSSExpert — one clipped-SwiGLU expert ────────────────────────

/// A single MoE expert: the three affine-int4 projections + their
/// per-output-row biases. The clipped-SwiGLU activation runs host-side
/// (see the file header) so the expert exposes the GEMVs separately.
public final class GPTOSSExpert: Module {
    let gateProj, upProj, downProj: GPTOSSExpertProjection

    init(gateProj: GPTOSSExpertProjection,
         upProj: GPTOSSExpertProjection,
         downProj: GPTOSSExpertProjection) {
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gateProj.linear.parameters() {
            out.append(("gate_proj.\(k)", v))
        }
        out.append(("gate_proj.bias", gateProj.bias))
        for (k, v) in upProj.linear.parameters() {
            out.append(("up_proj.\(k)", v))
        }
        out.append(("up_proj.bias", upProj.bias))
        for (k, v) in downProj.linear.parameters() {
            out.append(("down_proj.\(k)", v))
        }
        out.append(("down_proj.bias", downProj.bias))
        return out
    }
}

// ─── GPTOSSMoELayer — the block-sparse MoE feed-forward layer ────────
//
// `decode` runs: router GEMV → CPU top-K + softmax → per selected
// expert {gate/up GEMV → host clipped-SwiGLU → down GEMV} → combine.
//
// IMPORTANT — command-buffer contract. `decode` commits the passed
// `cmd` (the router needs the logits on the CPU, and the per-expert
// activation is host-side). The enclosing `GPTOSSLayer` obtains a fresh
// buffer afterwards. Mirrors `MoELayer`'s contract.

public final class GPTOSSMoELayer: Module {
    public let router: GPTOSSBiasedLinear
    public let experts: [GPTOSSExpert]
    public let topK: Int
    public let hidden: Int
    public let swigluLimit: Float
    public let dtype: DType

    init(router: GPTOSSBiasedLinear, experts: [GPTOSSExpert],
         topK: Int, hidden: Int, swigluLimit: Float, dtype: DType) {
        precondition(topK > 0 && topK <= experts.count,
                     "GPTOSSMoELayer: topK \(topK) out of range "
                     + "1…\(experts.count)")
        self.router = router
        self.experts = experts
        self.topK = topK
        self.hidden = hidden
        self.swigluLimit = swigluLimit
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in router.parameters() { out.append(("router.\(k)", v)) }
        for (e, expert) in experts.enumerated() {
            for (k, v) in expert.parameters() {
                out.append(("experts.\(e).\(k)", v))
            }
        }
        return out
    }

    /// Single-token MoE forward. Commits `cmd` (router CPU readback)
    /// and returns a fully-resident `[hidden]` tensor produced on a
    /// fresh, locally-committed buffer.
    func decode(_ x: Tensor, cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(x.elementCount == hidden,
                     "GPTOSSMoELayer.decode: input has \(x.elementCount) "
                     + "elements, expected hidden \(hidden)")

        // ── Router GEMV on the caller's buffer, then CPU sync ─────────
        let logitsTensor = router(x, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── CPU routing — top-K of raw logits, softmax over the K ─────
        // GPT-OSS routes top-K of the raw router logits then softmaxes
        // just those K (mlx-lm's `topK` then `softmax`).
        let logits = logitsTensor.toFloatArray()
        let order = (0..<logits.count).sorted { a, b in
            if logits[a] != logits[b] { return logits[a] > logits[b] }
            return a < b
        }
        let idx = Array(order.prefix(topK))
        let pickedLogits = idx.map { logits[$0] }
        let combineWeights = softmaxSmall(pickedLogits)

        // ── Per-expert clipped-SwiGLU ─────────────────────────────────
        // Each `runExpert` runs on its own command buffers (it commits
        // mid-way for the host-side activation) and returns a resident
        // tensor. The weighted combine then runs on one fresh buffer.
        var expertOuts: [Tensor] = []
        expertOuts.reserveCapacity(idx.count)
        for expertId in idx {
            expertOuts.append(runExpert(experts[expertId], x: x,
                                        device: device))
        }

        let work = device.makeCommandBuffer()
        var accumulator: Tensor?
        for (slot, expertOut) in expertOuts.enumerated() {
            let weightTensor = Tensor.filled(combineWeights[slot],
                                             shape: [hidden], dtype: dtype,
                                             device: device)
            let scaled = Ops.mul(expertOut, weightTensor, on: work)
            accumulator = accumulator.map { Ops.add($0, scaled, on: work) }
                ?? scaled
        }
        let result = accumulator!     // topK ≥ 1
        work.commit()
        work.waitUntilCompleted()
        return result
    }

    /// Run one expert: gate/up GEMVs on the GPU, the clipped-SwiGLU
    /// activation on the host, the down GEMV on the GPU. The clip has
    /// no GPU op in FFAI and the activation vector is small, so the
    /// host fold is the simplest correct path (see the file header).
    /// Owns its command buffers; returns a fully-resident tensor.
    private func runExpert(_ expert: GPTOSSExpert, x: Tensor,
                           device: Device) -> Tensor {
        // ── GPU phase 1: gate / up GEMVs (+ per-row expert bias) ──────
        let cmd = device.makeCommandBuffer()
        let gate = Ops.add(expert.gateProj.linear(x, on: cmd),
                           expert.gateProj.bias, on: cmd)
        let up = Ops.add(expert.upProj.linear(x, on: cmd),
                         expert.upProj.bias, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Host phase: clipped-SwiGLU ────────────────────────────────
        //   gate = clip(gate, max: limit)
        //   up   = clip(up, -limit … limit)
        //   glu  = gate · sigmoid(α·gate)
        //   act  = glu · (up + 1)
        let gateHost = readGPTOSSFloats(gate)
        let upHost = readGPTOSSFloats(up)
        let limit = swigluLimit
        var act = [Float](repeating: 0, count: gateHost.count)
        for i in 0..<gateHost.count {
            var g = gateHost[i]
            var u = upHost[i]
            if g > limit { g = limit }
            if u > limit { u = limit }
            if u < -limit { u = -limit }
            let glu = g / (1.0 + Foundation.exp(-gptOSSSwishAlpha * g))
            act[i] = glu * (u + 1.0)
        }

        // ── GPU phase 2: down GEMV (+ per-row bias) on a fresh buffer ─
        let phase2 = device.makeCommandBuffer()
        let actTensor = Tensor.empty(shape: [act.count], dtype: dtype,
                                     device: device)
        writeGPTOSSFloats(act, into: actTensor)
        let down = Ops.add(expert.downProj.linear(actTensor, on: phase2),
                           expert.downProj.bias, on: phase2)
        phase2.commit()
        phase2.waitUntilCompleted()
        return down
    }
}

/// Numerically-stable softmax over a small host vector.
private func softmaxSmall(_ x: [Float]) -> [Float] {
    guard let maxV = x.max() else { return [] }
    let exps = x.map { Foundation.exp($0 - maxV) }
    let sum = exps.reduce(0, +)
    guard sum > 0 else {
        return [Float](repeating: 1 / Float(x.count), count: x.count)
    }
    return exps.map { $0 / sum }
}
