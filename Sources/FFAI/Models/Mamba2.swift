// Mamba 2 family — selective-SSM / hybrid backbone.
//
// Phase 5e ships the dense decode path:
//   - one MTLCommandBuffer per token (same single-commit shape as Llama)
//   - constant-memory recurrent state via `Mamba2LayerCache`
//   - n_groups=1 only (B / C tensors shared across heads). Grouped B/C
//     (Mamba 2's `n_groups > 1`) and chunked-prefill parallel-scan are
//     follow-ups — see `planning/roadmap.md`.
//
// Architecture per layer (matching the official Mamba 2 SSD form +
// mlx-lm's reference impl):
//
//   residual = x_in
//   h        = RMSNorm(x_in)                                    [hidden]
//   proj     = in_proj(h)                                       [in_proj_dim]
//   z, xBC, dt_raw = split(proj, [d_inner, conv_dim, n_heads])
//   xBC      = silu(conv1d_causal_step(xBC) + bias)             [conv_dim]
//   x, B, C  = split(xBC, [d_inner, state_dim, state_dim])
//   dt       = softplus(dt_raw + dt_bias)                       [n_heads]
//   y        = ssm_step(x, A_eff, B, C, dt, state)              [d_inner]
//   y       += D_tiled * x                                       (skip)
//   y       *= silu(z)                                           (gating)
//   y        = mixer_norm(y)
//   y        = out_proj(y)                                      [hidden]
//   out      = residual + y
//
// Where:
//   d_inner    = expand * hidden
//   conv_dim   = d_inner + 2 * n_groups * state_dim
//   in_proj_dim = 2 * d_inner + 2 * n_groups * state_dim + n_heads
//   A_eff      = -exp(A_log)         (precomputed at load)
//   D_tiled    = D[h] tiled across head_dim → [d_inner]
//                (lets us reuse Ops.mul instead of a broadcast kernel)

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum Mamba2 {
    public static let modelTypes: Set<String> = ["mamba2"]
    public static let architectures: Set<String> = ["Mamba2ForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any Mamba2Variant.Type {
        return Mamba2Dense.self
    }
}

public protocol Mamba2Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Mamba2Model
}

public enum Mamba2Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f): return "Mamba2: required config field missing: \(f)"
        case .unsupportedConfig(let m): return "Mamba2: unsupported config: \(m)"
        }
    }
}

// ─── Mamba2Dense — single dense variant (130m / 370m / 780m / 1.3b / 2.7b)

public struct Mamba2Dense: Mamba2Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Mamba 2 defaults. Greedy by default (the published checkpoints
    /// are base-LM only — no chat tuning yet). Prefill step size is a
    /// non-op until chunked-prefill ships.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 256,
        temperature: 0.0,
        topP: 1.0,
        topK: 0,
        minP: 0.0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Mamba2Model {
        guard let hidden = config.hiddenSize,
              let nLayers = config.numLayers,
              let vocab = config.vocabSize
        else { throw Mamba2Error.missingConfig("hidden / layers / vocab") }
        guard let stateDim = config.int("state_size")
        else { throw Mamba2Error.missingConfig("state_size") }
        guard let nHeads = config.int("num_heads")
        else { throw Mamba2Error.missingConfig("num_heads") }
        guard let headDim = config.int("head_dim")
        else { throw Mamba2Error.missingConfig("head_dim") }
        let convKernel = config.int("conv_kernel") ?? 4
        let expand = config.int("expand") ?? 2
        let nGroups = config.int("n_groups") ?? 1
        let useConvBias = config.bool("use_conv_bias") ?? true
        let useBias = config.bool("use_bias") ?? false
        let eps = config.float("rms_norm_eps") ?? config.float("layer_norm_epsilon") ?? 1e-5
        let tieEmbed = config.tieWordEmbeddings

        guard nGroups == 1 else {
            throw Mamba2Error.unsupportedConfig("n_groups > 1 not yet supported (got \(nGroups))")
        }
        guard !useBias else {
            throw Mamba2Error.unsupportedConfig("use_bias=true on in_proj/out_proj not yet supported")
        }

        let dInner = expand * hidden
        precondition(dInner == nHeads * headDim,
                     "expand*hidden (\(dInner)) must equal n_heads*head_dim (\(nHeads * headDim))")
        let convDim = dInner + 2 * nGroups * stateDim

        // Activation/inference dtype — taken from the embedding weight
        // (Mamba 2 130m ships as fp32 today; future checkpoints may be
        // bf16). The rest of the layer storage is allocated to match.
        let embedW = try weights.tensor(named: "backbone.embeddings.weight")
        let activationDtype = embedW.dtype
        precondition(activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
                     "Mamba2: unexpected activation dtype \(activationDtype)")
        let embedTokens = AnyEmbedding(Embedding(weight: embedW))

        // Layers
        var layers: [Mamba2Layer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "backbone.layers.\(i)"

            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).norm.weight"),
                eps: Float(eps))

            let inProj = AnyLinear(Linear(
                weight: try weights.tensor(named: "\(p).mixer.in_proj.weight")))
            let outProj = AnyLinear(Linear(
                weight: try weights.tensor(named: "\(p).mixer.out_proj.weight")))

            // conv1d.weight ships as [conv_dim, 1, kernel_size]; the
            // metaltile kernel expects [kernel_size, conv_dim].
            // Transpose CPU-side at load (~28 KB at bf16 for 130m).
            let convWSrc = try weights.tensor(named: "\(p).mixer.conv1d.weight")
            // HF ships various shape labelings — [C, 1, K], [C, K, 1],
            // or [C, K]. All three share the same row-major byte layout
            // (channel-major, then kernel-tap). The transpose helper
            // only needs total count.
            precondition(convWSrc.elementCount == convDim * convKernel,
                         "conv1d.weight count mismatch: \(convWSrc.shape)")
            let convW = transposeConv1dWeight(src: convWSrc,
                                              kernel: convKernel,
                                              channels: convDim,
                                              dtype: activationDtype,
                                              device: device)

            let convB: Tensor = {
                if useConvBias {
                    return (try? weights.tensor(named: "\(p).mixer.conv1d.bias"))
                        ?? Tensor.empty(shape: [convDim], dtype: activationDtype, device: device).also { $0.zero() }
                }
                let t = Tensor.empty(shape: [convDim], dtype: activationDtype, device: device)
                t.zero()
                return t
            }()

            // A_eff = -exp(A_log) and dt_bias are per-head; D is per-head
            // tiled to [d_inner] so we can reuse Ops.mul instead of a
            // broadcast kernel.
            let aLog = try weights.tensor(named: "\(p).mixer.A_log")
            let dLog = try weights.tensor(named: "\(p).mixer.D")
            let dtBiasSrc = try weights.tensor(named: "\(p).mixer.dt_bias")

            let aEff = computeAEff(aLog: aLog, nHeads: nHeads,
                                   dtype: activationDtype, device: device)
            let dtBias = castVector(dtBiasSrc, count: nHeads,
                                    dtype: activationDtype, device: device)
            let dTiled = tileDOverHeadDim(d: dLog, nHeads: nHeads, headDim: headDim,
                                          dtype: activationDtype, device: device)

            let mixerNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).mixer.norm.weight"),
                eps: Float(eps))

            layers.append(Mamba2Layer(
                inputNorm: inputNorm,
                inProj: inProj, outProj: outProj,
                convW: convW, convB: convB,
                aEff: aEff, dtBias: dtBias, dTiled: dTiled,
                mixerNorm: mixerNorm,
                hidden: hidden, dInner: dInner, convDim: convDim,
                nHeads: nHeads, headDim: headDim, stateDim: stateDim,
                convKernel: convKernel, dtype: activationDtype
            ))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "backbone.norm_f.weight"),
            eps: Float(eps))

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: nil)
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        return Mamba2Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers,
            nHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
            stateDim: stateDim, convDim: convDim, convKernel: convKernel,
            dInner: dInner, vocab: vocab,
            dtype: activationDtype
        )
    }
}

// ─── Mamba2Layer ─────────────────────────────────────────────────────

public final class Mamba2Layer: Module {
    let inputNorm: RMSNorm
    let inProj, outProj: AnyLinear
    let convW: Tensor        // [kernel_size, conv_dim]   activation dtype
    let convB: Tensor        // [conv_dim]                activation dtype
    let aEff: Tensor         // [n_heads]                 activation dtype  (= -exp(A_log))
    let dtBias: Tensor       // [n_heads]                 activation dtype
    let dTiled: Tensor       // [d_inner]                 activation dtype  (D[h] tiled)
    let mixerNorm: RMSNorm
    let hidden, dInner, convDim, nHeads, headDim, stateDim, convKernel: Int
    let dtype: DType

    init(inputNorm: RMSNorm,
         inProj: AnyLinear, outProj: AnyLinear,
         convW: Tensor, convB: Tensor,
         aEff: Tensor, dtBias: Tensor, dTiled: Tensor,
         mixerNorm: RMSNorm,
         hidden: Int, dInner: Int, convDim: Int,
         nHeads: Int, headDim: Int, stateDim: Int,
         convKernel: Int, dtype: DType) {
        self.inputNorm = inputNorm
        self.inProj = inProj
        self.outProj = outProj
        self.convW = convW
        self.convB = convB
        self.aEff = aEff
        self.dtBias = dtBias
        self.dTiled = dTiled
        self.mixerNorm = mixerNorm
        self.hidden = hidden; self.dInner = dInner; self.convDim = convDim
        self.nHeads = nHeads; self.headDim = headDim; self.stateDim = stateDim
        self.convKernel = convKernel; self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inputNorm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in inProj.parameters() { out.append(("mixer.in_proj.\(k)", v)) }
        for (k, v) in outProj.parameters() { out.append(("mixer.out_proj.\(k)", v)) }
        // convW + convB are derived (transposed / synthesized) and don't
        // participate in checkpoint round-tripping today.
        for (k, v) in mixerNorm.parameters() { out.append(("mixer.norm.\(k)", v)) }
        return out
    }

    /// Single-token decode forward. All work queued on `cmd`; cache
    /// state mutations are kernel-driven (no host sync).
    func forward(_ h: Tensor, cache: Mamba2LayerCache,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // (1) pre-mixer RMSNorm
        let xNorm = inputNorm(h, on: cmd)

        // (2) in_proj → split into z / xBC / dt_raw
        let proj = inProj(xNorm, on: cmd)       // [in_proj_dim]
        let z = proj.slicedRows(start: 0, count: dInner)
        let xBC = proj.slicedRows(start: dInner, count: convDim)
        let dtRaw = proj.slicedRows(start: dInner + convDim, count: nHeads)

        // (3) conv1d causal step (rolling state in `cache.conv.state`),
        //     then SiLU activation.
        let convOut = Tensor.empty(shape: [convDim], dtype: dtype, device: device)
        Ops.conv1dCausalStep(
            x: xBC, w: convW, b: convB,
            state: cache.conv.state, into: convOut,
            nChannels: convDim, kernelSize: convKernel,
            on: cmd
        )
        let convAct = Ops.silu(convOut, on: cmd)

        // (4) split conv output into x / B / C
        let x = convAct.slicedRows(start: 0, count: dInner)
        let bVec = convAct.slicedRows(start: dInner, count: stateDim)
        let cVec = convAct.slicedRows(start: dInner + stateDim, count: stateDim)

        // (5) dt = softplus(dt_raw + dt_bias)
        let dtSum = Ops.add(dtRaw, dtBias, on: cmd)
        let dt = Ops.softplus(dtSum, on: cmd)

        // (6) selective scan step
        let y = Tensor.empty(shape: [dInner], dtype: dtype, device: device)
        Ops.ssmStep(
            x: x, a: aEff, b: bVec, c: cVec, dt: dt,
            state: cache.ssm.h, into: y,
            nHeads: nHeads, headDim: headDim, stateDim: stateDim,
            on: cmd
        )

        // (7) skip: y += D_tiled * x
        let dx = Ops.mul(dTiled, x, on: cmd)
        let ySkip = Ops.add(y, dx, on: cmd)

        // (8) gating: y *= silu(z)
        let zAct = Ops.silu(z, on: cmd)
        let yGated = Ops.mul(ySkip, zAct, on: cmd)

        // (9) mixer norm + out_proj
        let yNorm = mixerNorm(yGated, on: cmd)
        let yOut = outProj(yNorm, on: cmd)

        // (10) residual add
        let result = Ops.add(h, yOut, on: cmd)

        cache.advance()
        return result
    }
}

// ─── Mamba2Model ─────────────────────────────────────────────────────

public final class Mamba2Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [Mamba2Layer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim: Int
    public let stateDim, convDim, convKernel, dInner: Int
    public let vocab: Int
    public let maxSeq: Int = .max     // SSM has no length-bound state
    public let dtype: DType

    init(embedTokens: AnyEmbedding, layers: [Mamba2Layer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         stateDim: Int, convDim: Int, convKernel: Int, dInner: Int,
         vocab: Int, dtype: DType) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.stateDim = stateDim; self.convDim = convDim
        self.convKernel = convKernel; self.dInner = dInner
        self.vocab = vocab; self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() { out.append(("backbone.embeddings.\(k)", v)) }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("backbone.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("backbone.norm_f.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    public func makeLayerCaches(maxSeq _: Int?, device: Device) -> [any LayerCacheProtocol] {
        return (0..<nLayers).map { _ in
            Mamba2LayerCache(
                nHeads: nHeads, stateDim: stateDim, headDim: headDim,
                convChannels: convDim, convKernelSize: convKernel,
                dtype: dtype, device: device
            )
        }
    }

    /// Primitive: queue a single-token forward pass onto the caller's
    /// command buffer. No commit. The `LanguageModel` default
    /// extension composes this with the appropriate output kernel on
    /// the same cmdbuf (argmax for forwardSample, softmax-categorical
    /// for forwardSampleCategorical), giving Mamba 2 the 1-commit-per-
    /// token path automatically — no hand-rolled overrides needed.
    ///
    /// Mamba 2 ignores `position` because the per-layer Mamba2LayerCache
    /// already tracks recurrent state; the `_` underscore drops it.
    public func forward(tokenId: Int, position _: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, cache: caches[i] as! Mamba2LayerCache,
                              cmd: cmd, device: device)
        }

        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    // `forward`, `forwardSample`, `forwardSampleCategorical` come from
    // LanguageModel's default extension — they wrap `forward(...on cmd:)`
    // with a single command buffer and the appropriate output kernel.
}

// ─── Load-time host helpers ──────────────────────────────────────────

/// Read a tensor of dtype f32 / bf16 / f16 into a `[Float]` for
/// CPU-side derivation work at load time. Only used for small per-head
/// vectors / conv weight transposition — never on the hot path.
private func readFloats(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .bf16:
        let bits = t.toArray(as: UInt16.self)
        return bits.map { Float(bitPattern: UInt32($0) << 16) }
    case .f16:
        let halves = t.toArray(as: Float16.self)
        return halves.map { Float($0) }
    default:
        fatalError("Mamba2: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Write a `[Float]` into a fresh tensor of the requested dtype.
private func writeFloats(_ values: [Float], shape: [Int], dtype: DType,
                         device: Device) -> Tensor {
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    switch dtype {
    case .f32:
        t.copyIn(from: values)
    case .bf16:
        let bits = values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        t.copyIn(from: bits)
    case .f16:
        let halves = values.map { Float16($0) }
        t.copyIn(from: halves)
    default:
        fatalError("Mamba2: unsupported dtype for host conversion: \(dtype)")
    }
    return t
}

/// A_eff = -exp(A_log), per head, in the requested activation dtype.
private func computeAEff(aLog: Tensor, nHeads: Int,
                         dtype: DType, device: Device) -> Tensor {
    let floats = readFloats(aLog)
    precondition(floats.count == nHeads, "A_log expected [n_heads]")
    let aEff = floats.map { -Foundation.exp($0) }
    return writeFloats(aEff, shape: [nHeads], dtype: dtype, device: device)
}

/// Cast a per-head vector to the activation dtype, preserving values.
private func castVector(_ src: Tensor, count: Int,
                        dtype: DType, device: Device) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = readFloats(src)
    precondition(floats.count == count, "vector size mismatch")
    return writeFloats(floats, shape: [count], dtype: dtype, device: device)
}

/// Tile `D[h]` across `head_dim` channels, producing `[n_heads * head_dim]`
/// so the skip connection (`y += D * x`) reuses `Ops.mul` without needing
/// a broadcast kernel.
private func tileDOverHeadDim(d: Tensor, nHeads: Int, headDim: Int,
                              dtype: DType, device: Device) -> Tensor {
    let floats = readFloats(d)
    precondition(floats.count == nHeads, "D expected [n_heads]")
    var tiled = [Float](); tiled.reserveCapacity(nHeads * headDim)
    for h in 0..<nHeads {
        for _ in 0..<headDim { tiled.append(floats[h]) }
    }
    return writeFloats(tiled, shape: [nHeads * headDim],
                       dtype: dtype, device: device)
}

/// HF conv1d.weight ships as `[C, 1, K]` (depthwise grouped conv,
/// channel-major). The metaltile kernel expects `[K, C]` (kernel-tap
/// major). Transpose CPU-side; tiny (~K*C floats) so the cost is in the
/// noise at load time.
private func transposeConv1dWeight(src: Tensor,
                                   kernel K: Int, channels C: Int,
                                   dtype: DType, device: Device) -> Tensor {
    let floats = readFloats(src)
    precondition(floats.count == K * C, "conv1d.weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0..<C {
        for k in 0..<K {
            dst[k * C + c] = floats[c * K + k]
        }
    }
    return writeFloats(dst, shape: [K, C], dtype: dtype, device: device)
}

// ─── Small Tensor convenience ────────────────────────────────────────

private extension Tensor {
    /// Inline side-effect for declarative initializers; returns self.
    func also(_ apply: (Tensor) -> Void) -> Tensor { apply(self); return self }
}
