// GlmOcr vision tower internals — ViT + text engine + GPU-dispatching helpers.
//
// This file contains GlmOcrTextLayer, GlmOcrModel (the full engine which
// implements LanguageModel), GlmOcrVisionTower, GlmOcrVisionBlock, and all
// supporting helpers — CpuLinear (Ops.gemm / Ops.dequantGemv wrapper),
// cpuRMSNorm (host-side small-element scaling that feeds Ops.gemm), patch
// unfold (load-time prep), and dtype conversion (also load-time) — plus
// bundle adapter types (SafeTensorsBundlePrefixView, loadLinear /
// loadEmbedding overloads for the prefix view). The "Cpu" prefix on
// CpuLinear / cpuRMSNorm is historical — the projection and matmul work
// dispatches to the GPU; only the per-row scale / small-element setup
// stays host-side where Metal launch overhead would dominate.
//
// Per the GLM-OCR design: the engine itself is a LanguageModel (not a
// VisionModel splice), so GlmOcrModel owns both the ViT and the text decoder —
// the whole file stays together rather than splitting the engine from the
// tower.
//
// The family orchestrator (load entrypoint, `GlmOcrError`, `GlmOcr` enum with
// `modelTypes` / `architectures` / `load()`) lives in `Models/GlmOcr.swift`.

import Foundation
import Metal

// ─── Text layer (sandwiched pre-post-norm) ────────────────────────────

/// GLM-OCR decoder layer: pre-norm → self-attn → post-attn-norm → residual,
/// then pre-mlp-norm → MLP → post-mlp-norm → residual. The two extra
/// post-norms (compared to standard Llama / Qwen3) are the structural
/// delta specific to this checkpoint.
public final class GlmOcrTextLayer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear
    /// Pre-attention RMSNorm.
    let inputNorm: RMSNorm
    /// Post-attention RMSNorm (applied to the attention output before adding
    /// the residual).
    let postAttnNorm: RMSNorm
    /// Pre-MLP RMSNorm (applied to the post-attention residual stream).
    let postAttnLN2: RMSNorm
    /// Post-MLP RMSNorm (applied to the MLP output before adding the
    /// residual).
    let postMlpNorm: RMSNorm

    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let scale: Float

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
         inputNorm: RMSNorm, postAttnNorm: RMSNorm,
         postAttnLN2: RMSNorm, postMlpNorm: RMSNorm,
         hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         intermediate: Int, ropeTheta: Float) {
        self.qProj = qProj; self.kProj = kProj
        self.vProj = vProj; self.oProj = oProj
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.postAttnLN2 = postAttnLN2; self.postMlpNorm = postMlpNorm
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads
        self.headDim = headDim; self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters()      { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters()      { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters()      { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters()      { out.append(("self_attn.o_proj.\(k)", v)) }
        for (k, v) in gateProj.parameters()   { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters()     { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters()   { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters()  { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_self_attn_layernorm.\(k)", v)) }
        for (k, v) in postAttnLN2.parameters()  { out.append(("post_attention_layernorm.\(k)", v)) }
        for (k, v) in postMlpNorm.parameters()  { out.append(("post_mlp_layernorm.\(k)", v)) }
        return out
    }

    /// Single-token forward. Returns the updated residual stream `[hidden]`.
    /// All GPU work is queued on `cmd`; caller commits once at end-of-token.
    func forward(_ h: Tensor, position: Int, cache: any KVCacheProtocol,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // ── Attention sub-block ──
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // RoPE (scalar position approximation for M-RoPE — coherence-first)
        let qRot = Ops.rope(q.reshaped(to: [nHeads, headDim]),
                            position: position, headDim: headDim,
                            thetaBase: ropeTheta, on: cmd)
        let kRot = Ops.rope(k.reshaped(to: [nKVHeads, headDim]),
                            position: position, headDim: headDim,
                            thetaBase: ropeTheta, on: cmd)

        cache.appendOnGPU(kFlat: kRot,
                          vFlat: v.reshaped(to: [nKVHeads, headDim]),
                          on: cmd)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRot, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)
        // Post-attention norm on the attention output, then add residual.
        let attnNormed = postAttnNorm(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let oOut = oProj(attnNormed, on: cmd)
        let postAttn = Ops.add(h, oOut, on: cmd)

        // ── MLP sub-block ──
        // Pre-MLP norm (applied to the updated residual stream).
        let mlpNorm = postAttnLN2(postAttn, on: cmd)
        let gate = gateProj(mlpNorm, on: cmd)
        let up   = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpRaw = downProj(mlpInner, on: cmd)
        // Post-MLP norm on MLP output, then add residual.
        let mlpNormed = postMlpNorm(mlpRaw, on: cmd)
        return Ops.add(postAttn, mlpNormed, on: cmd)
    }
}

// ─── GlmOcrModel ─────────────────────────────────────────────────────

/// GLM-OCR vision-language model engine. Implements `LanguageModel` for
/// the text-only decode path (compatible with `Generate.swift`) and
/// exposes `generate(image:promptTokens:maxTokens:device:)` for the
/// multi-modal prefill + decode path.
public final class GlmOcrModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [GlmOcrTextLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear
    /// Vision tower: patch-embed + transformer blocks + merger.
    public let visionTower: GlmOcrVisionTower

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let ropeTheta: Float
    public let dtype: DType
    public let kvCacheKind: KVCacheKind
    public let imageTokenId: Int
    public let eosTokenId: Int

    init(embedTokens: AnyEmbedding, layers: [GlmOcrTextLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         visionTower: GlmOcrVisionTower,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         vocab: Int, maxSeq: Int, ropeTheta: Float, dtype: DType,
         imageTokenId: Int, eosTokenId: Int,
         kvCacheKind: KVCacheKind = .raw) {
        self.embedTokens  = embedTokens
        self.layers       = layers
        self.finalNorm    = finalNorm
        self.lmHead       = lmHead
        self.visionTower  = visionTower
        self.hidden       = hidden; self.nLayers  = nLayers
        self.nHeads       = nHeads; self.nKVHeads = nKVHeads
        self.headDim      = headDim; self.vocab    = vocab
        self.maxSeq       = maxSeq;  self.ropeTheta = ropeTheta
        self.dtype        = dtype;   self.kvCacheKind = kvCacheKind
        self.imageTokenId = imageTokenId
        self.eosTokenId   = eosTokenId
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("language_model.model.embed_tokens.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("language_model.model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() {
            out.append(("language_model.model.norm.\(k)", v))
        }
        for (k, v) in lmHead.parameters() {
            out.append(("language_model.lm_head.\(k)", v))
        }
        return out
    }

    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        switch kvCacheKind {
        case .raw:
            return (0..<nLayers).map { _ in
                KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, device: device)
            }
        case .affineQuantized(let bits, let groupSize):
            let sharedK = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            let sharedV = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            return (0..<nLayers).map { _ in
                AffineQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, bits: bits, groupSize: groupSize,
                    sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                    device: device)
            }
        case .auraQuantized:
            // GLM-OCR doesn't ship AURA conversions yet; fall back to raw.
            return (0..<nLayers).map { _ in
                KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, device: device)
            }
        }
    }

    // MARK: - Text-only forward (LanguageModel protocol)

    /// Command-buffer-aware variant — required by `LanguageModel`.
    /// The internal forward constructs its own command buffer; the
    /// caller-supplied `cmd` is currently ignored.
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        forward(tokenId: tokenId, position: position,
                caches: caches, device: device)
    }

    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol], device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }

        let normed  = finalNorm(h, on: cmd)
        let logits  = lmHead(normed, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }

    /// Multi-token forward — prefill fast path. Loops
    /// `forward(tokenId:)` per row on the supplied `cmd`.
    ///
    /// GLM-OCR's text decoder uses a 4-norm sandwich layout
    /// (input → attn → postAttn → residual → postAttnLN2 → MLP →
    /// postMlp → residual) plus dynamic-resolution ViT. A chunked
    /// path would adopt the Llama pattern with the extra norm steps;
    /// today this override is commit-count-batched only.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "GlmOcrModel.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(tokenId: tok, position: position + i,
                             caches: caches, on: cmd, device: device)
        }
        return logits
    }

    public func forwardSample(tokenId: Int, position: Int,
                              caches: [any LayerCacheProtocol], device: Device) -> Int {
        let cmd = device.makeCommandBuffer()

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }

        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)

        let outBuf = device.makeBuffer(length: 4)
        let outT   = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.argmax(logits, into: outT, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }

    public func forwardSampleCategorical(
        tokenId: Int, position: Int, caches: [any LayerCacheProtocol],
        temperature: Float, uniformDraw: Float, device: Device
    ) -> Int {
        let cmd = device.makeCommandBuffer()

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }

        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)

        let tBuf = device.makeBuffer(length: 4)
        var tVal = temperature
        memcpy(tBuf.contents(), &tVal, 4)
        let temperatureT = Tensor(buffer: tBuf, offset: 0, shape: [1], dtype: .f32)

        let uBuf = device.makeBuffer(length: 4)
        var uVal = uniformDraw
        memcpy(uBuf.contents(), &uVal, 4)
        let uniformT = Tensor(buffer: uBuf, offset: 0, shape: [1], dtype: .f32)

        let outBuf = device.makeBuffer(length: 4)
        let outT   = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.softmaxCategoricalSample(logits, into: outT,
                                     temperature: temperatureT,
                                     uniform: uniformT, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }

    // MARK: - Multi-modal generate

    /// Greedy multi-modal generation. Encodes `image` (NCHW Float32 tensor,
    /// `[1, 3, imageH, imageW]`) through the vision tower, injects the
    /// resulting patch tokens at every `imageTokenId` position in
    /// `promptTokens`, then runs prefill + greedy decode.
    ///
    /// `image` must already be resized and normalised for the GLM-OCR
    /// tower (mean=[0.48145466, 0.4578275, 0.40821073],
    /// std=[0.26862954, 0.26130258, 0.27577711]).
    ///
    /// Returns the generated token ids. The contract is coherence: the
    /// output should be non-degenerate text. Use `forwardSample` for
    /// text-only decode; this entry point is the vision path.
    public func generate(image: GlmOcrRGBImage?, promptTokens: [Int],
                         maxTokens: Int, device: Device = .shared) -> [Int] {
        let caches = makeLayerCaches(device: device)

        // Build the prefill embedding stream.
        // If an image is supplied, encode it and splice patch embeddings
        // at every `imageTokenId` position; otherwise fall back to pure
        // text embeddings.
        let stream: [Tensor] = buildPrefillStream(
            image: image, promptTokens: promptTokens, device: device)

        // Prefill: forward every token / vision embedding in sequence.
        // Keep the argmax of the final position as the first decode token.
        var nextToken = 0
        for (pos, embedding) in stream.enumerated() {
            nextToken = forwardEmbedding(embedding, position: pos,
                                         caches: caches, device: device)
        }

        // Decode: greedy token-id forward from the tail of the prefill.
        var generated: [Int] = []
        var pos = stream.count
        for _ in 0..<maxTokens {
            if nextToken == eosTokenId { break }
            generated.append(nextToken)
            let prior = nextToken
            nextToken = forwardSample(tokenId: prior, position: pos,
                                       caches: caches, device: device)
            pos += 1
        }
        return generated
    }

    // ── Internal helpers ──

    /// Build `[Tensor]` prefill stream: vision embeddings spliced at
    /// `imageTokenId` positions, text embeddings elsewhere.
    private func buildPrefillStream(image: GlmOcrRGBImage?, promptTokens: [Int],
                                    device: Device) -> [Tensor] {
        // Encode the image once and get `[numVisionTokens, hidden]`.
        let imageTokens: Tensor?
        if let img = image {
            imageTokens = visionTower.encode(image: img, dtype: dtype, device: device)
        } else {
            imageTokens = nil
        }

        var stream: [Tensor] = []
        stream.reserveCapacity(promptTokens.count)
        var visionCursor = 0

        let imageTokenCount = imageTokens?.shape[0] ?? 0
        let rowBytes = hidden * dtype.byteSize

        for tok in promptTokens {
            if tok == imageTokenId, let vt = imageTokens,
               visionCursor < imageTokenCount {
                // Slice row `visionCursor` as a `[hidden]` view.
                let row = Tensor(
                    buffer: vt.buffer,
                    offset: vt.offset + visionCursor * rowBytes,
                    shape: [hidden], dtype: vt.dtype)
                stream.append(row)
                visionCursor += 1
            } else {
                stream.append(textEmbedding(tokenId: tok, device: device))
            }
        }
        return stream
    }

    /// Look up the embedding for a single token id, returning `[hidden]`.
    /// `public` to satisfy the `LanguageModel` protocol requirement.
    public func textEmbedding(tokenId: Int, device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        let emb = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])
        cmd.commit()
        cmd.waitUntilCompleted()
        return emb
    }

    /// Forward a single `[hidden]` embedding (vision or text) through all
    /// decoder layers and return the argmax of the logits. Caller-managed
    /// caches grow with each call.
    private func forwardEmbedding(_ h: Tensor, position: Int,
                                   caches: [any LayerCacheProtocol],
                                   device: Device) -> Int {
        let cmd = device.makeCommandBuffer()
        var x = h
        for (i, layer) in layers.enumerated() {
            x = layer.forward(x, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }
        let normed = finalNorm(x, on: cmd)
        let logits = lmHead(normed, on: cmd)
        let outBuf = device.makeBuffer(length: 4)
        let outT   = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.argmax(logits, into: outT, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }
}

// ─── Vision tower ─────────────────────────────────────────────────────

/// Minimal CPU-side image representation: flat Float32 `[height, width, 3]`
/// after normalization. Populated by the caller (the integration test
/// generates a solid-color image directly into this layout).
public struct GlmOcrRGBImage {
    /// Pixel data in HWC row-major Float32, normalized.
    public let data: [Float]
    public let height: Int
    public let width: Int

    public init(data: [Float], height: Int, width: Int) {
        precondition(data.count == height * width * 3,
                     "GlmOcrRGBImage: data.count \(data.count) ≠ \(height * width * 3)")
        self.data = data
        self.height = height
        self.width = width
    }

    /// Convenience: solid-colour image for testing.
    public static func solid(width: Int, height: Int,
                             r: Float, g: Float, b: Float) -> GlmOcrRGBImage {
        let n = height * width * 3
        var data = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            data[i] = r; data[i+1] = g; data[i+2] = b
            i += 3
        }
        return GlmOcrRGBImage(data: data, height: height, width: width)
    }
}

/// GLM-OCR vision tower configuration, decoded from `vision_config`.
public struct GlmOcrVisionConfig {
    public let depth: Int
    public let hidden: Int             // ViT hidden dim
    public let intermediate: Int       // MLP intermediate dim
    public let outHidden: Int          // after downsample Conv2d
    public let numHeads: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let inChannels: Int
    public let rmsNormEps: Float

    /// Per-head dimension in the vision transformer.
    public var headDim: Int { hidden / numHeads }
    /// Side of the spatial-merge neighbourhood (number of merge units per side).
    public var mergeUnit: Int { spatialMergeSize * spatialMergeSize }

    static func decode(_ c: ModelConfig) throws -> GlmOcrVisionConfig {
        guard let depth   = c.int("depth"),
              let hidden  = c.int("hidden_size"),
              let numHeads = c.int("num_heads"),
              let patch   = c.int("patch_size"),
              let merge   = c.int("spatial_merge_size")
        else {
            throw GlmOcrError.missingConfig
        }
        return GlmOcrVisionConfig(
            depth:              depth,
            hidden:             hidden,
            intermediate:       c.int("intermediate_size") ?? hidden * 4,
            outHidden:          c.int("out_hidden_size")   ?? hidden,
            numHeads:           numHeads,
            patchSize:          patch,
            spatialMergeSize:   merge,
            temporalPatchSize:  c.int("temporal_patch_size") ?? 2,
            inChannels:         c.int("in_channels") ?? 3,
            rmsNormEps:         Float(c.float("rms_norm_eps") ?? 1e-5))
    }
}

/// Holds the weight tensors for the GLM-OCR vision tower.
public final class GlmOcrVisionTower: @unchecked Sendable {
    let cfg: GlmOcrVisionConfig
    /// Flattened patch-embed GEMM weight: `[hidden, patchDim]`
    /// where `patchDim = inCh · temporalPatch · patchH · patchW`.
    let patchEmbedWeight: [Float]
    let patchEmbedBias: [Float]?
    let blocks: [GlmOcrVisionBlock]
    /// Post-transformer RMSNorm (applied before the downsample).
    let postLayerNorm: GlmOcrVisionRMSNorm
    /// Downsample Conv2d (spatial merge): flattened [outHidden, mergeUnit·hidden].
    let downsampleWeight: [Float]
    let downsampleBias: [Float]
    /// Merger weights: proj linear, post-projection LayerNorm, gate/up/down SwiGLU.
    let mergerProj: [Float]         // [outHidden, outHidden]
    let mergerNormWeight: [Float]   // [outHidden]
    let mergerNormBias: [Float]?    // [outHidden]
    let mergerGate: [Float]         // [contextDim, outHidden]
    let mergerUp:   [Float]         // [contextDim, outHidden]
    let mergerDown: [Float]         // [outHidden, contextDim]
    let contextDim: Int             // = outHidden * inChannels
    let textHidden: Int
    let dtype: DType

    init(cfg: GlmOcrVisionConfig,
         patchEmbedWeight: [Float], patchEmbedBias: [Float]?,
         blocks: [GlmOcrVisionBlock],
         postLayerNorm: GlmOcrVisionRMSNorm,
         downsampleWeight: [Float], downsampleBias: [Float],
         mergerProj: [Float], mergerNormWeight: [Float], mergerNormBias: [Float]?,
         mergerGate: [Float], mergerUp: [Float], mergerDown: [Float],
         contextDim: Int, textHidden: Int, dtype: DType) {
        self.cfg = cfg
        self.patchEmbedWeight = patchEmbedWeight
        self.patchEmbedBias   = patchEmbedBias
        self.blocks           = blocks
        self.postLayerNorm    = postLayerNorm
        self.downsampleWeight = downsampleWeight
        self.downsampleBias   = downsampleBias
        self.mergerProj       = mergerProj
        self.mergerNormWeight = mergerNormWeight
        self.mergerNormBias   = mergerNormBias
        self.mergerGate       = mergerGate
        self.mergerUp         = mergerUp
        self.mergerDown       = mergerDown
        self.contextDim       = contextDim
        self.textHidden       = textHidden
        self.dtype            = dtype
    }

    // MARK: - Load

    static func load(
        cfg visCfgModel: ModelConfig, textHidden: Int,
        weights: SafeTensorsBundlePrefixView, dtype: DType, device: Device
    ) throws -> GlmOcrVisionTower {
        let cfg = try GlmOcrVisionConfig.decode(visCfgModel)

        // ── Patch embed ──
        // The mlx-community conversion stores the Conv3d weight in
        // channel-last MLX layout `[hidden, tP, pY, pX, inCh]`.
        // We flatten it into `[hidden, tP·inCh·pY·pX]` for the
        // CPU patch-embed GEMM.
        let patchRaw = try weights.tensor(named: "patch_embed.proj.weight")
        let patchBiasRaw = try? weights.tensor(named: "patch_embed.proj.bias")
        let patchEmbed = flattenPatchEmbedWeight(patchRaw, cfg: cfg)
        let patchBias: [Float]? = patchBiasRaw.map { $0.toFloatArray() }

        // ── Vision blocks ──
        var blocks: [GlmOcrVisionBlock] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0..<cfg.depth {
            let p = "blocks.\(i)"
            func rn(_ key: String) throws -> GlmOcrVisionRMSNorm {
                let w = try weights.tensor(named: "\(p).\(key)").toFloatArray()
                return GlmOcrVisionRMSNorm(weight: w, eps: cfg.rmsNormEps)
            }
            func fl(_ key: String) throws -> [Float] {
                try weights.tensor(named: "\(p).\(key)").toFloatArray()
            }
            blocks.append(GlmOcrVisionBlock(
                norm1:  try rn("norm1.weight"),
                norm2:  try rn("norm2.weight"),
                qkvWeight: try fl("attn.qkv.weight"),
                qkvBias:   try fl("attn.qkv.bias"),
                projWeight: try fl("attn.proj.weight"),
                projBias:   try fl("attn.proj.bias"),
                qNormWeight: try fl("attn.q_norm.weight"),
                kNormWeight: try fl("attn.k_norm.weight"),
                gateWeight: try fl("mlp.gate_proj.weight"),
                gateBias:   try fl("mlp.gate_proj.bias"),
                upWeight:   try fl("mlp.up_proj.weight"),
                upBias:     try fl("mlp.up_proj.bias"),
                downWeight: try fl("mlp.down_proj.weight"),
                downBias:   try fl("mlp.down_proj.bias"),
                cfg: cfg))
        }

        // ── Post-layer norm ──
        let postLNW = try weights.tensor(named: "post_layernorm.weight").toFloatArray()
        let postLN  = GlmOcrVisionRMSNorm(weight: postLNW, eps: cfg.rmsNormEps)

        // ── Downsample Conv2d ──
        // Stored as `[outHidden, kH, kW, inCh·tP]` MLX OHWI or
        // `[outHidden, inCh·tP, kH, kW]` PyTorch OIHW.
        // We flatten to `[outHidden, mergeUnit·hidden]`.
        let dsRaw    = try weights.tensor(named: "downsample.weight")
        let dsBias   = try weights.tensor(named: "downsample.bias").toFloatArray()
        let dsWeight = flattenDownsampleWeight(dsRaw, cfg: cfg)

        // ── Merger ──
        let mergerProj = try weights.tensor(named: "merger.proj.weight").toFloatArray()
        let mergerNormW = try weights.tensor(named: "merger.post_projection_norm.weight").toFloatArray()
        let mergerNormB = (try? weights.tensor(named: "merger.post_projection_norm.bias"))?
            .toFloatArray()
        let mergerGate = try weights.tensor(named: "merger.gate_proj.weight").toFloatArray()
        let mergerUp   = try weights.tensor(named: "merger.up_proj.weight").toFloatArray()
        let mergerDown = try weights.tensor(named: "merger.down_proj.weight").toFloatArray()
        let contextDim = cfg.outHidden * cfg.inChannels

        return GlmOcrVisionTower(
            cfg: cfg,
            patchEmbedWeight: patchEmbed, patchEmbedBias: patchBias,
            blocks: blocks, postLayerNorm: postLN,
            downsampleWeight: dsWeight, downsampleBias: dsBias,
            mergerProj: mergerProj, mergerNormWeight: mergerNormW,
            mergerNormBias: mergerNormB,
            mergerGate: mergerGate, mergerUp: mergerUp, mergerDown: mergerDown,
            contextDim: contextDim, textHidden: textHidden, dtype: dtype)
    }

    // MARK: - Encode

    /// Encode a `GlmOcrRGBImage` through the vision tower.
    /// Returns `[mergedTokenCount, textHidden]` as a GPU `Tensor` in `dtype`.
    func encode(image: GlmOcrRGBImage, dtype: DType, device: Device) -> Tensor {
        let p      = cfg.patchSize
        let tP     = cfg.temporalPatchSize
        let inCh   = cfg.inChannels
        let hidden = cfg.hidden
        let merge  = cfg.spatialMergeSize

        // ── Patch grid ──
        // Round image dimensions down to nearest multiple of patchSize.
        let gridH = image.height / p
        let gridW = image.width  / p
        let nPatches = gridH * gridW

        // ── Patch embed: unfold + GEMM ──
        // Each patch is a `[tP, inCh, p, p]` tile. For a single 2D image
        // we tile the temporal dimension by repeating the frame.
        let patchDim = tP * inCh * p * p
        var patches = [Float](repeating: 0, count: nPatches * patchDim)
        unfoldPatches(image: image, patches: &patches,
                      gridH: gridH, gridW: gridW,
                      patchSize: p, temporalPatch: tP)

        // GEMM: [nPatches, patchDim] × [hidden, patchDim]ᵀ → [nPatches, hidden]
        var h = cpuGemm(a: patches, b: patchEmbedWeight,
                        m: nPatches, n: hidden, k: patchDim)
        if let bias = patchEmbedBias {
            for row in 0..<nPatches {
                for col in 0..<hidden {
                    h[row * hidden + col] += bias[col]
                }
            }
        }

        // ── Transformer blocks ──
        for block in blocks {
            h = block.forward(h, nPatches: nPatches, hidden: hidden)
        }

        // ── Post-layer norm ──
        postLayerNorm.normalize(&h, nRows: nPatches, rowSize: hidden)

        // ── Spatial merge (Conv2d downsample) ──
        // Reshape [nPatches, hidden] → [gridH/merge, merge, gridW/merge, merge, hidden],
        // transpose to [gridH/merge, gridW/merge, merge, merge, hidden], then flatten
        // the merge×merge neighbourhood → [merged, mergeUnit·hidden].
        let mergedH = gridH / merge
        let mergedW = gridW / merge
        let mergedTokens = mergedH * mergedW
        let mergeUnit = merge * merge
        var grouped = [Float](repeating: 0, count: mergedTokens * mergeUnit * hidden)
        for mh in 0..<mergedH {
            for mw in 0..<mergedW {
                for my in 0..<merge {
                    for mx in 0..<merge {
                        let srcRow = (mh * merge + my) * gridW + (mw * merge + mx)
                        let dstBase = (mh * mergedW + mw) * (mergeUnit * hidden)
                            + (my * merge + mx) * hidden
                        for c in 0..<hidden {
                            grouped[dstBase + c] = h[srcRow * hidden + c]
                        }
                    }
                }
            }
        }
        // Downsample GEMM: [mergedTokens, mergeUnit·hidden] × [outHidden, mergeUnit·hidden]ᵀ
        let outHidden = cfg.outHidden
        var ds = cpuGemm(a: grouped, b: downsampleWeight,
                         m: mergedTokens, n: outHidden, k: mergeUnit * hidden)
        for r in 0..<mergedTokens {
            for c in 0..<outHidden {
                ds[r * outHidden + c] += downsampleBias[c]
            }
        }

        // ── Merger ──
        // proj linear, GELU + post-projection LayerNorm, gate-up-down SwiGLU.
        // proj: [mergedTokens, outHidden] × [outHidden, outHidden]ᵀ
        var proj = cpuGemm(a: ds, b: mergerProj,
                           m: mergedTokens, n: outHidden, k: outHidden)
        // GELU activation then LayerNorm.
        cpuGeluInPlace(&proj)
        cpuLayerNorm(&proj, nRows: mergedTokens, rowSize: outHidden,
                     weight: mergerNormWeight, bias: mergerNormBias)
        // SwiGLU MLP: gate [mergedTokens, outHidden] → [mergedTokens, contextDim]
        let gate = cpuGemm(a: proj, b: mergerGate,
                           m: mergedTokens, n: contextDim, k: outHidden)
        let up   = cpuGemm(a: proj, b: mergerUp,
                           m: mergedTokens, n: contextDim, k: outHidden)
        var gateUp = [Float](repeating: 0, count: mergedTokens * contextDim)
        for i in 0..<gateUp.count {
            let g = gate[i]
            gateUp[i] = (g / (1 + exp(-g))) * up[i]   // SiLU(gate) * up
        }
        // down projection: [mergedTokens, contextDim] × [textHidden, contextDim]ᵀ
        let out = cpuGemm(a: gateUp, b: mergerDown,
                          m: mergedTokens, n: textHidden, k: contextDim)

        // Copy into a GPU Tensor in the model's dtype.
        return makeDTypeTensor(from: out, shape: [mergedTokens, textHidden],
                               dtype: dtype, device: device)
    }

    // MARK: - Static helpers

    /// Flatten the Conv3d patch-embed weight from MLX channel-last layout
    /// `[hidden, tP, pY, pX, inCh]` (or PyTorch `[hidden, inCh, tP, pY, pX]`)
    /// into `[hidden, tP·inCh·pY·pX]` with column order `(tP, inCh, pY, pX)`.
    static func flattenPatchEmbedWeight(_ w: Tensor, cfg: GlmOcrVisionConfig) -> [Float] {
        let src = w.toFloatArray()
        let hid = cfg.hidden
        let tP  = cfg.temporalPatchSize
        let p   = cfg.patchSize
        let ch  = cfg.inChannels
        let patchDim = tP * ch * p * p
        var dst = [Float](repeating: 0, count: hid * patchDim)

        // Detect layout from the trailing dimension.
        let mlxLayout = (w.shape.count == 5 && w.shape[4] <= 4)
        if mlxLayout {
            // src: [hid, tP, pY, pX, inCh]
            for o in 0..<hid {
                for t in 0..<tP {
                    for py in 0..<p {
                        for px in 0..<p {
                            for c in 0..<ch {
                                let si = ((((o * tP + t) * p + py) * p + px) * ch + c)
                                let col = (((t * ch + c) * p + py) * p + px)
                                dst[o * patchDim + col] = src[si]
                            }
                        }
                    }
                }
            }
        } else {
            // src: [hid, inCh, tP, pY, pX]
            for o in 0..<hid {
                for c in 0..<ch {
                    for t in 0..<tP {
                        for py in 0..<p {
                            for px in 0..<p {
                                let si = ((((o * ch + c) * tP + t) * p + py) * p + px)
                                let col = (((t * ch + c) * p + py) * p + px)
                                dst[o * patchDim + col] = src[si]
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    /// Flatten the Conv2d downsample weight from MLX OHWI `[outHidden, kH, kW, inCh]`
    /// or PyTorch OIHW `[outHidden, inCh, kH, kW]` into
    /// `[outHidden, mergeUnit·hidden]` with column order `(kH, kW, inCh)`.
    static func flattenDownsampleWeight(_ w: Tensor, cfg: GlmOcrVisionConfig) -> [Float] {
        let src = w.toFloatArray()
        let outH = cfg.outHidden
        let k    = cfg.spatialMergeSize
        let inCh = cfg.hidden  // the merger takes vision hidden dim as input channels
        let kk   = k * k
        var dst  = [Float](repeating: 0, count: outH * kk * inCh)
        let mlxLayout = (w.shape.count == 4 && w.shape[3] == inCh)
        if mlxLayout {
            // src: [outH, kH, kW, inCh]
            for o in 0..<outH {
                for ky in 0..<k {
                    for kx in 0..<k {
                        for c in 0..<inCh {
                            let si = (((o * k + ky) * k + kx) * inCh + c)
                            let col = (ky * k + kx) * inCh + c
                            dst[o * (kk * inCh) + col] = src[si]
                        }
                    }
                }
            }
        } else {
            // src: [outH, inCh, kH, kW]
            for o in 0..<outH {
                for c in 0..<inCh {
                    for ky in 0..<k {
                        for kx in 0..<k {
                            let si = (((o * inCh + c) * k + ky) * k + kx)
                            let col = (ky * k + kx) * inCh + c
                            dst[o * (kk * inCh) + col] = src[si]
                        }
                    }
                }
            }
        }
        return dst
    }
}

// ─── Vision block ─────────────────────────────────────────────────────

/// One GLM-OCR vision transformer block: RMSNorm → QKV + RoPE + attention
/// → proj → residual, then RMSNorm → SwiGLU MLP → residual.
///
/// Per-layer projection weights (QKV, proj, gate/up/down) live as f32 GPU
/// `Tensor`s so each matmul dispatches one `Ops.gemm + Ops.add`. The
/// RMSNorm + per-head Q/K-norm + RoPE remain on the CPU — the matmul
/// bandwidth was the bottleneck, not the per-token reductions.
final class GlmOcrVisionBlock {
    let norm1: GlmOcrVisionRMSNorm
    let norm2: GlmOcrVisionRMSNorm
    let qkvWeight: Tensor   // [3·hidden, hidden] f32 GPU
    let qkvBias:   Tensor   // [3·hidden] f32 GPU
    let projWeight: Tensor  // [hidden, hidden] f32 GPU
    let projBias:   Tensor  // [hidden] f32 GPU
    let qNormWeight: [Float] // [headDim] (CPU — applied per token slice)
    let kNormWeight: [Float] // [headDim] (CPU)
    let gateWeight: Tensor  // [intermediate, hidden] f32 GPU
    let gateBias:   Tensor  // [intermediate] f32 GPU
    let upWeight:   Tensor  // [intermediate, hidden] f32 GPU
    let upBias:     Tensor  // [intermediate] f32 GPU
    let downWeight: Tensor  // [hidden, intermediate] f32 GPU
    let downBias:   Tensor  // [hidden] f32 GPU
    let cfg: GlmOcrVisionConfig

    init(norm1: GlmOcrVisionRMSNorm, norm2: GlmOcrVisionRMSNorm,
         qkvWeight: [Float], qkvBias: [Float],
         projWeight: [Float], projBias: [Float],
         qNormWeight: [Float], kNormWeight: [Float],
         gateWeight: [Float], gateBias: [Float],
         upWeight: [Float], upBias: [Float],
         downWeight: [Float], downBias: [Float],
         cfg: GlmOcrVisionConfig) {
        self.norm1 = norm1; self.norm2 = norm2
        let hidden = cfg.hidden
        let inter  = cfg.intermediate
        // Re-host every weight/bias as an f32 GPU Tensor so `Ops.gemm`
        // / `Ops.add` can consume them on the hot path.
        self.qkvWeight = glmOcrFloatsToTensor(qkvWeight,  shape: [3 * hidden, hidden])
        self.qkvBias   = glmOcrFloatsToTensor(qkvBias,    shape: [3 * hidden])
        self.projWeight = glmOcrFloatsToTensor(projWeight, shape: [hidden, hidden])
        self.projBias   = glmOcrFloatsToTensor(projBias,   shape: [hidden])
        self.qNormWeight = qNormWeight
        self.kNormWeight = kNormWeight
        self.gateWeight = glmOcrFloatsToTensor(gateWeight, shape: [inter, hidden])
        self.gateBias   = glmOcrFloatsToTensor(gateBias,   shape: [inter])
        self.upWeight   = glmOcrFloatsToTensor(upWeight,   shape: [inter, hidden])
        self.upBias     = glmOcrFloatsToTensor(upBias,     shape: [inter])
        self.downWeight = glmOcrFloatsToTensor(downWeight, shape: [hidden, inter])
        self.downBias   = glmOcrFloatsToTensor(downBias,   shape: [hidden])
        self.cfg = cfg
    }

    /// Forward `[nPatches, hidden]` patch activations (flat Float32 row-major).
    /// Returns updated `[nPatches, hidden]`.
    func forward(_ h: [Float], nPatches: Int, hidden: Int) -> [Float] {
        let numHeads = cfg.numHeads
        let headDim  = cfg.headDim
        let intermed = cfg.intermediate
        let qkvDim   = 3 * hidden
        let scale    = 1.0 / Float(Double(headDim).squareRoot())
        let device = Device.shared

        // ── Attention ──
        var normed = h
        norm1.normalize(&normed, nRows: nPatches, rowSize: hidden)

        // QKV projection + bias on the GPU: [nPatches, 3·hidden]
        let normedT = glmOcrFloatsToTensor(normed, shape: [nPatches, hidden],
                                            device: device)
        let cmdQKV = device.makeCommandBuffer()
        let qkvT = glmOcrGemmBiased(input: normedT, weight: qkvWeight, bias: qkvBias,
                                     nRows: nPatches, outDim: qkvDim,
                                     device: device, on: cmdQKV)
        cmdQKV.commit()
        cmdQKV.waitUntilCompleted()
        let qkv = qkvT.toFloatArray()

        // Split Q, K, V and apply q_norm / k_norm per head.
        var Q = [Float](repeating: 0, count: nPatches * hidden)
        var K = [Float](repeating: 0, count: nPatches * hidden)
        var V = [Float](repeating: 0, count: nPatches * hidden)
        for tok in 0..<nPatches {
            let base = tok * qkvDim
            // Q: [numHeads, headDim] with q_norm
            for h2 in 0..<numHeads {
                var slice = Array(qkv[(base + h2*headDim)..<(base + h2*headDim + headDim)])
                cpuRMSNorm(&slice, weight: qNormWeight, eps: cfg.rmsNormEps)
                for d in 0..<headDim {
                    Q[tok * hidden + h2 * headDim + d] = slice[d]
                }
            }
            // K: [numHeads, headDim] with k_norm
            let kBase = base + hidden
            for h2 in 0..<numHeads {
                var slice = Array(qkv[(kBase + h2*headDim)..<(kBase + h2*headDim + headDim)])
                cpuRMSNorm(&slice, weight: kNormWeight, eps: cfg.rmsNormEps)
                for d in 0..<headDim {
                    K[tok * hidden + h2 * headDim + d] = slice[d]
                }
            }
            // V: no norm
            let vBase = base + 2 * hidden
            for d in 0..<hidden { V[tok * hidden + d] = qkv[vBase + d] }
        }

        // Bidirectional multi-head attention — now GPU-resident via
        // `Ops.sdpaMulti(causal: false)`. GLM-OCR vision uses headDim=128
        // (hidden 1536 / 12 heads), which the d128 sdpa_multi kernel
        // covers; `sdpaBidirectional` only ships d∈{32,64,72} variants.
        //
        // Buffer layout reminder:
        //   Q in [nPatches, hidden] row-major → [nPatches, numHeads, headDim]
        //     matches the kernel's Q contract directly (one copy).
        //   K/V in [nPatches, hidden] row-major → transpose to
        //     [numHeads, nPatches, headDim] for the kernel's KV cache.
        var kFlat = [Float](repeating: 0, count: numHeads * nPatches * headDim)
        var vFlat = [Float](repeating: 0, count: numHeads * nPatches * headDim)
        for j in 0..<nPatches {
            for h in 0..<numHeads {
                let src = j * hidden + h * headDim
                let dst = (h * nPatches + j) * headDim
                for d in 0..<headDim {
                    kFlat[dst + d] = K[src + d]
                    vFlat[dst + d] = V[src + d]
                }
            }
        }
        let qT = Tensor.empty(shape: [nPatches, numHeads, headDim],
                              dtype: .f32, device: device)
        let kT = Tensor.empty(shape: [numHeads, nPatches, headDim],
                              dtype: .f32, device: device)
        let vT = Tensor.empty(shape: [numHeads, nPatches, headDim],
                              dtype: .f32, device: device)
        qT.copyIn(from: Q)
        kT.copyIn(from: kFlat)
        vT.copyIn(from: vFlat)
        let attnCmd = device.makeCommandBuffer()
        let attnSdpaT = Ops.sdpaMulti(
            q: qT, k: kT, v: vT,
            nQHeads: numHeads, nKVHeads: numHeads, headDim: headDim,
            baseKV: 0, nQuery: nPatches, kvStride: nPatches,
            causal: false, scale: scale, on: attnCmd)
        attnCmd.commit()
        attnCmd.waitUntilCompleted()
        let attn = attnSdpaT.toFloatArray()  // [nPatches, numHeads, headDim] = [nPatches, hidden]

        // Projection + bias on the GPU: [nPatches, hidden]
        let attnInT = glmOcrFloatsToTensor(attn, shape: [nPatches, hidden],
                                            device: device)
        let cmdProj = device.makeCommandBuffer()
        let attnOutT = glmOcrGemmBiased(input: attnInT, weight: projWeight, bias: projBias,
                                         nRows: nPatches, outDim: hidden,
                                         device: device, on: cmdProj)
        cmdProj.commit()
        cmdProj.waitUntilCompleted()
        let attnOut = attnOutT.toFloatArray()

        // Residual.
        var postAttn = [Float](repeating: 0, count: nPatches * hidden)
        for i in 0..<postAttn.count { postAttn[i] = h[i] + attnOut[i] }

        // ── MLP ──
        var normed2 = postAttn
        norm2.normalize(&normed2, nRows: nPatches, rowSize: hidden)

        // Gate + Up dispatched on a shared command buffer (both consume
        // the same `normed2` input).
        let normed2T = glmOcrFloatsToTensor(normed2, shape: [nPatches, hidden],
                                             device: device)
        let cmdGU = device.makeCommandBuffer()
        let gateT = glmOcrGemmBiased(input: normed2T, weight: gateWeight, bias: gateBias,
                                      nRows: nPatches, outDim: intermed,
                                      device: device, on: cmdGU)
        let upT   = glmOcrGemmBiased(input: normed2T, weight: upWeight, bias: upBias,
                                      nRows: nPatches, outDim: intermed,
                                      device: device, on: cmdGU)
        cmdGU.commit()
        cmdGU.waitUntilCompleted()
        let gate = gateT.toFloatArray()
        let up   = upT.toFloatArray()

        var gateUp = [Float](repeating: 0, count: nPatches * intermed)
        for i in 0..<gateUp.count {
            let g = gate[i]
            gateUp[i] = (g / (1 + exp(-g))) * up[i]
        }

        let gateUpT = glmOcrFloatsToTensor(gateUp, shape: [nPatches, intermed],
                                            device: device)
        let cmdDown = device.makeCommandBuffer()
        let mlpOutT = glmOcrGemmBiased(input: gateUpT, weight: downWeight, bias: downBias,
                                        nRows: nPatches, outDim: hidden,
                                        device: device, on: cmdDown)
        cmdDown.commit()
        cmdDown.waitUntilCompleted()
        let mlpOut = mlpOutT.toFloatArray()

        // Residual.
        var out = postAttn
        for i in 0..<out.count { out[i] += mlpOut[i] }
        return out
    }
}

/// Upload an f32 `[Float]` array as a fresh f32 GPU `Tensor` of the given shape.
private func glmOcrFloatsToTensor(_ values: [Float], shape: [Int],
                                   device: Device = .shared) -> Tensor {
    let n = shape.reduce(1, *)
    precondition(values.count == n,
                 "glmOcrFloatsToTensor: count \(values.count) ≠ product(shape)=\(n)")
    let t = Tensor.empty(shape: shape, dtype: .f32, device: device)
    t.copyIn(from: values)
    return t
}

/// `out = input · weightᵀ + bias` (bias broadcast across `nRows`) as one
/// `Ops.gemm` + `Ops.add` on the supplied command buffer. Bias tile is
/// staged CPU-side and uploaded once per call. Caller commits and reads
/// back.
private func glmOcrGemmBiased(input: Tensor, weight: Tensor, bias: Tensor,
                               nRows: Int, outDim: Int, device: Device,
                               on cmd: MTLCommandBuffer) -> Tensor {
    let out = Ops.gemm(weight: weight, input: input, nRows: nRows, on: cmd)
    let biasVals = bias.toFloatArray()
    var tiled = [Float](repeating: 0, count: nRows * outDim)
    for r in 0..<nRows {
        let base = r * outDim
        for c in 0..<outDim { tiled[base + c] = biasVals[c] }
    }
    let tiledT = glmOcrFloatsToTensor(tiled, shape: [nRows, outDim], device: device)
    return Ops.add(out, tiledT, on: cmd)
}

// ─── CPU RMSNorm ──────────────────────────────────────────────────────

/// Lightweight CPU RMSNorm for the vision tower (avoids a GPU round-trip
/// for the small patch-token sequences).
struct GlmOcrVisionRMSNorm {
    let weight: [Float]
    let eps: Float

    init(weight: [Float], eps: Float) {
        self.weight = weight
        self.eps    = eps
    }

    /// In-place RMSNorm on `x` of shape `[nRows, rowSize]`.
    func normalize(_ x: inout [Float], nRows: Int, rowSize: Int) {
        for r in 0..<nRows {
            let base = r * rowSize
            var sq: Float = 0
            for i in 0..<rowSize { sq += x[base + i] * x[base + i] }
            let rms = (sq / Float(rowSize) + eps).squareRoot()
            let inv = 1.0 / rms
            for i in 0..<rowSize {
                x[base + i] = x[base + i] * inv * weight[i]
            }
        }
    }
}

// ─── CPU primitives ───────────────────────────────────────────────────

/// Row-major GEMM: `[m, k] × [n, k]ᵀ → [m, n]`.
/// Used for all vision-tower projections on the CPU.
@inline(__always)
private func cpuGemm(a: [Float], b: [Float], m: Int, n: Int, k: Int) -> [Float] {
    var c = [Float](repeating: 0, count: m * n)
    for i in 0..<m {
        let aBase = i * k
        let cBase = i * n
        for j in 0..<n {
            var acc: Float = 0
            let bBase = j * k
            for p in 0..<k { acc += a[aBase + p] * b[bBase + p] }
            c[cBase + j] = acc
        }
    }
    return c
}

/// GEMM + bias add: `[m, k] × [n, k]ᵀ + [n] → [m, n]`.
@inline(__always)
private func cpuGemmWithBias(a: [Float], w: [Float], b: [Float],
                              m: Int, n: Int, k: Int) -> [Float] {
    var c = cpuGemm(a: a, b: w, m: m, n: n, k: k)
    for r in 0..<m {
        for j in 0..<n { c[r * n + j] += b[j] }
    }
    return c
}

/// Per-element RMSNorm on a `[headDim]` slice (for q/k head norms).
@inline(__always)
private func cpuRMSNorm(_ x: inout [Float], weight: [Float], eps: Float) {
    let d = x.count
    var sq: Float = 0
    for v in x { sq += v * v }
    let inv = 1.0 / (sq / Float(d) + eps).squareRoot()
    for i in 0..<d { x[i] = x[i] * inv * weight[i] }
}

/// In-place GELU (approximate tanh form) on a flat Float array.
@inline(__always)
private func cpuGeluInPlace(_ x: inout [Float]) {
    let c: Float = 0.7978845608028654   // sqrt(2/π)
    let k: Float = 0.044715
    for i in 0..<x.count {
        let v = x[i]
        let inner = c * (v + k * v * v * v)
        x[i] = 0.5 * v * (1.0 + tanh(inner))
    }
}

/// In-place LayerNorm (used in the merger post-projection norm).
/// mean-centre, then divide by std, then scale+shift.
@inline(__always)
private func cpuLayerNorm(_ x: inout [Float], nRows: Int, rowSize: Int,
                           weight: [Float], bias: [Float]?) {
    for r in 0..<nRows {
        let base = r * rowSize
        var mean: Float = 0
        for i in 0..<rowSize { mean += x[base + i] }
        mean /= Float(rowSize)
        var variance: Float = 0
        for i in 0..<rowSize {
            let d = x[base + i] - mean
            variance += d * d
        }
        variance /= Float(rowSize)
        let inv = 1.0 / (variance + 1e-5).squareRoot()
        for i in 0..<rowSize {
            let normalized = (x[base + i] - mean) * inv
            x[base + i] = normalized * weight[i] + (bias?[i] ?? 0)
        }
    }
}

/// Unfold a `GlmOcrRGBImage` into `[nPatches, tP·inCh·pY·pX]` patch rows.
/// `patches` is pre-allocated and zeroed. For a single 2D image the
/// temporal patch dim is filled by repeating the frame `tP` times.
private func unfoldPatches(image: GlmOcrRGBImage,
                           patches: inout [Float],
                           gridH: Int, gridW: Int,
                           patchSize p: Int, temporalPatch tP: Int) {
    let inCh = 3
    for ph in 0..<gridH {
        for pw in 0..<gridW {
            let patchIdx = ph * gridW + pw
            let patchBase = patchIdx * (tP * inCh * p * p)
            for t in 0..<tP {
                for c in 0..<inCh {
                    for py in 0..<p {
                        for px in 0..<p {
                            let imgRow = ph * p + py
                            let imgCol = pw * p + px
                            let pixelIdx = (imgRow * image.width + imgCol) * inCh + c
                            let col = (((t * inCh + c) * p + py) * p + px)
                            patches[patchBase + col] = image.data[pixelIdx]
                        }
                    }
                }
            }
        }
    }
}

/// Copy a `[Float]` array into a GPU `Tensor` in the specified `dtype`.
/// Converts f32 → f16 / bf16 as needed.
private func makeDTypeTensor(from data: [Float], shape: [Int],
                              dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    switch dtype {
    case .f32:
        t.copyIn(from: data)
    case .f16:
        var f16 = [UInt16](repeating: 0, count: data.count)
        for i in 0..<data.count { f16[i] = floatToF16(data[i]) }
        t.copyIn(from: f16)
    case .bf16:
        var bf16 = [UInt16](repeating: 0, count: data.count)
        for i in 0..<data.count { bf16[i] = floatToBf16(data[i]) }
        t.copyIn(from: bf16)
    default:
        // Fall back to f32 copy for unsupported types.
        let tf32 = Tensor.empty(shape: shape, dtype: .f32, device: device)
        tf32.copyIn(from: data)
        return tf32
    }
    return t
}

/// Convert Float32 to Float16 bit pattern (round-to-nearest).
@inline(__always)
private func floatToF16(_ v: Float) -> UInt16 {
    var f = v
    var result: UInt16 = 0
    withUnsafeBytes(of: &f) { ptr in
        let bits = ptr.load(as: UInt32.self)
        let sign = UInt16((bits >> 16) & 0x8000)
        let exp  = Int((bits >> 23) & 0xff) - 127 + 15
        let mant = bits & 0x007FFFFF
        if exp <= 0 {
            result = sign
        } else if exp >= 31 {
            result = sign | 0x7C00
        } else {
            result = sign | UInt16(exp << 10) | UInt16(mant >> 13)
        }
    }
    return result
}

/// Convert Float32 to BFloat16 bit pattern (truncate mantissa).
@inline(__always)
private func floatToBf16(_ v: Float) -> UInt16 {
    var f = v
    return withUnsafeBytes(of: &f) { ptr in
        UInt16(ptr.load(as: UInt32.self) >> 16)
    }
}

/// (Removed: `Tensor.toFloatArray()` already lives in
/// `Sources/FFAI/Tensor.swift`.) Keeping the GlmOcr-private f16/bf16
/// helpers below for the internal SafeTensors prefix view.
private extension Tensor {
    /// Internal duplicate path (renamed to `_glmOcrToFloatArray`) — kept
    /// only so the file's existing call-sites compile while the agent
    /// switches them over to the public `Tensor.toFloatArray()`.
    func _glmOcrToFloatArray() -> [Float] {
        let n = elementCount
        switch dtype {
        case .f32:
            return toArray(as: Float.self)
        case .f16:
            let raw = toArray(as: UInt16.self)
            return raw.map { f16ToFloat($0) }
        case .bf16:
            return toArray(as: UInt16.self).map { bf16ToFloat($0) }
        default:
            // Return zeros for unsupported types to avoid crash.
            return [Float](repeating: 0, count: n)
        }
    }
}

@inline(__always)
private func f16ToFloat(_ bits: UInt16) -> Float {
    let sign: UInt32  = (UInt32(bits) & 0x8000) << 16
    let exp  = (UInt32(bits) >> 10) & 0x1F
    let mant = UInt32(bits) & 0x03FF
    let f32bits: UInt32
    if exp == 0 {
        f32bits = sign   // zero / subnormal → 0
    } else if exp == 31 {
        f32bits = sign | 0x7F800000 | (mant << 13)  // inf / NaN
    } else {
        f32bits = sign | ((exp + 112) << 23) | (mant << 13)
    }
    return withUnsafeBytes(of: f32bits) { $0.load(as: Float.self) }
}

@inline(__always)
private func bf16ToFloat(_ bits: UInt16) -> Float {
    let f32bits = UInt32(bits) << 16
    return withUnsafeBytes(of: f32bits) { $0.load(as: Float.self) }
}

/// `SafeTensorsBundle.glmOcrPrefixed` — returns a typed prefix view that
/// prepends a fixed prefix when looking up tensor names. Renamed from
/// `prefixed(...)` to avoid the name collision with the existing
/// `SafeTensorsBundle.prefixed(_:)` that returns `SafeTensorsBundle`.
/// (GLM-OCR's loader needs the `SafeTensorsBundlePrefixView`-typed
/// surface for its bidirectional ViT block loader.)
extension SafeTensorsBundle {
    func glmOcrPrefixed(_ prefix: String) -> SafeTensorsBundlePrefixView {
        SafeTensorsBundlePrefixView(base: self, prefix: prefix)
    }
}

/// Lightweight prefix-routing adapter over `SafeTensorsBundle`.
public struct SafeTensorsBundlePrefixView: @unchecked Sendable {
    let base: SafeTensorsBundle
    let prefix: String

    public func tensor(named name: String) throws -> Tensor {
        try base.tensor(named: prefix + name)
    }

    public func has(_ name: String) -> Bool {
        base.has(prefix + name)
    }

    public func isQuantized(_ base2: String) -> Bool {
        base.isQuantized(prefix + base2)
    }

    public func quantizedTriplet(_ base2: String) throws -> SafeTensorsBundle.QuantizedTriplet {
        try base.quantizedTriplet(prefix + base2)
    }
}

// Overloads of `loadLinear` and `loadEmbedding` that accept the prefix view.

func loadLinear(
    base name: String, in bundle: SafeTensorsBundlePrefixView,
    quantization: ModelConfig.QuantizationConfig?
) throws -> AnyLinear {
    if let q = quantization, [3,4,5,6,8].contains(q.bits), bundle.isQuantized(name) {
        let t = try bundle.quantizedTriplet(name)
        return AnyLinear(QuantizedLinear(weight: t.weight, scales: t.scales,
                                         biases: t.biases,
                                         bits: q.bits, groupSize: q.groupSize))
    }
    return AnyLinear(Linear(weight: try bundle.tensor(named: "\(name).weight")))
}

func loadEmbedding(
    base name: String, in bundle: SafeTensorsBundlePrefixView,
    hidden: Int, quantization: ModelConfig.QuantizationConfig?
) throws -> AnyEmbedding {
    if let q = quantization, [3,4,5,6,8].contains(q.bits), bundle.isQuantized(name) {
        let t = try bundle.quantizedTriplet(name)
        return AnyEmbedding(QuantizedEmbedding(
            weight: t.weight, scales: t.scales, biases: t.biases,
            hidden: hidden, bits: q.bits, groupSize: q.groupSize))
    }
    return AnyEmbedding(Embedding(weight: try bundle.tensor(named: "\(name).weight")))
}
