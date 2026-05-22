// VLModel — vision-language model composition.
//
// A VLM is a vision encoder + a text backbone joined by a cross-modal
// token splice: the prompt contains image-placeholder tokens, the vision
// encoder turns an image into a run of patch-token embeddings, and the
// splice replaces the placeholder positions in the text embedding stream
// with those vision tokens. The fused stream then flows through the text
// backbone exactly as a text-only prompt would.
//
// This file is the family-agnostic core: the splice + a self-contained
// greedy generate. Each VL family file (Gemma3VL, Qwen25VL, …) builds a
// `VLModel` from its checkpoint — a `VisionEncoder` (Phase 6.5
// `VisionEncoder.swift`) plus the already-shipped text engine — and
// declares its image-placeholder token id.
//
// The text backbone is driven through `LanguageModel.forward(
// inputEmbedding:...)` — the embedding-input primitive the VL-target
// families implement. Decode after prefill is ordinary token-id forward.

import Foundation
import Metal

public enum VLModelError: Error, CustomStringConvertible {
    case engineLacksEmbeddingInput(String)
    case placeholderCountMismatch(expected: Int, found: Int)

    public var description: String {
        switch self {
        case .engineLacksEmbeddingInput(let name):
            return "VLModel: text engine \(name) does not support "
                + "embedding-input forward — VLM splice impossible"
        case .placeholderCountMismatch(let expected, let found):
            return "VLModel: prompt has \(found) image-placeholder tokens "
                + "but the vision encoder produced \(expected) image tokens"
        }
    }
}

/// A loaded vision-language model: a `VisionEncoder` + a text
/// `LanguageModel` engine + the image-placeholder token id its chat
/// template uses.
public final class VLModel: @unchecked Sendable {
    /// The vision tower.
    public let visionEncoder: VisionEncoder
    /// The text backbone (Gemma3Model, Qwen3Model, …).
    public let engine: any LanguageModel
    /// Token id the chat template emits as an image placeholder. Each
    /// occurrence is replaced by one vision token.
    public let imageTokenId: Int
    /// Per-channel image normalization the checkpoint expects.
    public let normalization: ImageNormalization
    /// Number of vision tokens one image contributes to the text stream.
    /// Usually `visionEncoder.config.numPatches`, but a family whose
    /// projector pools the patch grid (Gemma 3 VL: 4096 patches → 256
    /// tokens) passes the pooled count explicitly.
    public let imageTokenCount: Int

    public init(visionEncoder: VisionEncoder, engine: any LanguageModel,
                imageTokenId: Int,
                normalization: ImageNormalization,
                imageTokenCount: Int? = nil) throws {
        guard engine.supportsEmbeddingInput else {
            throw VLModelError.engineLacksEmbeddingInput(String(describing: type(of: engine)))
        }
        self.visionEncoder = visionEncoder
        self.engine = engine
        self.imageTokenId = imageTokenId
        self.normalization = normalization
        self.imageTokenCount = imageTokenCount ?? visionEncoder.config.numPatches
    }

    // ─── Cross-modal splice ──────────────────────────────────────────

    /// Build the spliced prompt-embedding stream.
    ///
    /// `promptTokens` is the tokenized prompt, with exactly one
    /// `imageTokenId` per image-token slot. `imageTokens` is the
    /// vision-encoder output, `[numImages * numPatches, hidden]`. Every
    /// `imageTokenId` position in the prompt takes the next vision row;
    /// every other position takes its text-token embedding.
    ///
    /// Returns one `[hidden]` `Tensor` per prompt position — the input
    /// the text backbone's `forward(inputEmbedding:...)` consumes.
    ///
    /// CPU-driven (the spec marks the splice "CPU-fine"); the per-token
    /// embedding lookups + the vision encode are the only GPU work.
    public func splice(promptTokens: [Int], imageTokens: Tensor,
                       device: Device = .shared) throws -> [Tensor] {
        let placeholderCount = promptTokens.filter { $0 == imageTokenId }.count
        let visionRows = imageTokens.shape[0]
        guard placeholderCount == visionRows else {
            throw VLModelError.placeholderCountMismatch(
                expected: visionRows, found: placeholderCount)
        }
        let hidden = engine.hidden
        precondition(imageTokens.shape == [visionRows, hidden],
                     "VLModel.splice: image tokens \(imageTokens.shape) "
                     + "≠ [\(visionRows), \(hidden)]")

        let imageRowBytes = hidden * imageTokens.dtype.byteSize
        var stream: [Tensor] = []
        stream.reserveCapacity(promptTokens.count)
        var visionCursor = 0
        for tok in promptTokens {
            if tok == imageTokenId {
                // Slice the next vision-encoder row as a [hidden] view.
                let row = Tensor(
                    buffer: imageTokens.buffer,
                    offset: imageTokens.offset + visionCursor * imageRowBytes,
                    shape: [hidden], dtype: imageTokens.dtype)
                stream.append(row)
                visionCursor += 1
            } else {
                stream.append(engine.textEmbedding(tokenId: tok, device: device))
            }
        }
        return stream
    }

    /// Encode an image into vision tokens in the text hidden dim.
    public func encodeImage(_ image: RGBImage, device: Device = .shared) -> Tensor {
        let cfg = visionEncoder.config
        let pixels = ImagePreprocessing.preprocess(
            image, targetW: cfg.imageSize, targetH: cfg.imageSize,
            normalization: normalization, dtype: visionEncoder.dtype,
            device: device)
        return visionEncoder.encode(image: pixels, device: device)
    }

    // ─── Generation ──────────────────────────────────────────────────

    /// Greedy multi-modal generation. Runs prefill over the spliced
    /// embedding stream (image tokens injected at the placeholder
    /// positions), then decodes `maxTokens` tokens greedily.
    ///
    /// Returns the generated token ids. Coherence-first: this is the
    /// minimal path the Phase 6.5 integration tests exercise; sampling
    /// filters / streaming are a later pass — the text-only `Generate`
    /// already has them and `VLModel` will route through it once the
    /// engine grows a public embedding-prefill entry point.
    public func generate(promptTokens: [Int], image: RGBImage?,
                         maxTokens: Int, eosTokenId: Int?,
                         device: Device = .shared) throws -> [Int] {
        let caches = engine.makeLayerCaches(device: device)

        // Build the prefill embedding stream.
        let stream: [Tensor]
        if let image {
            let imageTokens = encodeImage(image, device: device)
            stream = try splice(promptTokens: promptTokens,
                                imageTokens: imageTokens, device: device)
        } else {
            // Text-only prompt — every position is a text-token embedding.
            stream = promptTokens.map {
                engine.textEmbedding(tokenId: $0, device: device)
            }
        }

        // Prefill: forward every spliced embedding, keeping the argmax
        // of the final position as the first decoded token.
        var nextToken = 0
        for (pos, embedding) in stream.enumerated() {
            let cmd = device.makeCommandBuffer()
            let logits = engine.forward(
                inputEmbedding: embedding, position: pos,
                caches: caches, on: cmd, device: device)
            let outBuf = device.makeBuffer(length: 4)
            let outT = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
            Ops.argmax(logits, into: outT, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            nextToken = Int(outBuf.contents()
                .bindMemory(to: UInt32.self, capacity: 1).pointee)
        }

        // Decode: ordinary token-id forward from the prefill tail.
        var generated: [Int] = []
        var pos = stream.count
        for _ in 0..<maxTokens {
            if let eos = eosTokenId, nextToken == eos { break }
            generated.append(nextToken)
            let prior = nextToken
            nextToken = engine.forwardSample(
                tokenId: prior, position: pos, caches: caches, device: device)
            pos += 1
        }
        return generated
    }
}
