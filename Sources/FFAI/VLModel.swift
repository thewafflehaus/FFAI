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
// `VLModel` from its checkpoint — a `VisionEncoder` (.5
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
    case videoTokenIdMissing(String)
    case videoPlaceholderCountMismatch(expected: Int, found: Int)

    public var description: String {
        switch self {
        case .engineLacksEmbeddingInput(let name):
            return "VLModel: text engine \(name) does not support "
                + "embedding-input forward — VLM splice impossible"
        case .placeholderCountMismatch(let expected, let found):
            return "VLModel: prompt has \(found) image-placeholder tokens "
                + "but the vision encoder produced \(expected) image tokens"
        case .videoTokenIdMissing(let family):
            return "VLModel: \(family) was built without a videoTokenId; "
                + "generate(promptTokens:videoFrames:...) needs the family "
                + "loader to thread `video_token_index` through to VLModel.init"
        case .videoPlaceholderCountMismatch(let expected, let found):
            return "VLModel: prompt has \(found) video-placeholder tokens "
                + "but the vision encoder produced \(expected) video tokens"
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
    /// Token id the chat template emits as a video-frame placeholder
    /// (`<|video_pad|>` on Qwen 2/2.5/3 VL). Each occurrence is replaced
    /// by one of the per-temporal-patch merged vision tokens emitted by
    /// the vision tower's `encode(frames:)`. `nil` for families that
    /// don't wire the multi-frame temporal-patch path.
    public let videoTokenId: Int?
    /// Per-channel image normalization the checkpoint expects.
    public let normalization: ImageNormalization
    /// Number of vision tokens one image contributes to the text stream.
    /// Usually `visionEncoder.config.numPatches`, but a family whose
    /// projector pools the patch grid (Gemma 3 VL: 4096 patches → 256
    /// tokens) passes the pooled count explicitly.
    public let imageTokenCount: Int

    public init(visionEncoder: VisionEncoder, engine: any LanguageModel,
                imageTokenId: Int,
                videoTokenId: Int? = nil,
                normalization: ImageNormalization,
                imageTokenCount: Int? = nil) throws {
        guard engine.supportsEmbeddingInput else {
            throw VLModelError.engineLacksEmbeddingInput(String(describing: type(of: engine)))
        }
        self.visionEncoder = visionEncoder
        self.engine = engine
        self.imageTokenId = imageTokenId
        self.videoTokenId = videoTokenId
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

    /// Encode a sequence of video frames into vision tokens in the text
    /// hidden dim. Each frame is preprocessed (resize → normalize) the
    /// same way `encodeImage(...)` is — the vision tower's temporal
    /// folding (one temporal patch = `temporal_patch_size` consecutive
    /// frames) happens inside `VisionEncoder.encode(frames:)`. Throws
    /// `VisionEncoderError.videoUnsupported` for towers that don't wire
    /// the multi-frame path.
    public func encodeVideo(_ frames: [RGBImage],
                            device: Device = .shared) throws -> Tensor {
        precondition(!frames.isEmpty,
                     "VLModel.encodeVideo: expected at least one frame")
        let cfg = visionEncoder.config
        let pixels: [Tensor] = frames.map { frame in
            ImagePreprocessing.preprocess(
                frame, targetW: cfg.imageSize, targetH: cfg.imageSize,
                normalization: normalization, dtype: visionEncoder.dtype,
                device: device)
        }
        return try visionEncoder.encode(frames: pixels, device: device)
    }

    /// Cross-modal splice driver — builds the embedding stream by
    /// walking `promptTokens` and substituting vision-encoder rows at
    /// each image-placeholder / video-placeholder position. The image
    /// and video token streams are consumed independently (each carries
    /// its own cursor) so a prompt may interleave both.
    private func spliceMultimodal(promptTokens: [Int],
                                  imageTokens: Tensor?,
                                  videoTokens: Tensor?,
                                  device: Device) throws -> [Tensor] {
        let hidden = engine.hidden
        // Validate placeholder counts up-front so we fail fast on
        // mismatched prompts.
        let imagePlaceholderCount = promptTokens.filter { $0 == imageTokenId }.count
        let imageVisionRows = imageTokens?.shape[0] ?? 0
        guard imagePlaceholderCount == imageVisionRows else {
            throw VLModelError.placeholderCountMismatch(
                expected: imageVisionRows, found: imagePlaceholderCount)
        }
        let videoPlaceholderCount: Int
        if let videoTokenId {
            videoPlaceholderCount = promptTokens.filter { $0 == videoTokenId }.count
        } else {
            videoPlaceholderCount = 0
        }
        let videoVisionRows = videoTokens?.shape[0] ?? 0
        guard videoPlaceholderCount == videoVisionRows else {
            throw VLModelError.videoPlaceholderCountMismatch(
                expected: videoVisionRows, found: videoPlaceholderCount)
        }
        if let imageTokens {
            precondition(imageTokens.shape == [imageVisionRows, hidden],
                         "VLModel.splice: image tokens \(imageTokens.shape) "
                         + "≠ [\(imageVisionRows), \(hidden)]")
        }
        if let videoTokens {
            precondition(videoTokens.shape == [videoVisionRows, hidden],
                         "VLModel.splice: video tokens \(videoTokens.shape) "
                         + "≠ [\(videoVisionRows), \(hidden)]")
        }
        // Byte stride for slicing a single [hidden] row out of the
        // vision token buffer — same for image and video.
        let imageRowBytes = imageTokens.map {
            hidden * $0.dtype.byteSize
        } ?? 0
        let videoRowBytes = videoTokens.map {
            hidden * $0.dtype.byteSize
        } ?? 0

        var stream: [Tensor] = []
        stream.reserveCapacity(promptTokens.count)
        var imageCursor = 0
        var videoCursor = 0
        for tok in promptTokens {
            if tok == imageTokenId, let imageTokens {
                let row = Tensor(
                    buffer: imageTokens.buffer,
                    offset: imageTokens.offset + imageCursor * imageRowBytes,
                    shape: [hidden], dtype: imageTokens.dtype)
                stream.append(row)
                imageCursor += 1
            } else if let videoTokenId, tok == videoTokenId, let videoTokens {
                let row = Tensor(
                    buffer: videoTokens.buffer,
                    offset: videoTokens.offset + videoCursor * videoRowBytes,
                    shape: [hidden], dtype: videoTokens.dtype)
                stream.append(row)
                videoCursor += 1
            } else {
                stream.append(engine.textEmbedding(tokenId: tok, device: device))
            }
        }
        return stream
    }

    // ─── Generation ──────────────────────────────────────────────────

    /// Greedy multi-modal generation. Runs prefill over the spliced
    /// embedding stream (image tokens injected at the placeholder
    /// positions), then decodes `maxTokens` tokens greedily.
    ///
    /// Returns the generated token ids. Coherence-first: this is the
    /// minimal path the integration tests exercise; sampling
    /// filters / streaming are a later pass — the text-only `Generate`
    /// already has them and `VLModel` will route through it once the
    /// engine grows a public embedding-prefill entry point.
    public func generate(promptTokens: [Int], image: RGBImage?,
                         maxTokens: Int, eosTokenId: Int?,
                         eosTokenIds: [Int] = [],
                         device: Device = .shared) throws -> [Int] {
        let imageTokens = image.map { encodeImage($0, device: device) }
        return try runGreedy(
            promptTokens: promptTokens,
            imageTokens: imageTokens, videoTokens: nil,
            maxTokens: maxTokens, eosTokenId: eosTokenId,
            eosTokenIds: eosTokenIds, device: device)
    }

    /// Greedy multi-modal generation with a video prompt. `videoFrames`
    /// is the ordered list of RGB frames (the caller decides sampling —
    /// `VLMTestSupport.catVideoFrames(...)` produces 1–N evenly spaced
    /// frames); the vision tower folds them into temporal patches and
    /// the splice substitutes one vision-token row at every
    /// `<|video_pad|>` placeholder.
    ///
    /// Throws `VLModelError.videoTokenIdMissing` for VLMs that were
    /// built without a `videoTokenId` (`Gemma3VL`, `Paligemma`,
    /// `FastVLM`, …), or
    /// `VLModelError.videoPlaceholderCountMismatch` if the prompt
    /// doesn't contain exactly `(T/temporal_patch_size) × spatial`
    /// `<|video_pad|>` placeholders.
    public func generate(promptTokens: [Int], videoFrames: [RGBImage],
                         maxTokens: Int, eosTokenId: Int?,
                         eosTokenIds: [Int] = [],
                         device: Device = .shared) throws -> [Int] {
        guard videoTokenId != nil else {
            throw VLModelError.videoTokenIdMissing(
                String(describing: type(of: visionEncoder)))
        }
        let videoTokens = try encodeVideo(videoFrames, device: device)
        return try runGreedy(
            promptTokens: promptTokens,
            imageTokens: nil, videoTokens: videoTokens,
            maxTokens: maxTokens, eosTokenId: eosTokenId,
            eosTokenIds: eosTokenIds, device: device)
    }

    /// Shared greedy-decode driver — builds the prefill embedding
    /// stream (with optional pre-encoded image / video tokens), runs
    /// prefill, then decodes tokens one at a time. Identical control
    /// flow to the legacy `generate(promptTokens:image:...)` path.
    private func runGreedy(promptTokens: [Int],
                           imageTokens: Tensor?, videoTokens: Tensor?,
                           maxTokens: Int, eosTokenId: Int?,
                           eosTokenIds: [Int],
                           device: Device) throws -> [Int] {
        let caches = engine.makeLayerCaches(device: device)

        // Build the prefill embedding stream.
        let stream: [Tensor]
        if imageTokens != nil || videoTokens != nil {
            stream = try spliceMultimodal(
                promptTokens: promptTokens,
                imageTokens: imageTokens, videoTokens: videoTokens,
                device: device)
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
        // Stop on any EOS — Gemma 3+ etc. publish `eos_token_id` as a
        // list; the caller passes both `eosTokenId` (back-compat single
        // id) and the optional `eosTokenIds` list.
        var stopSet: Set<Int> = Set(eosTokenIds)
        if let single = eosTokenId { stopSet.insert(single) }
        var generated: [Int] = []
        var pos = stream.count
        for _ in 0..<maxTokens {
            if stopSet.contains(nextToken) { break }
            generated.append(nextToken)
            let prior = nextToken
            nextToken = engine.forwardSample(
                tokenId: prior, position: pos, caches: caches, device: device)
            pos += 1
        }
        return generated
    }
}
