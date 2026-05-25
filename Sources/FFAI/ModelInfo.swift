// ModelInfo — programmatic probe of a loaded `Model`.
//
// Every field `ffai inspect` prints is also available through this
// struct so tools, GUIs, telemetry agents, and tests can read the
// model's shape without parsing CLI output or downcasting the engine.
//
// `Model.info` builds a fresh `ModelInfo` per call (cheap — every
// field is a property lookup or a one-shot tensor walk). Callers that
// need it repeatedly should cache it themselves.

import Foundation

public struct ModelInfo: Sendable {

    // ── Identity ────────────────────────────────────────────────────────

    /// HuggingFace repo id (or local path) the model was loaded from.
    /// `nil` for models constructed in tests that bypass `Model.load`.
    public let modelId: String?

    /// `architectures[0]` from the checkpoint's `config.json`, e.g.
    /// `"Gemma2ForCausalLM"` or `"Qwen2ForCausalLM"`.
    public let architecture: String?

    /// `model_type` from `config.json`, e.g. `"gemma2"` or `"qwen2"`.
    public let modelType: String?

    /// FFAI engine class name (e.g. `"LlamaModel"`, `"Gemma2Model"`).
    /// Useful for "which family file is driving this" answers without
    /// having to enumerate every typed convenience accessor.
    public let family: String

    // ── Architecture shape ──────────────────────────────────────────────

    /// Activation dtype the engine runs in (f32 / f16 / bf16).
    public let dtype: DType

    public let hidden: Int
    public let nLayers: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let vocab: Int
    /// `max_position_embeddings` — the longest context the model was
    /// trained on (not the runtime KV-cache cap).
    public let maxSeq: Int

    /// GQA fan-out: how many query heads share each KV head. `1` for
    /// vanilla MHA, `2/4/6` typical for Llama / Qwen / Gemma GQA.
    public var gqaFanOut: Int { nHeads / max(nKVHeads, 1) }

    /// Affine-quantization scheme (bits + group size) declared in the
    /// checkpoint's root `config.json`. `nil` for full-precision
    /// checkpoints (raw bf16 / f16 / f32).
    public let quantization: ModelConfig.QuantizationConfig?

    // ── Generation defaults ─────────────────────────────────────────────

    /// Family-specific defaults — temperature, topP/topK, prefill chunk
    /// size, etc. Callers that just want sensible defaults can pass
    /// this straight to `Model.generate(parameters:)`.
    public let defaultGenerationParameters: GenerationParameters

    // ── Capabilities ────────────────────────────────────────────────────

    /// Capabilities the checkpoint physically supports (e.g. `.visionIn`
    /// for a VLM, `.audioIn` for an STT model, always `.textIn`/`.textOut`).
    public let availableCapabilities: Set<Capability>

    /// Capabilities currently enabled — a runtime knob that the caller
    /// can toggle via `Model.enable(_:)` / `Model.disable(_:)`.
    public let enabledCapabilities: Set<Capability>

    // ── Weight footprint ────────────────────────────────────────────────

    /// Total parameters across every weight tensor the engine exposes
    /// via `parameters()`. For quantized models this is the *packed*
    /// element count (uint32 lanes) for the weight tensor plus the
    /// scales/biases element counts — i.e. it represents the on-GPU
    /// footprint, not the dequantized parameter count.
    public let parameterCount: Int

    /// Total bytes across every weight tensor (sum of `tensor.byteCount`
    /// for each entry in `engine.parameters()`). Same packing rules as
    /// `parameterCount` — represents what's resident in GPU memory.
    public let parameterBytes: Int

    // ── Tokenization ────────────────────────────────────────────────────

    public let bosTokenId: Int?
    public let eosTokenIds: [Int]
    /// `tie_word_embeddings` from the checkpoint config. When `true` the
    /// LM-head shares weights with the embedding table.
    public let tieWordEmbeddings: Bool

    /// Whether the engine supports the `forward(inputEmbedding:...)`
    /// VLM-splice path. `true` for Gemma 2 / 3 / 4, Llama, Qwen 3.5 etc.
    public let supportsEmbeddingInput: Bool

    // ── VLM ────────────────────────────────────────────────────────────

    /// `true` when the checkpoint is wrapped in a `VLModel` (a vision
    /// tower + cross-modal splice on top of the text engine). Mirrors
    /// `Model.vlModel != nil`.
    public let isVLM: Bool

    /// Number of vision tokens spliced per image (e.g. 256 for Gemma 3 VL,
    /// 64 for MiniCPM-V 4.6). `nil` for non-VLM models.
    public let imageTokenCount: Int?
}

extension Model {
    /// Build a fresh `ModelInfo` snapshot from this loaded `Model`.
    ///
    /// Cheap — every field is a property read or a one-shot walk over
    /// `engine.parameters()` to sum the weight footprint. Callers that
    /// need this repeatedly should cache it themselves.
    public var info: ModelInfo {
        var paramCount = 0
        var paramBytes = 0
        for (_, t) in engine.parameters() {
            paramCount += t.elementCount
            paramBytes += t.byteCount
        }
        // VLM image-token count is exposed on VLModel; it's also
        // mirror-stored on PaligemmaModel / Gemma3VLComposedEncoder /
        // etc. for the engine-internal-vision families. The VLModel
        // path covers the modern composition; engine-internal-vision
        // checkpoints (Paligemma) get a `numImageTokens` accessor
        // we can pick up via `as?` cast — but to keep this generic,
        // we just read from `vlModel` and leave nil otherwise.
        return ModelInfo(
            modelId: modelDirectory.lastPathComponent.contains("--")
                ? modelDirectory.lastPathComponent
                : modelDirectory.path,
            architecture: config.architecture,
            modelType: config.modelType,
            family: String(describing: type(of: engine)),
            dtype: engine.dtype,
            hidden: engine.hidden,
            nLayers: engine.nLayers,
            nHeads: engine.nHeads,
            nKVHeads: engine.nKVHeads,
            headDim: engine.headDim,
            vocab: engine.vocab,
            maxSeq: engine.maxSeq,
            quantization: config.quantization,
            defaultGenerationParameters: defaultGenerationParameters,
            availableCapabilities: availableCapabilities,
            enabledCapabilities: enabledCapabilities,
            parameterCount: paramCount,
            parameterBytes: paramBytes,
            bosTokenId: config.bosTokenId,
            eosTokenIds: config.eosTokenIds,
            tieWordEmbeddings: config.tieWordEmbeddings,
            supportsEmbeddingInput: engine.supportsEmbeddingInput,
            isVLM: vlModel != nil,
            imageTokenCount: vlModel?.imageTokenCount
        )
    }
}
