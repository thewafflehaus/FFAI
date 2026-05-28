// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Marvis — Conversational Speech Model (CSM) text-to-speech.
//
// Marvis-TTS is built on Sesame's CSM architecture: a dual-transformer
// that turns a text prompt + a reference-speaker segment into a stream
// of Mimi neural-codec frames. A Mimi decoder turns those frames into a
// 24 kHz waveform.
//
//   text + ref-audio ──tokenize──▶ interleaved [text | audio-codebook]
//        frames ──backbone (Llama transformer)──▶ per-frame hidden
//        ──codebook-0 head──▶ codebook 0 ──┐
//        ──depth decoder (small Llama)─────┼─▶ codebooks 1…K-1
//        ──Mimi decoder──▶ waveform
//
// The model is **autoregressive over audio frames**: each step the
// backbone consumes the previous frame's summed codebook embeddings and
// emits a hidden state; the codebook-0 head samples the first code; the
// depth decoder then autoregressively samples the remaining K-1 codes
// for that frame, conditioned on the backbone hidden + codebook-0.
//
// ## Scope note
//
// FFAI's contribution is the **CSM acoustic model**: the backbone and
// depth-decoder transformers (built on FFAI's `LlamaLayer` blocks — the
// same dense-transformer code path the Llama family uses), the
// text / audio-codebook embedding tables, the codebook-0 + per-codebook
// audio heads, and the frame-generation loop.
//
// The **Mimi neural codec** (audio tokenizer + waveform decoder) is a
// separate codec port. `generateFrames` is the supported entry point —
// it returns the `[K, nFrames]` Mimi code matrix a caller feeds to an
// external Mimi decoder. `synthesize` wires the decoder when one is set
// and reports the codec as unavailable otherwise. See `MarvisError`.

import Foundation
import Metal
import Tokenizers

// ─── Configuration ───────────────────────────────────────────────────

/// One CSM sub-transformer's hyper-parameters (backbone or depth
/// decoder). Both are standard Llama dense transformers — the same
/// shape FFAI's `LlamaLayer` consumes.
public struct CSMTransformerConfig: Sendable {
    public let hidden: Int
    public let nLayers: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let intermediate: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: Ops.RoPEScaling

    public init(
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int,
        headDim: Int, intermediate: Int, rmsNormEps: Float,
        ropeTheta: Float, ropeScaling: Ops.RoPEScaling
    ) {
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.intermediate = intermediate
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
    }
}

/// Marvis / CSM model hyper-parameters, decoded from `config.json`.
public struct MarvisConfig: Sendable {
    /// Backbone transformer config.
    public let backbone: CSMTransformerConfig
    /// Depth-decoder transformer config.
    public let decoder: CSMTransformerConfig
    /// Text-side vocabulary size.
    public let textVocabSize: Int
    /// Per-codebook audio vocabulary size.
    public let audioVocabSize: Int
    /// Number of Mimi codebooks per frame (`K`).
    public let audioNumCodebooks: Int
    /// Output waveform sample rate (24 kHz for Mimi).
    public let sampleRate: Int
    /// MLX affine quantization, when the checkpoint is `-4bit` / `-8bit`.
    /// `nil` for full-precision weights. Threaded through the load path
    /// so `QuantizedLinear` / `QuantizedEmbedding` are bound for the
    /// matching weights — without this, mlx-community's quantized
    /// Marvis checkpoints silently bind U32-packed tensors to a dense
    /// `Linear` and produce garbage output.
    public let quantization: ModelConfig.QuantizationConfig?

    public init(
        backbone: CSMTransformerConfig, decoder: CSMTransformerConfig,
        textVocabSize: Int, audioVocabSize: Int,
        audioNumCodebooks: Int, sampleRate: Int = 24_000,
        quantization: ModelConfig.QuantizationConfig? = nil
    ) {
        self.backbone = backbone
        self.decoder = decoder
        self.textVocabSize = textVocabSize
        self.audioVocabSize = audioVocabSize
        self.audioNumCodebooks = audioNumCodebooks
        self.sampleRate = sampleRate
        self.quantization = quantization
    }

    /// Decode a CSM `config.json`. The depth decoder is nested under
    /// `depth_decoder_config`; the backbone fields are top-level.
    public static func from(_ config: ModelConfig) -> MarvisConfig? {
        guard let hidden = config.hiddenSize,
            let nLayers = config.numLayers,
            let nHeads = config.numAttentionHeads,
            let intermediate = config.intermediateSize
        else { return nil }
        let headDim = config.headDim ?? (hidden / nHeads)
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let eps = Float(config.rmsNormEps ?? 1e-5)
        let theta = Float(config.ropeTheta ?? 500_000)
        let scaling = csmRoPEScaling(config.nested("rope_scaling"), headDim: headDim)

        let backbone = CSMTransformerConfig(
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, intermediate: intermediate,
            rmsNormEps: eps, ropeTheta: theta, ropeScaling: scaling)

        // Depth decoder — its own nested config block.
        let decoder: CSMTransformerConfig
        if let d = config.nested("depth_decoder_config"),
            let dHidden = (d["hidden_size"] as? Int),
            let dLayers = (d["num_hidden_layers"] as? Int),
            let dHeads = (d["num_attention_heads"] as? Int),
            let dInter = (d["intermediate_size"] as? Int)
        {
            let dHeadDim = (d["head_dim"] as? Int) ?? (dHidden / dHeads)
            let dKV = (d["num_key_value_heads"] as? Int) ?? dHeads
            let dEps = Float((d["rms_norm_eps"] as? Double) ?? 1e-5)
            let dTheta = Float(
                (d["rope_theta"] as? Int).map(Double.init)
                    ?? (d["rope_theta"] as? Double) ?? 500_000)
            decoder = CSMTransformerConfig(
                hidden: dHidden, nLayers: dLayers, nHeads: dHeads,
                nKVHeads: dKV, headDim: dHeadDim, intermediate: dInter,
                rmsNormEps: dEps, ropeTheta: dTheta,
                ropeScaling: csmRoPEScaling(
                    d["rope_scaling"] as? [String: Any],
                    headDim: dHeadDim))
        } else {
            // Single-transformer fallback — decoder mirrors the backbone.
            decoder = backbone
        }

        guard let textVocab = config.int("text_vocab_size") ?? config.vocabSize,
            let audioVocab = config.int("audio_vocab_size"),
            let nCodebooks = config.int("audio_num_codebooks")
        else { return nil }

        return MarvisConfig(
            backbone: backbone, decoder: decoder,
            textVocabSize: textVocab, audioVocabSize: audioVocab,
            audioNumCodebooks: nCodebooks,
            sampleRate: config.int("sample_rate") ?? 24_000,
            quantization: config.quantization)
    }
}

/// Build `Ops.RoPEScaling` from a CSM `rope_scaling` block (Llama3
/// scaling). Returns `.none` when absent or not the `llama3` type.
private func csmRoPEScaling(_ rs: [String: Any]?, headDim: Int)
    -> Ops.RoPEScaling
{
    guard let rs = rs else { return .none }
    let type = (rs["rope_type"] as? String) ?? (rs["type"] as? String)
    guard type == "llama3" else { return .none }
    func f(_ k: String, _ d: Float) -> Float {
        if let v = rs[k] as? Double { return Float(v) }
        if let v = rs[k] as? Int { return Float(v) }
        return d
    }
    return Ops.RoPEScaling(
        scaleFactor: f("factor", 32),
        lowFreqFactor: f("low_freq_factor", 1),
        highFreqFactor: f("high_freq_factor", 4),
        originalMaxPosition: f("original_max_position_embeddings", 8192))
}

// ─── CSM transformer ─────────────────────────────────────────────────

/// A CSM sub-transformer — a stack of FFAI `LlamaLayer` blocks plus a
/// final RMSNorm. Unlike `LlamaModel` it has no embedding table or
/// lm_head: it consumes an embedding-stream `[hidden]` row per timestep
/// and returns the post-norm hidden state. The backbone and the depth
/// decoder are both instances of this.
public final class CSMTransformer: @unchecked Sendable {
    public let config: CSMTransformerConfig
    public let layers: [LlamaLayer]
    public let finalNorm: RMSNorm
    public let dtype: DType

    public init(
        config: CSMTransformerConfig, layers: [LlamaLayer],
        finalNorm: RMSNorm, dtype: DType
    ) {
        self.config = config
        self.layers = layers
        self.finalNorm = finalNorm
        self.dtype = dtype
    }

    /// One per-layer KV cache for a generation session.
    public func makeCaches(maxSeq: Int, device: Device) -> [KVCache] {
        (0 ..< config.nLayers).map { _ in
            KVCache(
                nKVHeads: config.nKVHeads, headDim: config.headDim,
                contextLength: maxSeq, dtype: dtype, device: device)
        }
    }

    /// Run one timestep through the layer stack. `h` is the `[hidden]`
    /// embedding row; `position` its absolute index. Returns the
    /// post-norm hidden state. All work is queued on `cmd`.
    public func forward(
        _ h: Tensor, position: Int,
        caches: [KVCache], on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
        var x = h
        for (i, layer) in layers.enumerated() {
            x = layer.forward(
                x, position: position, cache: caches[i],
                cmd: cmd, device: device)
        }
        return finalNorm(x, on: cmd)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum MarvisError: Error, CustomStringConvertible {
    /// No Mimi decoder wired — `synthesize` needs the codec tail.
    /// `generateFrames` works regardless.
    case codecUnavailable
    case missingConfig
    case noFrames

    public var description: String {
        switch self {
        case .codecUnavailable:
            return "Marvis: no Mimi decoder is wired in this build; use "
                + "generateFrames to obtain the Mimi code matrix and "
                + "decode it with an external Mimi codec"
        case .missingConfig:
            return "Marvis: required config field missing"
        case .noFrames:
            return "Marvis: generation produced no audio frames"
        }
    }
}

// ─── Mimi codec boundary ─────────────────────────────────────────────

/// The Mimi neural-codec decoder boundary. A Mimi decoder turns the
/// `[K, nFrames]` code matrix CSM emits into a 24 kHz waveform. The
/// concrete codec is a separate port; Marvis accepts any conforming
/// decoder so the acoustic model and the codec can land independently.
public protocol MimiDecoding: Sendable {
    /// Decode a `[K, nFrames]` Mimi code matrix into a `[outLen]`
    /// waveform.
    func decode(codes: [[Int]], device: Device) -> Tensor
}

// ─── Model ───────────────────────────────────────────────────────────

/// A loaded Marvis / CSM TTS model. Owns both transformers, the
/// embedding tables, the audio heads and, when wired, a Mimi decoder.
public final class MarvisModel: @unchecked Sendable {
    public let config: MarvisConfig
    public let backbone: CSMTransformer
    public let decoder: CSMTransformer
    /// Text-token embedding table `[textVocab, backboneHidden]`.
    /// `AnyEmbedding` so quantized (`-4bit` / `-8bit`) checkpoints work.
    public let textEmbeddings: AnyEmbedding
    /// Audio-codebook embedding table — all K codebooks share one table
    /// of `[K * audioVocab, backboneHidden]`, indexed with a per-
    /// codebook offset. `AnyEmbedding` for quantization support.
    public let audioEmbeddings: AnyEmbedding
    /// Backbone-hidden → decoder-hidden projection.
    public let projection: AnyLinear
    /// Codebook-0 logits head `[audioVocab, backboneHidden]`.
    public let codebook0Head: AnyLinear
    /// Per-codebook audio heads for codebooks 1…K-1, stacked as
    /// `[K-1, decoderHidden, audioVocab]`.
    public let audioHead: Tensor
    public let tokenizer: any Tokenizer
    let dtype: DType
    /// The Mimi codec decoder — `nil` until the codec port lands.
    public var mimiDecoder: (any MimiDecoding)?

    public init(
        config: MarvisConfig, backbone: CSMTransformer,
        decoder: CSMTransformer, textEmbeddings: AnyEmbedding,
        audioEmbeddings: AnyEmbedding, projection: AnyLinear,
        codebook0Head: AnyLinear, audioHead: Tensor,
        tokenizer: any Tokenizer, dtype: DType,
        mimiDecoder: (any MimiDecoding)? = nil
    ) {
        self.config = config
        self.backbone = backbone
        self.decoder = decoder
        self.textEmbeddings = textEmbeddings
        self.audioEmbeddings = audioEmbeddings
        self.projection = projection
        self.codebook0Head = codebook0Head
        self.audioHead = audioHead
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.mimiDecoder = mimiDecoder
    }

    public var sampleRate: Int { config.sampleRate }

    // ─── Embedding helpers ───────────────────────────────────────────

    /// Embed a text token id into the backbone hidden dim.
    private func embedText(
        _ tokenId: Int, on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
        let buf = device.makeBuffer(length: 4)
        var v = UInt32(tokenId)
        memcpy(buf.contents(), &v, 4)
        let idT = Tensor(buffer: buf, offset: 0, shape: [1], dtype: .u32)
        return textEmbeddings(idT, on: cmd).reshaped(to: [config.backbone.hidden])
    }

    /// Embed an audio-codebook code. CSM packs all K codebooks into one
    /// table; codebook `cb`'s code `c` is row `cb * audioVocab + c`.
    private func embedAudio(
        codebook cb: Int, code: Int,
        on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
        let row = cb * config.audioVocabSize + code
        let buf = device.makeBuffer(length: 4)
        var v = UInt32(row)
        memcpy(buf.contents(), &v, 4)
        let idT = Tensor(buffer: buf, offset: 0, shape: [1], dtype: .u32)
        return audioEmbeddings(idT, on: cmd).reshaped(to: [config.backbone.hidden])
    }

    // ─── Frame generation ────────────────────────────────────────────

    /// Generate the Mimi code matrix for a prompt. Runs the CSM
    /// autoregressive loop: the backbone consumes the text prompt then,
    /// frame-by-frame, the previous frame's summed codebook embeddings;
    /// the codebook-0 head + depth decoder produce one `[K]` code vector
    /// per frame. Generation stops at an all-zero (EOS) frame or
    /// `maxFrames`.
    ///
    /// Returns `[K, nFrames]` — the Mimi code matrix, one row per
    /// codebook. A speaker prefix `[speaker]` frames the text the way
    /// CSM was trained.
    public func generateFrames(
        text: String, speaker: Int = 0,
        maxFrames: Int = 750,
        temperature: Float = 0.9,
        seed: UInt64 = 0,
        device: Device = .shared
    ) throws -> [[Int]] {
        let K = config.audioNumCodebooks
        let maxSeq = 2048
        let backboneCaches = backbone.makeCaches(maxSeq: maxSeq, device: device)

        // Prefill the backbone with the text prompt. CSM frames text as
        // `[speaker]text`; each text token is its own backbone timestep.
        let promptText = "[\(speaker)]" + text
        let textIds = tokenizer.encode(text: promptText)
        var position = 0
        var backboneHidden = Tensor.empty(
            shape: [config.backbone.hidden],
            dtype: dtype, device: device)
        for tid in textIds {
            let cmd = device.makeCommandBuffer()
            let emb = embedText(tid, on: cmd, device: device)
            backboneHidden = backbone.forward(
                emb, position: position,
                caches: backboneCaches,
                on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            position += 1
        }

        var rng = SeededRandomNumberGenerator(seed: seed)
        var frames: [[Int]] = []

        for _ in 0 ..< maxFrames {
            // Codebook 0 from the backbone hidden state.
            let frame = try generateFrame(
                backboneHidden: backboneHidden, temperature: temperature,
                rng: &rng, device: device)

            // EOS — CSM signals end-of-utterance with an all-zero frame.
            if frame.allSatisfy({ $0 == 0 }) { break }
            frames.append(frame)

            // Feed the frame back into the backbone: sum its K codebook
            // embeddings into one timestep input.
            let cmd = device.makeCommandBuffer()
            var summed = embedAudio(
                codebook: 0, code: frame[0],
                on: cmd, device: device)
            for cb in 1 ..< K {
                let e = embedAudio(
                    codebook: cb, code: frame[cb],
                    on: cmd, device: device)
                summed = Ops.add(summed, e, on: cmd)
            }
            backboneHidden = backbone.forward(
                summed, position: position,
                caches: backboneCaches,
                on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            position += 1
        }

        guard !frames.isEmpty else { throw MarvisError.noFrames }
        // Transpose `[nFrames, K]` → `[K, nFrames]` — one row per codebook.
        var codes = [[Int]](repeating: [], count: K)
        for frame in frames {
            for cb in 0 ..< K { codes[cb].append(frame[cb]) }
        }
        return codes
    }

    /// Generate one `[K]` frame: the codebook-0 head samples code 0 from
    /// the backbone hidden, then the depth decoder autoregressively
    /// samples codebooks 1…K-1.
    private func generateFrame(
        backboneHidden: Tensor, temperature: Float,
        rng: inout SeededRandomNumberGenerator,
        device: Device
    ) throws -> [Int] {
        let K = config.audioNumCodebooks
        var frame = [Int](repeating: 0, count: K)

        // Codebook 0 — backbone hidden → codebook-0 head.
        let cmd0 = device.makeCommandBuffer()
        let c0Logits = codebook0Head(backboneHidden, on: cmd0)
        cmd0.commit()
        cmd0.waitUntilCompleted()
        frame[0] = sampleLogits(c0Logits, temperature: temperature, rng: &rng)

        if K == 1 { return frame }

        // Depth decoder — autoregressively sample codebooks 1…K-1. Its
        // first two timesteps are the projected backbone hidden and the
        // codebook-0 embedding; thereafter each sampled code's embedding.
        let decoderCaches = decoder.makeCaches(maxSeq: K + 1, device: device)
        var decPosition = 0

        let cmdP = device.makeCommandBuffer()
        let projHidden = projection(backboneHidden, on: cmdP)
        _ = decoder.forward(
            projHidden, position: decPosition,
            caches: decoderCaches, on: cmdP, device: device)
        cmdP.commit()
        cmdP.waitUntilCompleted()
        decPosition += 1

        var lastCode = frame[0]
        for cb in 1 ..< K {
            let cmd = device.makeCommandBuffer()
            // Embed the previous codebook's code; project to decoder dim.
            let emb = embedAudio(
                codebook: cb - 1, code: lastCode,
                on: cmd, device: device)
            let projEmb = projection(emb, on: cmd)
            let decHidden = decoder.forward(
                projEmb, position: decPosition,
                caches: decoderCaches,
                on: cmd, device: device)
            // Codebook `cb`'s head is audio_head row `cb - 1`:
            // `[decoderHidden, audioVocab]`. logits = decHidden · head.
            let headRow = audioHeadRow(cb - 1)
            // gemv wants weight `[out, in]`; the head is `[in, out]`, so
            // matmul as decHidden(in) · head(in,out) — done via gemv on
            // the transposed view is unavailable, so use a row-major
            // gemm over a single row.
            let logits = Ops.gemm(
                weight: headRow, input: decHidden,
                nRows: 1, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
            let code = sampleLogits(logits, temperature: temperature, rng: &rng)
            frame[cb] = code
            lastCode = code
            decPosition += 1
        }
        return frame
    }

    /// Slice audio-head plane `i` — the gemm weight `[audioVocab,
    /// decoderHidden]` for codebook `i + 1`. `audioHead` is stored
    /// pre-transposed to `[K-1, audioVocab, decoderHidden]` at load time
    /// (see `loadAudioHead`) precisely so this is a contiguous slice and
    /// a row-major gemm yields `[audioVocab]` logits directly.
    private func audioHeadRow(_ i: Int) -> Tensor {
        let decHidden = config.decoder.hidden
        let audioVocab = config.audioVocabSize
        let plane = decHidden * audioVocab
        return Tensor(
            buffer: audioHead.buffer,
            offset: audioHead.offset + i * plane * audioHead.dtype.byteSize,
            shape: [audioVocab, decHidden], dtype: audioHead.dtype)
    }

    /// Sample a token id from `[vocab]` logits. `temperature == 0` is
    /// greedy argmax; otherwise a temperature-scaled categorical draw.
    private func sampleLogits(
        _ logits: Tensor, temperature: Float,
        rng: inout SeededRandomNumberGenerator
    ) -> Int {
        if temperature <= 0 { return Sampling.argmax(logits) }
        let values = Sampling.decodeF32(logits).map { $0 / temperature }
        // Softmax + inverse-CDF draw.
        let maxV = values.max() ?? 0
        var exps = values.map { Foundation.exp($0 - maxV) }
        let sum = exps.reduce(0, +)
        if sum > 0 { for i in exps.indices { exps[i] /= sum } }
        let draw = Float.random(in: 0 ..< 1, using: &rng)
        var acc: Float = 0
        for (i, p) in exps.enumerated() {
            acc += p
            if draw <= acc { return i }
        }
        return exps.count - 1
    }

    /// Full text→waveform synthesis. Requires a Mimi decoder; throws
    /// `MarvisError.codecUnavailable` when one is not wired.
    public func synthesize(
        text: String, speaker: Int = 0,
        maxFrames: Int = 750,
        temperature: Float = 0.9,
        seed: UInt64 = 0,
        device: Device = .shared
    ) throws -> Tensor {
        guard let decoder = mimiDecoder else {
            throw MarvisError.codecUnavailable
        }
        let codes = try generateFrames(
            text: text, speaker: speaker,
            maxFrames: maxFrames,
            temperature: temperature,
            seed: seed, device: device)
        return decoder.decode(codes: codes, device: device)
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension MarvisModel {
    public static let modelTypes: Set<String> = ["csm", "marvis"]
    public static let architectures: Set<String> = ["CSMForConditionalGeneration"]

    /// Whether a decoded `config.json` describes a Marvis / CSM
    /// checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        // Structural fallback — CSM's distinguishing config fields.
        return config.has("audio_num_codebooks")
            && (config.has("depth_decoder_config") || config.has("backbone_flavor"))
    }

    /// Load a Marvis / CSM checkpoint from a resolved snapshot directory.
    /// Both transformers are built from the checkpoint weights; the Mimi
    /// codec decoder is left unset (separate port — see scope note).
    public static func load(directory: URL, device: Device = .shared)
        async throws -> MarvisModel
    {
        let config = try ModelConfig.load(from: directory)
        guard let mc = MarvisConfig.from(config) else {
            throw MarvisError.missingConfig
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        let tokenizer = try await TokenizerLoader().load(from: directory)
        return try build(
            config: mc, bundle: bundle, tokenizer: tokenizer,
            device: device)
    }

    /// Assemble a `MarvisModel` from a decoded config + a weight bundle.
    /// Factored out so tests can drive it directly.
    public static func build(
        config mc: MarvisConfig,
        bundle: SafeTensorsBundle,
        tokenizer: any Tokenizer,
        device: Device = .shared
    ) throws -> MarvisModel {
        // CSM checkpoints prefix the two transformers `backbone.` and
        // `decoder.` (mlx conversions also carry a leading `model.`).
        let prefix =
            bundle.has("model.backbone.layers.0.input_layernorm.weight")
            ? "model." : ""
        // Dtype probe — the unquantized text-embedding scales (or weight
        // itself if not quantized) reveals the activation precision.
        let dtypeProbe =
            bundle.isQuantized("\(prefix)text_embeddings")
            ? try bundle.tensor(named: "\(prefix)text_embeddings.scales").dtype
            : try bundle.tensor(named: "\(prefix)text_embeddings.weight").dtype
        let dtype = dtypeProbe

        // mlx-community publishes every Marvis checkpoint quantized
        // (`-4bit` / `-8bit`); plumb `quantization` from the parsed config
        // so the load path actually instantiates `QuantizedLinear` /
        // `QuantizedEmbedding` instead of binding U32-packed tensors to
        // dense Linear/Embedding modules (which silently produces garbage).
        let quant = mc.quantization

        let backboneT = try buildTransformer(
            base: "\(prefix)backbone", config: mc.backbone,
            bundle: bundle, dtype: dtype, quantization: quant)
        let decoderT = try buildTransformer(
            base: "\(prefix)decoder", config: mc.decoder,
            bundle: bundle, dtype: dtype, quantization: quant)

        let textEmb = try loadEmbedding(
            base: "\(prefix)text_embeddings", in: bundle,
            hidden: mc.backbone.hidden, quantization: quant)
        let audioEmb = try loadEmbedding(
            base: "\(prefix)audio_embeddings", in: bundle,
            hidden: mc.backbone.hidden, quantization: quant)
        let projection = try loadLinear(
            base: "\(prefix)projection", in: bundle, quantization: quant)
        let codebook0 = try loadLinear(
            base: "\(prefix)codebook0_head", in: bundle, quantization: quant)
        // audio_head is stored `[K-1, decoderHidden, audioVocab]`; the
        // per-codebook gemm wants `[audioVocab, decoderHidden]`. Transpose
        // each plane once at load so `audioHeadRow` is a contiguous slice.
        let audioHead = try loadAudioHead(
            try bundle.tensor(named: "\(prefix)audio_head"),
            decoderHidden: mc.decoder.hidden, audioVocab: mc.audioVocabSize,
            dtype: dtype, device: device)

        return MarvisModel(
            config: mc, backbone: backboneT, decoder: decoderT,
            textEmbeddings: textEmb, audioEmbeddings: audioEmb,
            projection: projection, codebook0Head: codebook0,
            audioHead: audioHead, tokenizer: tokenizer, dtype: dtype)
    }

    /// Build one CSM sub-transformer (a `LlamaLayer` stack + final norm)
    /// from `base`-prefixed checkpoint weights.
    private static func buildTransformer(
        base: String, config c: CSMTransformerConfig,
        bundle: SafeTensorsBundle, dtype: DType,
        quantization q: ModelConfig.QuantizationConfig?
    ) throws -> CSMTransformer {
        var layers: [LlamaLayer] = []
        layers.reserveCapacity(c.nLayers)
        for i in 0 ..< c.nLayers {
            let p = "\(base).layers.\(i)"
            layers.append(
                LlamaLayer(
                    qProj: try loadLinear(
                        base: "\(p).self_attn.q_proj", in: bundle, quantization: q),
                    kProj: try loadLinear(
                        base: "\(p).self_attn.k_proj", in: bundle, quantization: q),
                    vProj: try loadLinear(
                        base: "\(p).self_attn.v_proj", in: bundle, quantization: q),
                    oProj: try loadLinear(
                        base: "\(p).self_attn.o_proj", in: bundle, quantization: q),
                    gateProj: try loadLinear(
                        base: "\(p).mlp.gate_proj", in: bundle, quantization: q),
                    upProj: try loadLinear(base: "\(p).mlp.up_proj", in: bundle, quantization: q),
                    downProj: try loadLinear(
                        base: "\(p).mlp.down_proj", in: bundle, quantization: q),
                    inputNorm: RMSNorm(
                        weight: try bundle.tensor(named: "\(p).input_layernorm.weight"),
                        eps: c.rmsNormEps),
                    postAttnNorm: RMSNorm(
                        weight: try bundle.tensor(named: "\(p).post_attention_layernorm.weight"),
                        eps: c.rmsNormEps),
                    hidden: c.hidden, nHeads: c.nHeads, nKVHeads: c.nKVHeads,
                    headDim: c.headDim, intermediate: c.intermediate,
                    ropeTheta: c.ropeTheta, ropeScaling: c.ropeScaling))
        }
        let finalNorm = RMSNorm(
            weight: try bundle.tensor(named: "\(base).norm.weight"),
            eps: c.rmsNormEps)
        return CSMTransformer(
            config: c, layers: layers,
            finalNorm: finalNorm, dtype: dtype)
    }

    /// Transpose the stored `audio_head` from
    /// `[K-1, decoderHidden, audioVocab]` to `[K-1, audioVocab,
    /// decoderHidden]` — the per-codebook gemm weight layout. The
    /// transpose runs once on the CPU at load time; generation then
    /// slices contiguous `[audioVocab, decoderHidden]` planes. The
    /// result is materialized in `dtype` — the activation dtype the
    /// decoder hidden runs in — so the per-codebook gemm dtype-matches.
    private static func loadAudioHead(
        _ stored: Tensor, decoderHidden: Int,
        audioVocab: Int, dtype: DType,
        device: Device
    )
        throws -> Tensor
    {
        let planeElems = decoderHidden * audioVocab
        precondition(
            stored.elementCount % planeElems == 0,
            "Marvis.loadAudioHead: audio_head shape "
                + "\(stored.shape) is not a multiple of "
                + "decoderHidden*audioVocab")
        let nPlanes = stored.elementCount / planeElems
        let src = stored.toFloatArray()
        var dst = [Float](repeating: 0, count: src.count)
        for p in 0 ..< nPlanes {
            let base = p * planeElems
            // [decoderHidden, audioVocab] → [audioVocab, decoderHidden].
            for d in 0 ..< decoderHidden {
                for v in 0 ..< audioVocab {
                    dst[base + v * decoderHidden + d] = src[base + d * audioVocab + v]
                }
            }
        }
        let out = Tensor.empty(
            shape: [nPlanes, audioVocab, decoderHidden],
            dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(dst, into: out)
        return out
    }
}
