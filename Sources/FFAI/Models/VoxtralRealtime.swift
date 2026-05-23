// VoxtralRealtime — Mistral AI's streaming speech-to-text family
// (Voxtral-Mini-4B-Realtime). Architecture:
//
//   waveform  ──Slaney log-Mel──▶ [nMels, nFrames]
//             ──causal Conv1d stem (2 × stride-1/2)──▶ [nFrames', d]
//             ──32 × RoPE causal transformer layers (sliding window 750)──▶
//             ──RMSNorm──▶ ──downsample×4──▶ ──2-layer GELU MLP──▶
//             audio features [nAudioTokens, decoderDim]
//
//   BOS + nLeftPad × pad + nDelay × pad  ←→  audio features (element-wise add)
//   ──Mistral text decoder (26 layers, GQA 32/8, AdaRMSNorm time cond)──▶
//   transcript tokens
//
// The audio encoder uses a custom Slaney mel filterbank (fMax = 8 kHz,
// norm = "slaney", mel scale = slaney) rather than the Whisper HTK
// filterbank. Normalization: log10, clamp to (globalLogMelMax − 8),
// then (x + 4) / 4.
//
// Conv1d weights in the checkpoint are [outCh, kernelSize, inCh]
// (MLX NLC); FFAI's Ops.audioConv1d expects [outCh, inCh, k] (NCL),
// so they are transposed on load.
//
// Detection: `model_type == "voxtral_realtime"`.

import Foundation
import Metal

// ─── Configuration ────────────────────────────────────────────────────

/// Audio-encoding front-end parameters for VoxtralRealtime.
public struct VoxtralRealtimeAudioConfig: Sendable {
    /// Sample rate in Hz (always 16000).
    public let samplingRate: Int
    /// Frames per second (always 12.5 → 80 samples per frame at hop 160).
    public let frameRate: Float
    /// Number of Mel filterbank bins (always 128).
    public let numMelBins: Int
    /// STFT hop length in samples (always 160).
    public let hopLength: Int
    /// STFT window size in samples (always 400).
    public let windowSize: Int
    /// Voxtral's Slaney log-Mel clamp ceiling (default 1.5).
    public let globalLogMelMax: Float

    public init(
        samplingRate: Int = 16_000,
        frameRate: Float = 12.5,
        numMelBins: Int = 128,
        hopLength: Int = 160,
        windowSize: Int = 400,
        globalLogMelMax: Float = 1.5
    ) {
        self.samplingRate = samplingRate
        self.frameRate = frameRate
        self.numMelBins = numMelBins
        self.hopLength = hopLength
        self.windowSize = windowSize
        self.globalLogMelMax = globalLogMelMax
    }
}

/// Voxtral audio encoder (Mistal-based streaming transformer) parameters.
public struct VoxtralRealtimeEncoderConfig: Sendable {
    /// Encoder hidden dim.
    public let dim: Int
    /// Number of transformer layers.
    public let nLayers: Int
    /// Number of query heads.
    public let nHeads: Int
    /// Per-head key/query dimension.
    public let headDim: Int
    /// Feed-forward intermediate dim.
    public let hiddenDim: Int
    /// Number of KV heads (GQA).
    public let nKVHeads: Int
    /// RMSNorm epsilon.
    public let normEps: Float
    /// RoPE base frequency.
    public let ropeTheta: Float
    /// Sliding window size (attention scope).
    public let slidingWindow: Int
    /// Whether convolutions / attention are causal.
    public let causal: Bool
    /// Whether attention projections carry biases.
    public let useBiases: Bool
    /// Temporal downsampling factor applied after the transformer.
    public let downsampleFactor: Int

    public init(
        dim: Int = 1280,
        nLayers: Int = 32,
        nHeads: Int = 32,
        headDim: Int = 64,
        hiddenDim: Int = 5120,
        nKVHeads: Int = 32,
        normEps: Float = 1e-5,
        ropeTheta: Float = 1_000_000,
        slidingWindow: Int = 750,
        causal: Bool = true,
        useBiases: Bool = true,
        downsampleFactor: Int = 4
    ) {
        self.dim = dim
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.nKVHeads = nKVHeads
        self.normEps = normEps
        self.ropeTheta = ropeTheta
        self.slidingWindow = slidingWindow
        self.causal = causal
        self.useBiases = useBiases
        self.downsampleFactor = downsampleFactor
    }
}

/// Voxtral text decoder parameters.
public struct VoxtralRealtimeDecoderConfig: Sendable {
    /// Decoder hidden dim.
    public let dim: Int
    /// Number of decoder layers.
    public let nLayers: Int
    /// Number of query heads.
    public let nHeads: Int
    /// Number of KV heads (GQA).
    public let nKVHeads: Int
    /// Per-head dimension.
    public let headDim: Int
    /// Feed-forward intermediate dim.
    public let hiddenDim: Int
    /// Vocabulary size.
    public let vocabSize: Int
    /// RMSNorm epsilon.
    public let normEps: Float
    /// RoPE base frequency.
    public let ropeTheta: Float
    /// Decoder sliding window size.
    public let slidingWindow: Int
    /// Whether embedding weights are tied to lm_head.
    public let tiedEmbeddings: Bool
    /// Whether AdaRMSNorm time conditioning is applied in every layer.
    public let adaRmsNormTCond: Bool
    /// Bottleneck dim for AdaRMSNorm time conditioning.
    public let adaRmsNormTCondDim: Int

    public init(
        dim: Int = 3072,
        nLayers: Int = 26,
        nHeads: Int = 32,
        nKVHeads: Int = 8,
        headDim: Int = 128,
        hiddenDim: Int = 9216,
        vocabSize: Int = 131072,
        normEps: Float = 1e-5,
        ropeTheta: Float = 1_000_000,
        slidingWindow: Int = 8192,
        tiedEmbeddings: Bool = true,
        adaRmsNormTCond: Bool = true,
        adaRmsNormTCondDim: Int = 32
    ) {
        self.dim = dim
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.vocabSize = vocabSize
        self.normEps = normEps
        self.ropeTheta = ropeTheta
        self.slidingWindow = slidingWindow
        self.tiedEmbeddings = tiedEmbeddings
        self.adaRmsNormTCond = adaRmsNormTCond
        self.adaRmsNormTCondDim = adaRmsNormTCondDim
    }
}

/// Top-level VoxtralRealtime configuration, decoded from `config.json`.
public struct VoxtralRealtimeConfig: Sendable {
    public let audioConfig: VoxtralRealtimeAudioConfig
    public let encoderConfig: VoxtralRealtimeEncoderConfig
    public let decoderConfig: VoxtralRealtimeDecoderConfig
    /// Transcription delay in milliseconds (default 480 ms).
    public let transcriptionDelayMs: Int
    /// BOS token id (Tekken vocabulary, always 1).
    public let bosTokenId: Int
    /// EOS token id.
    public let eosTokenId: Int
    /// Streaming pad token id used to fill the delay prefix.
    public let streamingPadTokenId: Int
    /// Number of left-pad tokens prepended to every prompt.
    public let nLeftPadTokens: Int

    public init(
        audioConfig: VoxtralRealtimeAudioConfig = VoxtralRealtimeAudioConfig(),
        encoderConfig: VoxtralRealtimeEncoderConfig = VoxtralRealtimeEncoderConfig(),
        decoderConfig: VoxtralRealtimeDecoderConfig = VoxtralRealtimeDecoderConfig(),
        transcriptionDelayMs: Int = 480,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        streamingPadTokenId: Int = 32,
        nLeftPadTokens: Int = 32
    ) {
        self.audioConfig = audioConfig
        self.encoderConfig = encoderConfig
        self.decoderConfig = decoderConfig
        self.transcriptionDelayMs = transcriptionDelayMs
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.streamingPadTokenId = streamingPadTokenId
        self.nLeftPadTokens = nLeftPadTokens
    }

    /// Decode from a `ModelConfig` (top-level `config.json` dict).
    public static func from(_ mc: ModelConfig) -> VoxtralRealtimeConfig? {
        guard mc.modelType == "voxtral_realtime" else { return nil }

        // Helper: integer from a nested sub-dict key.
        func ni(_ dict: [String: Any]?, _ key: String) -> Int? {
            guard let d = dict else { return nil }
            if let v = d[key] as? Int { return v }
            if let v = d[key] as? Double { return Int(v) }
            return nil
        }
        func nf(_ dict: [String: Any]?, _ key: String) -> Float? {
            guard let d = dict else { return nil }
            if let v = d[key] as? Double { return Float(v) }
            if let v = d[key] as? Int { return Float(v) }
            return nil
        }
        func nb(_ dict: [String: Any]?, _ key: String) -> Bool? {
            (dict)?[key] as? Bool
        }

        let encRaw   = mc.nested("encoder_args")
        let decRaw   = mc.nested("decoder")
        // audio_encoding_args is nested inside encoder_args in the checkpoint.
        let audioRaw = encRaw.flatMap { $0["audio_encoding_args"] as? [String: Any] }
            ?? mc.nested("audio_encoding_args")

        let audioConfig = VoxtralRealtimeAudioConfig(
            samplingRate:   ni(audioRaw, "sampling_rate") ?? 16_000,
            frameRate:      nf(audioRaw, "frame_rate")    ?? 12.5,
            numMelBins:     ni(audioRaw, "num_mel_bins")  ?? 128,
            hopLength:      ni(audioRaw, "hop_length")    ?? 160,
            windowSize:     ni(audioRaw, "window_size")   ?? 400,
            globalLogMelMax: nf(audioRaw, "global_log_mel_max") ?? 1.5
        )

        let encoderConfig = VoxtralRealtimeEncoderConfig(
            dim:             ni(encRaw, "dim")             ?? 1280,
            nLayers:         ni(encRaw, "n_layers")        ?? 32,
            nHeads:          ni(encRaw, "n_heads")         ?? 32,
            headDim:         ni(encRaw, "head_dim")        ?? 64,
            hiddenDim:       ni(encRaw, "hidden_dim")      ?? 5120,
            nKVHeads:        ni(encRaw, "n_kv_heads")      ?? 32,
            normEps:         nf(encRaw, "norm_eps")        ?? 1e-5,
            ropeTheta:       nf(encRaw, "rope_theta")      ?? 1_000_000,
            slidingWindow:   ni(encRaw, "sliding_window")  ?? 750,
            causal:          nb(encRaw, "causal")          ?? true,
            useBiases:       nb(encRaw, "use_biases")      ?? true,
            downsampleFactor: ni(encRaw, "downsample_factor") ?? 4
        )

        let decoderConfig = VoxtralRealtimeDecoderConfig(
            dim:             ni(decRaw, "dim")             ?? 3072,
            nLayers:         ni(decRaw, "n_layers")        ?? 26,
            nHeads:          ni(decRaw, "n_heads")         ?? 32,
            nKVHeads:        ni(decRaw, "n_kv_heads")      ?? 8,
            headDim:         ni(decRaw, "head_dim")        ?? 128,
            hiddenDim:       ni(decRaw, "hidden_dim")      ?? 9216,
            vocabSize:       ni(decRaw, "vocab_size")      ?? 131072,
            normEps:         nf(decRaw, "norm_eps")        ?? 1e-5,
            ropeTheta:       nf(decRaw, "rope_theta")      ?? 1_000_000,
            slidingWindow:   ni(decRaw, "sliding_window")  ?? 8192,
            tiedEmbeddings:  nb(decRaw, "tied_embeddings") ?? true,
            adaRmsNormTCond: nb(decRaw, "ada_rms_norm_t_cond") ?? true,
            adaRmsNormTCondDim: ni(decRaw, "ada_rms_norm_t_cond_dim") ?? 32
        )

        let delayMs = (mc.raw["transcription_delay_ms"] as? Int)
            ?? (mc.raw["transcription_delay_ms"] as? Double).map(Int.init) ?? 480
        let bosId   = (mc.raw["bos_token_id"]           as? Int) ?? 1
        let eosId   = (mc.raw["eos_token_id"]           as? Int) ?? 2
        let padId   = (mc.raw["streaming_pad_token_id"] as? Int) ?? 32
        let nLeft   = (mc.raw["n_left_pad_tokens"]      as? Int) ?? 32

        return VoxtralRealtimeConfig(
            audioConfig: audioConfig,
            encoderConfig: encoderConfig,
            decoderConfig: decoderConfig,
            transcriptionDelayMs: delayMs,
            bosTokenId: bosId,
            eosTokenId: eosId,
            streamingPadTokenId: padId,
            nLeftPadTokens: nLeft
        )
    }
}

// ─── Encoder weights ──────────────────────────────────────────────────

/// Weights for one VoxtralRealtime encoder attention block.
// @unchecked Sendable: RMSNorm and AnyLinear wrap MTLBuffer which is thread-safe
// after construction; no mutation occurs after load.
public struct VoxtralEncoderAttentionWeights: @unchecked Sendable {
    /// Query projection [nHeads * headDim, dim]. Optional bias [nHeads * headDim].
    public let wqWeight: Tensor
    public let wqBias: Tensor?
    /// Key projection [nKVHeads * headDim, dim]. No bias on wk.
    public let wkWeight: Tensor
    /// Value projection [dim, dim]. Optional bias.
    public let wvWeight: Tensor
    public let wvBias: Tensor?
    /// Output projection [dim, nHeads * headDim]. Optional bias.
    public let woWeight: Tensor
    public let woBias: Tensor?
}

/// Weights for one VoxtralRealtime encoder transformer layer.
// @unchecked Sendable: RMSNorm wraps MTLBuffer; immutable after construction.
public struct VoxtralEncoderLayerWeights: @unchecked Sendable {
    public let attnNorm: RMSNorm
    public let attn: VoxtralEncoderAttentionWeights
    public let ffnNorm: RMSNorm
    /// w1 / w3 for SwiGLU gate; w2 for output.
    /// Stored as Linear (encoder FFN weights are never quantized in the
    /// mlx-community Voxtral checkpoints; only decoder layers are quantized).
    public let w1Weight: Linear
    public let w2Weight: Linear
    public let w3Weight: Linear
}

// ─── Decoder weights ──────────────────────────────────────────────────

/// Weights for the AdaRMSNorm time-conditioning scale in each decoder layer.
// @unchecked Sendable: AnyLinear wraps MTLBuffer; immutable after construction.
public struct VoxtralAdaRMSNormWeights: @unchecked Sendable {
    /// ada_down: [bottleneckDim, dim]; ada_up: [dim, bottleneckDim].
    public let adaDown: AnyLinear
    public let adaUp: AnyLinear
}

/// Weights for one VoxtralRealtime decoder layer.
// @unchecked Sendable: RMSNorm and AnyLinear wrap MTLBuffer; immutable after construction.
public struct VoxtralDecoderLayerWeights: @unchecked Sendable {
    public let attnNorm: RMSNorm
    public let wqWeight: AnyLinear
    public let wkWeight: AnyLinear
    public let wvWeight: AnyLinear
    public let woWeight: AnyLinear
    public let ffnNorm: RMSNorm
    /// Optional time conditioning (present on every layer in Mini-4B).
    public let adaRmsNorm: VoxtralAdaRMSNormWeights?
    public let w1Weight: AnyLinear
    public let w2Weight: AnyLinear
    public let w3Weight: AnyLinear
}

// ─── VoxtralRealtimeModel ─────────────────────────────────────────────

/// A loaded VoxtralRealtime speech-to-text model.
///
/// The model has two parts:
///   * An audio encoder: causal Conv1d stem → 32-layer sliding-window
///     RoPE transformer → downsample×4 → 2-layer GELU MLP.
///   * A Mistral text decoder with AdaRMSNorm time conditioning.
///
/// `transcribe(waveform:maxTokens:device:)` is the main entry point.
/// `encodeAudio(waveform:device:)` returns the audio feature tensor
/// `[nAudioTokens, decoderDim]` which can be inspected independently.
public final class VoxtralRealtimeModel: @unchecked Sendable {
    public let config: VoxtralRealtimeConfig

    // ── Encoder weights ──────────────────────────────────────────────
    /// Causal Conv1d stem layer 0: weight [dim, numMelBins, 3] (NCL after load).
    let conv0Weight: Tensor
    let conv0Bias: Tensor
    /// Causal Conv1d stem layer 1: weight [dim, dim, 3] (NCL), stride 2.
    let conv1Weight: Tensor
    let conv1Bias: Tensor
    let encoderLayers: [VoxtralEncoderLayerWeights]
    let encoderNorm: RMSNorm
    /// Audio→language projection 0: weight [decoderDim, dim*downsampleFactor] — GELU.
    /// Stored as plain Linear: audio projection weights are never quantized.
    let audioProj0: Linear
    /// Audio→language projection 2: weight [decoderDim, decoderDim].
    let audioProj2: Linear

    // ── Decoder weights ──────────────────────────────────────────────
    let tokEmbeddings: AnyEmbedding
    let decoderLayers: [VoxtralDecoderLayerWeights]
    let decoderNorm: RMSNorm
    /// lm_head (tied to tokEmbeddings in Mini-4B).
    let lmHead: AnyLinear

    /// Precomputed AdaRMSNorm scales — one per decoder layer.
    /// Computed once per transcription call (depends on delay tokens).
    private var cachedAdaScales: [[Float]?] = []
    private var cachedDelayTokens: Int = -1

    let dtype: DType

    // ─── Tekken tokenizer ─────────────────────────────────────────────

    /// Custom Tekken byte-pair tokenizer loaded from `tekken.json`.
    private struct TekkenTokenizer: @unchecked Sendable {
        // Vocab entries in base64-encoded raw bytes.
        struct VocabEntry: Decodable {
            let tokenBytes: String
            enum CodingKeys: String, CodingKey { case tokenBytes = "token_bytes" }
        }
        struct Config: Decodable {
            let defaultNumSpecialTokens: Int?
            enum CodingKeys: String, CodingKey {
                case defaultNumSpecialTokens = "default_num_special_tokens"
            }
        }
        struct SpecialToken: Decodable {
            let rank: Int?
        }
        struct TekkenFile: Decodable {
            let vocab: [VocabEntry]
            let config: Config?
            let specialTokens: [SpecialToken]?
            enum CodingKeys: String, CodingKey {
                case vocab, config
                case specialTokens = "special_tokens"
            }
        }

        let vocab: [VocabEntry]
        let nSpecial: Int
        let specialIds: Set<Int>

        init(url: URL) throws {
            let data = try Data(contentsOf: url)
            let parsed = try JSONDecoder().decode(TekkenFile.self, from: data)
            vocab = parsed.vocab
            nSpecial = parsed.config?.defaultNumSpecialTokens ?? 1000
            specialIds = Set((parsed.specialTokens ?? []).compactMap { $0.rank })
        }

        func decode(tokenIds: [Int]) -> String {
            var out: [UInt8] = []
            out.reserveCapacity(tokenIds.count * 2)
            for id in tokenIds {
                guard id >= nSpecial, !specialIds.contains(id) else { continue }
                let vocabId = id - nSpecial
                guard vocabId >= 0, vocabId < vocab.count else { continue }
                if let bytes = Data(base64Encoded: vocab[vocabId].tokenBytes) {
                    out.append(contentsOf: bytes)
                }
            }
            return String(decoding: out, as: UTF8.self)
        }
    }

    private var tekkenTokenizer: TekkenTokenizer?

    public init(
        config: VoxtralRealtimeConfig,
        conv0Weight: Tensor, conv0Bias: Tensor,
        conv1Weight: Tensor, conv1Bias: Tensor,
        encoderLayers: [VoxtralEncoderLayerWeights],
        encoderNorm: RMSNorm,
        audioProj0: Linear,
        audioProj2: Linear,
        tokEmbeddings: AnyEmbedding,
        decoderLayers: [VoxtralDecoderLayerWeights],
        decoderNorm: RMSNorm,
        lmHead: AnyLinear,
        dtype: DType
    ) {
        self.config = config
        self.conv0Weight = conv0Weight
        self.conv0Bias = conv0Bias
        self.conv1Weight = conv1Weight
        self.conv1Bias = conv1Bias
        self.encoderLayers = encoderLayers
        self.encoderNorm = encoderNorm
        self.audioProj0 = audioProj0
        self.audioProj2 = audioProj2
        self.tokEmbeddings = tokEmbeddings
        self.decoderLayers = decoderLayers
        self.decoderNorm = decoderNorm
        self.lmHead = lmHead
        self.dtype = dtype
    }

    // ─── Audio encoding ───────────────────────────────────────────────

    /// Compute Voxtral's Slaney mel spectrogram from a mono 16 kHz waveform.
    ///
    /// Voxtral uses:
    ///   - Slaney mel scale (fMax = 8 kHz) — different from Whisper's HTK.
    ///   - Periodic Hann window.
    ///   - Normalization: log10, clamp to (globalLogMelMax − 8), (x+4)/4.
    ///
    /// Returns `[nMels, nFrames]` (frequency-major, matching the Conv1d input).
    private func computeVoxtralMel(
        waveform: [Float],
        device: Device
    ) -> [Float] {
        let ac = config.audioConfig
        let nMels = ac.numMelBins
        let nFFT  = ac.windowSize
        let hop   = ac.hopLength

        // ── Reflect-pad by windowSize/2 on both sides ──
        let pad = nFFT / 2
        let padded = AudioPreprocessing.reflectPad(waveform, pad: pad)

        // ── Periodic Hann window (n denominator, not n-1) ──
        var window = [Float](repeating: 0, count: nFFT)
        let twoPi = 2.0 * Double.pi
        for i in 0..<nFFT {
            window[i] = Float(0.5 * (1.0 - cos(twoPi * Double(i) / Double(nFFT))))
        }

        // ── Slaney mel filterbank [nMels, nFreq] ──
        // Uses fMax = 8000 Hz (slaney convention) rather than Nyquist.
        let slaneyConfig = AudioFrontEndConfig(
            sampleRate: ac.samplingRate, nFFT: nFFT, hopLength: hop,
            nMels: nMels, fMin: 0.0, fMax: 8000.0)
        let filterbank = AudioPreprocessing.melFilterbank(slaneyConfig)
        let nFreq = slaneyConfig.nFreq

        // ── GPU mel spectrogram (natural-log output) ──
        let nFrames = AudioPreprocessing.frameCount(
            paddedSamples: padded.count, cfg: slaneyConfig)
        guard nFrames > 0 else { return [] }

        // Drop the last frame (match reference: drop the +1 frame from reflect-pad).
        let nFramesUsed = max(nFrames - 1, 1)

        // Use the Whisper-normalised=false path so we can apply Voxtral norms.
        let winT = Tensor.empty(shape: [nFFT], dtype: .f32, device: device)
        AudioPreprocessing.copyFloats(window, into: winT)
        let melT = Tensor.empty(shape: [nMels, nFreq], dtype: .f32, device: device)
        AudioPreprocessing.copyFloats(filterbank, into: melT)

        let audioT = Tensor.empty(shape: [padded.count], dtype: .f32, device: device)
        AudioPreprocessing.copyFloats(padded, into: audioT)

        let cmdMel = device.makeCommandBuffer()
        let rawMel = Ops.melSpectrogram(
            audio: audioT, window: winT, melWeight: melT,
            nFFT: nFFT, nMels: nMels, hopLength: hop,
            nFrames: nFrames, on: cmdMel)
        cmdMel.commit(); cmdMel.waitUntilCompleted()

        // rawMel is [nFrames, nMels] in natural log; we need [nMels, nFrames].
        let rawVals = rawMel.toFloatArray() // [nFrames * nMels]

        // Convert natural log → log10, apply Voxtral normalization, transpose.
        let invLn10: Float = 1.0 / 2.302_585_092_994_046
        let globalMax = config.audioConfig.globalLogMelMax
        let floor = globalMax - 8.0
        var melFreqMajor = [Float](repeating: 0, count: nMels * nFramesUsed)

        for f in 0..<nFramesUsed {
            for m in 0..<nMels {
                var v = rawVals[f * nMels + m] * invLn10
                if v < floor { v = floor }
                // Voxtral normalization: (x + 4) / 4 — same affine as Whisper.
                v = (v + 4.0) / 4.0
                // Transpose: [nFrames, nMels] → [nMels, nFrames].
                melFreqMajor[m * nFramesUsed + f] = v
            }
        }
        return melFreqMajor  // [nMels * nFramesUsed] row-major (freq, time)
    }

    /// Run the causal Conv1d stem: two stacked causal convolutions.
    ///
    /// Causal padding: prepend (kernelSize − stride) zeros to each input so
    /// frame t only attends to frames ≤ t. FFAI's `audioConv1d` takes
    /// `[batch, inCh, inLen]` (NCL) and produces `[batch, outCh, outLen]`.
    ///
    /// Returns `[seqLen, dim]` (time-major) for the transformer.
    private func causalConvStem(
        mel: [Float], nFrames: Int,
        device: Device
    ) -> [Float] {
        let ec = config.encoderConfig
        let nMels = config.audioConfig.numMelBins

        // ── Layer 0: [1, nMels, nFrames] → [1, dim, nFrames] (stride 1) ──
        let k0 = 3
        let pad0 = k0 - 1  // causal padding: k - stride (stride=1)
        let inLen0 = nFrames + pad0

        // Prepare padded input: zeros left-pad.
        var padded0 = [Float](repeating: 0, count: nMels * inLen0)
        for m in 0..<nMels {
            for t in 0..<nFrames {
                padded0[m * inLen0 + pad0 + t] = mel[m * nFrames + t]
            }
        }

        // Upload [1, nMels, inLen0].
        let in0T = Tensor.empty(shape: [1, nMels, inLen0], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(padded0, into: in0T)

        let cmd0 = device.makeCommandBuffer()
        // outLen = (inLen + 2*0 - k) / stride + 1 = (nFrames + k0 - 1 - k0) / 1 + 1 = nFrames.
        let outLen0 = nFrames
        let out0T = Ops.audioConv1d(
            input: in0T, weight: conv0Weight, bias: conv0Bias,
            batch: 1, inCh: nMels, inLen: inLen0, outCh: ec.dim,
            k: k0, stride: 1, pad: 0, on: cmd0)
        // Apply GELU on the conv output.
        let gelu0 = Ops.gelu(
            out0T.reshaped(to: [ec.dim * outLen0]), on: cmd0)
        cmd0.commit(); cmd0.waitUntilCompleted()
        let x0 = gelu0.toFloatArray()

        // ── Layer 1: [1, dim, nFrames] → [1, dim, ceil(nFrames/2)] (stride 2) ──
        let k1 = 3
        let stride1 = 2
        let pad1 = k1 - stride1  // causal: k - stride
        let inLen1 = outLen0 + pad1

        // x0 is [dim * outLen0] (NCL flattened); re-pad for causal conv.
        // x0 layout: [dim, nFrames] (outCh-major = NCL without batch).
        var padded1 = [Float](repeating: 0, count: ec.dim * inLen1)
        for c in 0..<ec.dim {
            for t in 0..<outLen0 {
                padded1[c * inLen1 + pad1 + t] = x0[c * outLen0 + t]
            }
        }

        let in1T = Tensor.empty(shape: [1, ec.dim, inLen1], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(padded1, into: in1T)

        let cmd1 = device.makeCommandBuffer()
        let out1T = Ops.audioConv1d(
            input: in1T, weight: conv1Weight, bias: conv1Bias,
            batch: 1, inCh: ec.dim, inLen: inLen1, outCh: ec.dim,
            k: k1, stride: stride1, pad: 0, on: cmd1)

        // Trim so length is divisible by downsampleFactor (matches reference).
        let rawLen = out1T.shape[2]  // [1, dim, outLen1]
        let ds = ec.downsampleFactor
        let trimLen = rawLen - (rawLen % ds)

        let gelu1 = Ops.gelu(
            out1T.reshaped(to: [ec.dim * rawLen]), on: cmd1)
        cmd1.commit(); cmd1.waitUntilCompleted()
        let x1flat = gelu1.toFloatArray()  // [dim * rawLen]

        // Convert from NCL [dim, rawLen] → time-major [trimLen, dim].
        var timeMajor = [Float](repeating: 0, count: trimLen * ec.dim)
        // NCL flattened: x1flat[c * rawLen + t] for c in dim, t in rawLen.
        // We skip the first (rawLen % ds) frames and output [trimLen, dim].
        let skip = rawLen - trimLen
        for t in 0..<trimLen {
            for c in 0..<ec.dim {
                timeMajor[t * ec.dim + c] = x1flat[c * rawLen + skip + t]
            }
        }
        return timeMajor  // [trimLen * dim]
    }

    /// Run the encoder transformer over `[seqLen, dim]` input.
    /// Uses interleaved RoPE (Voxtral convention). Returns `[seqLen, dim]`.
    ///
    /// For inputs longer than `slidingWindow` the sequence is processed in
    /// causal chunks to bound peak memory.
    private func runEncoderTransformer(
        seqVals: [Float], seqLen: Int,
        device: Device
    ) -> [Float] {
        let ec = config.encoderConfig
        let sw = ec.slidingWindow

        if seqLen <= sw {
            return runEncoderChunk(
                seqVals: seqVals, seqLen: seqLen,
                startPos: 0, device: device)
        }

        // Chunked processing: process sliding-window chunks, preserving the
        // last (slidingWindow - 1) frames as KV context carry-over.
        // The approach here mirrors the reference: process all frames but
        // use per-layer KV caches that trim to the sliding window.
        // For simplicity we use the same chunk-and-concatenate approach.
        var caches: [[(keys: [Float], values: [Float])?]]
            = Array(repeating: Array(repeating: nil, count: ec.nLayers), count: 1)
        var allOutputs = [Float]()
        allOutputs.reserveCapacity(seqLen * ec.dim)

        var chunkStart = 0
        while chunkStart < seqLen {
            let chunkEnd = min(chunkStart + sw, seqLen)
            let chunkLen = chunkEnd - chunkStart
            let chunkVals = Array(seqVals[chunkStart * ec.dim ..< chunkEnd * ec.dim])
            let (outVals, newCaches) = runEncoderChunkWithCache(
                seqVals: chunkVals, seqLen: chunkLen,
                startPos: chunkStart,
                layerCaches: caches[0],
                device: device)
            allOutputs += outVals
            caches[0] = newCaches
            chunkStart = chunkEnd
        }
        return allOutputs
    }

    /// Process one chunk through all encoder layers without KV caching.
    private func runEncoderChunk(
        seqVals: [Float], seqLen: Int, startPos: Int,
        device: Device
    ) -> [Float] {
        var h = seqVals
        for layer in encoderLayers {
            h = runEncoderLayer(
                layer, seq: h, seqLen: seqLen,
                startPos: startPos, device: device).0
        }
        return h
    }

    /// Process one chunk with KV caches (for chunked full-sequence encoding).
    private func runEncoderChunkWithCache(
        seqVals: [Float], seqLen: Int, startPos: Int,
        layerCaches: [(keys: [Float], values: [Float])?],
        device: Device
    ) -> ([Float], [(keys: [Float], values: [Float])?]) {
        let ec = config.encoderConfig
        var h = seqVals
        var newCaches = [(keys: [Float], values: [Float])?]()
        newCaches.reserveCapacity(ec.nLayers)
        for (i, layer) in encoderLayers.enumerated() {
            let (newH, newCache) = runEncoderLayer(
                layer, seq: h, seqLen: seqLen,
                startPos: startPos,
                cache: layerCaches[i],
                device: device)
            h = newH
            newCaches.append(newCache)
        }
        return (h, newCaches)
    }

    /// One VoxtralRealtime encoder layer: RMSNorm → attention → residual →
    /// RMSNorm → SwiGLU FFN → residual.
    ///
    /// Attention uses interleaved RoPE (x1 * cos - x2 * sin, x2 * cos + x1 * sin
    /// on adjacent pairs), matching the reference `voxtralApplyInterleavedRoPE`.
    private func runEncoderLayer(
        _ layer: VoxtralEncoderLayerWeights,
        seq seqVals: [Float],
        seqLen: Int,
        startPos: Int,
        cache: (keys: [Float], values: [Float])? = nil,
        device: Device
    ) -> ([Float], (keys: [Float], values: [Float])?) {
        let ec = config.encoderConfig
        let H = ec.dim
        let nH = ec.nHeads
        let nKVH = ec.nKVHeads
        let hd = ec.headDim
        let sw = ec.slidingWindow

        // ── Upload + RMSNorm ──
        let seqT = Tensor.empty(shape: [seqLen, H], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(seqVals, into: seqT)

        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(
            seqT, weight: layer.attnNorm.weight, eps: layer.attnNorm.eps,
            nRows: seqLen, rowSize: H, on: cmd)

        // ── Q / K / V projections ──
        // Use GEMM for multi-row projections (wqWeight/wkWeight/wvWeight are Tensors).
        let q = Ops.gemm(weight: layer.attn.wqWeight, input: normed, nRows: seqLen, on: cmd)
        let k = Ops.gemm(weight: layer.attn.wkWeight, input: normed, nRows: seqLen, on: cmd)
        let v = Ops.gemm(weight: layer.attn.wvWeight, input: normed, nRows: seqLen, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        // ── Add biases if present ──
        let qVals = addRowBiasIfPresent(
            q.toFloatArray(), bias: layer.attn.wqBias?.toFloatArray(),
            nRows: seqLen, rowSize: nH * hd)
        let kVals = addRowBiasIfPresent(
            k.toFloatArray(), bias: nil,  // wk has no bias
            nRows: seqLen, rowSize: nKVH * hd)
        let vVals = addRowBiasIfPresent(
            v.toFloatArray(), bias: layer.attn.wvBias?.toFloatArray(),
            nRows: seqLen, rowSize: nKVH * hd)

        // ── Interleaved RoPE on Q and K ──
        let qRot = voxtralInterleavedRoPE(
            vals: qVals, seqLen: seqLen, nHeads: nH,
            headDim: hd, startPos: startPos, theta: ec.ropeTheta)
        let kRot = voxtralInterleavedRoPE(
            vals: kVals, seqLen: seqLen, nHeads: nKVH,
            headDim: hd, startPos: startPos, theta: ec.ropeTheta)

        // ── KV cache management + sliding window trimming ──
        var allK = kRot
        var allV = vVals
        var kvOffset = 0
        if let c = cache {
            allK = c.keys + kRot
            allV = c.values + vVals
        }
        var kvLen = allK.count / (nKVH * hd)
        if kvLen > sw {
            let trim = kvLen - sw
            kvOffset = trim
            allK = Array(allK[(trim * nKVH * hd)...])
            allV = Array(allV[(trim * nKVH * hd)...])
            kvLen = sw
        }
        let newCache: (keys: [Float], values: [Float])? = (keys: allK, values: allV)

        // ── CPU bidirectional / causal multi-head attention ──
        let attnCtx = cpuEncoderAttention(
            q: qRot, k: allK, v: allV,
            seqLen: seqLen, kvLen: kvLen,
            kvOffset: kvOffset, startPos: startPos,
            nQHeads: nH, nKVHeads: nKVH, headDim: hd,
            slidingWindow: sw, causal: ec.causal,
            scale: 1.0 / Float(Double(hd).squareRoot()),
            device: device)

        // ── Output projection + residual ──
        let cmd2 = device.makeCommandBuffer()
        let outProjT = Tensor.empty(shape: [seqLen, H], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(attnCtx, into: outProjT)
        let outProj = Ops.gemm(weight: layer.attn.woWeight, input: outProjT,
                               nRows: seqLen, on: cmd2)
        let attnPlusBias = addRowBiasIfPresent(
            outProj.toFloatArray(), bias: layer.attn.woBias?.toFloatArray(),
            nRows: seqLen, rowSize: H)

        // Residual: h = x + attn_out.
        var hVals = seqVals
        for i in 0..<(seqLen * H) { hVals[i] += attnPlusBias[i] }

        // ── RMSNorm + SwiGLU FFN ──
        let hT2 = Tensor.empty(shape: [seqLen, H], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(hVals, into: hT2)
        let normed2 = Ops.rmsNormRows(
            hT2, weight: layer.ffnNorm.weight, eps: layer.ffnNorm.eps,
            nRows: seqLen, rowSize: H, on: cmd2)
        let ff1 = Ops.gemm(weight: layer.w1Weight.weight,
                           input: normed2, nRows: seqLen, on: cmd2)
        let ff3 = Ops.gemm(weight: layer.w3Weight.weight,
                           input: normed2, nRows: seqLen, on: cmd2)
        cmd2.commit(); cmd2.waitUntilCompleted()

        // SwiGLU: gate * up (SiLU on gate, element-wise multiply with up).
        let ff1Vals = ff1.toFloatArray()
        let ff3Vals = ff3.toFloatArray()
        let hidDim = ec.hiddenDim
        var gatedVals = [Float](repeating: 0, count: seqLen * hidDim)
        for i in 0..<gatedVals.count {
            let g = ff1Vals[i]
            // SiLU: g * sigmoid(g).
            let silu = g * (1.0 / (1.0 + exp(-g)))
            gatedVals[i] = silu * ff3Vals[i]
        }
        let gatedT = Tensor.empty(shape: [seqLen, hidDim], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(gatedVals, into: gatedT)

        let cmd3 = device.makeCommandBuffer()
        let ff2Out = Ops.gemm(
            weight: layer.w2Weight.weight,
            input: gatedT, nRows: seqLen, on: cmd3)
        let ff2Vals = addRowBiasIfPresent(
            ff2Out.toFloatArray(), bias: layer.w2Weight.bias?.toFloatArray(),
            nRows: seqLen, rowSize: H)
        cmd3.commit(); cmd3.waitUntilCompleted()

        // Residual: out = h + ffn_out.
        var outVals = hVals
        for i in 0..<(seqLen * H) { outVals[i] += ff2Vals[i] }

        return (outVals, newCache)
    }

    /// CPU multi-head attention for the encoder.
    ///
    /// Supports both causal (sliding window) and bidirectional modes.
    /// Handles GQA by repeating KV heads across query groups.
    private func cpuEncoderAttention(
        q: [Float], k: [Float], v: [Float],
        seqLen: Int, kvLen: Int,
        kvOffset: Int, startPos: Int,
        nQHeads: Int, nKVHeads: Int, headDim: Int,
        slidingWindow: Int, causal: Bool,
        scale: Float,
        device: Device
    ) -> [Float] {
        let qHeadDim = nQHeads * headDim
        let kHeadDim = nKVHeads * headDim
        let groupSize = nQHeads / nKVHeads  // heads per KV group (GQA)

        var out = [Float](repeating: 0, count: seqLen * qHeadDim)
        out.withUnsafeMutableBufferPointer { outBuf in
            let outPtr = outBuf.baseAddress!
            q.withUnsafeBufferPointer { qBuf in
            k.withUnsafeBufferPointer { kBuf in
            v.withUnsafeBufferPointer { vBuf in
                let qb = qBuf.baseAddress!
                let kb = kBuf.baseAddress!
                let vb = vBuf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: nQHeads * seqLen) { work in
                    let qHead = work / seqLen
                    let qRow  = work % seqLen
                    let kvHead = qHead / groupSize
                    let qOff  = qRow * qHeadDim + qHead * headDim
                    let qPos  = startPos + qRow

                    // Compute attention scores over all KV positions.
                    var scores = [Float](repeating: 0, count: kvLen)
                    var maxScore = -Float.greatestFiniteMagnitude
                    for j in 0..<kvLen {
                        let kvPos = kvOffset + j
                        // Causal mask: only attend to kvPos <= qPos.
                        if causal && kvPos > qPos { continue }
                        // Sliding window mask.
                        if causal && qPos - kvPos >= slidingWindow { continue }
                        let kOff = j * kHeadDim + kvHead * headDim
                        var dot: Float = 0
                        for d in 0..<headDim { dot += qb[qOff + d] * kb[kOff + d] }
                        let s = dot * scale
                        scores[j] = s
                        if s > maxScore { maxScore = s }
                    }
                    // Softmax.
                    var sumExp: Float = 0
                    for j in 0..<kvLen {
                        let kvPos = kvOffset + j
                        if causal && kvPos > qPos { scores[j] = -Float.greatestFiniteMagnitude; continue }
                        if causal && qPos - kvPos >= slidingWindow { scores[j] = -Float.greatestFiniteMagnitude; continue }
                        let e = exp(scores[j] - maxScore)
                        scores[j] = e; sumExp += e
                    }
                    let inv = sumExp > 0 ? 1.0 / sumExp : 0
                    let oOff = qRow * qHeadDim + qHead * headDim
                    for j in 0..<kvLen {
                        let w = scores[j] * inv
                        if w == 0 { continue }
                        let vOff = j * kHeadDim + kvHead * headDim
                        for d in 0..<headDim { outPtr[oOff + d] += w * vb[vOff + d] }
                    }
                }
            }}}
        }
        let result = Tensor.empty(shape: [seqLen, qHeadDim], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(out, into: result)
        return out
    }

    /// Downsample `[seqLen, dim]` by factor `ds` and project through two
    /// linear layers with GELU in between.
    ///
    /// Matches reference `downsampleAndProject`: reshape to [seqLen/ds, dim*ds],
    /// apply projection0 (→ decoderDim) + GELU, then projection2 (→ decoderDim).
    private func downsampleAndProject(
        seqVals: [Float], seqLen: Int,
        device: Device
    ) -> ([Float], Int) {
        let ec = config.encoderConfig
        let ds = ec.downsampleFactor
        let decoderDim = config.decoderConfig.dim

        let dsLen = seqLen / ds
        guard dsLen > 0 else { return ([], 0) }

        // Reshape [seqLen, dim] → [dsLen, dim*ds] (row-major flatten).
        let flatDim = ec.dim * ds
        var reshaped = [Float](repeating: 0, count: dsLen * flatDim)
        for i in 0..<dsLen {
            for j in 0..<ds {
                let srcRow = i * ds + j
                for c in 0..<ec.dim {
                    reshaped[i * flatDim + j * ec.dim + c] = seqVals[srcRow * ec.dim + c]
                }
            }
        }
        let reshapedT = Tensor.empty(shape: [dsLen, flatDim], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(reshaped, into: reshapedT)

        // Projection 0 → GELU.
        // audioProj0.weight is [decoderDim, dim*downsampleFactor]; use Ops.gemm
        // for multi-row input (dsLen rows).
        let cmd = device.makeCommandBuffer()
        let proj0Out = Ops.gemm(weight: audioProj0.weight,
                                input: reshapedT, nRows: dsLen, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        let p0Vals = proj0Out.toFloatArray()  // [dsLen * decoderDim]
        var p0Act = [Float](repeating: 0, count: dsLen * decoderDim)
        // GELU approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
        let gk: Float = 0.7978845608
        let gc: Float = 0.044715
        for i in 0..<p0Act.count {
            let x = p0Vals[i]
            let ginner = gk * (x + gc * x * x * x)
            p0Act[i] = 0.5 * x * (1 + tanh(ginner))
        }

        let p0T = Tensor.empty(shape: [dsLen, decoderDim], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(p0Act, into: p0T)

        // Projection 2.
        let cmd2 = device.makeCommandBuffer()
        let proj2Out = Ops.gemm(weight: audioProj2.weight,
                                input: p0T, nRows: dsLen, on: cmd2)
        cmd2.commit(); cmd2.waitUntilCompleted()

        return (proj2Out.toFloatArray(), dsLen)
    }

    /// Encode a 16 kHz mono waveform into audio feature tokens
    /// `[nAudioTokens, decoderDim]` ready to be added to decoder embeddings.
    public func encodeAudio(
        waveform: [Float],
        device: Device = .shared
    ) -> Tensor {
        let ac = config.audioConfig
        let ec = config.encoderConfig

        // ── Step 1: Slaney log-Mel spectrogram ──
        let melVals = computeVoxtralMel(waveform: waveform, device: device)
        let nFrames = melVals.count / ac.numMelBins

        guard nFrames > 0 else {
            // Return empty [0, decoderDim] tensor for zero-length audio.
            return Tensor.empty(shape: [0, config.decoderConfig.dim],
                                dtype: dtype, device: device)
        }

        // ── Step 2: Causal Conv1d stem ──
        let convOut = causalConvStem(mel: melVals, nFrames: nFrames, device: device)
        let convSeqLen = convOut.count / ec.dim

        // ── Step 3: Encoder transformer ──
        // Each layer applies its own pre-norm internally. encoderNorm is applied
        // once post-transformer (matches reference VoxtralRealtimeAudioEncoder).
        let transformerOut = runEncoderTransformer(
            seqVals: convOut, seqLen: convSeqLen, device: device)

        // Post-transformer RMSNorm.
        let encoderOut = applyRMSNormRows(
            vals: transformerOut, nRows: convSeqLen, rowSize: ec.dim,
            weight: encoderNorm.weight, eps: encoderNorm.eps, device: device)

        // ── Step 4: Downsample + project ──
        let (adapterVals, nAudioTokens) = downsampleAndProject(
            seqVals: encoderOut, seqLen: convSeqLen, device: device)

        guard nAudioTokens > 0 else {
            return Tensor.empty(shape: [0, config.decoderConfig.dim],
                                dtype: dtype, device: device)
        }

        let result = Tensor.empty(
            shape: [nAudioTokens, config.decoderConfig.dim],
            dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(adapterVals, into: result)
        return result
    }

    // ─── Transcription ────────────────────────────────────────────────

    /// Transcribe a 16 kHz mono waveform to text using the Tekken tokenizer.
    ///
    /// - Parameters:
    ///   - waveform: 16 kHz mono PCM samples.
    ///   - maxTokens: Maximum tokens to generate.
    ///   - delayMs: Transcription delay in ms; `nil` uses config default (480 ms).
    ///   - device: Metal device.
    /// - Returns: Decoded transcript string, or empty string if the tokenizer
    ///   has not been loaded (load the model via `VoxtralRealtimeModel.load`).
    public func transcribe(
        waveform: [Float],
        maxTokens: Int = 4096,
        delayMs: Int? = nil,
        device: Device = .shared
    ) -> String {
        let dc = config.decoderConfig
        let ac = config.audioConfig

        // ── 1. Compute adapter output (audio features) ──
        let adapterOut = encodeAudio(waveform: waveform, device: device)
        let nAudioTotal = adapterOut.shape[0]

        // ── 2. Compute delay and padding ──
        let resolvedDelayMs = delayMs ?? config.transcriptionDelayMs
        let nDelay = numDelayTokens(delayMs: resolvedDelayMs, sampleRate: ac.samplingRate)
        let nLeft  = config.nLeftPadTokens
        let promptLength = 1 + nLeft + nDelay  // BOS + pad×nLeft + pad×nDelay

        // ── 3. Precompute AdaRMSNorm scales ──
        ensureAdaScales(delayTokens: nDelay)

        // ── 4. Build prompt embeddings and add adapter output ──
        // Prompt token ids: [BOS, pad×(nLeft + nDelay)].
        let promptIds = [config.bosTokenId]
            + [Int](repeating: config.streamingPadTokenId,
                    count: nLeft + nDelay)
        let maxSeq = promptLength + max(nAudioTotal - promptLength, 0) + maxTokens + 16

        // Embed prompt tokens.
        let idsTensor = Tensor.empty(shape: [promptLength], dtype: .u32, device: device)
        idsTensor.copyIn(from: promptIds.map { UInt32($0) })
        let cmd = device.makeCommandBuffer()
        let promptEmbeds = tokEmbeddings(idsTensor, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()

        // Add adapter output to prompt embeddings (element-wise).
        // adapterOut[i] + tokEmbed[i] for i in 0..<promptLength.
        let adapterVals = adapterOut.toFloatArray()
        let embedVals   = promptEmbeds.toFloatArray()
        let hidden = dc.dim
        var prefixVals = [Float](repeating: 0, count: promptLength * hidden)
        let usableAdapter = min(promptLength, nAudioTotal)
        for i in 0..<usableAdapter {
            for c in 0..<hidden {
                prefixVals[i * hidden + c] = adapterVals[i * hidden + c]
                    + embedVals[i * hidden + c]
            }
        }
        for i in usableAdapter..<promptLength {
            for c in 0..<hidden {
                prefixVals[i * hidden + c] = embedVals[i * hidden + c]
            }
        }

        // ── 5. Prefill decoder ──
        let nLayers   = dc.nLayers
        let nKVHeads  = dc.nKVHeads
        let hd        = dc.headDim

        var caches = (0..<nLayers).map { _ in
            KVCache(nKVHeads: nKVHeads, headDim: hd, maxSeq: maxSeq,
                    dtype: dtype, device: device)
        }

        // Feed prompt one token at a time.
        var lastLogits: Tensor? = nil
        for pos in 0..<promptLength {
            let rowEmbed = Tensor.empty(shape: [hidden], dtype: dtype, device: device)
            AudioPreprocessing.copyFloats(
                Array(prefixVals[pos * hidden ..< (pos + 1) * hidden]), into: rowEmbed)
            lastLogits = forwardOneDecoderToken(
                embed: rowEmbed, caches: &caches, device: device)
        }
        guard var logits = lastLogits else { return "" }

        // ── 6. Streaming decode ──
        // Positions promptLength..nAudioTotal use adapter output + token embed.
        var generated: [Int] = []
        let eos = config.eosTokenId

        for pos in promptLength..<(nAudioTotal + maxTokens) {
            // Greedy sample.
            let logitVals = logits.toFloatArray()
            var best = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for (i, v) in logitVals.enumerated() where v > bestVal {
                bestVal = v; best = i
            }
            if best == eos { break }
            if generated.count >= maxTokens { break }
            generated.append(best)

            // Backstop: prevent infinite repetition loops.
            if generated.count >= 24, Set(generated.suffix(24)).count <= 3 { break }

            // Embed the generated token.
            let nextIdT = Tensor.empty(shape: [1], dtype: .u32, device: device)
            nextIdT.copyIn(from: [UInt32(best)])
            let cmdEmb = device.makeCommandBuffer()
            let tokenEmbed = tokEmbeddings(nextIdT, on: cmdEmb)
            cmdEmb.commit(); cmdEmb.waitUntilCompleted()
            let tokenEmbVals = tokenEmbed.toFloatArray()  // [hidden]

            // Combine token embed with adapter output (if in audio span).
            let inputVals: [Float]
            if pos < nAudioTotal {
                var combined = [Float](repeating: 0, count: hidden)
                for c in 0..<hidden {
                    combined[c] = adapterVals[pos * hidden + c] + tokenEmbVals[c]
                }
                inputVals = combined
            } else {
                inputVals = tokenEmbVals
            }

            let inputT = Tensor.empty(shape: [hidden], dtype: dtype, device: device)
            AudioPreprocessing.copyFloats(inputVals, into: inputT)
            logits = forwardOneDecoderToken(embed: inputT, caches: &caches, device: device)
        }

        return tekkenTokenizer?.decode(tokenIds: generated)
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    // ─── Decoder forward pass ─────────────────────────────────────────

    /// Forward a single embedding `[hidden]` through all Voxtral decoder
    /// layers and return `[vocabSize]` logits.
    private func forwardOneDecoderToken(
        embed: Tensor,
        caches: inout [KVCache],
        device: Device
    ) -> Tensor {
        let dc = config.decoderConfig
        let H = dc.dim
        var h = embed.shape.count == 1 ? embed : embed.reshaped(to: [H])

        let offset = caches[0].length

        for (i, layer) in decoderLayers.enumerated() {
            h = runDecoderLayer(
                layer, h: h, offset: offset,
                adaScale: cachedAdaScales.indices.contains(i)
                    ? cachedAdaScales[i] : nil,
                cache: caches[i], device: device)
        }

        // Post-decoder RMSNorm → lm_head.
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNorm(h, weight: decoderNorm.weight,
                                 eps: decoderNorm.eps, on: cmd)
        let logits = lmHead(normed, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return logits
    }

    /// One Voxtral decoder layer with GQA, AdaRMSNorm time conditioning,
    /// interleaved RoPE, and SwiGLU FFN.
    private func runDecoderLayer(
        _ layer: VoxtralDecoderLayerWeights,
        h hIn: Tensor,
        offset: Int,
        adaScale: [Float]?,
        cache: KVCache,
        device: Device
    ) -> Tensor {
        let dc = config.decoderConfig
        let H  = dc.dim
        let nH = dc.nHeads
        let nKVH = dc.nKVHeads
        let hd = dc.headDim
        let theta = dc.ropeTheta
        let scale = 1.0 / Float(Double(hd).squareRoot())

        // ── Pre-norm + QKV projections (gemv, seqLen=1) ──
        let cmd1 = device.makeCommandBuffer()
        let normed = Ops.rmsNorm(hIn, weight: layer.attnNorm.weight,
                                 eps: layer.attnNorm.eps, on: cmd1)
        let q = layer.wqWeight(normed, on: cmd1)   // [nH * hd]
        let k = layer.wkWeight(normed, on: cmd1)   // [nKVH * hd]
        let v = layer.wvWeight(normed, on: cmd1)   // [nKVH * hd]
        cmd1.commit(); cmd1.waitUntilCompleted()

        // ── Interleaved RoPE (single position) ──
        let qVals = voxtralInterleavedRoPEStep(
            vals: q.toFloatArray(), nHeads: nH, headDim: hd,
            position: offset, theta: theta)
        let kVals = voxtralInterleavedRoPEStep(
            vals: k.toFloatArray(), nHeads: nKVH, headDim: hd,
            position: offset, theta: theta)

        let qT = Tensor.empty(shape: [nH, hd], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(qVals, into: qT)
        let kT = Tensor.empty(shape: [nKVH, hd], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(kVals, into: kT)
        let vT = v.reshaped(to: [nKVH, hd])

        // ── KV cache append + sdpaDecode ──
        let cmd2 = device.makeCommandBuffer()
        cache.appendOnGPU(kFlat: kT, vFlat: vT, on: cmd2)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd2)
        let attnOut = Ops.sdpaDecode(
            q: qT, k: cacheK, v: cacheV,
            nQHeads: nH, nKVHeads: nKVH, headDim: hd,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd2)
        let attnFlat = attnOut.reshaped(to: [nH * hd])

        // ── Output projection + residual ──
        let oOut = layer.woWeight(attnFlat, on: cmd2)
        let postAttn = Ops.add(hIn, oOut, on: cmd2)

        // ── FFN with AdaRMSNorm time conditioning ──
        var ffnIn = Ops.rmsNorm(postAttn, weight: layer.ffnNorm.weight,
                                eps: layer.ffnNorm.eps, on: cmd2)

        // AdaRMSNorm scale (if present): ffnIn = ffnIn * (1 + adaScale).
        if let scale = adaScale, layer.adaRmsNorm != nil {
            cmd2.commit(); cmd2.waitUntilCompleted()
            let scaleVals = ffnIn.toFloatArray()
            var scaled = [Float](repeating: 0, count: H)
            for i in 0..<H { scaled[i] = scaleVals[i] * (1.0 + scale[i]) }
            let scaledT = Tensor.empty(shape: [H], dtype: dtype, device: device)
            AudioPreprocessing.copyFloats(scaled, into: scaledT)
            ffnIn = scaledT
            let cmd3 = device.makeCommandBuffer()
            let gate  = layer.w1Weight(ffnIn, on: cmd3)
            let up    = layer.w3Weight(ffnIn, on: cmd3)
            let gated = Ops.mul(Ops.silu(gate, on: cmd3), up, on: cmd3)
            let down  = layer.w2Weight(gated, on: cmd3)
            let result = Ops.add(postAttn, down, on: cmd3)
            cmd3.commit(); cmd3.waitUntilCompleted()
            return result.reshaped(to: [H])
        }

        let gate  = layer.w1Weight(ffnIn, on: cmd2)
        let up    = layer.w3Weight(ffnIn, on: cmd2)
        let gated = Ops.mul(Ops.silu(gate, on: cmd2), up, on: cmd2)
        let down  = layer.w2Weight(gated, on: cmd2)
        let result = Ops.add(postAttn, down, on: cmd2)
        cmd2.commit(); cmd2.waitUntilCompleted()
        return result.reshaped(to: [H])
    }

    // ─── AdaRMSNorm scale caching ─────────────────────────────────────

    /// Compute and cache AdaRMSNorm scales for a given delay token count.
    /// These are constant for a given delay so we only recompute on change.
    private func ensureAdaScales(delayTokens: Int) {
        guard delayTokens != cachedDelayTokens else { return }
        let dc = config.decoderConfig
        let H  = dc.dim

        // Time embedding: [H] = concat(cos, sin) of half-dim frequencies.
        let halfDim = H / 2
        let theta: Float = 10000.0
        var tEmbed = [Float](repeating: 0, count: H)
        let t = Float(delayTokens)
        for i in 0..<halfDim {
            let invFreq = exp(-log(theta) * Float(i) / Float(halfDim))
            let angle = t * invFreq
            tEmbed[i]            = cos(angle)
            tEmbed[halfDim + i]  = sin(angle)
        }

        // Compute per-layer AdaRMSNorm scales.
        var scales = [[Float]?]()
        scales.reserveCapacity(decoderLayers.count)
        for layer in decoderLayers {
            guard let ada = layer.adaRmsNorm else {
                scales.append(nil)
                continue
            }
            // ada_down: [bottleneck, H] → GELU → ada_up: [H, bottleneck].
            let tT = Tensor.empty(shape: [H], dtype: dtype, device: .shared)
            AudioPreprocessing.copyFloats(tEmbed, into: tT)
            let cmd = Device.shared.makeCommandBuffer()
            let down = ada.adaDown(tT, on: cmd)
            let geluOut = Ops.gelu(down, on: cmd)
            let scaleOut = ada.adaUp(geluOut, on: cmd)
            cmd.commit(); cmd.waitUntilCompleted()
            scales.append(scaleOut.toFloatArray())
        }
        cachedAdaScales = scales
        cachedDelayTokens = delayTokens
    }

    // ─── Timing helpers ───────────────────────────────────────────────

    /// Number of audio tokens corresponding to a given delay in ms.
    private func numDelayTokens(delayMs: Int, sampleRate: Int) -> Int {
        let sampleLen = Int(Double(delayMs) / 1000.0 * Double(sampleRate))
        let hop = config.audioConfig.hopLength
        let perTok = Int(Float(sampleRate) / config.audioConfig.frameRate)
        let frames: Int
        if sampleLen % hop != 0 {
            frames = Int(ceil(Double(sampleLen) / Double(hop) - 1.0))
        } else {
            frames = sampleLen / hop
        }
        return Int(ceil(Double(frames) / Double(perTok / hop)))
    }

    // ─── CPU math helpers ─────────────────────────────────────────────

    /// Apply row-broadcast bias addition if bias is non-nil.
    private func addRowBiasIfPresent(
        _ vals: [Float], bias: [Float]?,
        nRows: Int, rowSize: Int
    ) -> [Float] {
        guard let b = bias else { return vals }
        var out = vals
        for r in 0..<nRows {
            for c in 0..<rowSize { out[r * rowSize + c] += b[c] }
        }
        return out
    }

    /// Apply row-wise RMSNorm (GPU, nRows × rowSize).
    private func applyRMSNormRows(
        vals: [Float], nRows: Int, rowSize: Int,
        weight: Tensor, eps: Float,
        device: Device
    ) -> [Float] {
        let t = Tensor.empty(shape: [nRows, rowSize], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(vals, into: t)
        let cmd = device.makeCommandBuffer()
        let normed = Ops.rmsNormRows(t, weight: weight, eps: eps,
                                     nRows: nRows, rowSize: rowSize, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return normed.toFloatArray()
    }
}

// ─── Voxtral interleaved RoPE ─────────────────────────────────────────

/// Apply Voxtral's interleaved RoPE to a multi-row sequence tensor.
/// x shape: [seqLen, nHeads * headDim]. Returns rotated [seqLen, nHeads * headDim].
///
/// Voxtral uses adjacent-pair interleaving: for each head, pairs (x[2i], x[2i+1])
/// are rotated by (cos, sin) at position `startPos + row`.
/// This differs from the split-half RoPE used in Llama.
private func voxtralInterleavedRoPE(
    vals: [Float], seqLen: Int, nHeads: Int,
    headDim: Int, startPos: Int,
    theta: Float
) -> [Float] {
    let H = nHeads * headDim
    var out = vals
    let halfDim = headDim / 2

    // Precompute inverse frequencies for adjacent-pair rotation.
    var invFreqs = [Float](repeating: 0, count: halfDim)
    for i in 0..<halfDim {
        invFreqs[i] = exp(-log(theta) * Float(2 * i) / Float(headDim))
    }

    for row in 0..<seqLen {
        let pos = Float(startPos + row)
        for h in 0..<nHeads {
            let base = row * H + h * headDim
            for i in 0..<halfDim {
                let angle = pos * invFreqs[i]
                let c = cos(angle)
                let s = sin(angle)
                let x1 = vals[base + 2 * i]
                let x2 = vals[base + 2 * i + 1]
                out[base + 2 * i]     = x1 * c - x2 * s
                out[base + 2 * i + 1] = x2 * c + x1 * s
            }
        }
    }
    return out
}

/// Apply Voxtral interleaved RoPE to a single-position flat vector (decode step).
/// vals: [nHeads * headDim]. Returns rotated vector.
private func voxtralInterleavedRoPEStep(
    vals: [Float], nHeads: Int, headDim: Int,
    position: Int, theta: Float
) -> [Float] {
    return voxtralInterleavedRoPE(
        vals: vals, seqLen: 1, nHeads: nHeads,
        headDim: headDim, startPos: position, theta: theta)
}

// ─── Registry detection + loader ─────────────────────────────────────

extension VoxtralRealtimeModel {
    public static let modelType = "voxtral_realtime"

    /// Whether a decoded `config.json` describes a VoxtralRealtime checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        config.modelType == modelType
    }

    /// Load a VoxtralRealtime checkpoint from a resolved snapshot directory.
    public static func load(
        directory: URL, device: Device = .shared
    ) throws -> VoxtralRealtimeModel {
        let mc = try ModelConfig.load(from: directory)
        guard let vc = VoxtralRealtimeConfig.from(mc) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a VoxtralRealtime config")
        }
        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        let model = try build(config: vc, bundle: bundle, rootConfig: mc)

        // Load the Tekken tokenizer if present alongside the weights.
        let tekkenURL = directory.appendingPathComponent("tekken.json")
        if FileManager.default.fileExists(atPath: tekkenURL.path) {
            model.tekkenTokenizer = try TekkenTokenizer(url: tekkenURL)
        }

        // Precompute AdaRMSNorm scales for the default delay.
        model.ensureAdaScales(delayTokens: model.numDelayTokens(
            delayMs: vc.transcriptionDelayMs,
            sampleRate: vc.audioConfig.samplingRate))

        return model
    }

    /// Assemble a `VoxtralRealtimeModel` from a decoded config + weight bundle.
    public static func build(
        config vc: VoxtralRealtimeConfig,
        bundle: SafeTensorsBundle,
        rootConfig: ModelConfig? = nil
    ) throws -> VoxtralRealtimeModel {
        let ec = vc.encoderConfig
        let dc = vc.decoderConfig
        let quant = rootConfig?.quantization

        // Detect dtype from a probe tensor.
        let probKey = "encoder.transformer_layers.0.attention_norm.weight"
        let dtype = try bundle.tensor(named: probKey).dtype

        // ── Conv1d stem ──
        // Checkpoint stores conv weights as [outCh, kernelSize, inCh] (NLC).
        // FFAI's audioConv1d expects [outCh, inCh, k] (NCL).
        // Transpose: [o, k, i] → [o, i, k].
        func loadConv1dWeight(_ key: String) throws -> Tensor {
            let raw = try bundle.tensor(named: key)
            // raw shape: [outCh, kernelSize, inCh] (from MLX safetensors).
            let outCh = raw.shape[0]
            let kSz   = raw.shape[1]
            let inCh  = raw.shape[2]
            let rawVals = raw.toFloatArray()
            var transposed = [Float](repeating: 0, count: outCh * inCh * kSz)
            for o in 0..<outCh {
                for k in 0..<kSz {
                    for i in 0..<inCh {
                        let src = o * kSz * inCh + k * inCh + i
                        let dst = o * inCh * kSz + i * kSz + k
                        transposed[dst] = rawVals[src]
                    }
                }
            }
            let t = Tensor.empty(shape: [outCh, inCh, kSz], dtype: dtype,
                                 device: .shared)
            AudioPreprocessing.copyFloats(transposed, into: t)
            return t
        }

        let conv0Weight = try loadConv1dWeight("encoder.conv_layers_0_conv.conv.weight")
        let conv0Bias   = try bundle.tensor(named: "encoder.conv_layers_0_conv.conv.bias")
        let conv1Weight = try loadConv1dWeight("encoder.conv_layers_1_conv.conv.weight")
        let conv1Bias   = try bundle.tensor(named: "encoder.conv_layers_1_conv.conv.bias")

        // ── Encoder transformer layers ──
        var encLayers: [VoxtralEncoderLayerWeights] = []
        encLayers.reserveCapacity(ec.nLayers)
        for i in 0..<ec.nLayers {
            let p = "encoder.transformer_layers.\(i)"

            let attnNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).attention_norm.weight"),
                eps: ec.normEps)
            let ffnNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).ffn_norm.weight"),
                eps: ec.normEps)

            // Encoder attention weights.
            // In published mlx-community Voxtral checkpoints the encoder weights
            // are stored as plain fp16 (not quantized), so we load tensors directly.
            let wqW = try bundle.tensor(named: "\(p).attention.wq.weight")
            let wqB: Tensor? = bundle.has("\(p).attention.wq.bias")
                ? try bundle.tensor(named: "\(p).attention.wq.bias") : nil
            let wkW = try bundle.tensor(named: "\(p).attention.wk.weight")
            let wvW = try bundle.tensor(named: "\(p).attention.wv.weight")
            let wvB: Tensor? = bundle.has("\(p).attention.wv.bias")
                ? try bundle.tensor(named: "\(p).attention.wv.bias") : nil
            let woW = try bundle.tensor(named: "\(p).attention.wo.weight")
            let woB: Tensor? = bundle.has("\(p).attention.wo.bias")
                ? try bundle.tensor(named: "\(p).attention.wo.bias") : nil

            let attn = VoxtralEncoderAttentionWeights(
                wqWeight: wqW, wqBias: wqB,
                wkWeight: wkW,
                wvWeight: wvW, wvBias: wvB,
                woWeight: woW, woBias: woB)

            // Encoder FFN weights are not quantized in mlx-community checkpoints;
            // load as plain tensors and construct Linear directly.
            let w1W = try bundle.tensor(named: "\(p).feed_forward_w1.weight")
            let w2W = try bundle.tensor(named: "\(p).feed_forward_w2.weight")
            let w3W = try bundle.tensor(named: "\(p).feed_forward_w3.weight")
            let w1B: Tensor? = bundle.has("\(p).feed_forward_w1.bias")
                ? try bundle.tensor(named: "\(p).feed_forward_w1.bias") : nil
            let w2B: Tensor? = bundle.has("\(p).feed_forward_w2.bias")
                ? try bundle.tensor(named: "\(p).feed_forward_w2.bias") : nil
            let w3B: Tensor? = bundle.has("\(p).feed_forward_w3.bias")
                ? try bundle.tensor(named: "\(p).feed_forward_w3.bias") : nil

            encLayers.append(VoxtralEncoderLayerWeights(
                attnNorm: attnNorm, attn: attn, ffnNorm: ffnNorm,
                w1Weight: Linear(weight: w1W, bias: w1B),
                w2Weight: Linear(weight: w2W, bias: w2B),
                w3Weight: Linear(weight: w3W, bias: w3B)))
        }

        let encoderNorm = RMSNorm(
            weight: try bundle.tensor(named: "encoder.transformer_norm.weight"),
            eps: ec.normEps)

        // Audio→language projection layers (no bias on either; never quantized).
        let audioProj0 = Linear(
            weight: try bundle.tensor(named: "encoder.audio_language_projection_0.weight"))
        let audioProj2 = Linear(
            weight: try bundle.tensor(named: "encoder.audio_language_projection_2.weight"))

        // ── Decoder token embeddings ──
        let tokEmbeddings = try loadEmbedding(
            base: "decoder.tok_embeddings", in: bundle,
            hidden: dc.dim, quantization: quant)

        // ── Decoder layers ──
        var decLayers: [VoxtralDecoderLayerWeights] = []
        decLayers.reserveCapacity(dc.nLayers)
        for i in 0..<dc.nLayers {
            let p = "decoder.layers.\(i)"

            let attnNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).attention_norm.weight"),
                eps: dc.normEps)
            let ffnNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).ffn_norm.weight"),
                eps: dc.normEps)

            let wq = try loadLinear(base: "\(p).attention.wq", in: bundle, quantization: quant)
            let wk = try loadLinear(base: "\(p).attention.wk", in: bundle, quantization: quant)
            let wv = try loadLinear(base: "\(p).attention.wv", in: bundle, quantization: quant)
            let wo = try loadLinear(base: "\(p).attention.wo", in: bundle, quantization: quant)
            let w1 = try loadLinear(base: "\(p).feed_forward_w1", in: bundle, quantization: quant)
            let w2 = try loadLinear(base: "\(p).feed_forward_w2", in: bundle, quantization: quant)
            let w3 = try loadLinear(base: "\(p).feed_forward_w3", in: bundle, quantization: quant)

            // AdaRMSNorm (ada_down/ada_up) — never quantized.
            let adaRmsNorm: VoxtralAdaRMSNormWeights?
            if dc.adaRmsNormTCond,
               bundle.has("\(p).ada_rms_norm_t_cond.ada_down.weight") {
                let adaDown = try loadLinear(
                    base: "\(p).ada_rms_norm_t_cond.ada_down", in: bundle,
                    quantization: nil)
                let adaUp = try loadLinear(
                    base: "\(p).ada_rms_norm_t_cond.ada_up", in: bundle,
                    quantization: nil)
                adaRmsNorm = VoxtralAdaRMSNormWeights(adaDown: adaDown, adaUp: adaUp)
            } else {
                adaRmsNorm = nil
            }

            decLayers.append(VoxtralDecoderLayerWeights(
                attnNorm: attnNorm, wqWeight: wq, wkWeight: wk,
                wvWeight: wv, woWeight: wo, ffnNorm: ffnNorm,
                adaRmsNorm: adaRmsNorm,
                w1Weight: w1, w2Weight: w2, w3Weight: w3))
        }

        let decoderNorm = RMSNorm(
            weight: try bundle.tensor(named: "decoder.norm.weight"),
            eps: dc.normEps)

        // lm_head — tied to tok_embeddings in Mini-4B (tiedEmbeddings == true).
        let lmHead: AnyLinear
        if !dc.tiedEmbeddings, bundle.has("decoder.lm_head.weight") {
            lmHead = try loadLinear(base: "decoder.lm_head", in: bundle,
                                    quantization: quant)
        } else if let q = quant, bundle.isQuantized("decoder.tok_embeddings") {
            let t = try bundle.quantizedTriplet("decoder.tok_embeddings")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: bits, groupSize: q.groupSize))
        } else {
            lmHead = AnyLinear(Linear(weight: tokEmbeddings.weight))
        }

        return VoxtralRealtimeModel(
            config: vc,
            conv0Weight: conv0Weight, conv0Bias: conv0Bias,
            conv1Weight: conv1Weight, conv1Bias: conv1Bias,
            encoderLayers: encLayers,
            encoderNorm: encoderNorm,
            audioProj0: audioProj0,
            audioProj2: audioProj2,
            tokEmbeddings: tokEmbeddings,
            decoderLayers: decLayers,
            decoderNorm: decoderNorm,
            lmHead: lmHead,
            dtype: dtype)
    }
}
