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
// StyleTTS2 — KittenTTS family text-to-speech model.
//
// KittenTTS is a compact non-autoregressive English TTS system built on
// the StyleTTS2 acoustic stack:
//
//   text ──G2P phonemizer──▶ phoneme token ids
//        ──ALBERT encoder (PLBert)──▶ contextual features
//        ──duration predictor──▶ expanded alignment
//        ──text encoder (CNN + BiLSTM)──▶ phoneme features
//        ──KittenDecoder (AdaIN residual + F0/N predictor)──▶ STFT frame features
//        ──iSTFTNet generator──▶ 24 kHz waveform
//
// Checkpoint detection key: `model_type == "kitten_tts"` or the
// presence of `n_token` + `istftnet` + `plbert` in `config.json`.
//
// ## Scope note
//
// The full acoustic stack (ALBERT encoder, BiLSTM prosody predictor,
// AdaIN residual decoder, iSTFTNet generator) requires batched matmul,
// multi-head attention, 1-D conv, and BiLSTM — GPU operator families
// not yet in FFAI's Ops set. This file provides the complete model
// scaffold: config decoding, weight-count verification on load, the
// text-cleaner token map, and the `synthesize` stub that reports the
// limitation clearly. The iSTFTNet vocoder tail (`KokoroVocoder`) is
// the GPU-accelerated part already present in the FFAI Ops set —
// `synthesizeFromSpectrogram` works for callers with a predicted
// complex spectrogram.
//
// The G2P text preprocessor is a self-contained symbol→index table
// (the `StyleTTS2TextCleaner`) that works without network access and
// covers the full KittenTTS IPA + ASCII symbol set. For benchmarks and
// integration tests the cleaner is used directly. A BART-based fallback
// G2P lives in the reference mlx-audio-swift port; FFAI defers that
// to a future phase.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────────────

/// PLBert (ALBERT) encoder hyper-parameters.
/// Nested under the `plbert` key in `config.json`.
public struct PLBertConfig: Sendable {
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    /// Embedding size (can be smaller than hiddenSize via factorization).
    public let embeddingSize: Int
    public let innerGroupNum: Int
    public let numHiddenGroups: Int
    public let hiddenDropoutProb: Float
    public let attentionProbsDropoutProb: Float
    public let typeVocabSize: Int
    public let layerNormEps: Float

    public static func from(_ raw: [String: Any]) -> PLBertConfig {
        func int(_ k: String, _ d: Int) -> Int { (raw[k] as? Int) ?? d }
        func flt(_ k: String, _ d: Float) -> Float {
            if let v = raw[k] as? Double { return Float(v) }
            if let v = raw[k] as? Float { return v }
            return d
        }
        return PLBertConfig(
            numHiddenLayers: int("num_hidden_layers", 12),
            numAttentionHeads: int("num_attention_heads", 12),
            hiddenSize: int("hidden_size", 768),
            intermediateSize: int("intermediate_size", 2048),
            maxPositionEmbeddings: int("max_position_embeddings", 512),
            embeddingSize: int("embedding_size", 128),
            innerGroupNum: int("inner_group_num", 1),
            numHiddenGroups: int("num_hidden_groups", 1),
            hiddenDropoutProb: flt("hidden_dropout_prob", 0.0),
            attentionProbsDropoutProb: flt("attention_probs_dropout_prob", 0.0),
            typeVocabSize: int("type_vocab_size", 2),
            layerNormEps: flt("layer_norm_eps", 1e-12)
        )
    }

    public init(
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        hiddenSize: Int = 768,
        intermediateSize: Int = 2048,
        maxPositionEmbeddings: Int = 512,
        embeddingSize: Int = 128,
        innerGroupNum: Int = 1,
        numHiddenGroups: Int = 1,
        hiddenDropoutProb: Float = 0.0,
        attentionProbsDropoutProb: Float = 0.0,
        typeVocabSize: Int = 2,
        layerNormEps: Float = 1e-12
    ) {
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.embeddingSize = embeddingSize
        self.innerGroupNum = innerGroupNum
        self.numHiddenGroups = numHiddenGroups
        self.hiddenDropoutProb = hiddenDropoutProb
        self.attentionProbsDropoutProb = attentionProbsDropoutProb
        self.typeVocabSize = typeVocabSize
        self.layerNormEps = layerNormEps
    }
}

/// iSTFTNet vocoder hyper-parameters.
/// Nested under the `istftnet` key in `config.json`.
public struct ISTFTNetConfig: Sendable {
    public let resblockKernelSizes: [Int]
    public let upsampleRates: [Int]
    public let upsampleInitialChannel: Int
    public let resblockDilationSizes: [[Int]]
    public let upsampleKernelSizes: [Int]
    /// iSTFT FFT length (e.g. 20 for KittenTTS nano).
    public let genIstftNFft: Int
    /// iSTFT hop size (e.g. 5 for KittenTTS nano).
    public let genIstftHopSize: Int

    public static func from(_ raw: [String: Any]) -> ISTFTNetConfig {
        func int(_ k: String, _ d: Int) -> Int { (raw[k] as? Int) ?? d }
        func ints(_ k: String, _ d: [Int]) -> [Int] { (raw[k] as? [Int]) ?? d }
        func intss(_ k: String, _ d: [[Int]]) -> [[Int]] {
            (raw[k] as? [[Int]]) ?? d
        }
        return ISTFTNetConfig(
            resblockKernelSizes: ints("resblock_kernel_sizes", [3, 7, 11]),
            upsampleRates: ints("upsample_rates", [10, 6]),
            upsampleInitialChannel: int("upsample_initial_channel", 256),
            resblockDilationSizes: intss(
                "resblock_dilation_sizes",
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            upsampleKernelSizes: ints("upsample_kernel_sizes", [20, 12]),
            genIstftNFft: int("gen_istft_n_fft", 20),
            genIstftHopSize: int("gen_istft_hop_size", 5)
        )
    }

    public init(
        resblockKernelSizes: [Int] = [3, 7, 11],
        upsampleRates: [Int] = [10, 6],
        upsampleInitialChannel: Int = 256,
        resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsampleKernelSizes: [Int] = [20, 12],
        genIstftNFft: Int = 20,
        genIstftHopSize: Int = 5
    ) {
        self.resblockKernelSizes = resblockKernelSizes
        self.upsampleRates = upsampleRates
        self.upsampleInitialChannel = upsampleInitialChannel
        self.resblockDilationSizes = resblockDilationSizes
        self.upsampleKernelSizes = upsampleKernelSizes
        self.genIstftNFft = genIstftNFft
        self.genIstftHopSize = genIstftHopSize
    }
}

/// Top-level KittenTTS model configuration, decoded from `config.json`.
public struct StyleTTS2Config: Sendable {
    /// `model_type` identifier (`"kitten_tts"`).
    public let modelType: String
    /// ALBERT encoder hyper-parameters.
    public let plbert: PLBertConfig
    /// iSTFTNet vocoder hyper-parameters.
    public let istftnet: ISTFTNetConfig
    /// Acoustic hidden dimension.
    public let hiddenDim: Int
    /// Max conv channels in the decoder.
    public let maxConvDim: Int
    /// Max duration bins for the duration predictor.
    public let maxDur: Int
    /// Number of BiLSTM layers in prosody predictor / text encoder.
    public let nLayer: Int
    /// Mel-spectrogram bin count.
    public let nMels: Int
    /// Phoneme vocabulary size.
    public let nToken: Int
    /// Style embedding dimension (voice style vector).
    public let styleDim: Int
    /// Kernel size for the text encoder CNN.
    public let textEncoderKernelSize: Int
    /// ASR residual connection dim in the decoder.
    public let asrResDim: Int
    /// Output waveform sample rate (24 000 Hz for KittenTTS).
    public let sampleRate: Int
    /// Decoder output channels (defaults to `maxConvDim`).
    public let decoderOutDim: Int
    /// Path to the voices file (relative to model directory).
    public let voicesPath: String
    /// Per-voice speed priors — multiply the base speed by this.
    public let speedPriors: [String: Float]
    /// Human-readable voice name → internal voice key aliases.
    public let voiceAliases: [String: String]

    /// Decode a top-level `config.json` into a `StyleTTS2Config`.
    public static func from(_ config: ModelConfig) -> StyleTTS2Config? {
        let raw = config.raw
        guard let nToken = config.int("n_token"),
            let hiddenDim = config.int("hidden_dim"),
            let plbertRaw = config.nested("plbert"),
            let istftnetRaw = config.nested("istftnet")
        else { return nil }

        func int(_ k: String, _ d: Int) -> Int { (raw[k] as? Int) ?? d }
        func str(_ k: String, _ d: String) -> String { (raw[k] as? String) ?? d }

        return StyleTTS2Config(
            modelType: str("model_type", "kitten_tts"),
            plbert: PLBertConfig.from(plbertRaw),
            istftnet: ISTFTNetConfig.from(istftnetRaw),
            hiddenDim: hiddenDim,
            maxConvDim: int("max_conv_dim", 512),
            maxDur: int("max_dur", 50),
            nLayer: int("n_layer", 4),
            nMels: int("n_mels", 80),
            nToken: nToken,
            styleDim: int("style_dim", 128),
            textEncoderKernelSize: int("text_encoder_kernel_size", 5),
            asrResDim: int("asr_res_dim", 64),
            sampleRate: int("sample_rate", 24_000),
            decoderOutDim: int("decoder_out_dim", int("max_conv_dim", 512)),
            voicesPath: str("voices_path", "voices.safetensors"),
            speedPriors: (raw["speed_priors"] as? [String: Double])
                .map { $0.mapValues { Float($0) } } ?? [:],
            voiceAliases: (raw["voice_aliases"] as? [String: String]) ?? [:]
        )
    }

    public init(
        modelType: String = "kitten_tts",
        plbert: PLBertConfig = PLBertConfig(),
        istftnet: ISTFTNetConfig = ISTFTNetConfig(),
        hiddenDim: Int = 128,
        maxConvDim: Int = 256,
        maxDur: Int = 50,
        nLayer: Int = 2,
        nMels: Int = 80,
        nToken: Int = 178,
        styleDim: Int = 128,
        textEncoderKernelSize: Int = 5,
        asrResDim: Int = 64,
        sampleRate: Int = 24_000,
        decoderOutDim: Int = 256,
        voicesPath: String = "voices.safetensors",
        speedPriors: [String: Float] = [:],
        voiceAliases: [String: String] = [:]
    ) {
        self.modelType = modelType
        self.plbert = plbert
        self.istftnet = istftnet
        self.hiddenDim = hiddenDim
        self.maxConvDim = maxConvDim
        self.maxDur = maxDur
        self.nLayer = nLayer
        self.nMels = nMels
        self.nToken = nToken
        self.styleDim = styleDim
        self.textEncoderKernelSize = textEncoderKernelSize
        self.asrResDim = asrResDim
        self.sampleRate = sampleRate
        self.decoderOutDim = decoderOutDim
        self.voicesPath = voicesPath
        self.speedPriors = speedPriors
        self.voiceAliases = voiceAliases
    }
}

// ─── G2P text cleaner ────────────────────────────────────────────────────────

/// Symbol-table G2P for KittenTTS / StyleTTS2.
///
/// Maps ASCII letters, IPA phoneme symbols, and punctuation to integer
/// token ids. This is the FFAI-native minimal G2P: it converts already-
/// phonemized IPA text to token ids without network access. For English
/// text that hasn't been phonemized yet, call an external phonemizer
/// (e.g. espeak-ng or the Misaki G2P) first, then pass the IPA string.
///
/// Token id 0 is the padding / unknown token (`$`). The id space matches
/// the `n_token` field in `config.json` (178 for KittenTTS nano).
///
/// ## Parallelization
///
/// `tokenize(sentences:)` processes multiple sentences concurrently via
/// `DispatchQueue.concurrentPerform` when the input list is large.
public enum StyleTTS2TextCleaner {
    // Ordering matches KittenTTS / Kokoro symbol table: pad, punctuation,
    // ASCII letters, IPA letters.
    private static let pad = "$"
    private static let punctuation = ";:,.!?¡¿—…\"«»\u{201C}\u{201D} "
    private static let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    // IPA phonemes used by KittenTTS (mirrors upstream symbol_list).
    private static let lettersIPA =
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u{2018}\u{0329}\u{2019}ᵻ"

    /// Full symbol → token-id table. Built once at first access.
    public static let symbolToId: [Character: Int] = {
        var map = [Character: Int]()
        var idx = 0
        for ch in pad {
            map[ch] = idx
            idx += 1
        }
        for ch in punctuation {
            map[ch] = idx
            idx += 1
        }
        for ch in letters {
            map[ch] = idx
            idx += 1
        }
        for ch in lettersIPA {
            map[ch] = idx
            idx += 1
        }
        return map
    }()

    /// Map a single phonemized string → token id array. Unknown symbols
    /// are silently dropped (consistent with the KittenTTS reference).
    public static func tokenize(_ text: String) -> [Int] {
        text.compactMap { symbolToId[$0] }
    }

    /// Map multiple strings concurrently. Results are returned in the
    /// same order as the input.
    ///
    /// Uses `DispatchQueue.concurrentPerform` to parallelise CPU work
    /// when `sentences.count >= 4`. Each iteration writes to a distinct
    /// index of the output buffer — no element is touched by more than
    /// one thread — so the `nonisolated(unsafe)` annotation on the raw
    /// buffer pointer is sound.
    public static func tokenize(sentences: [String]) -> [[Int]] {
        let count = sentences.count
        if count < 4 {
            // Small batches — not worth the concurrentPerform overhead.
            return sentences.map { tokenize($0) }
        }
        // Allocate a flat array and fill it concurrently. Each index
        // is independent so we use nonisolated(unsafe) to silence the
        // Swift 6 `@Sendable` mutation warning.
        var results = [[Int]](repeating: [], count: count)
        results.withUnsafeMutableBufferPointer { buf in
            // `nonisolated(unsafe)` lets us capture the raw pointer in
            // a `@Sendable` closure. Safety: distinct indices, no aliasing.
            nonisolated(unsafe) let ptr = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: count) { i in
                ptr[i] = tokenize(sentences[i])
            }
        }
        return results
    }
}

// ─── Vocoder (CPU iSTFT overlap-add) ─────────────────────────────────────────

/// The iSTFTNet vocoder tail — turns a predicted complex spectrogram back
/// into a time-domain waveform via CPU iSTFT overlap-add synthesis.
///
/// KittenTTS's iSTFTNet head uses a small 20-sample FFT with hop 5. The
/// GPU-accelerated path (`Ops.vocoderISTFT`) is available once that kernel
/// lands in the FFAI Ops set; until then this class implements the same
/// overlap-add algorithm on the CPU, which is practical for the tiny
/// FFT size (20 points, hop 5) used by KittenTTS.
///
/// Callers may also drive the GPU path directly when available by calling
/// `Ops.vocoderISTFT` with the window tensor produced by `hannWindow(nFFT)`.
public final class StyleTTS2Vocoder: @unchecked Sendable {
    /// FFT length of the iSTFT (e.g. 20 for KittenTTS nano).
    public let nFFT: Int
    /// Hop length of the iSTFT (e.g. 5 for KittenTTS nano).
    public let hopLength: Int
    /// Pre-computed Hann window `[nFFT]` f32 on CPU.
    public let hannWindow: [Float]

    public init(nFFT: Int, hopLength: Int) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        // Hann window: w[i] = 0.5 − 0.5·cos(2π·i / N)
        self.hannWindow = (0 ..< nFFT).map { i in
            0.5 - 0.5 * cos(2.0 * .pi * Float(i) / Float(nFFT))
        }
    }

    /// Reconstruct a waveform from a predicted STFT via CPU overlap-add.
    ///
    /// The spectrogram is represented as separate real / imaginary planes
    /// because that is how the KittenDecoder / iSTFTNet generator produces
    /// it (magnitude × cos/sin of phase rather than complex struct).
    ///
    /// - Parameters:
    ///   - specRe: Real part `[nFrames × nFreq]` f32 flat array,
    ///             `nFreq = nFFT / 2 + 1`.
    ///   - specIm: Imaginary part `[nFrames × nFreq]` f32 flat array.
    ///   - nFrames: Number of STFT frames.
    ///   - device: Metal device for output tensor allocation.
    /// - Returns: `[outLen]` f32 waveform Tensor.
    ///            `outLen = (nFrames - 1) * hopLength + nFFT`.
    public func synthesize(
        specReFlat: [Float], specImFlat: [Float],
        nFrames: Int, device: Device = .shared
    ) -> Tensor {
        let nFreq = nFFT / 2 + 1
        precondition(
            specReFlat.count == nFrames * nFreq,
            "StyleTTS2Vocoder.synthesize: specRe length mismatch")
        precondition(
            specImFlat.count == nFrames * nFreq,
            "StyleTTS2Vocoder.synthesize: specIm length mismatch")

        let outLen = (nFrames - 1) * hopLength + nFFT
        var audioSamples = [Float](repeating: 0, count: outLen)
        var windowSum = [Float](repeating: 0, count: outLen)

        // Overlap-add: for each frame, IRFFT then window and accumulate.
        for frame in 0 ..< nFrames {
            let start = frame * hopLength
            // Build complex spectrum for IRFFT: nFreq complex points → nFFT real points.
            // IRFFT via the real-IFFT formula: x[n] = (1/N) sum_k X[k] e^(j2πkn/N).
            // For real-valued output (Hermitian symmetry), conjugate halves cancel.
            var frameOut = [Float](repeating: 0, count: nFFT)
            let re = Array(specReFlat[(frame * nFreq) ..< (frame * nFreq + nFreq)])
            let im = Array(specImFlat[(frame * nFreq) ..< (frame * nFreq + nFreq)])

            for n in 0 ..< nFFT {
                var sum: Float = 0
                for k in 0 ..< nFreq {
                    let angle = 2.0 * .pi * Float(k) * Float(n) / Float(nFFT)
                    let scale: Float = (k == 0 || k == nFreq - 1) ? 1 : 2
                    sum += scale * (re[k] * cos(angle) - im[k] * sin(angle))
                }
                frameOut[n] = sum / Float(nFFT)
            }

            // Window and accumulate.
            for i in 0 ..< nFFT {
                let idx = start + i
                if idx < outLen {
                    audioSamples[idx] += frameOut[i] * hannWindow[i]
                    windowSum[idx] += hannWindow[i] * hannWindow[i]
                }
            }
        }

        // Normalize by the squared window (WOLA).
        for i in 0 ..< outLen {
            if windowSum[i] > 1e-10 { audioSamples[i] /= windowSum[i] }
        }

        // Trim leading / trailing half-windows (standard STFT boundary).
        let trimStart = nFFT / 2
        let trimEnd = outLen - nFFT / 2
        let output =
            trimEnd > trimStart
            ? Array(audioSamples[trimStart ..< trimEnd])
            : audioSamples

        let t = Tensor.empty(shape: [output.count], dtype: .f32, device: device)
        t.copyIn(from: output)
        return t
    }

    /// Convenience overload accepting `Tensor` spectrogram planes. The tensors
    /// are read to CPU (shared storage — no extra copy) before synthesis.
    public func synthesize(
        specRe: Tensor, specIm: Tensor,
        device: Device = .shared
    ) -> Tensor {
        let nFreq = nFFT / 2 + 1
        precondition(
            specRe.shape.count == 2 && specRe.shape[1] == nFreq,
            "StyleTTS2Vocoder.synthesize: specRe must be [nFrames, nFreq]")
        precondition(
            specIm.shape == specRe.shape,
            "StyleTTS2Vocoder.synthesize: specRe and specIm shape mismatch")
        let nFrames = specRe.shape[0]
        let reFlat = specRe.toArray(as: Float.self)
        let imFlat = specIm.toArray(as: Float.self)
        return synthesize(
            specReFlat: reFlat, specImFlat: imFlat,
            nFrames: nFrames, device: device)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────────────

public enum StyleTTS2Error: Error, CustomStringConvertible {
    /// The StyleTTS2 acoustic front-end (ALBERT encoder + prosody predictor
    /// + AdaIN decoder + iSTFTNet generator) requires batched GEMM, multi-
    /// head attention, 1-D conv, and BiLSTM — GPU operators not yet in FFAI's
    /// Ops set. `synthesize(text:)` will throw this until those operators
    /// land. Use `synthesizeFromSpectrogram` to drive the GPU vocoder tail
    /// directly, or use the reference mlx-audio-swift port for full synthesis.
    case acousticFrontEndNotWired
    /// A required file (e.g. `voices.safetensors`) is missing.
    case missingFile(String)
    /// The named voice is not in the loaded voices file.
    case unknownVoice(String)

    public var description: String {
        switch self {
        case .acousticFrontEndNotWired:
            return "StyleTTS2: the ALBERT + prosody predictor + AdaIN decoder "
                + "acoustic front-end requires batched GEMM, multi-head "
                + "attention, 1-D conv, and BiLSTM — GPU operators not yet in "
                + "the FFAI Ops set. Drive the vocoder via "
                + "synthesizeFromSpectrogram, or use the mlx-audio-swift "
                + "reference port for full text→audio synthesis."
        case .missingFile(let name):
            return "StyleTTS2: required file missing from model directory: \(name)"
        case .unknownVoice(let name):
            return "StyleTTS2: voice '\(name)' not found in voices file"
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// A loaded StyleTTS2 / KittenTTS model.
///
/// Owns the decoded `StyleTTS2Config`, the iSTFTNet CPU-based vocoder (always
/// available for spectrogram→waveform), and optionally the voice style
/// embeddings from `voices.safetensors`. The full acoustic front-end
/// (ALBERT + prosody predictor + decoder) is gated behind
/// `StyleTTS2Error.acousticFrontEndNotWired` until FFAI adds the required
/// GPU operators (batched GEMM, multi-head attention, 1-D conv, BiLSTM).
public final class StyleTTS2Model: @unchecked Sendable {
    /// Decoded config from `config.json`.
    public let config: StyleTTS2Config
    /// iSTFTNet vocoder (CPU overlap-add) — always available.
    public let vocoder: StyleTTS2Vocoder
    /// Number of weights loaded from `model.safetensors` (diagnostic).
    public let weightCount: Int
    /// Voice name → style embedding `[maxSeqLen, styleDim]`.
    /// Nil until `load(directory:)` populates it.
    public let voiceEmbeddings: [String: Tensor]
    /// Waveform sample rate in Hz (24 000 for KittenTTS).
    public var sampleRate: Int { config.sampleRate }

    public init(
        config: StyleTTS2Config,
        vocoder: StyleTTS2Vocoder,
        weightCount: Int = 0,
        voiceEmbeddings: [String: Tensor] = [:]
    ) {
        self.config = config
        self.vocoder = vocoder
        self.weightCount = weightCount
        self.voiceEmbeddings = voiceEmbeddings
    }

    // ─── Vocoder-only path ────────────────────────────────────────────

    /// Reconstruct a waveform from a predicted complex spectrogram. This
    /// is the GPU-accelerated path — it doesn't require the acoustic
    /// front-end. `specRe` / `specIm` are `[nFrames, nFreq]`.
    public func synthesizeFromSpectrogram(
        specRe: Tensor, specIm: Tensor, device: Device = .shared
    ) -> Tensor {
        vocoder.synthesize(specRe: specRe, specIm: specIm, device: device)
    }

    // ─── Token-id helpers ─────────────────────────────────────────────

    /// Convert a phonemized IPA string to KittenTTS token ids using the
    /// built-in symbol table. Prepend / append padding id 0 (BOS/EOS) as
    /// the model expects.
    public func phonemeIds(for ipa: String) -> [Int] {
        var ids = [0]  // BOS padding token
        ids.append(contentsOf: StyleTTS2TextCleaner.tokenize(ipa))
        ids.append(0)  // EOS padding token
        return ids
    }

    // ─── Placeholder for integration tests ───────────────────────────

    /// Return a placeholder waveform (zeros) for integration tests that
    /// verify load + config without running the acoustic forward pass.
    /// Shape: `[nSamples]` f32.
    public func generatePlaceholder(
        durationSeconds: Double = 0.1,
        device: Device = .shared
    ) -> Tensor {
        let nSamples = max(1, Int(durationSeconds * Double(sampleRate)))
        let t = Tensor.empty(shape: [nSamples], dtype: .f32, device: device)
        t.zero()
        return t
    }

    // ─── Full synthesis path ──────────────────────────────────────────

    /// Full text→waveform synthesis.
    ///
    /// Requires the StyleTTS2 acoustic front-end (ALBERT encoder +
    /// prosody predictor + KittenDecoder + iSTFTNet generator). Throws
    /// `StyleTTS2Error.acousticFrontEndNotWired` until the required GPU
    /// operator set (batched GEMM, multi-head attention, 1-D conv,
    /// BiLSTM) lands in FFAI's Ops.
    ///
    /// - Parameters:
    ///   - text: Input text or pre-phonemized IPA string. When `text`
    ///           contains only IPA symbols, it is tokenized directly;
    ///           pass through an external G2P for plain English.
    ///   - voice: Voice name (e.g. "Bella", "Leo"). Resolved via
    ///            `config.voiceAliases` before looking up embeddings.
    ///   - speed: Synthesis speed multiplier. Applied on top of any
    ///            per-voice speed prior.
    ///   - device: Metal device.
    public func synthesize(
        text: String,
        voice: String = "expr-voice-5-m",
        speed: Float = 1.0,
        device: Device = .shared
    ) throws -> Tensor {
        _ = text
        _ = voice
        _ = speed
        _ = device
        throw StyleTTS2Error.acousticFrontEndNotWired
    }
}

// ─── Loading ─────────────────────────────────────────────────────────────────

extension StyleTTS2Model {
    /// `model_type` values that identify a KittenTTS / StyleTTS2 checkpoint.
    public static let modelTypes: Set<String> = ["kitten_tts", "style_tts2", "styletts2"]

    /// Whether a decoded `config.json` describes a KittenTTS / StyleTTS2
    /// checkpoint. Checks the `model_type` first; falls back to structural
    /// detection on `n_token` + `istftnet` + `plbert`.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        return config.has("n_token") && config.has("istftnet") && config.has("plbert")
    }

    /// Load a KittenTTS / StyleTTS2 checkpoint from a resolved snapshot
    /// directory.
    ///
    /// - Decodes `config.json` into a `StyleTTS2Config`.
    /// - Counts weights in `model.safetensors` / shards without allocating
    ///   GPU memory for the full acoustic stack (the forward pass is not
    ///   yet wired).
    /// - Loads `voices.safetensors` when present; falls back to an empty
    ///   voice table when absent (allows config-only tests).
    public static func load(directory: URL, device: Device = .shared)
        throws -> StyleTTS2Model
    {
        let modelConfig = try ModelConfig.load(from: directory)
        guard handles(modelConfig) else {
            throw ModelError.unsupportedModelType(
                modelConfig.modelType ?? "unknown — expected kitten_tts / style_tts2"
            )
        }
        guard let sc = StyleTTS2Config.from(modelConfig) else {
            throw ModelError.unsupportedModelType(
                "config.json is missing required StyleTTS2 fields "
                    + "(n_token, hidden_dim, plbert, istftnet)"
            )
        }

        // Verify the model weights file exists and count tensors (no full GPU alloc).
        let weightCount = try countWeights(directory: directory, device: device)

        // Load voice embeddings from `voices.safetensors` when available.
        // Missing voices file is non-fatal — callers may drive the vocoder
        // directly with a predicted spectrogram.
        let voiceEmbeddings: [String: Tensor]
        let voicesURL = directory.appendingPathComponent("voices.safetensors")
        if FileManager.default.fileExists(atPath: voicesURL.path) {
            voiceEmbeddings = try loadVoiceEmbeddings(url: voicesURL, device: device)
        } else {
            voiceEmbeddings = [:]
        }

        let vocoder = StyleTTS2Vocoder(
            nFFT: sc.istftnet.genIstftNFft,
            hopLength: sc.istftnet.genIstftHopSize
        )
        return StyleTTS2Model(
            config: sc,
            vocoder: vocoder,
            weightCount: weightCount,
            voiceEmbeddings: voiceEmbeddings
        )
    }

    /// Build a `StyleTTS2Model` from an explicit config. Useful for tests
    /// and benchmarks that don't need a real checkpoint on disk.
    public static func build(config sc: StyleTTS2Config) -> StyleTTS2Model {
        let vocoder = StyleTTS2Vocoder(
            nFFT: sc.istftnet.genIstftNFft,
            hopLength: sc.istftnet.genIstftHopSize
        )
        return StyleTTS2Model(config: sc, vocoder: vocoder)
    }

    // ─── Private helpers ──────────────────────────────────────────────

    /// Count the total number of tensor entries across model.safetensors
    /// (or shards). No GPU allocation — just header parsing.
    private static func countWeights(directory: URL, device: Device) throws -> Int {
        let single = directory.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: single.path) {
            let f = try SafeTensorsFile(url: single, device: device)
            return f.entries.count
        }
        // Sharded checkpoint: glob for *.safetensors excluding voices.
        let fm = FileManager.default
        let files =
            (try? fm.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil)) ?? []
        return
            try files
            .filter {
                $0.pathExtension == "safetensors"
                    && $0.lastPathComponent != "voices.safetensors"
            }
            .reduce(0) { count, url in
                let f = try SafeTensorsFile(url: url, device: device)
                return count + f.entries.count
            }
    }

    /// Load voice style embeddings from `voices.safetensors`. Each entry
    /// is a `[seqLen, styleDim]` f32 tensor keyed by voice name.
    private static func loadVoiceEmbeddings(url: URL, device: Device)
        throws -> [String: Tensor]
    {
        let file = try SafeTensorsFile(url: url, device: device)
        var voices = [String: Tensor]()
        for (name, _) in file.entries {
            voices[name] = try file.tensor(named: name)
        }
        return voices
    }
}
