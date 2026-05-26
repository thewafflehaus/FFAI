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
// Kokoro — a StyleTTS2-family text-to-speech model. Kokoro turns a
// phonemized text sequence into a waveform through the StyleTTS2
// pipeline:
//
//   phonemes ──text encoder (PLBert)──▶ token features
//            ──prosody predictor (+ voice style)──▶ durations + F0/N
//            ──length-regulated decoder──▶ predicted STFT (re, im)
//            ──iSTFTNet vocoder──▶ waveform
//
// FFAI ships the `vocoder` (iSTFT overlap-add) Metal kernel — the
// waveform-synthesis tail — and `Ops.vocoderISTFT` wraps it. This file
// is the FFAI Kokoro model: a `KokoroVocoder` that runs the iSTFTNet
// tail on the GPU, plus the surrounding model scaffold.
//
// ## Scope note
//
// The full StyleTTS2 acoustic stack (the PLBert text encoder, the
// duration / F0 / N prosody predictors, the AdaIN-residual decoder) is
// a large port. FFAI's contribution today is the GPU vocoder tail
// and the model-level plumbing: `Kokoro.synthesize` consumes a
// predicted complex spectrogram and produces a waveform. The acoustic
// front-end is loaded from the checkpoint when present; when it is
// not yet wired, `synthesizeFromSpectrogram` is the supported entry
// point and `synthesize(phonemes:)` reports the front-end as
// unavailable.
//
// ## Voices
//
// The default voice is `af_heart` — the standard Kokoro v1.0
// reference voice. Override per call via
// `AudioGenerationParameters.voice` or persist a different active
// voice with `setVoice(_:)`. The full Kokoro v1.0 voice catalogue is
// exposed via `availableVoices`; each voice is a 256-dim style vector
// the prosody predictor + decoder consume. Voice vectors are not
// bundled with FFAI — they're resolved on first use from the
// HuggingFace cache (the Kokoro repo ships them alongside the model
// weights) so we don't redistribute weights under a different licence.
//
// ## Phonemizer (text-side front-end)
//
// Kokoro takes *phonemes*, not raw text. The text-to-phoneme step
// runs through the `Phonemizer` protocol (`Audio/Phonemizer.swift`);
// callers register a provider via `PhonemizerRegistry.shared`. Misaki
// G2P (Apache 2.0) is the upstream Kokoro choice for English; for
// other languages users typically plug in espeak-ng (GPL — not
// bundled) or a neural G2P model. FFAI ships the registry surface and
// the AudioModel-level `voice` plumbing so the acoustic-stack port
// (when it lands) drops straight in.

import Foundation
import Metal

// ─── Configuration ───────────────────────────────────────────────────

/// Kokoro / iSTFTNet hyper-parameters, decoded from `config.json`.
public struct KokoroConfig: Sendable {
    /// Phoneme vocabulary size (`n_token`).
    public let nToken: Int
    /// Acoustic hidden dimension (`hidden_dim`).
    public let hidden: Int
    /// Mel bins the acoustic decoder predicts (`n_mels`).
    public let nMels: Int
    /// Output waveform sample rate (24 kHz for Kokoro).
    public let sampleRate: Int
    /// iSTFT FFT length — Kokoro's iSTFTNet head uses a tiny 20-sample
    /// FFT with hop 5 (`gen_istft_n_fft`, `gen_istft_hop_size`).
    public let istftNFFT: Int
    /// iSTFT hop length.
    public let istftHop: Int

    public init(
        nToken: Int, hidden: Int, nMels: Int,
        sampleRate: Int = 24_000, istftNFFT: Int = 20,
        istftHop: Int = 5
    ) {
        self.nToken = nToken
        self.hidden = hidden
        self.nMels = nMels
        self.sampleRate = sampleRate
        self.istftNFFT = istftNFFT
        self.istftHop = istftHop
    }

    /// Build a `KokoroConfig` from a decoded `config.json`. Kokoro's
    /// config nests the iSTFTNet block under `istftnet`.
    public static func from(_ config: ModelConfig) -> KokoroConfig? {
        guard let nToken = config.int("n_token"),
            let hidden = config.int("hidden_dim")
        else { return nil }
        let nMels = config.int("n_mels") ?? 80
        let sampleRate = config.int("sample_rate") ?? 24_000
        var nFFT = 20
        var hop = 5
        if let istft = config.nested("istftnet") {
            if let f = istft["gen_istft_n_fft"] as? Int { nFFT = f }
            if let h = istft["gen_istft_hop_size"] as? Int { hop = h }
        }
        return KokoroConfig(
            nToken: nToken, hidden: hidden, nMels: nMels,
            sampleRate: sampleRate, istftNFFT: nFFT,
            istftHop: hop)
    }
}

// ─── iSTFTNet vocoder ────────────────────────────────────────────────

/// The iSTFTNet vocoder tail — turns a predicted complex spectrogram
/// back into a time-domain waveform. This is the GPU-accelerated part
/// of Kokoro: `Ops.vocoderISTFT` wraps the fused iSTFT overlap-add
/// kernel.
public final class KokoroVocoder: @unchecked Sendable {
    /// FFT length of the iSTFT.
    public let nFFT: Int
    /// Hop length of the iSTFT.
    public let hopLength: Int
    /// Synthesis window `[nFFT]` (Hann). Built once at init.
    public let window: Tensor

    public init(
        nFFT: Int, hopLength: Int, dtype: DType = .f32,
        device: Device = .shared
    ) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        let win = AudioPreprocessing.hannWindow(nFFT)
        let w = Tensor.empty(shape: [nFFT], dtype: dtype, device: device)
        AudioPreprocessing.copyFloats(win, into: w)
        self.window = w
    }

    /// Reconstruct a waveform from a predicted STFT. `specRe` / `specIm`
    /// are `[nFrames, nFreq]` real / imaginary planes
    /// (`nFreq = nFFT/2 + 1`). Returns the `[outLen]` waveform with
    /// `outLen = (nFrames - 1) * hopLength + nFFT`.
    public func synthesize(
        specRe: Tensor, specIm: Tensor,
        device: Device = .shared
    ) -> Tensor {
        let nFreq = nFFT / 2 + 1
        precondition(
            specRe.shape.count == 2 && specRe.shape[1] == nFreq,
            "KokoroVocoder.synthesize: specRe must be [nFrames, nFreq]")
        let nFrames = specRe.shape[0]
        let cmd = device.makeCommandBuffer()
        let waveform = Ops.vocoderISTFT(
            specRe: specRe, specIm: specIm, window: window,
            nFrames: nFrames, nFFT: nFFT, hopLength: hopLength, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return waveform
    }
}

// ─── Kokoro model ────────────────────────────────────────────────────

public enum KokoroError: Error, CustomStringConvertible {
    /// The acoustic front-end (text encoder + prosody predictor +
    /// decoder) is not yet wired in this build — `synthesize(phonemes:)`
    /// needs it. `synthesizeFromSpectrogram` works regardless.
    case acousticFrontEndUnavailable

    public var description: String {
        switch self {
        case .acousticFrontEndUnavailable:
            return "Kokoro: the StyleTTS2 acoustic front-end is not wired "
                + "in this build; drive the vocoder via "
                + "synthesizeFromSpectrogram or supply a predicted "
                + "spectrogram from an external acoustic model"
        }
    }
}

/// A loaded Kokoro TTS model. Owns the iSTFTNet vocoder tail (always
/// available — it is the FFAI-accelerated part) and, when the
/// checkpoint's acoustic stack is wired, the text encoder + prosody
/// predictor + decoder.
public final class KokoroModel: @unchecked Sendable {
    public let config: KokoroConfig
    public let vocoder: KokoroVocoder
    /// The phoneme→id table from the checkpoint's `config.json` `vocab`.
    public let phonemeVocab: [String: Int]
    let dtype: DType
    /// Currently-active voice name (mutates via `setVoice(_:)`).
    /// Defaults to `Kokoro.defaultVoice` at init.
    public private(set) var currentVoice: String

    public init(
        config: KokoroConfig, vocoder: KokoroVocoder,
        phonemeVocab: [String: Int], dtype: DType
    ) {
        self.config = config
        self.vocoder = vocoder
        self.phonemeVocab = phonemeVocab
        self.dtype = dtype
        self.currentVoice = KokoroModel.defaultVoice
    }

    // ─── Voice catalogue ─────────────────────────────────────────────

    /// Default voice — Kokoro v1.0's reference female "American Heart"
    /// voice. Picked because it's the most-cited demo voice in the
    /// Kokoro paper + community samples.
    public static let defaultVoice = "af_heart"

    /// Kokoro v1.0 voice catalogue. Prefix convention:
    ///   a* = American English   b* = British English
    ///   e* = Spanish            f* = French
    ///   h* = Hindi              i* = Italian
    ///   j* = Japanese           p* = Portuguese (BR)
    ///   z* = Mandarin Chinese
    /// Second letter: f = female, m = male.
    ///
    /// Voice vectors (256-dim style vectors per voice) are not
    /// bundled here — Kokoro resolves them from the HF cache on first
    /// `setVoice(_:)` once the acoustic stack ships.
    public static let availableVoices: [String] = [
        // American English
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
        // British English
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        // Other languages — abbreviated prefix only; full list grows
        // with Kokoro upstream voice packs.
        "ef_dora", "em_alex", "em_santa",
        "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
        "jm_kumo",
        "pf_dora", "pm_alex", "pm_santa",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]

    /// Activate `name` as the model's current voice. Validates against
    /// `availableVoices`; throws `AudioGenerationError.voiceNotAvailable`
    /// for unknown names. The actual style-vector load (from the HF
    /// cache) lands with the acoustic stack — this is the API surface.
    public func setVoice(_ name: String) throws {
        let resolved = (name == "default") ? KokoroModel.defaultVoice : name
        guard KokoroModel.availableVoices.contains(resolved) else {
            throw AudioGenerationError.voiceNotAvailable(
                requested: name, available: KokoroModel.availableVoices)
        }
        currentVoice = resolved
    }

    /// Synthesize a waveform from a predicted complex spectrogram.
    /// This is the GPU vocoder path — the FFAI-accelerated tail of
    /// Kokoro. `specRe` / `specIm` are `[nFrames, nFreq]`.
    public func synthesizeFromSpectrogram(
        specRe: Tensor, specIm: Tensor, device: Device = .shared
    ) -> Tensor {
        vocoder.synthesize(specRe: specRe, specIm: specIm, device: device)
    }

    /// Map a phoneme string into checkpoint token ids using the
    /// `config.json` `vocab` table. An external Misaki G2P produces the
    /// phoneme string; this is the id mapping Kokoro applies before the
    /// acoustic encoder. Unknown phonemes are dropped.
    public func phonemeIds(for phonemes: String) -> [Int] {
        phonemes.compactMap { phonemeVocab[String($0)] }
    }

    /// Full text→waveform synthesis. Requires the acoustic front-end;
    /// throws `KokoroError.acousticFrontEndUnavailable` when it is not
    /// wired (see the scope note at the top of this file).
    public func synthesize(
        phonemeIds: [Int],
        device: Device = .shared
    ) throws -> Tensor {
        _ = phonemeIds
        _ = device
        throw KokoroError.acousticFrontEndUnavailable
    }
}

// ─── Loading ─────────────────────────────────────────────────────────

extension KokoroModel {
    public static let modelTypes: Set<String> = ["kokoro", "style_tts2"]

    /// Whether a decoded `config.json` describes a Kokoro checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        return config.has("istftnet") && config.has("n_token")
    }

    /// Load a Kokoro checkpoint from a resolved snapshot directory. The
    /// vocoder tail is always constructed; the acoustic stack is wired
    /// when present (see scope note).
    public static func load(directory: URL, device: Device = .shared)
        throws -> KokoroModel
    {
        let config = try ModelConfig.load(from: directory)
        guard let kc = KokoroConfig.from(config) else {
            throw ModelError.unsupportedModelType(
                "config.json is not a Kokoro config")
        }
        // The phoneme vocabulary lives inline in config.json.
        let phonemeVocab = (config.raw["vocab"] as? [String: Int]) ?? [:]
        return build(config: kc, phonemeVocab: phonemeVocab, device: device)
    }

    /// Assemble a `KokoroModel` from a decoded config. Factored out so
    /// tests can drive the vocoder path without a checkpoint.
    public static func build(
        config kc: KokoroConfig,
        phonemeVocab: [String: Int] = [:],
        dtype: DType = .f32,
        device: Device = .shared
    ) -> KokoroModel {
        let vocoder = KokoroVocoder(
            nFFT: kc.istftNFFT, hopLength: kc.istftHop,
            dtype: dtype, device: device)
        return KokoroModel(
            config: kc, vocoder: vocoder,
            phonemeVocab: phonemeVocab, dtype: dtype)
    }
}
