// AudioPreprocessing — CPU-side waveform handling for the audio
// front-end. The heavy STFT + Mel projection runs on the GPU
// (`Ops.melSpectrogram`); this file is the framing math around it:
//
//   * load a waveform from a file or raw PCM,
//   * resample to the model's expected sample rate,
//   * build the analysis window + Mel filterbank,
//   * reflect-pad so every STFT frame is in-bounds.
//
// The Mel filterbank + Hann window are pure functions of the front-end
// config, so a model builds them once at load time and reuses them for
// every clip.

import Foundation
import Metal
#if canImport(AVFoundation)
import AVFoundation
#endif

/// Front-end geometry shared by every speech model FFAI supports.
/// Whisper, Qwen-Omni audio-in and Kokoro all use the standard
/// log-Mel STFT; only the constants differ.
public struct AudioFrontEndConfig: Sendable {
    /// Target sample rate in Hz (Whisper / Qwen-Omni: 16000).
    public let sampleRate: Int
    /// STFT window / FFT length in samples.
    public let nFFT: Int
    /// Hop between consecutive frames in samples.
    public let hopLength: Int
    /// Number of Mel filterbank bins.
    public let nMels: Int
    /// Lowest Mel-filter edge frequency in Hz.
    public let fMin: Double
    /// Highest Mel-filter edge frequency in Hz (`nil` → Nyquist).
    public let fMax: Double?

    public init(sampleRate: Int, nFFT: Int, hopLength: Int, nMels: Int,
                fMin: Double = 0, fMax: Double? = nil) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels
        self.fMin = fMin
        self.fMax = fMax
    }

    /// Non-redundant real-FFT bin count — `Ops.melSpectrogram`'s
    /// `nFreq` invariant.
    public var nFreq: Int { nFFT / 2 + 1 }

    /// The Whisper / Qwen-Omni front-end: 16 kHz, 400-sample FFT,
    /// 160-sample hop (10 ms), 80 Mel bins.
    public static let whisper = AudioFrontEndConfig(
        sampleRate: 16_000, nFFT: 400, hopLength: 160, nMels: 80)

    /// Whisper large-v3 uses 128 Mel bins (everything else identical).
    public static let whisperLargeV3 = AudioFrontEndConfig(
        sampleRate: 16_000, nFFT: 400, hopLength: 160, nMels: 128)
}

public enum AudioPreprocessing {

    // ─── Window + filterbank construction ────────────────────────────

    /// Periodic Hann window of length `n` — the analysis window every
    /// log-Mel front-end uses. "Periodic" (divisor `n`, not `n - 1`) is
    /// the librosa / PyTorch STFT default; matching it is what keeps
    /// FFAI's Mel features aligned with the reference checkpoints.
    public static func hannWindow(_ n: Int) -> [Float] {
        guard n > 1 else { return [Float](repeating: 1, count: max(n, 0)) }
        var w = [Float](repeating: 0, count: n)
        let twoPi = 2.0 * Double.pi
        for i in 0..<n {
            w[i] = Float(0.5 - 0.5 * cos(twoPi * Double(i) / Double(n)))
        }
        return w
    }

    /// Convert a frequency in Hz to the Mel scale (HTK formula — the
    /// convention Whisper / transformers use).
    private static func hzToMel(_ hz: Double) -> Double {
        2595.0 * log10(1.0 + hz / 700.0)
    }

    /// Convert a Mel value back to Hz (HTK inverse).
    private static func melToHz(_ mel: Double) -> Double {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    /// Build a `[nMels, nFreq]` row-major Mel filterbank — the triangular
    /// projection from the linear power spectrum onto the Mel scale.
    /// Slaney-normalised (each triangle scaled by `2 / (f_hi - f_lo)`),
    /// matching `librosa.filters.mel(norm="slaney")` and the filterbank
    /// baked into the Whisper / transformers feature extractors.
    public static func melFilterbank(_ cfg: AudioFrontEndConfig) -> [Float] {
        let nFreq = cfg.nFreq
        let fMax = cfg.fMax ?? Double(cfg.sampleRate) / 2.0
        // FFT bin centre frequencies (Hz): bin k → k * sr / nFFT.
        var fftFreqs = [Double](repeating: 0, count: nFreq)
        for k in 0..<nFreq {
            fftFreqs[k] = Double(k) * Double(cfg.sampleRate) / Double(cfg.nFFT)
        }
        // `nMels + 2` equally-spaced points on the Mel scale → triangle
        // edges. Triangle m spans [pts[m], pts[m+2]] peaking at pts[m+1].
        let melLo = hzToMel(cfg.fMin)
        let melHi = hzToMel(fMax)
        var edgeHz = [Double](repeating: 0, count: cfg.nMels + 2)
        for i in 0..<(cfg.nMels + 2) {
            let mel = melLo + (melHi - melLo) * Double(i) / Double(cfg.nMels + 1)
            edgeHz[i] = melToHz(mel)
        }

        var bank = [Float](repeating: 0, count: cfg.nMels * nFreq)
        for m in 0..<cfg.nMels {
            let lo = edgeHz[m]
            let ctr = edgeHz[m + 1]
            let hi = edgeHz[m + 2]
            // Slaney normalisation — keeps total filter energy uniform.
            let enorm = 2.0 / (hi - lo)
            for k in 0..<nFreq {
                let f = fftFreqs[k]
                // Rising edge lo→ctr, falling edge ctr→hi.
                let lower = (f - lo) / max(ctr - lo, 1e-9)
                let upper = (hi - f) / max(hi - ctr, 1e-9)
                let tri = max(0.0, min(lower, upper))
                bank[m * nFreq + k] = Float(tri * enorm)
            }
        }
        return bank
    }

    // ─── Sinusoidal position table ───────────────────────────────────

    /// Build a `[length, dim]` sinusoidal position-embedding table —
    /// the fixed (non-learned) positional encoding Whisper bakes into
    /// its checkpoint as `embed_positions.weight` and Qwen-Omni's audio
    /// encoder computes at runtime. `dim` must be even.
    ///
    /// Row `p`, even index `2i`: `sin(p / 10000^(2i/dim))`; odd index
    /// `2i+1`: `cos(...)` — the original Transformer formulation, which
    /// is also what Whisper / Qwen-Omni use for the audio encoder.
    public static func sinusoidalPositions(length: Int, dim: Int)
        -> [Float] {
        precondition(dim % 2 == 0,
                     "sinusoidalPositions: dim must be even, got \(dim)")
        var table = [Float](repeating: 0, count: length * dim)
        let half = dim / 2
        // log-spaced inverse frequencies, matching Whisper's
        // `log(10000) / (half - 1)` increment.
        let logTimescale = log(10_000.0) / Double(max(half - 1, 1))
        for p in 0..<length {
            for i in 0..<half {
                let invFreq = exp(-logTimescale * Double(i))
                let angle = Double(p) * invFreq
                table[p * dim + i] = Float(sin(angle))
                table[p * dim + half + i] = Float(cos(angle))
            }
        }
        return table
    }

    // ─── Reflect padding ─────────────────────────────────────────────

    /// Reflect-pad a mono waveform by `pad` samples on each side — the
    /// `mel_spectrogram` kernel does no bounds check on the frame walk,
    /// so the caller pads to keep every frame in-bounds. Whisper pads by
    /// `nFFT / 2`. Reflection mirrors the signal across the boundary
    /// sample (numpy `mode="reflect"`).
    public static func reflectPad(_ x: [Float], pad: Int) -> [Float] {
        guard pad > 0 else { return x }
        guard x.count > 1 else {
            // Degenerate signal — constant-pad with the single sample
            // (reflection is undefined for length ≤ 1).
            let v = x.first ?? 0
            return [Float](repeating: v, count: pad) + x
                + [Float](repeating: v, count: pad)
        }
        let n = x.count
        // Reflection index folds an out-of-range index back into [0, n).
        func reflectIndex(_ i: Int) -> Int {
            var j = i
            let period = 2 * (n - 1)
            j = ((j % period) + period) % period
            return j < n ? j : period - j
        }
        var out = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad {
            out[i] = x[reflectIndex(pad - i)]            // left mirror
        }
        for i in 0..<n {
            out[pad + i] = x[i]
        }
        for i in 0..<pad {
            out[pad + n + i] = x[reflectIndex(n - 2 - i)] // right mirror
        }
        return out
    }

    /// Number of STFT frames a (post-pad) signal of `paddedSamples`
    /// samples produces. Mirrors the kernel's frame walk:
    /// frame f covers `[f*hop, f*hop + nFFT)`.
    public static func frameCount(paddedSamples: Int,
                                  cfg: AudioFrontEndConfig) -> Int {
        guard paddedSamples >= cfg.nFFT else { return 0 }
        return (paddedSamples - cfg.nFFT) / cfg.hopLength + 1
    }

    // ─── Resampling ──────────────────────────────────────────────────

    /// Linear-interpolation resample of a mono waveform from `srcRate`
    /// to `dstRate`. Linear interpolation is adequate for STT/TTS
    /// front-ends — the log-Mel projection that follows is far lossier
    /// than the resampler. A polyphase resampler is a quality
    /// follow-up.
    public static func resample(_ x: [Float], from srcRate: Int,
                                 to dstRate: Int) -> [Float] {
        guard srcRate != dstRate, srcRate > 0, dstRate > 0, !x.isEmpty else {
            return x
        }
        let ratio = Double(dstRate) / Double(srcRate)
        let dstCount = Int((Double(x.count) * ratio).rounded())
        guard dstCount > 0 else { return [] }
        var out = [Float](repeating: 0, count: dstCount)
        for i in 0..<dstCount {
            let srcPos = Double(i) / ratio
            let i0 = Int(srcPos.rounded(.down))
            let frac = Float(srcPos - Double(i0))
            let a = x[min(i0, x.count - 1)]
            let b = x[min(i0 + 1, x.count - 1)]
            out[i] = a + (b - a) * frac
        }
        return out
    }

    // ─── File loading ────────────────────────────────────────────────

    public enum AudioError: Error, CustomStringConvertible {
        case decodeFailed(String)
        case unsupportedFormat(String)

        public var description: String {
            switch self {
            case .decodeFailed(let m): return "Audio decode failed: \(m)"
            case .unsupportedFormat(let m): return "Unsupported audio format: \(m)"
            }
        }
    }

    /// Load an audio file as a mono `[Float]` waveform resampled to
    /// `targetRate`. Uses AVFoundation, so any container the OS decodes
    /// (wav / mp3 / m4a / flac / …) works. Multi-channel input is
    /// down-mixed to mono by averaging.
    #if canImport(AVFoundation)
    public static func loadWaveform(url: URL, targetRate: Int) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioError.decodeFailed("\(url.lastPathComponent): \(error)")
        }
        let srcFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0 else { return [] }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: srcFormat,
                                            frameCapacity: frameCount) else {
            throw AudioError.decodeFailed("could not allocate PCM buffer")
        }
        do {
            try file.read(into: buffer)
        } catch {
            throw AudioError.decodeFailed("read failed: \(error)")
        }
        guard let channelData = buffer.floatChannelData else {
            throw AudioError.unsupportedFormat("non-float PCM not supported")
        }
        let channels = Int(srcFormat.channelCount)
        let n = Int(buffer.frameLength)
        var mono = [Float](repeating: 0, count: n)
        // Down-mix to mono by averaging the channels.
        for c in 0..<channels {
            let ptr = channelData[c]
            for i in 0..<n {
                mono[i] += ptr[i]
            }
        }
        if channels > 1 {
            let inv = 1.0 / Float(channels)
            for i in 0..<n { mono[i] *= inv }
        }
        return resample(mono, from: Int(srcFormat.sampleRate), to: targetRate)
    }
    #endif

    // ─── End-to-end log-Mel ──────────────────────────────────────────

    /// Whisper's log-Mel dynamic-range floor: values more than this many
    /// decades below the clip maximum are clamped up to it.
    private static let whisperLogFloorDecades: Float = 8.0
    /// Whisper's log-Mel affine normalisation — `(log10 + 4) / 4` maps
    /// the clamped log-Mel into roughly `[-1, +1]`.
    private static let whisperNormOffset: Float = 4.0
    private static let whisperNormScale: Float = 4.0
    /// `ln(10)` — converts the kernel's natural log into a base-10 log.
    private static let lnOf10: Float = 2.302_585_092_994_046

    /// Compute the `[nFrames, nMels]` log-Mel spectrogram of a mono
    /// waveform on the GPU. Reflect-pads by `nFFT/2`, uploads the
    /// padded signal + window + filterbank, dispatches
    /// `Ops.melSpectrogram`, returns the result Tensor.
    ///
    /// `window` and `melWeight` are caller-supplied so a model can build
    /// them once at load time and reuse across clips; pass `nil` to have
    /// this helper build them per call (convenient for tests).
    ///
    /// When `whisperNormalize` is true (the default — every shipped STT /
    /// audio-in tower is Whisper-derived and expects it), the raw kernel
    /// output is post-processed exactly as OpenAI's reference
    /// `log_mel_spectrogram` does: the natural-log kernel output is
    /// rescaled to base-10, clamped to `max - 8` decades, then mapped by
    /// `(x + 4) / 4` into roughly `[-1, +1]`. Without this step the conv
    /// stem — trained on the normalised features — sees a wrong scale
    /// and offset and produces finite-but-meaningless audio features.
    ///
    /// - Important: the `cmd` commit contract differs by mode.
    ///   * `whisperNormalize == true`  — this helper **commits and waits
    ///     on `cmd`** itself (the CPU-side normalisation needs the kernel
    ///     result). The returned Tensor is CPU-synced; the caller MUST
    ///     NOT commit `cmd` again.
    ///   * `whisperNormalize == false` — the kernel is only *queued* on
    ///     `cmd`; the caller commits + waits as usual.
    public static func logMelSpectrogram(
        waveform: [Float], cfg: AudioFrontEndConfig,
        window: Tensor? = nil, melWeight: Tensor? = nil,
        dtype: DType = .f32, whisperNormalize: Bool = true,
        device: Device = .shared,
        on cmd: MTLCommandBuffer
    ) -> Tensor {
        precondition(dtype == .f32 || dtype == .f16,
                     "logMelSpectrogram: dtype must be f32 or f16")
        let pad = cfg.nFFT / 2
        let padded = reflectPad(waveform, pad: pad)
        let nFrames = frameCount(paddedSamples: padded.count, cfg: cfg)
        precondition(nFrames > 0, "logMelSpectrogram: waveform too short")

        // Upload the padded waveform.
        let audioT = Tensor.empty(shape: [padded.count], dtype: dtype,
                                  device: device)
        copyFloats(padded, into: audioT)

        // Window + filterbank — build if not supplied.
        let winT: Tensor
        if let w = window {
            winT = w
        } else {
            winT = Tensor.empty(shape: [cfg.nFFT], dtype: dtype, device: device)
            copyFloats(hannWindow(cfg.nFFT), into: winT)
        }
        let melT: Tensor
        if let m = melWeight {
            melT = m
        } else {
            melT = Tensor.empty(shape: [cfg.nMels, cfg.nFreq], dtype: dtype,
                                device: device)
            copyFloats(melFilterbank(cfg), into: melT)
        }

        let raw = Ops.melSpectrogram(
            audio: audioT, window: winT, melWeight: melT,
            nFFT: cfg.nFFT, nMels: cfg.nMels, hopLength: cfg.hopLength,
            nFrames: nFrames, on: cmd)
        guard whisperNormalize else { return raw }
        // The GPU kernel emits a *natural*-log mel; Whisper's front-end
        // is a base-10 log with a dynamic-range clamp + affine norm.
        // Apply that post-processing on the CPU (nFrames*nMels is tiny).
        return applyWhisperLogMelNorm(raw, device: device, on: cmd)
    }

    /// Post-process the raw natural-log mel from `Ops.melSpectrogram`
    /// into Whisper's normalised log-Mel. Mirrors OpenAI's reference
    /// `log_mel_spectrogram`:
    ///
    ///   log10  = ln_mel / ln(10)
    ///   log10  = max(log10, log10.max() - 8)
    ///   out    = (log10 + 4) / 4
    ///
    /// The kernel must already have been dispatched on `cmd`; this waits
    /// for it, reads the result back, normalises on the CPU and uploads
    /// the result with the same dtype.
    private static func applyWhisperLogMelNorm(
        _ rawNaturalLog: Tensor, device: Device, on cmd: MTLCommandBuffer
    ) -> Tensor {
        cmd.commit()
        cmd.waitUntilCompleted()
        var vals = rawNaturalLog.toFloatArray()
        // Natural log → base-10 log.
        let invLn10 = 1.0 / lnOf10
        var maxLog = -Float.greatestFiniteMagnitude
        for i in vals.indices {
            vals[i] *= invLn10
            if vals[i] > maxLog { maxLog = vals[i] }
        }
        // Dynamic-range clamp + affine normalisation.
        let floor = maxLog - whisperLogFloorDecades
        for i in vals.indices {
            let clamped = max(vals[i], floor)
            vals[i] = (clamped + whisperNormOffset) / whisperNormScale
        }
        let out = Tensor.empty(shape: rawNaturalLog.shape,
                               dtype: rawNaturalLog.dtype, device: device)
        copyFloats(vals, into: out)
        return out
    }

    /// Convert a tensor to a different storage dtype via a CPU
    /// round-trip through `[Float]`. Returns `source` unchanged when the
    /// dtype already matches. Used by the audio encoders to bridge the
    /// f32-only `mel_spectrogram` kernel output to a bf16 model: the Mel
    /// front-end runs in f32, then the spectrogram is cast to the
    /// model's activation dtype before the conv stem.
    static func castTensor(_ source: Tensor, to dtype: DType,
                           device: Device = .shared) -> Tensor {
        guard source.dtype != dtype else { return source }
        let values = source.toFloatArray()
        let out = Tensor.empty(shape: source.shape, dtype: dtype,
                               device: device)
        copyFloats(values, into: out)
        return out
    }

    /// Copy a `[Float]` array into a tensor, converting to the tensor's
    /// dtype (f32 / f16 / bf16). Small helper used by the upload paths.
    static func copyFloats(_ values: [Float], into t: Tensor) {
        precondition(values.count == t.elementCount,
                     "copyFloats: count mismatch \(values.count) vs \(t.elementCount)")
        switch t.dtype {
        case .f32:
            t.copyIn(from: values)
        case .f16:
            t.copyIn(from: values.map { Float16($0) })
        case .bf16:
            t.copyIn(from: values.map { v -> UInt16 in
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            })
        default:
            fatalError("copyFloats: unsupported dtype \(t.dtype)")
        }
    }
}
