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
// DeepFilterNetDSP — STFT / iSTFT helpers for DeepFilterNet speech enhancement.
//
// DeepFilterNet operates in the STFT domain: the encoder takes an
// STFT-power-spectrum + ERB features, and the decoder produces a masked /
// deep-filtered spectrum that is inverted back to time via iSTFT.
//
// This file provides:
//   * `DeepFilterNetSTFT.stft`  — short-time Fourier transform (offline)
//   * `DeepFilterNetSTFT.istft` — inverse STFT (overlap-add, offline)
//   * ERB filterbank construction utilities used by the feature extractor
//   * Vorbis window construction
//
// STFT uses vDSP real FFT (radix-2) internally. For DeepFilterNet the FFT
// size is always 960 (= 64 × 15), which is NOT a power of 2. We use the
// Apple Accelerate split-radix FFT (vDSP_fft_zrip) with the next power-of-2
// zero-padded buffer, then keep only the first fftSize/2+1 bins — this
// exactly mirrors libDF's RFFT convention.
//
// All signal math lives on the CPU; the neural-network forward runs via
// FFAI's Tensor / SafeTensors / Ops stack.
//
// Parallelism: CPU loops over frames are parallelised with
// DispatchQueue.concurrentPerform where the frame count warrants it.

import Accelerate
import Foundation

// MARK: - STFT Output

/// Per-frame complex spectrum for a mono signal.
/// Layout: `real[frame * freqBins + bin]`, `imag[frame * freqBins + bin]`.
public struct DeepFilterNetSpectrum: Sendable {
    public let real: [Float]
    public let imag: [Float]
    /// Number of frames (time axis).
    public let nFrames: Int
    /// Number of complex frequency bins (fftSize / 2 + 1).
    public let freqBins: Int
}

// MARK: - STFT / iSTFT

public enum DeepFilterNetSTFT {

    // MARK: - Vorbis window

    /// Vorbis analysis window — the window DeepFilterNet uses for both
    /// analysis and synthesis. Matches `libDF::windowVorbis`.
    ///
    /// `window[i] = sin(π/2 · sin²(π(i+0.5)/(size/2)))`
    public static func vorbisWindow(size: Int) -> [Float] {
        let half = max(1, size / 2)
        var window = [Float](repeating: 0, count: size)
        for i in 0 ..< size {
            let inner = sinf(0.5 * Float.pi * (Float(i) + 0.5) / Float(half))
            window[i] = sinf(0.5 * Float.pi * inner * inner)
        }
        return window
    }

    // MARK: - STFT

    /// Short-time Fourier transform (causal, no centre-padding).
    ///
    /// Mirrors `libDF::stft` / `MossFormer2DSP.stft`:
    ///   * Prepend `hopSize` zeros (initial frame alignment).
    ///   * Process frames with step `hopSize`.
    ///   * The output is scaled by `wnorm = 2·hopSize / fftSize²`.
    ///
    /// - Parameters:
    ///   - audio: Mono PCM in `[-1, 1]`.
    ///   - fftSize: FFT length (e.g. 960 for DeepFilterNet3).
    ///   - hopSize: Frame hop (e.g. 480).
    ///   - window: Vorbis window of length `fftSize`.
    /// - Returns: Complex spectrum `[nFrames, freqBins]`.
    public static func stft(
        audio: [Float],
        fftSize: Int,
        hopSize: Int,
        window: [Float]
    ) -> DeepFilterNetSpectrum {
        precondition(window.count == fftSize, "stft: window length must equal fftSize")

        // Prepend hopSize zeros to match libDF frame alignment.
        let padded =
            [Float](repeating: 0, count: hopSize) + audio
            + [Float](repeating: 0, count: fftSize)

        let freqBins = fftSize / 2 + 1
        let nFrames = max(0, (padded.count - fftSize) / hopSize + 1)
        let wnorm = Float(2 * hopSize) / Float(fftSize * fftSize)

        guard nFrames > 0 else {
            return DeepFilterNetSpectrum(real: [], imag: [], nFrames: 0, freqBins: freqBins)
        }

        // Allocate output arrays (nFrames × freqBins).
        var outReal = [Float](repeating: 0, count: nFrames * freqBins)
        var outImag = [Float](repeating: 0, count: nFrames * freqBins)

        // Use next power-of-2 for the FFT plan (vDSP requires power-of-2).
        let fftPow2 = nextPow2(fftSize)
        let log2N = vDSP_Length(log2(Double(fftPow2)).rounded())
        guard let fftSetup = vDSP_create_fftsetup(log2N, FFTRadix(kFFTRadix2)) else {
            return DeepFilterNetSpectrum(real: [], imag: [], nFrames: 0, freqBins: freqBins)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Process frames (parallelised for large frame counts).
        let concurrentThreshold = 16
        if nFrames >= concurrentThreshold {
            // Parallel: each frame is independent. Allocate per-frame scratch.
            var realBuffers = [[Float]](
                repeating: [Float](repeating: 0, count: fftPow2), count: nFrames)
            var imagBuffers = [[Float]](
                repeating: [Float](repeating: 0, count: fftPow2), count: nFrames)
            DispatchQueue.concurrentPerform(iterations: nFrames) { frame in
                let offset = frame * hopSize
                var frameBuffer = [Float](repeating: 0, count: fftPow2)
                // Apply window and fill frame buffer (zero-pad to fftPow2).
                for i in 0 ..< fftSize {
                    frameBuffer[i] = padded[offset + i] * window[i]
                }
                // Real FFT via split complex.
                var splitComplex = DSPSplitComplex(
                    realp: &realBuffers[frame],
                    imagp: &imagBuffers[frame]
                )
                frameBuffer.withUnsafeBufferPointer { buf in
                    buf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftPow2 / 2) {
                        cPtr in
                        vDSP_ctoz(cPtr, 2, &splitComplex, 1, vDSP_Length(fftPow2 / 2))
                    }
                }
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2N, FFTDirection(FFT_FORWARD))
                // Scale: vDSP forward FFT has an implicit factor of 2 vs DFT convention.
                // We apply wnorm here (matches libDF's fft_norm).
                let scale = wnorm / 2.0  // /2 because vDSP RFFT packs Nyquist in imag[0]
                realBuffers[frame][0] *= scale
                imagBuffers[frame][0] = 0  // DC bin is purely real
                for b in 1 ..< (freqBins - 1) {
                    realBuffers[frame][b] *= scale
                    imagBuffers[frame][b] *= scale
                }
                // Nyquist (freqBins-1) lives in splitComplex.imagp[0] for power-of-2 FFT.
                if freqBins - 1 < fftPow2 / 2 {
                    realBuffers[frame][freqBins - 1] *= scale
                    imagBuffers[frame][freqBins - 1] *= scale
                } else {
                    // Nyquist packed in imagp[0] by vDSP convention.
                    realBuffers[frame][freqBins - 1] = imagBuffers[frame][0] * scale
                    imagBuffers[frame][freqBins - 1] = 0
                }
            }
            // Gather results into flat arrays.
            for frame in 0 ..< nFrames {
                let base = frame * freqBins
                for b in 0 ..< freqBins {
                    outReal[base + b] = realBuffers[frame][b]
                    outImag[base + b] = imagBuffers[frame][b]
                }
            }
        } else {
            // Serial path for small frame counts.
            var frameBuffer = [Float](repeating: 0, count: fftPow2)
            var realScratch = [Float](repeating: 0, count: fftPow2)
            var imagScratch = [Float](repeating: 0, count: fftPow2)
            for frame in 0 ..< nFrames {
                let offset = frame * hopSize
                // Apply window and zero-pad.
                for i in 0 ..< fftSize {
                    frameBuffer[i] = padded[offset + i] * window[i]
                }
                if fftPow2 > fftSize {
                    for i in fftSize ..< fftPow2 { frameBuffer[i] = 0 }
                }
                var splitComplex = DSPSplitComplex(realp: &realScratch, imagp: &imagScratch)
                frameBuffer.withUnsafeBufferPointer { buf in
                    buf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftPow2 / 2) {
                        cPtr in
                        vDSP_ctoz(cPtr, 2, &splitComplex, 1, vDSP_Length(fftPow2 / 2))
                    }
                }
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2N, FFTDirection(FFT_FORWARD))
                let scale = wnorm / 2.0
                let base = frame * freqBins
                outReal[base] = realScratch[0] * scale
                outImag[base] = 0
                for b in 1 ..< (freqBins - 1) {
                    outReal[base + b] = realScratch[b] * scale
                    outImag[base + b] = imagScratch[b] * scale
                }
                // Nyquist
                if freqBins - 1 < fftPow2 / 2 {
                    outReal[base + freqBins - 1] = realScratch[freqBins - 1] * scale
                    outImag[base + freqBins - 1] = imagScratch[freqBins - 1] * scale
                } else {
                    outReal[base + freqBins - 1] = imagScratch[0] * scale
                    outImag[base + freqBins - 1] = 0
                }
            }
        }

        return DeepFilterNetSpectrum(
            real: outReal, imag: outImag,
            nFrames: nFrames, freqBins: freqBins
        )
    }

    // MARK: - iSTFT

    /// Inverse STFT — overlap-add synthesis.
    ///
    /// Mirrors `libDF::istft` / `MossFormer2DSP.istft`:
    ///   * Un-apply wnorm: multiply each frame spectrum by 1/wnorm before IRFFT.
    ///   * Apply Vorbis window to the time-domain frame.
    ///   * Overlap-add with step `hopSize`.
    ///   * Remove the initial `delay = fftSize - hopSize` samples (algorithmic
    ///     delay incurred by the analysis prepend).
    ///
    /// - Parameters:
    ///   - spectrum: Enhanced spectrum from the model forward pass.
    ///   - fftSize: Must match the STFT fftSize.
    ///   - hopSize: Must match the STFT hopSize.
    ///   - window: Vorbis window (same as STFT).
    ///   - origLen: Expected output length (samples clipped to this).
    /// - Returns: Enhanced mono PCM, clipped to `origLen`.
    public static func istft(
        spectrum: DeepFilterNetSpectrum,
        fftSize: Int,
        hopSize: Int,
        window: [Float],
        origLen: Int
    ) -> [Float] {
        precondition(window.count == fftSize, "istft: window length must equal fftSize")
        let nFrames = spectrum.nFrames
        let freqBins = spectrum.freqBins
        guard nFrames > 0, freqBins == fftSize / 2 + 1 else { return [] }

        let wnorm = Float(2 * hopSize) / Float(fftSize * fftSize)
        // Un-apply wnorm: the forward STFT multiplied by wnorm, so the inverse
        // must divide it out before IRFFT.
        let invWnorm = 1.0 / wnorm

        let fftPow2 = nextPow2(fftSize)
        let log2N = vDSP_Length(log2(Double(fftPow2)).rounded())
        guard let fftSetup = vDSP_create_fftsetup(log2N, FFTRadix(kFFTRadix2)) else { return [] }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Output buffer (with analysis prepend for the delay).
        let totalSamples = nFrames * hopSize + fftSize
        var outputBuf = [Float](repeating: 0, count: totalSamples)

        // Process frames serially (overlap-add can't easily be parallelised).
        var realScratch = [Float](repeating: 0, count: fftPow2)
        var imagScratch = [Float](repeating: 0, count: fftPow2)
        var timeBuf = [Float](repeating: 0, count: fftPow2)

        for frame in 0 ..< nFrames {
            let base = frame * freqBins
            // Fill split complex (DC, positive frequencies, Nyquist).
            realScratch[0] = spectrum.real[base] * invWnorm * 2.0  // ×2 for vDSP convention
            imagScratch[0] = 0
            for b in 1 ..< (freqBins - 1) {
                realScratch[b] = spectrum.real[base + b] * invWnorm * 2.0
                imagScratch[b] = spectrum.imag[base + b] * invWnorm * 2.0
            }
            // Nyquist
            if freqBins - 1 < fftPow2 / 2 {
                realScratch[freqBins - 1] = spectrum.real[base + freqBins - 1] * invWnorm * 2.0
                imagScratch[freqBins - 1] = spectrum.imag[base + freqBins - 1] * invWnorm * 2.0
            } else {
                // Pack Nyquist into imagp[0] as vDSP requires.
                imagScratch[0] = spectrum.real[base + freqBins - 1] * invWnorm * 2.0
            }
            // Zero upper half (not needed for IRFFT but keep scratch clean).
            for b in freqBins ..< (fftPow2 / 2 + 1) {
                realScratch[b] = 0
                imagScratch[b] = 0
            }
            var splitComplex = DSPSplitComplex(realp: &realScratch, imagp: &imagScratch)
            vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2N, FFTDirection(FFT_INVERSE))
            // Unpack split complex to interleaved.
            timeBuf.withUnsafeMutableBufferPointer { buf in
                buf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftPow2 / 2) {
                    cPtr in
                    vDSP_ztoc(&splitComplex, 1, cPtr, 2, vDSP_Length(fftPow2 / 2))
                }
            }
            // Normalise by fftPow2 (vDSP IRFFT scaling).
            let normFactor = 1.0 / Float(fftPow2)
            let frameOffset = frame * hopSize
            for i in 0 ..< fftSize {
                outputBuf[frameOffset + i] += timeBuf[i] * normFactor * window[i]
            }
        }

        // Remove algorithmic delay and clip to origLen.
        let delay = fftSize - hopSize
        let end = min(delay + origLen, outputBuf.count)
        guard end > delay else { return [] }
        var out = Array(outputBuf[delay ..< end])
        // Clip to [-1, 1].
        for i in out.indices {
            out[i] = min(max(out[i], -1.0), 1.0)
        }
        return out
    }

    // MARK: - Internal helpers

    /// Next power of 2 >= `n`.
    static func nextPow2(_ n: Int) -> Int {
        guard n > 1 else { return 1 }
        var p = 1
        while p < n { p <<= 1 }
        return p
    }
}

// MARK: - ERB utilities (shared with DeepFilterNetModel)

/// Frequency-to-ERB conversion (libDF convention).
/// `ERB(f) = 9.265 · log1p(f / (24.7 · 9.265))`
func dfFreqToErb(_ freqHz: Float) -> Float {
    9.265 * log1p(freqHz / (24.7 * 9.265))
}

/// ERB-to-frequency inverse (libDF convention).
func dfErbToFreq(_ erb: Float) -> Float {
    24.7 * 9.265 * (expf(erb / 9.265) - 1.0)
}

/// Compute the ERB band widths matching libDF's `compute_band_widths`.
/// Returns an array of `nbBands` integers summing to `fftSize/2 + 1`.
func dfErbBandWidths(
    sampleRate: Int,
    fftSize: Int,
    nbBands: Int,
    minNbFreqs: Int
) -> [Int] {
    guard sampleRate > 0, fftSize > 0, nbBands > 0 else { return [] }
    let nyq = sampleRate / 2
    let freqWidth = Float(sampleRate) / Float(fftSize)
    let erbLow = dfFreqToErb(0)
    let erbHigh = dfFreqToErb(Float(nyq))
    let step = (erbHigh - erbLow) / Float(nbBands)

    var widths = [Int](repeating: 0, count: nbBands)
    var prevFreq = 0
    var freqOver = 0
    let minBins = max(1, minNbFreqs)

    for i in 1 ... nbBands {
        let f = dfErbToFreq(erbLow + Float(i) * step)
        let fb = Int((f / freqWidth).rounded())
        var nbFreqs = fb - prevFreq - freqOver
        if nbFreqs < minBins {
            freqOver = minBins - nbFreqs
            nbFreqs = minBins
        } else {
            freqOver = 0
        }
        widths[i - 1] = max(1, nbFreqs)
        prevFreq = fb
    }

    // The last band gets +1 (FFT includes bin at fftSize/2).
    widths[nbBands - 1] += 1
    let target = fftSize / 2 + 1
    let total = widths.reduce(0, +)
    if total > target {
        widths[nbBands - 1] -= (total - target)
    } else if total < target {
        widths[nbBands - 1] += (target - total)
    }
    return widths
}

/// Linearly spaced `[Float]` from `start` to `end` (inclusive).
func dfLinspace(start: Float, end: Float, count: Int) -> [Float] {
    guard count > 1 else { return [start] }
    let step = (end - start) / Float(count - 1)
    return (0 ..< count).map { start + Float($0) * step }
}
