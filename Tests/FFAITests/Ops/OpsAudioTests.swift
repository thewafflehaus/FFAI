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
// GPU correctness tests for the audio Ops wrappers — melSpectrogram,
// audioConv1d, vocoderISTFT. Each asserts the wrapper output against a
// small CPU reference computed the same way the kernel describes its
// math in `ffai/{mel_spectrogram,audio_conv1d,vocoder}.rs`.

import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("Ops — audio")
struct OpsAudioTests {

    // ─── melSpectrogram ──────────────────────────────────────────────

    @Test("melSpectrogram f32 — matches a CPU direct-DFT reference")
    func melSpectrogramMatchesReference() {
        autoreleasepool {
            // Small synthetic front-end: nFFT=8, hop=4, nMels=3.
            let nFFT = 8, hop = 4, nMels = 3
            let nFreq = nFFT / 2 + 1   // 5
            let nFrames = 3
            let nSamples = (nFrames - 1) * hop + nFFT  // 16

            // A deterministic input waveform.
            var audio = [Float](repeating: 0, count: nSamples)
            for i in 0..<nSamples { audio[i] = sin(Float(i) * 0.7) }
            // A simple analysis window + filterbank.
            let window = AudioPreprocessing.hannWindow(nFFT)
            var melW = [Float](repeating: 0, count: nMels * nFreq)
            for m in 0..<nMels {
                for k in 0..<nFreq {
                    melW[m * nFreq + k] = Float((m + 1) * (k + 1)) * 0.01
                }
            }
            let logEps: Float = 1e-10

            // CPU reference — exactly the kernel's described math.
            var expected = [Float](repeating: 0, count: nFrames * nMels)
            for f in 0..<nFrames {
                let frameStart = f * hop
                for m in 0..<nMels {
                    var melAcc: Float = 0
                    for k in 0..<nFreq {
                        let angleStep = -2.0 * Float.pi / Float(nFFT) * Float(k)
                        var re: Float = 0, im: Float = 0
                        for t in 0..<nFFT {
                            let xw = audio[frameStart + t] * window[t]
                            let angle = angleStep * Float(t)
                            re += xw * cos(angle)
                            im += xw * sin(angle)
                        }
                        let power = re * re + im * im
                        melAcc += melW[m * nFreq + k] * power
                    }
                    expected[f * nMels + m] = log(melAcc + logEps)
                }
            }

            let audioT = Tensor.empty(shape: [nSamples], dtype: .f32)
            audioT.copyIn(from: audio)
            let winT = Tensor.empty(shape: [nFFT], dtype: .f32)
            winT.copyIn(from: window)
            let melT = Tensor.empty(shape: [nMels, nFreq], dtype: .f32)
            melT.copyIn(from: melW)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.melSpectrogram(
                    audio: audioT, window: winT, melWeight: melT,
                    nFFT: nFFT, nMels: nMels, hopLength: hop,
                    nFrames: nFrames, logEps: logEps, on: cb)
            }
            #expect(out.shape == [nFrames, nMels])
            let got = out.toArray(as: Float.self)
            for i in 0..<got.count {
                #expect(abs(got[i] - expected[i]) < 1e-2,
                        "mel[\(i)] got \(got[i]) expected \(expected[i])")
            }
        }
    }

    @Test("melSpectrogram — full preprocessing pipeline yields finite output")
    func melSpectrogramPipelineFinite() {
        autoreleasepool {
            // A 0.5 s 16 kHz sine — exercises the real Whisper config.
            let cfg = AudioFrontEndConfig.whisper
            let n = cfg.sampleRate / 2
            var wave = [Float](repeating: 0, count: n)
            for i in 0..<n {
                wave[i] = 0.3 * sin(2.0 * Float.pi * 220.0 * Float(i)
                                    / Float(cfg.sampleRate))
            }
            var out: Tensor!
            // `whisperNormalize: false` keeps the kernel only *queued*
            // on `cb` so `runAndWait` owns the commit (the normalised
            // path commits internally — that would double-commit here).
            runAndWait { cb in
                out = AudioPreprocessing.logMelSpectrogram(
                    waveform: wave, cfg: cfg, whisperNormalize: false,
                    on: cb)
            }
            #expect(out.shape[1] == cfg.nMels)
            let got = out.toFloatArray()
            #expect(got.allSatisfy { $0.isFinite })
            // A non-silent sine should produce non-constant log-Mel.
            #expect(Set(got.map { ($0 * 100).rounded() }).count > 1)
        }
    }

    // ─── audioConv1d ─────────────────────────────────────────────────

    @Test("audioConv1d f32 — matches a CPU reference, stride 1, pad 1")
    func audioConv1dStride1() {
        autoreleasepool {
            let batch = 1, inCh = 2, inLen = 5, outCh = 3, k = 3
            let stride = 1, pad = 1
            let outLen = (inLen + 2 * pad - k) / stride + 1  // 5

            var input = [Float](repeating: 0, count: batch * inCh * inLen)
            for i in 0..<input.count { input[i] = Float(i) * 0.5 - 1.0 }
            var weight = [Float](repeating: 0, count: outCh * inCh * k)
            for i in 0..<weight.count { weight[i] = Float(i % 5) * 0.3 - 0.5 }
            var bias = [Float](repeating: 0, count: outCh)
            for i in 0..<outCh { bias[i] = Float(i) * 0.1 }

            // CPU reference — the kernel's NCL conv math.
            var expected = [Float](repeating: 0, count: batch * outCh * outLen)
            for n in 0..<batch {
                for oc in 0..<outCh {
                    for op in 0..<outLen {
                        var acc = bias[oc]
                        let p0 = op * stride
                        for ic in 0..<inCh {
                            for kx in 0..<k {
                                let p = p0 + kx
                                if p >= pad && p < pad + inLen {
                                    let ix = p - pad
                                    let x = input[n * inCh * inLen + ic * inLen + ix]
                                    let w = weight[oc * inCh * k + ic * k + kx]
                                    acc += x * w
                                }
                            }
                        }
                        expected[n * outCh * outLen + oc * outLen + op] = acc
                    }
                }
            }

            let inT = Tensor.empty(shape: [batch, inCh, inLen], dtype: .f32)
            inT.copyIn(from: input)
            let wT = Tensor.empty(shape: [outCh, inCh, k], dtype: .f32)
            wT.copyIn(from: weight)
            let bT = Tensor.empty(shape: [outCh], dtype: .f32)
            bT.copyIn(from: bias)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.audioConv1d(
                    input: inT, weight: wT, bias: bT,
                    batch: batch, inCh: inCh, inLen: inLen, outCh: outCh,
                    k: k, stride: stride, pad: pad, on: cb)
            }
            #expect(out.shape == [batch, outCh, outLen])
            let got = out.toArray(as: Float.self)
            for i in 0..<got.count {
                #expect(abs(got[i] - expected[i]) < 1e-3,
                        "conv[\(i)] got \(got[i]) expected \(expected[i])")
            }
        }
    }

    @Test("audioConv1d f32 — stride-2 downsample halves the time axis")
    func audioConv1dStride2() {
        autoreleasepool {
            // Whisper's second stem conv: k=3, stride=2, pad=1.
            let batch = 1, inCh = 1, inLen = 8, outCh = 1, k = 3
            let stride = 2, pad = 1
            let outLen = (inLen + 2 * pad - k) / stride + 1  // 4

            let input = (0..<inLen).map { Float($0) }
            let weight = [Float](repeating: 1, count: k)  // sum-of-window
            let bias: [Float] = [0]

            let inT = Tensor.empty(shape: [batch, inCh, inLen], dtype: .f32)
            inT.copyIn(from: input)
            let wT = Tensor.empty(shape: [outCh, inCh, k], dtype: .f32)
            wT.copyIn(from: weight)
            let bT = Tensor.empty(shape: [outCh], dtype: .f32)
            bT.copyIn(from: bias)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.audioConv1d(
                    input: inT, weight: wT, bias: bT,
                    batch: batch, inCh: inCh, inLen: inLen, outCh: outCh,
                    k: k, stride: stride, pad: pad, on: cb)
            }
            #expect(out.shape == [1, 1, outLen])
            // Sum-of-3-window with pad 1: op=0 covers [-1,0,1]→0+0+1,
            // op=1 covers [1,2,3], op=2 [3,4,5], op=3 [5,6,7].
            let got = out.toArray(as: Float.self)
            #expect(got == [1, 6, 12, 18])
        }
    }

    // ─── vocoderISTFT ────────────────────────────────────────────────

    @Test("vocoderISTFT f32 — round-trips a DC spectrum to a constant")
    func vocoderISTFTDCRoundTrip() {
        autoreleasepool {
            // A pure-DC spectrum (only bin 0 non-zero) inverse-transforms
            // to a constant signal. With a rectangular window the COLA
            // normalisation divides it straight back out, so the
            // reconstruction equals the DC level.
            let nFFT = 8, hop = 4, nFrames = 3
            let nFreq = nFFT / 2 + 1
            let outLen = (nFrames - 1) * hop + nFFT

            var specRe = [Float](repeating: 0, count: nFrames * nFreq)
            let specIm = [Float](repeating: 0, count: nFrames * nFreq)
            // DC bin = nFFT so the (1/nFFT) inverse-DFT scale yields 1.0.
            for f in 0..<nFrames { specRe[f * nFreq + 0] = Float(nFFT) }
            let window = [Float](repeating: 1, count: nFFT)  // rectangular

            let reT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
            reT.copyIn(from: specRe)
            let imT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
            imT.copyIn(from: specIm)
            let winT = Tensor.empty(shape: [nFFT], dtype: .f32)
            winT.copyIn(from: window)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.vocoderISTFT(
                    specRe: reT, specIm: imT, window: winT,
                    nFrames: nFrames, nFFT: nFFT, hopLength: hop, on: cb)
            }
            #expect(out.shape == [outLen])
            let got = out.toArray(as: Float.self)
            // Every covered sample should reconstruct to ~1.0.
            for v in got {
                #expect(v.isFinite)
                #expect(abs(v - 1.0) < 1e-3, "got \(v) expected 1.0")
            }
        }
    }

    @Test("vocoderISTFT f32 — matches a CPU inverse-DFT overlap-add reference")
    func vocoderISTFTMatchesReference() {
        autoreleasepool {
            let nFFT = 8, hop = 4, nFrames = 3
            let nFreq = nFFT / 2 + 1
            let outLen = (nFrames - 1) * hop + nFFT
            let nyquist = nFFT / 2

            var specRe = [Float](repeating: 0, count: nFrames * nFreq)
            var specIm = [Float](repeating: 0, count: nFrames * nFreq)
            for i in 0..<specRe.count {
                specRe[i] = Float(i % 7) * 0.2 - 0.5
                specIm[i] = Float(i % 5) * 0.15 - 0.3
            }
            let window = AudioPreprocessing.hannWindow(nFFT)

            // CPU reference — the kernel's per-output-sample gather.
            var expected = [Float](repeating: 0, count: outLen)
            for t in 0..<outLen {
                var num: Float = 0, den: Float = 0
                let fHi = min(t / hop, nFrames - 1)
                let fLo = t + 1 > nFFT
                    ? (t + 1 - nFFT + hop - 1) / hop : 0
                if fLo <= fHi {
                    for f in fLo...fHi {
                        let tau = t - f * hop
                        let angleStep = 2.0 * Float.pi / Float(nFFT) * Float(tau)
                        var sample: Float = 0
                        for k in 0..<nFreq {
                            let re = specRe[f * nFreq + k]
                            let im = specIm[f * nFreq + k]
                            let angle = angleStep * Float(k)
                            let contrib = re * cos(angle) - im * sin(angle)
                            let w: Float = (k == 0 || k == nyquist) ? 1 : 2
                            sample += w * contrib
                        }
                        sample /= Float(nFFT)
                        let win = window[tau]
                        num += sample * win
                        den += win * win
                    }
                }
                expected[t] = den > 1e-8 ? num / den : 0
            }

            let reT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
            reT.copyIn(from: specRe)
            let imT = Tensor.empty(shape: [nFrames, nFreq], dtype: .f32)
            imT.copyIn(from: specIm)
            let winT = Tensor.empty(shape: [nFFT], dtype: .f32)
            winT.copyIn(from: window)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.vocoderISTFT(
                    specRe: reT, specIm: imT, window: winT,
                    nFrames: nFrames, nFFT: nFFT, hopLength: hop, on: cb)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == outLen)
            for i in 0..<outLen {
                #expect(abs(got[i] - expected[i]) < 1e-2,
                        "istft[\(i)] got \(got[i]) expected \(expected[i])")
            }
        }
    }

    // ─── Preprocessing helpers ───────────────────────────────────────

    @Test("reflectPad mirrors the signal across the boundary")
    func reflectPadMirrors() {
        let x: [Float] = [1, 2, 3, 4]
        let padded = AudioPreprocessing.reflectPad(x, pad: 2)
        // numpy reflect: [3,2, 1,2,3,4, 3,2]
        #expect(padded == [3, 2, 1, 2, 3, 4, 3, 2])
    }

    @Test("resample changes length by the rate ratio")
    func resampleLength() {
        let x = [Float](repeating: 0.5, count: 100)
        let up = AudioPreprocessing.resample(x, from: 8000, to: 16000)
        #expect(up.count == 200)
        let down = AudioPreprocessing.resample(x, from: 16000, to: 8000)
        #expect(down.count == 50)
        // Constant signal stays constant through linear interpolation.
        #expect(up.allSatisfy { abs($0 - 0.5) < 1e-5 })
    }

    @Test("melFilterbank rows are non-negative and overlap")
    func melFilterbankShape() {
        let cfg = AudioFrontEndConfig.whisper
        let bank = AudioPreprocessing.melFilterbank(cfg)
        #expect(bank.count == cfg.nMels * cfg.nFreq)
        #expect(bank.allSatisfy { $0 >= 0 })
        // Each Mel filter should have at least one non-zero tap.
        for m in 0..<cfg.nMels {
            let row = bank[(m * cfg.nFreq)..<((m + 1) * cfg.nFreq)]
            #expect(row.contains { $0 > 0 }, "Mel filter \(m) is all-zero")
        }
    }
}
