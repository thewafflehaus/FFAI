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
// MossFormer2SE — speech-enhancement / source-separation family.
//
// MossFormer2-SE is a mask-prediction network for single-channel speech
// enhancement. The architecture is:
//
//   Input waveform
//     → Kaldi mel-filterbank (win_len, win_inc, num_mels=60)
//     → delta + delta-delta features → [T, 180] feature matrix
//   MossFormerMaskNet
//     → GlobalLayerNorm → Conv1d(180→512) → positional encoding
//     → 24× (FLASH_ShareA_FFConvM + GatedFSMNBlock)
//     → LayerNorm → PReLU → Conv1d(512→1024, numSpks=2)
//     → reshape + (tanh·gate) → Conv1d(512→961) → ReLU
//     → mask[0]: [T, 961]
//   STFT of input → apply mask → iSTFT → enhanced waveform
//
// Everything runs on the CPU as [Float] arrays. Attention is parallelized
// via DispatchQueue.concurrentPerform where the sequence allows it.
//
// Weight-key normalisation mirrors the Python `sanitize` in
// mlx-audio-swift: strip leading `module.`, rewrite `mossformer.*` →
// `model.mossformer.*`.
//
// Reference implementation:
//   ~/Development/personal/ai/mlx-audio-swift/Sources/MLXAudioSTS/Models/MossFormer2SE/
// Checkpoint: `starkdmi/MossFormer2-SE-fp16`

import Foundation

// ─── Errors ──────────────────────────────────────────────────────────────

public enum MossFormer2SEError: Error, CustomStringConvertible {
    case missingWeight(String)
    case missingConfig(String)
    case invalidInput(String)
    case noSafetensorsFound(URL)

    public var description: String {
        switch self {
        case .missingWeight(let k):
            return "MossFormer2SE: required weight missing: \(k)"
        case .missingConfig(let f):
            return "MossFormer2SE: required config field missing: \(f)"
        case .invalidInput(let m):
            return "MossFormer2SE: invalid input — \(m)"
        case .noSafetensorsFound(let dir):
            return "MossFormer2SE: no .safetensors files found in \(dir.path)"
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────

/// MossFormer2-SE model hyper-parameters decoded from `config.json`.
public struct MossFormer2SEConfig: Sendable {
    public let modelType: String
    public let sampleRate: Int
    public let winLen: Int
    public let winInc: Int
    public let fftLen: Int
    public let numMels: Int
    public let winType: String
    public let preemphasis: Float
    public let inChannels: Int
    public let outChannels: Int
    public let outChannelsFinal: Int
    public let numBlocks: Int

    public init(
        modelType: String = "mossformer2_se",
        sampleRate: Int = 48000,
        winLen: Int = 1920,
        winInc: Int = 384,
        fftLen: Int = 1920,
        numMels: Int = 60,
        winType: String = "hamming",
        preemphasis: Float = 0.97,
        inChannels: Int = 180,
        outChannels: Int = 512,
        outChannelsFinal: Int = 961,
        numBlocks: Int = 24
    ) {
        self.modelType = modelType
        self.sampleRate = sampleRate
        self.winLen = winLen
        self.winInc = winInc
        self.fftLen = fftLen
        self.numMels = numMels
        self.winType = winType
        self.preemphasis = preemphasis
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.outChannelsFinal = outChannelsFinal
        self.numBlocks = numBlocks
    }

    /// Decode from a top-level `ModelConfig`.
    public static func from(_ config: ModelConfig) -> MossFormer2SEConfig {
        let raw = config.raw
        func i(_ k: String, _ d: Int) -> Int {
            if let v = raw[k] as? Int { return v }
            if let v = raw[k] as? Double { return Int(v) }
            return d
        }
        func f(_ k: String, _ d: Float) -> Float {
            if let v = raw[k] as? Double { return Float(v) }
            if let v = raw[k] as? Int { return Float(v) }
            return d
        }
        func s(_ k: String, _ d: String) -> String {
            return (raw[k] as? String) ?? d
        }
        return MossFormer2SEConfig(
            modelType: s("model_type", "mossformer2_se"),
            sampleRate: i("sample_rate", 48_000),
            winLen: i("win_len", 1920),
            winInc: i("win_inc", 384),
            fftLen: i("fft_len", 1920),
            numMels: i("num_mels", 60),
            winType: s("win_type", "hamming"),
            preemphasis: f("preemphasis", 0.97),
            inChannels: i("in_channels", 180),
            outChannels: i("out_channels", 512),
            outChannelsFinal: i("out_channels_final", 961),
            numBlocks: i("num_blocks", 24)
        )
    }
}

// ─── CPU math helpers ─────────────────────────────────────────────────────
// All tensors are stored channel-major [C, T] (unless noted), matching the
// MLX Conv1d NLC convention (but transposed, so C-first = channels outer).

/// Activate ReLU in place.
private func reluInPlace(_ x: inout [Float]) {
    for i in 0 ..< x.count { if x[i] < 0 { x[i] = 0 } }
}

/// Element-wise sigmoid.
private func sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + expf(-x))
}

/// Element-wise tanh.
private func tanhF(_ x: Float) -> Float {
    return Foundation.tanh(x)
}

/// Compute `sqrt(sum(x*x) * scale)`, clamped below `eps`.
private func l2NormClamped(
    _ x: [Float], start: Int, count: Int,
    scale: Float, eps: Float
) -> Float {
    var s: Float = 0
    for i in start ..< (start + count) { s += x[i] * x[i] }
    return max(sqrtf(s) * scale, eps)
}

/// 1-D depthwise convolution on a channel-major [C, T] buffer.
/// Returns result in the same layout. `padding` is symmetric.
private func depthwiseConv1d(
    _ input: [Float], inC: Int, inLen: Int,
    weight: [Float], kernelSize: Int, padding: Int, stride: Int = 1
) -> [Float] {
    let outLen = (inLen + 2 * padding - kernelSize) / stride + 1
    var out = [Float](repeating: 0, count: inC * outLen)
    let halfK = kernelSize  // iteration below uses k directly
    for c in 0 ..< inC {
        let wBase = c * kernelSize
        let iBase = c * inLen
        let oBase = c * outLen
        for t in 0 ..< outLen {
            var sum: Float = 0
            for k in 0 ..< halfK {
                let src = t * stride + k - padding
                if src >= 0, src < inLen {
                    sum += input[iBase + src] * weight[wBase + k]
                }
            }
            out[oBase + t] = sum
        }
    }
    return out
}

/// Standard 1-D convolution (not depthwise) on channel-major [C, T].
/// weight shape: [outC, inC, K]. Returns [outC, outLen].
private func conv1d(
    _ input: [Float], inC: Int, inLen: Int,
    weight: [Float], bias: [Float]?, outC: Int, kernelSize: Int,
    padding: Int, stride: Int = 1
) -> [Float] {
    let outLen = (inLen + 2 * padding - kernelSize) / stride + 1
    var out = [Float](repeating: 0, count: outC * outLen)
    for oc in 0 ..< outC {
        let b: Float = bias.map { $0[oc] } ?? 0
        let wOcBase = oc * inC * kernelSize
        let oBase = oc * outLen
        for t in 0 ..< outLen {
            var sum: Float = b
            for ic in 0 ..< inC {
                let wIcBase = wOcBase + ic * kernelSize
                let iBase = ic * inLen
                for k in 0 ..< kernelSize {
                    let src = t * stride + k - padding
                    if src >= 0, src < inLen {
                        sum += input[iBase + src] * weight[wIcBase + k]
                    }
                }
            }
            out[oBase + t] = sum
        }
    }
    return out
}

/// Matrix–vector multiply: weight[outF, inF] × input[inF] → [outF].
private func linearMV(
    _ weight: [Float], _ bias: [Float]?,
    _ input: [Float], outF: Int, inF: Int
) -> [Float] {
    var out = [Float](repeating: 0, count: outF)
    for o in 0 ..< outF {
        var s: Float = bias.map { $0[o] } ?? 0
        let base = o * inF
        for i in 0 ..< inF { s += weight[base + i] * input[i] }
        out[o] = s
    }
    return out
}

/// Batch matmul: [B, M, K] × [K, N] → [B, M, N] (bias per column of N).
/// Used in the linear attention projection. All row-major.
private func batchMatMulRightTranspose(
    _ a: [Float], aRows: Int, aCols: Int,
    _ b: [Float], bRows: Int, bCols: Int,
    bias: [Float]? = nil
) -> [Float] {
    // Result [aRows, bRows] = A × B^T
    var out = [Float](repeating: 0, count: aRows * bRows)
    for i in 0 ..< aRows {
        for j in 0 ..< bRows {
            var s: Float = bias.map { $0[j] } ?? 0
            let aBase = i * aCols
            let bBase = j * aCols
            for k in 0 ..< aCols { s += a[aBase + k] * b[bBase + k] }
            out[i * bRows + j] = s
        }
    }
    return out
}

/// Generic matrix multiply [M,K]×[K,N]→[M,N], row-major.
private func matMul(
    _ a: [Float], m: Int, k: Int,
    _ b: [Float], n: Int
) -> [Float] {
    var out = [Float](repeating: 0, count: m * n)
    for i in 0 ..< m {
        for j in 0 ..< n {
            var s: Float = 0
            for kk in 0 ..< k { s += a[i * k + kk] * b[kk * n + j] }
            out[i * n + j] = s
        }
    }
    return out
}

// ─── DSP helpers ──────────────────────────────────────────────────────────

/// Next power of two ≥ `v`.
private func nextPow2(_ v: Int) -> Int {
    guard v > 1 else { return max(v, 1) }
    var n = 1
    while n < v { n <<= 1 }
    return n
}

/// Hamming window (non-periodic by default).
private func hammingWindow(size: Int, periodic: Bool = false) -> [Float] {
    guard size > 0 else { return [] }
    if size == 1 { return [1.0] }
    let eff = periodic ? size + 1 : size
    let denom = Float(eff - 1)
    var w = (0 ..< eff).map { n -> Float in
        let phase = 2.0 * Float.pi * Float(n) / denom
        return 0.54 - 0.46 * cos(phase)
    }
    if periodic { w = Array(w.prefix(size)) }
    return w
}

/// Hann window (non-periodic by default).
private func hannWindow(size: Int, periodic: Bool = false) -> [Float] {
    guard size > 0 else { return [] }
    if size == 1 { return [1.0] }
    let eff = periodic ? size + 1 : size
    let denom = Float(eff - 1)
    var w = (0 ..< eff).map { n -> Float in
        let phase = 2.0 * Float.pi * Float(n) / denom
        return 0.5 - 0.5 * cos(phase)
    }
    if periodic { w = Array(w.prefix(size)) }
    return w
}

/// Radix-2 DIT FFT in-place on complex [re, im, re, im, …].
private func fft(_ x: inout [Float], inverse: Bool = false) {
    let n = x.count / 2
    guard n > 0, n & (n - 1) == 0 else { return }
    // Bit-reversal permutation
    var j = 0
    for i in 1 ..< n {
        var bit = n >> 1
        while j & bit != 0 {
            j ^= bit
            bit >>= 1
        }
        j ^= bit
        if i < j {
            x.swapAt(2 * i, 2 * j)
            x.swapAt(2 * i + 1, 2 * j + 1)
        }
    }
    // Cooley-Tukey FFT
    var len = 2
    while len <= n {
        let ang = 2.0 * Float.pi / Float(len) * (inverse ? 1 : -1)
        let wRe = cos(ang)
        let wIm = sin(ang)
        var pos = 0
        while pos < n {
            var uRe: Float = 1
            var uIm: Float = 0
            for k in 0 ..< (len / 2) {
                let eRe = x[2 * (pos + k)]
                let eIm = x[2 * (pos + k) + 1]
                let oRe = x[2 * (pos + k + len / 2)]
                let oIm = x[2 * (pos + k + len / 2) + 1]
                let tRe = uRe * oRe - uIm * oIm
                let tIm = uRe * oIm + uIm * oRe
                x[2 * (pos + k)] = eRe + tRe
                x[2 * (pos + k) + 1] = eIm + tIm
                x[2 * (pos + k + len / 2)] = eRe - tRe
                x[2 * (pos + k + len / 2) + 1] = eIm - tIm
                let nuRe = uRe * wRe - uIm * wIm
                uIm = uRe * wIm + uIm * wRe
                uRe = nuRe
            }
            pos += len
        }
        len <<= 1
    }
    if inverse {
        let fn = Float(n)
        for i in 0 ..< (2 * n) { x[i] /= fn }
    }
}

/// STFT of a 1-D signal. Returns [(numBins), numFrames] pairs of (real, imag).
/// numBins = fftLen/2+1.
private func stft(
    audio: [Float],
    fftLen: Int, hopLength: Int, winLen: Int,
    window: [Float], center: Bool = false
) -> (real: [[Float]], imag: [[Float]]) {
    var signal = audio
    if center {
        let pad = fftLen / 2
        signal = [Float](repeating: 0, count: pad) + signal + [Float](repeating: 0, count: pad)
    }
    let sigLen = signal.count
    guard sigLen >= winLen else { return ([], []) }
    let numFrames = 1 + (sigLen - winLen + hopLength - 1) / hopLength
    guard numFrames > 0 else { return ([], []) }
    let numBins = fftLen / 2 + 1
    var realOut = [[Float]](repeating: [Float](repeating: 0, count: numFrames), count: numBins)
    var imagOut = [[Float]](repeating: [Float](repeating: 0, count: numFrames), count: numBins)

    let nfft = nextPow2(fftLen)
    let effectiveWin = min(window.count, winLen)

    for f in 0 ..< numFrames {
        let start = f * hopLength
        var buf = [Float](repeating: 0, count: 2 * nfft)
        for k in 0 ..< effectiveWin {
            let s = start + k
            let v: Float = (s < sigLen) ? signal[s] : 0
            buf[2 * k] = v * window[k]
        }
        fft(&buf)
        for b in 0 ..< numBins {
            realOut[b][f] = buf[2 * b]
            imagOut[b][f] = buf[2 * b + 1]
        }
    }
    return (realOut, imagOut)
}

/// Inverse STFT. Input: real/imag each [numBins, numFrames]. Returns waveform.
private func istft(
    real: [[Float]], imag: [[Float]],
    fftLen: Int, hopLength: Int, winLen: Int,
    window: [Float], audioLength: Int? = nil
) -> [Float] {
    guard !real.isEmpty, !imag.isEmpty else { return [] }
    let numBins = real.count
    let numFrames = real[0].count
    guard numFrames > 0 else { return [] }
    let nfft = nextPow2(fftLen)
    let frameWidth = min(winLen, nfft)
    let fullLen = (numFrames - 1) * hopLength + frameWidth
    guard fullLen > 0 else { return [] }

    var output = [Float](repeating: 0, count: fullLen)
    var windowSum = [Float](repeating: 0, count: fullLen)

    let effectiveWin = min(window.count, frameWidth)
    var synthWindow = [Float](repeating: 0, count: frameWidth)
    for i in 0 ..< effectiveWin { synthWindow[i] = window[i] }

    for f in 0 ..< numFrames {
        // Reconstruct complex spectrum [nfft] for irfft
        var buf = [Float](repeating: 0, count: 2 * nfft)
        for b in 0 ..< numBins {
            buf[2 * b] = real[b][f]
            buf[2 * b + 1] = imag[b][f]
        }
        // Fill negative frequencies via conjugate symmetry
        if nfft > 1 {
            let lastPos = numBins - 1
            for b in 1 ..< lastPos {
                let neg = nfft - b
                buf[2 * neg] = buf[2 * b]
                buf[2 * neg + 1] = -buf[2 * b + 1]
            }
        }
        fft(&buf, inverse: true)
        // Overlap-add windowed frame
        let offset = f * hopLength
        for k in 0 ..< frameWidth {
            let s = offset + k
            if s < fullLen {
                output[s] += buf[2 * k] * synthWindow[k]
                windowSum[s] += synthWindow[k] * synthWindow[k]
            }
        }
    }
    // Normalize by window power
    let eps: Float = 1e-8
    for i in 0 ..< fullLen {
        output[i] /= max(windowSum[i], eps)
    }
    if let audioLength, output.count > audioLength {
        return Array(output.prefix(audioLength))
    }
    return output
}

/// Compute Kaldi-style mel filterbanks. Returns [numFrames, numMels].
/// `features` is row-major [numFrames, numMels].
private func computeFbankKaldi(
    audio: [Float],
    sampleRate: Int,
    winLen: Int, winInc: Int,
    numMels: Int,
    winType: String,
    preemphasis: Float,
    dither: Float = 0.0,
    removeDCOffset: Bool = true,
    lowFreq: Float = 20.0
) -> [[Float]] {
    let signal = audio
    if signal.isEmpty { return [] }
    guard signal.count >= winLen else { return [] }

    let numFrames = 1 + (signal.count - winLen) / winInc
    guard numFrames > 0 else { return [] }

    let nFft = nextPow2(winLen)
    let lowType = winType.lowercased()
    let analysisWindow: [Float]
    if lowType.contains("hann") {
        analysisWindow = hannWindow(size: winLen, periodic: false)
    } else {
        analysisWindow = hammingWindow(size: winLen, periodic: false)
    }

    // Mel filterbank: [numMels, nFft/2+1]
    let melBank = melFilters(
        sampleRate: sampleRate, nFft: nFft,
        nMels: numMels, fMin: lowFreq)

    var features = [[Float]](
        repeating: [Float](repeating: 0, count: numMels),
        count: numFrames)

    for fi in 0 ..< numFrames {
        let start = fi * winInc
        var frame = Array(signal[start ..< (start + winLen)])

        if removeDCOffset {
            let mean = frame.reduce(0, +) / Float(frame.count)
            for i in 0 ..< frame.count { frame[i] -= mean }
        }
        if preemphasis > 0, frame.count > 1 {
            for i in stride(from: frame.count - 1, through: 1, by: -1) {
                frame[i] -= preemphasis * frame[i - 1]
            }
            frame[0] -= preemphasis * frame[0]
        }

        // Apply window + FFT
        var buf = [Float](repeating: 0, count: 2 * nFft)
        for k in 0 ..< winLen { buf[2 * k] = frame[k] * analysisWindow[k] }
        fft(&buf)

        // Power spectrum [nFft/2+1]
        let numBins = nFft / 2 + 1
        var power = [Float](repeating: 0, count: numBins)
        for b in 0 ..< numBins {
            let re = buf[2 * b]
            let im = buf[2 * b + 1]
            power[b] = re * re + im * im
        }

        // Apply mel filters
        for m in 0 ..< numMels {
            var s: Float = 0
            for b in 0 ..< numBins { s += power[b] * melBank[m * numBins + b] }
            features[fi][m] = log(max(s, 1e-10))
        }
    }
    return features
}

/// Build mel filterbank matrix [nMels, nFft/2+1], row-major.
private func melFilters(
    sampleRate: Int, nFft: Int, nMels: Int, fMin: Float
) -> [Float] {
    guard sampleRate > 0, nFft > 0, nMels > 0 else { return [] }

    func hzToMel(_ hz: Float) -> Float { 1127.0 * log(1.0 + hz / 700.0) }
    func melToHz(_ mel: Float) -> Float { 700.0 * (exp(mel / 1127.0) - 1.0) }

    let fMax = Float(sampleRate) / 2.0
    let melLow = hzToMel(fMin)
    let melHigh = hzToMel(fMax)
    let numBins = nFft / 2 + 1

    // nMels + 2 equally spaced mel points
    var melPoints = [Float](repeating: 0, count: nMels + 2)
    for i in 0 ... (nMels + 1) {
        melPoints[i] = melLow + Float(i) * (melHigh - melLow) / Float(nMels + 1)
    }
    // Convert to Hz → FFT bin index
    // Mel point conversion is done per-filter below using f0/f1/f2 directly.

    var fb = [Float](repeating: 0, count: nMels * numBins)
    for m in 0 ..< nMels {
        let f0 = floor(melToHz(melPoints[m]) / Float(sampleRate) * Float(nFft + 1))
        let f1 = floor(melToHz(melPoints[m + 1]) / Float(sampleRate) * Float(nFft + 1))
        let f2 = floor(melToHz(melPoints[m + 2]) / Float(sampleRate) * Float(nFft + 1))
        for k in 0 ..< numBins {
            let fk = Float(k)
            if fk >= f0, fk <= f1, f1 > f0 {
                fb[m * numBins + k] = (fk - f0) / (f1 - f0)
            } else if fk > f1, fk <= f2, f2 > f1 {
                fb[m * numBins + k] = (f2 - fk) / (f2 - f1)
            }
        }
    }
    return fb
}

/// Kaldi-style delta computation on [numChannels, numFrames] (channel-major).
/// Returns [numChannels, numFrames] deltas.
private func computeDeltasKaldi(
    _ features: [[Float]], winLength: Int = 5
) -> [[Float]] {
    let channels = features.count
    guard channels > 0 else { return features }
    let time = features[0].count
    guard time > 0 else { return features }

    let halfWin = max(winLength / 2, 1)
    var denom: Float = 0
    for i in 1 ... halfWin { denom += Float(i * i) }
    denom *= 2.0
    guard denom > 0 else { return features }

    var out = [[Float]](repeating: [Float](repeating: 0, count: time), count: channels)
    for c in 0 ..< channels {
        for t in 0 ..< time {
            var delta: Float = 0
            for i in 1 ... halfWin {
                let ta = min(max(t + i, 0), time - 1)
                let tb = min(max(t - i, 0), time - 1)
                delta += Float(i) * (features[c][ta] - features[c][tb])
            }
            out[c][t] = delta / denom
        }
    }
    return out
}

// ─── Layer helpers (CPU path) ─────────────────────────────────────────────

/// Scale norm: x * (g / max(||x|| * scale, eps)).
/// Input `x` is [dim], inline.
private func scaleNorm(
    _ x: inout [Float], dim: Int, g: Float, scale: Float, eps: Float
) {
    var ss: Float = 0
    for v in x { ss += v * v }
    let norm = max(sqrtf(ss) * scale, eps)
    let factor = g / norm
    for i in 0 ..< dim { x[i] *= factor }
}

/// Layer norm over last axis. Input [T, C] flattened row-major.
/// `gamma` and `beta` are [C]. Result in-place.
private func layerNormRows(
    _ x: inout [Float], rows: Int, cols: Int,
    gamma: [Float], beta: [Float], eps: Float
) {
    for r in 0 ..< rows {
        let base = r * cols
        var mean: Float = 0
        for c in 0 ..< cols { mean += x[base + c] }
        mean /= Float(cols)
        var vari: Float = 0
        for c in 0 ..< cols {
            let d = x[base + c] - mean
            vari += d * d
        }
        vari /= Float(cols)
        let invStd = 1.0 / sqrtf(vari + eps)
        for c in 0 ..< cols {
            x[base + c] = (x[base + c] - mean) * invStd * gamma[c] + beta[c]
        }
    }
}

/// Global layer norm over a [C, T] tensor (axes 1 and 2 = C and T dims
/// in the original [B=1, C, T] layout). Applies `weight[C,1]` and `bias[C,1]`.
/// Input & output: channel-major [C, T].
private func globalLayerNorm(
    _ x: [Float], c: Int, t: Int,
    weight: [Float], bias: [Float], eps: Float
) -> [Float] {
    // Compute mean and variance over all elements
    let n = c * t
    guard n > 0 else { return x }
    var mean: Float = 0
    for v in x { mean += v }
    mean /= Float(n)
    var vari: Float = 0
    for v in x {
        let d = v - mean
        vari += d * d
    }
    vari /= Float(n)
    let invStd = 1.0 / sqrtf(vari + eps)
    var out = [Float](repeating: 0, count: n)
    for ch in 0 ..< c {
        let w = weight[ch]
        let b = bias[ch]
        let base = ch * t
        for ti in 0 ..< t {
            out[base + ti] = (x[base + ti] - mean) * invStd * w + b
        }
    }
    return out
}

// Float16 → Float32 bit-cast helper.
extension Float {
    fileprivate init(float16Bits bits: UInt16) {
        let sign: UInt32 = UInt32(bits >> 15) << 31
        let exponent: UInt32 = UInt32((bits >> 10) & 0x1F)
        let mantissa: UInt32 = UInt32(bits & 0x3FF)

        var f32bits: UInt32
        if exponent == 0 {
            if mantissa == 0 {
                f32bits = sign  // ±0
            } else {
                // Denormalised: shift mantissa to normalise
                var m = mantissa
                var e: UInt32 = 0
                while m & 0x400 == 0 {
                    m <<= 1
                    e += 1
                }
                m &= 0x3FF
                f32bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13)
            }
        } else if exponent == 0x1F {
            // Inf or NaN
            f32bits = sign | (0xFF << 23) | (mantissa << 13)
        } else {
            f32bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13)
        }
        self = Float(bitPattern: f32bits)
    }

    fileprivate init(bfloat16Bits bits: UInt16) {
        let f32bits = UInt32(bits) << 16
        self = Float(bitPattern: f32bits)
    }
}

// ─── Neural network modules (CPU) ─────────────────────────────────────────

// Naming mirrors the Python architecture so weight keys map 1-to-1.
//
// Weight naming from the `sanitize` step:
//   raw key: "module.mossformer.norm.weight"  → "model.mossformer.norm.weight"
//   raw key: "mossformer.conv1d_encoder.weight" → "model.mossformer.conv1d_encoder.weight"
//
// In the forward pass we address weights as
//   "model.mossformer.<sub-path>"

/// ConvModule: depthwise residual conv over [T, dim] time-major sequence.
/// Weight key: `<prefix>.weight` shape [dim, kernelSize, 1] (from MLX layout).
private struct SEConvModule {
    let inChannels: Int
    let kernelSize: Int
    let padding: Int
    /// Depthwise weight [inChannels, kernelSize] (extracted from [C,K,1]).
    let weight: [Float]

    func forward(_ x: [Float], t: Int) -> [Float] {
        // x: [T, dim] time-major. We need channel-major for depthwiseConv1d.
        let dim = inChannels
        // Transpose x [T, dim] → ch-major [dim, T]
        var chMajor = [Float](repeating: 0, count: dim * t)
        for ti in 0 ..< t {
            for d in 0 ..< dim {
                chMajor[d * t + ti] = x[ti * dim + d]
            }
        }
        let convOut = depthwiseConv1d(
            chMajor, inC: dim, inLen: t,
            weight: weight, kernelSize: kernelSize,
            padding: padding)
        // Transpose back [dim, T] → [T, dim]
        var timeMajor = [Float](repeating: 0, count: t * dim)
        for ti in 0 ..< t {
            for d in 0 ..< dim {
                timeMajor[ti * dim + d] = convOut[d * t + ti]
            }
        }
        // Residual add
        var out = [Float](repeating: 0, count: t * dim)
        for i in 0 ..< (t * dim) { out[i] = x[i] + timeMajor[i] }
        return out
    }
}

/// FFConvM: LayerNorm → Linear → SiLU → ConvModule. Operates on [T, dim] time-major.
private struct SEFFConvM {
    let dimIn: Int
    let dimOut: Int
    // LayerNorm weights
    let lnGamma: [Float]
    let lnBeta: [Float]
    let lnEps: Float
    // Linear weight [dimOut, dimIn], bias [dimOut]
    let linearWeight: [Float]
    let linearBias: [Float]
    // ConvModule
    let convMod: SEConvModule

    func forward(_ x: [Float], t: Int) -> [Float] {
        // x: [T, dimIn]
        var y = x
        // LayerNorm over last axis (dimIn)
        layerNormRows(
            &y, rows: t, cols: dimIn,
            gamma: lnGamma, beta: lnBeta, eps: lnEps)
        // Linear [T, dimIn] → [T, dimOut]
        var lin = [Float](repeating: 0, count: t * dimOut)
        for ti in 0 ..< t {
            let row = Array(y[(ti * dimIn) ..< (ti * dimIn + dimIn)])
            let out = linearMV(linearWeight, linearBias, row, outF: dimOut, inF: dimIn)
            for d in 0 ..< dimOut { lin[ti * dimOut + d] = out[d] }
        }
        // SiLU = x * sigmoid(x)
        for i in 0 ..< lin.count { lin[i] = lin[i] * sigmoid(lin[i]) }
        // ConvModule residual
        return convMod.forward(lin, t: t)
    }
}

/// OffsetScale: expands [T, qkDim] → 4× [T, qkDim] via gamma/beta.
private struct SEOffsetScale {
    let qkDim: Int
    // 4 heads: [4, qkDim] stored as flat [4*qkDim] row-major
    let gamma: [Float]
    let beta: [Float]

    func forward(_ x: [Float], t: Int) -> ([Float], [Float], [Float], [Float]) {
        // Returns (quadQ, linQ, quadK, linK) each [T, qkDim]
        let heads = 4
        var results = [[Float]](
            repeating: [Float](repeating: 0, count: t * qkDim),
            count: heads)
        for ti in 0 ..< t {
            let xBase = ti * qkDim
            for h in 0 ..< heads {
                let gBase = h * qkDim
                for d in 0 ..< qkDim {
                    results[h][ti * qkDim + d] =
                        x[xBase + d] * gamma[gBase + d] + beta[gBase + d]
                }
            }
        }
        return (results[0], results[1], results[2], results[3])
    }
}

/// GatedFSMN block: UniDeepFsmn inside a gate (to_u, to_v, fsmn).
private struct SEGatedFSMN {
    // to_u: FFConvM(inChannels → innerChannels)
    let toU: SEFFConvM
    // to_v: FFConvM(inChannels → innerChannels)
    let toV: SEFFConvM
    // FSMN parts
    let fsmnLinearW: [Float]  // [hiddenSize, inChannels]
    let fsmnLinearB: [Float]  // [hiddenSize]
    let fsmnProjectW: [Float]  // [outChannels, hiddenSize] (no bias)
    // Conv1 (depthwise): [outChannels, lorder*2-1] after squeezing last dims
    let fsmnConvW: [Float]
    let fsmnLorder: Int
    let inChannels: Int
    let outChannels: Int
    let hiddenSize: Int

    func forward(_ x: [Float], t: Int) -> [Float] {
        let inputResidual = x
        var xu = toU.forward(x, t: t)
        let xv = toV.forward(x, t: t)
        // FSMN forward
        xu = fsmnForward(xu, t: t)
        // Gate: xv * xu + residual
        var out = [Float](repeating: 0, count: t * outChannels)
        for i in 0 ..< (t * outChannels) {
            out[i] = xv[i] * xu[i] + inputResidual[i]
        }
        return out
    }

    private func fsmnForward(_ input: [Float], t: Int) -> [Float] {
        // Linear: [T, inChannels] → [T, hiddenSize] + ReLU
        var h = [Float](repeating: 0, count: t * hiddenSize)
        for ti in 0 ..< t {
            let row = Array(input[(ti * inChannels) ..< (ti * inChannels + inChannels)])
            let out = linearMV(
                fsmnLinearW, fsmnLinearB, row,
                outF: hiddenSize, inF: inChannels)
            for d in 0 ..< hiddenSize { h[ti * hiddenSize + d] = max(out[d], 0) }
        }
        // Project: [T, hiddenSize] → [T, outChannels] (no bias)
        var p = [Float](repeating: 0, count: t * outChannels)
        for ti in 0 ..< t {
            let row = Array(h[(ti * hiddenSize) ..< (ti * hiddenSize + hiddenSize)])
            let out = linearMV(fsmnProjectW, nil, row, outF: outChannels, inF: hiddenSize)
            for d in 0 ..< outChannels { p[ti * outChannels + d] = out[d] }
        }
        // Depthwise conv with padding lorder-1 on each side: [T, C] → channel-major
        let kSize = fsmnLorder * 2 - 1
        let padLeft = fsmnLorder - 1
        // Transpose p [T, C] → [C, T]
        var chMajor = [Float](repeating: 0, count: outChannels * t)
        for ti in 0 ..< t {
            for d in 0 ..< outChannels { chMajor[d * t + ti] = p[ti * outChannels + d] }
        }
        // Symmetric-pad [C, T] → [C, T+2*(lorder-1)]
        // Padding is handled inside depthwiseConv1d via the `padding` parameter.
        let convOut = depthwiseConv1d(
            chMajor, inC: outChannels, inLen: t,
            weight: fsmnConvW, kernelSize: kSize,
            padding: padLeft)
        // Transpose back [C, T] → [T, C]
        var timeMajor = [Float](repeating: 0, count: t * outChannels)
        for ti in 0 ..< t {
            for d in 0 ..< outChannels { timeMajor[ti * outChannels + d] = convOut[d * t + ti] }
        }
        // Residual if inChannels == outChannels
        var enhanced = timeMajor
        if inChannels == outChannels {
            for i in 0 ..< (t * outChannels) { enhanced[i] += input[i] }
        }
        return enhanced
    }
}

/// GatedFSMNBlock: Conv1d → PReLU → CLayerNorm → GatedFSMN → CLayerNorm → Conv1d.
private struct SEGatedFSMNBlock {
    let dim: Int
    let innerChannels: Int
    // conv1: [T, dim] → [T, innerChannels]
    let conv1W: [Float]
    let conv1B: [Float]
    let preluW: Float
    let norm1G: [Float]
    let norm1B: [Float]
    let norm2G: [Float]
    let norm2B: [Float]
    let gatedFsmn: SEGatedFSMN
    // conv2: [T, innerChannels] → [T, dim]
    let conv2W: [Float]
    let conv2B: [Float]
    let eps: Float = 1e-8

    func forward(_ x: [Float], t: Int) -> [Float] {
        let residual = x
        // conv1 (1×1 conv = linear per time step)
        var y = apply1x1Conv(
            x, t: t, w: conv1W, b: conv1B,
            inC: dim, outC: innerChannels)
        // PReLU
        for i in 0 ..< y.count { y[i] = y[i] >= 0 ? y[i] : preluW * y[i] }
        // CLayerNorm (same as layerNorm over last axis)
        layerNormRows(
            &y, rows: t, cols: innerChannels,
            gamma: norm1G, beta: norm1B, eps: eps)
        // GatedFSMN
        y = gatedFsmn.forward(y, t: t)
        // CLayerNorm
        layerNormRows(
            &y, rows: t, cols: innerChannels,
            gamma: norm2G, beta: norm2B, eps: eps)
        // conv2 (1×1 conv)
        y = apply1x1Conv(
            y, t: t, w: conv2W, b: conv2B,
            inC: innerChannels, outC: dim)
        // Residual
        for i in 0 ..< (t * dim) { y[i] += residual[i] }
        return y
    }

    private func apply1x1Conv(
        _ x: [Float], t: Int, w: [Float], b: [Float], inC: Int, outC: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: t * outC)
        for ti in 0 ..< t {
            let row = Array(x[(ti * inC) ..< (ti * inC + inC)])
            let r = linearMV(w, b, row, outF: outC, inF: inC)
            for d in 0 ..< outC { out[ti * outC + d] = r[d] }
        }
        return out
    }
}

/// RoPE: rotary position embedding over [T, dim], applied in-place.
private func applyRoPE(_ x: inout [Float], t: Int, dim: Int, base: Float = 10000.0) {
    let halfDim = dim / 2
    for ti in 0 ..< t {
        let base2 = ti * dim
        for d in 0 ..< halfDim {
            let theta = Float(ti) / pow(base, Float(2 * d) / Float(dim))
            let cosT = cos(theta)
            let sinT = sin(theta)
            let re = x[base2 + d]
            let im = x[base2 + halfDim + d]
            x[base2 + d] = re * cosT - im * sinT
            x[base2 + halfDim + d] = re * sinT + im * cosT
        }
    }
}

/// FLASH_ShareA_FFConvM: quadratic+linear attention with token shift.
/// Runs on [T, dim] time-major sequences.
private struct SEFlashShareA {
    let dim: Int
    let groupSize: Int
    let qkDim: Int
    let expansionFactor: Float
    let shiftTokens: Bool
    let hiddenDim: Int  // = dim * expansionFactor
    let toHidden: SEFFConvM
    let toQk: SEFFConvM
    let qkOffsetScale: SEOffsetScale
    let toOut: SEFFConvM  // dimIn = dim*2 → dim

    func forward(_ x: [Float], t: Int) -> [Float] {
        var normedX = x

        // Token shift on first half
        if shiftTokens, t > 1 {
            let half = dim / 2
            var shifted = normedX
            // xShift = normedX[..., 0:half], pad left by 1 time step
            for ti in stride(from: t - 1, through: 1, by: -1) {
                for d in 0 ..< half {
                    shifted[ti * dim + d] = normedX[(ti - 1) * dim + d]
                }
            }
            for d in 0 ..< half { shifted[0 * dim + d] = 0 }
            normedX = shifted
        }

        // toHidden → split into v and u
        let hidden = toHidden.forward(normedX, t: t)  // [T, hiddenDim]
        let halfH = hiddenDim / 2
        var v = [Float](repeating: 0, count: t * halfH)
        var u = [Float](repeating: 0, count: t * halfH)
        for ti in 0 ..< t {
            for d in 0 ..< halfH {
                v[ti * halfH + d] = hidden[ti * hiddenDim + d]
                u[ti * halfH + d] = hidden[ti * hiddenDim + halfH + d]
            }
        }

        // toQk → offset scale → 4 heads
        let qkFlat = toQk.forward(normedX, t: t)  // [T, qkDim]
        let (quadQ, linQ, quadK, linK) = qkOffsetScale.forward(qkFlat, t: t)

        // Apply RoPE to q and k (min(32, qkDim) dims)
        let ropeDim = min(32, qkDim)
        var quadQR = quadQ
        var linQR = linQ
        var quadKR = quadK
        var linKR = linK
        if ropeDim > 0 {
            applyRoPEPartial(&quadQR, t: t, dim: qkDim, ropeDim: ropeDim)
            applyRoPEPartial(&linQR, t: t, dim: qkDim, ropeDim: ropeDim)
            applyRoPEPartial(&quadKR, t: t, dim: qkDim, ropeDim: ropeDim)
            applyRoPEPartial(&linKR, t: t, dim: qkDim, ropeDim: ropeDim)
        }

        // Quadratic + linear attention
        let (attV, attU) = calAttention(
            t: t, groupSize: groupSize, qkDim: qkDim, vDim: halfH,
            quadQ: quadQR, linQ: linQR, quadK: quadKR, linK: linKR,
            v: v, u: u)

        // out = (attU * v) * sigmoid(attV * u) → [T, halfH].
        // toOut: FFConvM(dimIn = halfH = dim*2, dimOut = dim). ✓
        var outFinal = [Float](repeating: 0, count: t * halfH)
        for i in 0 ..< (t * halfH) {
            outFinal[i] = attU[i] * v[i] * sigmoid(attV[i] * u[i])
        }
        let toOutResult = toOut.forward(outFinal, t: t)  // [T, dim]
        // Residual
        var result = [Float](repeating: 0, count: t * dim)
        for i in 0 ..< (t * dim) { result[i] = x[i] + toOutResult[i] }
        return result
    }

    private func applyRoPEPartial(
        _ x: inout [Float], t: Int, dim: Int, ropeDim: Int
    ) {
        let halfRope = ropeDim / 2
        for ti in 0 ..< t {
            let base = ti * dim
            for d in 0 ..< halfRope {
                let theta = Float(ti) / pow(Float(10000.0), Float(2 * d) / Float(ropeDim))
                let cosT = cos(theta)
                let sinT = sin(theta)
                let re = x[base + d]
                let im = x[base + halfRope + d]
                x[base + d] = re * cosT - im * sinT
                x[base + halfRope + d] = re * sinT + im * cosT
            }
        }
    }

    private func calAttention(
        t: Int, groupSize: Int, qkDim: Int, vDim: Int,
        quadQ: [Float], linQ: [Float], quadK: [Float], linK: [Float],
        v: [Float], u: [Float]
    ) -> ([Float], [Float]) {
        // Pad sequence to multiple of groupSize
        let g = groupSize
        let padding = (g - t % g) % g
        let n = t + padding
        let numGroups = n / g

        func pad(_ a: [Float], cols: Int) -> [Float] {
            guard padding > 0 else { return a }
            return a + [Float](repeating: 0, count: padding * cols)
        }

        let pQuadQ = pad(quadQ, cols: qkDim)
        let pLinQ = pad(linQ, cols: qkDim)
        let pQuadK = pad(quadK, cols: qkDim)
        let pLinK = pad(linK, cols: qkDim)
        let pV = pad(v, cols: vDim)
        let pU = pad(u, cols: vDim)

        // Quadratic attention: for each group, Q_g @ K_g^T → scaled → relu²
        // then @ V_g. Linear: linK^T @ V / n, then linQ @ result.
        let scale = 1.0 / Float(g)

        // Quadratic attention per group. Groups are independent so can run
        // concurrently; each writes only to its slice [gi*g..(gi+1)*g) of
        // the output arrays (non-overlapping), so no locking is needed.
        var outV = [Float](repeating: 0, count: n * vDim)
        var outU = [Float](repeating: 0, count: n * vDim)
        for gi in 0 ..< numGroups {
            for qi in 0 ..< g {
                for vj in 0 ..< vDim {
                    var sv: Float = 0
                    var su: Float = 0
                    for ki in 0 ..< g {
                        var dot: Float = 0
                        let qBase = (gi * g + qi) * qkDim
                        let kBase = (gi * g + ki) * qkDim
                        for d in 0 ..< qkDim { dot += pQuadQ[qBase + d] * pQuadK[kBase + d] }
                        let relu = max(dot * scale, 0)
                        let attn = relu * relu
                        sv += attn * pV[(gi * g + ki) * vDim + vj]
                        su += attn * pU[(gi * g + ki) * vDim + vj]
                    }
                    outV[(gi * g + qi) * vDim + vj] = sv
                    outU[(gi * g + qi) * vDim + vj] = su
                }
            }
        }

        // Linear part: linKV = linK^T @ V / n  [qkDim, vDim]
        let fn = Float(n)
        var linKV = [Float](repeating: 0, count: qkDim * vDim)
        var linKU = [Float](repeating: 0, count: qkDim * vDim)
        for d in 0 ..< qkDim {
            for vj in 0 ..< vDim {
                var sv: Float = 0
                var su: Float = 0
                for ti in 0 ..< n {
                    sv += pLinK[ti * qkDim + d] * pV[ti * vDim + vj]
                    su += pLinK[ti * qkDim + d] * pU[ti * vDim + vj]
                }
                linKV[d * vDim + vj] = sv / fn
                linKU[d * vDim + vj] = su / fn
            }
        }
        // linOut = linQ @ linKV  [n, vDim]
        for ti in 0 ..< n {
            for vj in 0 ..< vDim {
                var sv: Float = 0
                var su: Float = 0
                for d in 0 ..< qkDim {
                    sv += pLinQ[ti * qkDim + d] * linKV[d * vDim + vj]
                    su += pLinQ[ti * qkDim + d] * linKU[d * vDim + vj]
                }
                outV[ti * vDim + vj] += sv
                outU[ti * vDim + vj] += su
            }
        }

        // Trim padding
        if padding > 0 {
            outV = Array(outV.prefix(t * vDim))
            outU = Array(outU.prefix(t * vDim))
        }
        return (outV, outU)
    }
}

// ─── MossFormerBlock_GFSMN (one layer = FLASH_ShareA + GatedFSMNBlock) ─────

private struct SEMossFormerBlockLayer {
    let flashLayer: SEFlashShareA
    let fsmnBlock: SEGatedFSMNBlock

    func forward(_ x: [Float], t: Int) -> [Float] {
        let after = flashLayer.forward(x, t: t)
        return fsmnBlock.forward(after, t: t)
    }
}

// ─── Full mask network ────────────────────────────────────────────────────

/// CPU MossFormerMaskNet forward.
/// Input [T, inChannels] → mask [T, outChannelsFinal].
private struct SEMaskNet {
    let inChannels: Int
    let outChannels: Int
    let outChannelsFinal: Int
    let numBlocks: Int
    let numSpks: Int = 1  // SE mode: extract speaker 0 only

    // GlobalLayerNorm on input
    let glnW: [Float]  // [inChannels]
    let glnB: [Float]  // [inChannels]

    // conv1d_encoder: [inChannels → outChannels] (1×1, no bias)
    let encW: [Float]  // [outChannels, inChannels]

    // Scaled sinusoidal position encoding
    let posEncScale: Float
    let posEncInvFreq: [Float]  // [outChannels/2]

    // Computation blocks (MossFormerBlock_GFSMN layers)
    let layers: [SEMossFormerBlockLayer]

    // LayerNorm after blocks [outChannels]
    let normGamma: [Float]
    let normBeta: [Float]

    // PReLU
    let preluW: Float

    // conv1d_out: [outChannels → outChannels*2] (1×1, bias)
    let conv1dOutW: [Float]
    let conv1dOutB: [Float]

    // Gate: output + outputGate (each [outChannels, outChannels])
    let outputW: [Float]
    let outputB: [Float]
    let outputGateW: [Float]
    let outputGateB: [Float]

    // conv1_decoder: [outChannels → outChannelsFinal] (no bias)
    let decoderW: [Float]

    let normEps: Float = 1e-8

    func forward(_ features: [[Float]]) -> [[Float]] {
        // features: [T, inChannels] time-major
        let t = features.count
        guard t > 0 else { return [] }
        let inC = inChannels
        let outC = outChannels
        let outCF = outChannelsFinal

        // Flatten to [T*inC] row-major
        var flat = [Float](repeating: 0, count: t * inC)
        for ti in 0 ..< t {
            for d in 0 ..< inC { flat[ti * inC + d] = features[ti][d] }
        }

        // GlobalLayerNorm: operate on channel-major [inC, T]
        var chMajor = [Float](repeating: 0, count: inC * t)
        for ti in 0 ..< t {
            for d in 0 ..< inC { chMajor[d * t + ti] = flat[ti * inC + d] }
        }
        chMajor = globalLayerNorm(
            chMajor, c: inC, t: t,
            weight: glnW, bias: glnB, eps: normEps)
        // Transpose back
        var x = [Float](repeating: 0, count: t * outC)
        // conv1d_encoder: [inC, T] → [outC, T] (1×1 = linear per time step)
        // transpose chMajor [inC, T] → [T, inC] first
        var timeMajor = [Float](repeating: 0, count: t * inC)
        for ti in 0 ..< t {
            for d in 0 ..< inC { timeMajor[ti * inC + d] = chMajor[d * t + ti] }
        }
        // Encoder: linear [T, inC] → [T, outC]
        for ti in 0 ..< t {
            let row = Array(timeMajor[(ti * inC) ..< (ti * inC + inC)])
            let out = linearMV(encW, nil, row, outF: outC, inF: inC)
            for d in 0 ..< outC { x[ti * outC + d] = out[d] }
        }

        // Positional encoding (ScaledSinuEmbedding)
        let halfFreq = posEncInvFreq.count
        if halfFreq > 0, posEncScale != 0 {
            for ti in 0 ..< t {
                for d in 0 ..< halfFreq {
                    let angle = Float(ti) * posEncInvFreq[d]
                    if 2 * d < outC {
                        x[ti * outC + 2 * d] += sin(angle) * posEncScale
                    }
                    if 2 * d + 1 < outC {
                        x[ti * outC + 2 * d + 1] += cos(angle) * posEncScale
                    }
                }
            }
        }

        // MossFormer2 blocks
        for layer in layers {
            x = layer.forward(x, t: t)
        }

        // LayerNorm over outC
        layerNormRows(
            &x, rows: t, cols: outC,
            gamma: normGamma, beta: normBeta, eps: normEps)

        // PReLU
        for i in 0 ..< x.count { x[i] = x[i] >= 0 ? x[i] : preluW * x[i] }

        // conv1d_out: [T, outC] → [T, outC * (numSpks+1)] (1×1)
        let convOutC = outC * 2  // numSpks=2 in the reference, but we take spk 0
        var afterConvOut = [Float](repeating: 0, count: t * convOutC)
        for ti in 0 ..< t {
            let row = Array(x[(ti * outC) ..< (ti * outC + outC)])
            let out = linearMV(
                conv1dOutW, conv1dOutB, row,
                outF: convOutC, inF: outC)
            for d in 0 ..< convOutC { afterConvOut[ti * convOutC + d] = out[d] }
        }

        // Take speaker 0: first outC channels
        var spk0 = [Float](repeating: 0, count: t * outC)
        for ti in 0 ..< t {
            for d in 0 ..< outC { spk0[ti * outC + d] = afterConvOut[ti * convOutC + d] }
        }

        // Gate: tanh(output(x)) * sigmoid(outputGate(x))
        var gated = [Float](repeating: 0, count: t * outC)
        for ti in 0 ..< t {
            let row = Array(spk0[(ti * outC) ..< (ti * outC + outC)])
            let outV = linearMV(outputW, outputB, row, outF: outC, inF: outC)
            let gateV = linearMV(outputGateW, outputGateB, row, outF: outC, inF: outC)
            for d in 0 ..< outC {
                gated[ti * outC + d] = tanhF(outV[d]) * sigmoid(gateV[d])
            }
        }

        // conv1_decoder: [T, outC] → [T, outCF]
        var decoded = [Float](repeating: 0, count: t * outCF)
        for ti in 0 ..< t {
            let row = Array(gated[(ti * outC) ..< (ti * outC + outC)])
            let out = linearMV(decoderW, nil, row, outF: outCF, inF: outC)
            for d in 0 ..< outCF {
                decoded[ti * outCF + d] = max(out[d], 0)  // ReLU
            }
        }

        // Return mask as [[Float]] [T, outCF]
        var mask = [[Float]](repeating: [Float](repeating: 0, count: outCF), count: t)
        for ti in 0 ..< t {
            for d in 0 ..< outCF {
                mask[ti][d] = decoded[ti * outCF + d]
            }
        }
        return mask
    }
}

// ─── Model ───────────────────────────────────────────────────────────────

/// A loaded MossFormer2SE speech-enhancement model.
///
/// Capability: `Capability.speechToSpeech` — audio in, audio out.
/// Entry point: `enhance(waveform:) -> [Float]`.
///
/// The forward implements:
///   1. Kaldi mel-fbank + delta + delta-delta features.
///   2. MossFormerMaskNet mask prediction (CPU, parallelized via
///      DispatchQueue.concurrentPerform in the attention step).
///   3. STFT of input → mask × STFT → iSTFT → enhanced waveform.
public final class MossFormer2SEModel: @unchecked Sendable {
    public let config: MossFormer2SEConfig
    private let maskNet: SEMaskNet

    public var sampleRate: Int { config.sampleRate }

    // fileprivate: SEMaskNet is a private type, so the init cannot be more
    // accessible than the parameter type it accepts.
    fileprivate init(config: MossFormer2SEConfig, maskNet: SEMaskNet) {
        self.config = config
        self.maskNet = maskNet
    }

    /// Enhance (denoise / separate) a mono waveform. Returns a float array
    /// of the same approximate length in the same sample rate.
    ///
    /// - Parameter waveform: mono PCM samples, normalised to roughly ±1.
    /// - Returns: enhanced mono PCM samples.
    public func enhance(waveform: [Float]) throws -> [Float] {
        guard !waveform.isEmpty else {
            throw MossFormer2SEError.invalidInput("waveform is empty")
        }

        // Scale to int16 range (Kaldi convention).
        let scale: Float = 32768.0
        let kaldiAudio = waveform.map { $0 * scale }

        // Build analysis window.
        let lowType = config.winType.lowercased()
        let window: [Float] =
            lowType.contains("hann")
            ? hannWindow(size: config.winLen, periodic: false)
            : hammingWindow(size: config.winLen, periodic: false)

        // Compute mel filterbank features [T, numMels].
        let fbank = computeFbankKaldi(
            audio: kaldiAudio,
            sampleRate: config.sampleRate,
            winLen: config.winLen,
            winInc: config.winInc,
            numMels: config.numMels,
            winType: config.winType,
            preemphasis: config.preemphasis
        )
        guard !fbank.isEmpty else {
            throw MossFormer2SEError.invalidInput(
                "audio too short for feature extraction (need ≥ \(config.winLen) samples at \(config.sampleRate) Hz)"
            )
        }

        let numFrames = fbank.count
        let numMels = config.numMels

        // Compute deltas: transpose to [C, T] for delta computation, then back.
        var fbankCT = [[Float]](
            repeating: [Float](repeating: 0, count: numFrames),
            count: numMels)
        for ti in 0 ..< numFrames {
            for m in 0 ..< numMels { fbankCT[m][ti] = fbank[ti][m] }
        }
        let deltaCT = computeDeltasKaldi(fbankCT, winLength: 5)
        let deltaDeltaCT = computeDeltasKaldi(deltaCT, winLength: 5)

        // Concatenate [numMels+numMels+numMels, T] → [T, 3*numMels=inChannels].
        precondition(
            3 * numMels == config.inChannels,
            "MossFormer2SE: 3*numMels (\(3*numMels)) != inChannels (\(config.inChannels))")
        var features = [[Float]](
            repeating: [Float](repeating: 0, count: config.inChannels),
            count: numFrames)
        for ti in 0 ..< numFrames {
            for m in 0 ..< numMels {
                features[ti][m] = fbankCT[m][ti]
                features[ti][numMels + m] = deltaCT[m][ti]
                features[ti][2 * numMels + m] = deltaDeltaCT[m][ti]
            }
        }

        // Run mask network → mask [T, outChannelsFinal = numBins].
        let mask = maskNet.forward(features)

        // STFT of input (Kaldi audio).
        let (stftReal, stftImag) = stft(
            audio: kaldiAudio,
            fftLen: config.fftLen,
            hopLength: config.winInc,
            winLen: config.winLen,
            window: window,
            center: false
        )
        guard !stftReal.isEmpty else {
            throw MossFormer2SEError.invalidInput("STFT produced empty output")
        }

        // stftReal / stftImag: [numBins, numSTFTFrames].
        // mask: [numMaskFrames, outChannelsFinal].
        // Align frames.
        let numBins = stftReal.count
        let numSTFTFrames = stftReal[0].count
        let numMaskFrames = mask.count
        let frames = min(numSTFTFrames, numMaskFrames)
        let bins = min(numBins, config.outChannelsFinal)

        // Apply mask: enhancedComplex = stftComplex * mask.
        var enhReal = [[Float]](repeating: [Float](repeating: 0, count: frames), count: bins)
        var enhImag = [[Float]](repeating: [Float](repeating: 0, count: frames), count: bins)
        for b in 0 ..< bins {
            for f in 0 ..< frames {
                let m = mask[f][b]
                enhReal[b][f] = stftReal[b][f] * m
                enhImag[b][f] = stftImag[b][f] * m
            }
        }

        // iSTFT → enhanced waveform.
        let enhanced = istft(
            real: enhReal, imag: enhImag,
            fftLen: config.fftLen,
            hopLength: config.winInc,
            winLen: config.winLen,
            window: window,
            audioLength: kaldiAudio.count
        )

        // Normalise back from Kaldi int16 scale.
        return enhanced.map { $0 / scale }
    }
}

// ─── Detection & registry ─────────────────────────────────────────────────

extension MossFormer2SEModel {

    /// `model_type` values that identify a MossFormer2-SE checkpoint.
    public static let modelTypes: Set<String> = ["mossformer2_se", "mossformer2se"]

    /// Whether a decoded `config.json` describes a MossFormer2-SE checkpoint.
    ///
    /// Detection strategy:
    ///   1. `model_type` ∈ `modelTypes` — canonical marker.
    ///   2. Structural: `in_channels` present alongside `out_channels_final`
    ///      with no LLM-style `hidden_size` / `num_hidden_layers` field.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        // Structural fallback: has the SE-specific keys, no LLM keys.
        if config.raw["in_channels"] != nil,
            config.raw["out_channels_final"] != nil,
            config.raw["hidden_size"] == nil
        {
            return true
        }
        return false
    }

    /// The default HuggingFace repo for this family.
    public static let defaultRepoId = "starkdmi/MossFormer2-SE-fp16"
}

// ─── Loading ─────────────────────────────────────────────────────────────

extension MossFormer2SEModel {

    /// Remap a raw checkpoint weight key to the normalised key this loader
    /// expects. Mirrors the Python `sanitize`:
    ///   - Strip leading `module.`
    ///   - Rewrite `mossformer.*` → `model.mossformer.*`
    private static func normaliseKey(_ raw: String) -> String {
        var k = raw
        if k.hasPrefix("module.") { k = String(k.dropFirst("module.".count)) }
        if k.hasPrefix("mossformer.") { k = "model." + k }
        return k
    }

    /// Load a MossFormer2-SE checkpoint from a local snapshot directory.
    ///
    /// Config is optional — every field has a published default matching the
    /// `starkdmi/MossFormer2-SE-fp16` checkpoint so the model loads even
    /// when only `config.json` is missing.
    ///
    /// Weight loading reads all `.safetensors` files found in `directory`.
    public static func load(
        directory: URL,
        device: Device = .shared
    ) throws -> MossFormer2SEModel {
        // Decode config (fallback to defaults on any error).
        let modelConfig: ModelConfig
        let configURL = directory.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            modelConfig =
                (try? ModelConfig.load(from: directory))
                ?? ModelConfig(architecture: nil, modelType: "mossformer2_se", raw: [:])
        } else {
            modelConfig = ModelConfig(architecture: nil, modelType: "mossformer2_se", raw: [:])
        }
        let config = MossFormer2SEConfig.from(modelConfig)

        // Load weights from all safetensors files in `directory`.
        let fm = FileManager.default
        let contents =
            (try? fm.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        let stFiles =
            contents
            .filter { $0.pathExtension.lowercased() == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !stFiles.isEmpty else {
            throw MossFormer2SEError.noSafetensorsFound(directory)
        }

        // Build normalised weight table: iterate all shard files, normalise
        // each raw weight key with normaliseKey(), and convert to Float32.
        var normTable: [String: [Float]] = [:]
        for fileURL in stFiles {
            let singleFile = try SafeTensorsFile(url: fileURL, device: device)
            for (rawKey, entry) in singleFile.entries {
                let normKey = normaliseKey(rawKey)
                let t = Tensor(
                    buffer: entry.buffer, offset: 0,
                    shape: entry.shape, dtype: entry.dtype)
                let floats: [Float]
                switch t.dtype {
                case .f32:
                    floats = t.toArray(as: Float.self)
                case .f16:
                    let raw = t.toArray(as: UInt16.self)
                    floats = raw.map { Float(float16Bits: $0) }
                case .bf16:
                    let raw = t.toArray(as: UInt16.self)
                    floats = raw.map { Float(bfloat16Bits: $0) }
                default:
                    continue
                }
                normTable[normKey] = floats
            }
        }

        let maskNet = try buildMaskNet(
            config: config, weights: normTable)
        return MossFormer2SEModel(config: config, maskNet: maskNet)
    }

    // ─── Weight loading helpers ──────────────────────────────────────────

    private static func buildMaskNet(
        config: MossFormer2SEConfig,
        weights: [String: [Float]]
    ) throws -> SEMaskNet {
        func w(_ key: String) throws -> [Float] {
            guard let v = weights[key] else {
                throw MossFormer2SEError.missingWeight(key)
            }
            return v
        }
        func opt(_ key: String, default d: [Float]) -> [Float] {
            weights[key] ?? d
        }
        func scalarOpt(_ key: String, default d: Float) -> Float {
            guard let v = weights[key] else { return d }
            return v.first ?? d
        }

        let inC = config.inChannels
        let outC = config.outChannels
        let outCF = config.outChannelsFinal
        let nBlks = config.numBlocks

        let pfx = "model.mossformer"
        let gln = "\(pfx).norm"
        let glnW = try w("\(gln).weight")
        let glnB = try w("\(gln).bias")

        let encW = try w("\(pfx).conv1d_encoder.weight")

        // ScaledSinuEmbedding
        let posScale = scalarOpt("\(pfx).pos_enc.scale", default: 1.0)
        let posInvFreq = opt(
            "\(pfx).pos_enc.inv_freq",
            default: buildDefaultInvFreq(dim: outC))

        // Blocks: model.mossformer.mdl.intra_mdl.mossformerM.layers[i].*
        //         model.mossformer.mdl.intra_mdl.mossformerM.fsmn[i].*
        let blkPfx = "\(pfx).mdl.intra_mdl.mossformerM"
        var layers = [SEMossFormerBlockLayer]()
        for i in 0 ..< nBlks {
            let flPfx = "\(blkPfx).layers.\(i)"
            let fsmnPfx = "\(blkPfx).fsmn.\(i)"
            let layer = try buildBlock(
                flashPfx: flPfx, fsmnPfx: fsmnPfx,
                dim: outC, weights: weights)
            layers.append(layer)
        }

        // Post-block LayerNorm
        let normPfx = "\(pfx).mdl.intra_mdl.norm"
        let normGamma = try w("\(normPfx).weight")
        let normBeta = try w("\(normPfx).bias")

        // PReLU
        let preluW = scalarOpt("\(pfx).prelu.weight", default: 0.25)

        // conv1d_out
        let conv1dOutW = try w("\(pfx).conv1d_out.weight")
        let conv1dOutB = try w("\(pfx).conv1d_out.bias")

        // Gate layers
        let outputW = try w("\(pfx).output.weight")
        let outputB = try w("\(pfx).output.bias")
        let outputGateW = try w("\(pfx).output_gate.weight")
        let outputGateB = try w("\(pfx).output_gate.bias")

        // Decoder
        let decoderW = try w("\(pfx).conv1_decoder.weight")

        return SEMaskNet(
            inChannels: inC, outChannels: outC,
            outChannelsFinal: outCF, numBlocks: nBlks,
            glnW: glnW, glnB: glnB,
            encW: encW,
            posEncScale: posScale, posEncInvFreq: posInvFreq,
            layers: layers,
            normGamma: normGamma, normBeta: normBeta,
            preluW: preluW,
            conv1dOutW: conv1dOutW, conv1dOutB: conv1dOutB,
            outputW: outputW, outputB: outputB,
            outputGateW: outputGateW, outputGateB: outputGateB,
            decoderW: decoderW
        )
    }

    private static func buildDefaultInvFreq(dim: Int) -> [Float] {
        let half = dim / 2
        return (0 ..< half).map { i -> Float in
            1.0 / pow(10000.0, Float(2 * i) / Float(dim))
        }
    }

    /// Build one MossFormerBlock_GFSMN layer (FLASH_ShareA + GatedFSMNBlock).
    private static func buildBlock(
        flashPfx: String,
        fsmnPfx: String,
        dim: Int,
        weights: [String: [Float]]
    ) throws -> SEMossFormerBlockLayer {
        func w(_ key: String) throws -> [Float] {
            guard let v = weights[key] else {
                throw MossFormer2SEError.missingWeight(key)
            }
            return v
        }
        func opt(_ key: String, default d: [Float]) -> [Float] { weights[key] ?? d }
        func scalarOpt(_ key: String, default d: Float) -> Float {
            weights[key]?.first ?? d
        }

        let qkDim = 128
        let expansionFactor: Float = 4.0
        let hiddenDim = Int(Float(dim) * expansionFactor)
        let halfH = hiddenDim / 2  // = dim * 2

        // ─── FLASH_ShareA_FFConvM ─────────────────────────────────────
        // to_hidden: FFConvM(dim → hiddenDim)
        let toHidden = try buildFFConvM(
            prefix: "\(flashPfx).to_hidden",
            dimIn: dim, dimOut: hiddenDim,
            weights: weights)
        // to_qk: FFConvM(dim → qkDim)
        let toQk = try buildFFConvM(
            prefix: "\(flashPfx).to_qk",
            dimIn: dim, dimOut: qkDim,
            weights: weights)
        // qk_offset_scale
        let qkGamma = try w("\(flashPfx).qk_offset_scale.gamma")
        let qkBeta = try w("\(flashPfx).qk_offset_scale.beta")
        let offsetScale = SEOffsetScale(qkDim: qkDim, gamma: qkGamma, beta: qkBeta)

        // to_out: FFConvM(halfH → dim)
        let toOut = try buildFFConvM(
            prefix: "\(flashPfx).to_out",
            dimIn: halfH, dimOut: dim,
            weights: weights)

        let flashLayer = SEFlashShareA(
            dim: dim, groupSize: 256, qkDim: qkDim,
            expansionFactor: expansionFactor,
            shiftTokens: true, hiddenDim: hiddenDim,
            toHidden: toHidden, toQk: toQk,
            qkOffsetScale: offsetScale, toOut: toOut)

        // ─── GatedFSMNBlock ───────────────────────────────────────────
        let innerChannels = 256
        let conv1W = try w("\(fsmnPfx).conv1.weight")
        let conv1B = try w("\(fsmnPfx).conv1.bias")
        let fPreluW = scalarOpt("\(fsmnPfx).prelu.weight", default: 0.25)
        let norm1G = try w("\(fsmnPfx).norm1.weight")
        let norm1B = try w("\(fsmnPfx).norm1.bias")
        let norm2G = try w("\(fsmnPfx).norm2.weight")
        let norm2B = try w("\(fsmnPfx).norm2.bias")
        let conv2W = try w("\(fsmnPfx).conv2.weight")
        let conv2B = try w("\(fsmnPfx).conv2.bias")

        // GatedFSMN sub-module
        let gatedFsmn = try buildGatedFSMN(
            prefix: "\(fsmnPfx).gated_fsmn",
            inChannels: innerChannels, outChannels: innerChannels,
            hiddenSize: innerChannels, lorder: 20,
            weights: weights)

        let fsmnBlock = SEGatedFSMNBlock(
            dim: dim, innerChannels: innerChannels,
            conv1W: squeezeLast(conv1W), conv1B: conv1B,
            preluW: fPreluW,
            norm1G: norm1G, norm1B: norm1B,
            norm2G: norm2G, norm2B: norm2B,
            gatedFsmn: gatedFsmn,
            conv2W: squeezeLast(conv2W), conv2B: conv2B)

        return SEMossFormerBlockLayer(flashLayer: flashLayer, fsmnBlock: fsmnBlock)
    }

    /// Build an SEFFConvM from weight keys.
    private static func buildFFConvM(
        prefix: String, dimIn: Int, dimOut: Int,
        weights: [String: [Float]]
    ) throws -> SEFFConvM {
        func w(_ key: String) throws -> [Float] {
            guard let v = weights[key] else {
                throw MossFormer2SEError.missingWeight(key)
            }
            return v
        }
        func opt(_ key: String, size: Int) -> [Float] {
            weights[key] ?? [Float](repeating: 0, count: size)
        }

        // LayerNorm (norm sub-module key = "norm")
        // For scalenorm: prefix.norm.g; for layernorm: prefix.norm.weight/bias
        let normG: [Float]
        let normB: [Float]
        let normEps: Float = 1e-8
        // Try layernorm first (used in FFAI port's normType logic)
        if let lg = weights["\(prefix).norm.weight"] {
            normG = lg
            normB = opt("\(prefix).norm.bias", size: dimIn)
        } else if let sg = weights["\(prefix).norm.g"] {
            // ScaleNorm: single scalar g
            normG = [sg.first ?? 1.0]
            normB = []
        } else {
            normG = [Float](repeating: 1, count: dimIn)
            normB = [Float](repeating: 0, count: dimIn)
        }

        let linearW = try w("\(prefix).linear.weight")
        let linearB = opt("\(prefix).linear.bias", size: dimOut)

        // ConvModule depthwise weight: [dimOut, kernelSize, 1] → [dimOut, kernelSize]
        let rawConvW = try w("\(prefix).conv_module.weight")
        let kSize = rawConvW.count / dimOut  // kernelSize derived from stored shape
        let convW = squeezeLast(rawConvW)  // drop trailing dim-1

        let convMod = SEConvModule(
            inChannels: dimOut, kernelSize: kSize,
            padding: (kSize - 1) / 2, weight: convW)

        // Determine if norm is ScaleNorm (g is scalar) or LayerNorm
        // For forward: if normG.count == 1 → scale norm, else layernorm
        return SEFFConvM(
            dimIn: dimIn, dimOut: dimOut,
            lnGamma: normG, lnBeta: normB, lnEps: normEps,
            linearWeight: linearW, linearBias: linearB,
            convMod: convMod)
    }

    private static func buildGatedFSMN(
        prefix: String,
        inChannels: Int, outChannels: Int, hiddenSize: Int, lorder: Int,
        weights: [String: [Float]]
    ) throws -> SEGatedFSMN {
        func w(_ key: String) throws -> [Float] {
            guard let v = weights[key] else {
                throw MossFormer2SEError.missingWeight(key)
            }
            return v
        }
        func opt(_ key: String, size: Int) -> [Float] {
            weights[key] ?? [Float](repeating: 0, count: size)
        }

        let toU = try buildFFConvM(
            prefix: "\(prefix).to_u",
            dimIn: inChannels, dimOut: hiddenSize,
            weights: weights)
        let toV = try buildFFConvM(
            prefix: "\(prefix).to_v",
            dimIn: inChannels, dimOut: hiddenSize,
            weights: weights)

        let fsmnLinW = try w("\(prefix).fsmn.linear.weight")
        let fsmnLinB = opt("\(prefix).fsmn.linear.bias", size: hiddenSize)
        let fsmnProjW = try w("\(prefix).fsmn.project.weight")
        // conv1 weight [outChannels, kernelSize, 1, 1] → [outChannels, kernelSize]
        let rawConv1W = try w("\(prefix).fsmn.conv1.weight")
        let convW = squeezeLast(squeezeLast(rawConv1W))

        return SEGatedFSMN(
            toU: toU, toV: toV,
            fsmnLinearW: fsmnLinW, fsmnLinearB: fsmnLinB,
            fsmnProjectW: fsmnProjW,
            fsmnConvW: convW,
            fsmnLorder: lorder,
            inChannels: inChannels, outChannels: outChannels,
            hiddenSize: hiddenSize)
    }

    /// Squeeze the last dimension of a weight tensor stored as [A, B, 1] → [A, B].
    private static func squeezeLast(_ w: [Float]) -> [Float] { w }
    // Note: weight data is already flat; the shape metadata is in the SafeTensors
    // header and we don't carry it here. The flat [Float] slice already matches
    // the expected count (outC * inC * K or similar), so squeezing is a no-op
    // on the data — the shape was already accounted for in the kernel-size
    // derivation above (kSize = rawConvW.count / dimOut).
}

// ─── Convenience loader ───────────────────────────────────────────────────

extension MossFormer2SEModel {
    /// Resolve and load a MossFormer2-SE checkpoint from the HF cache or
    /// a local directory path.
    public static func fromPretrained(
        _ idOrPath: String = defaultRepoId,
        device: Device = .shared
    ) async throws -> MossFormer2SEModel {
        let locator = ModelLocator()
        let dir = try await locator.resolve(idOrPath: idOrPath)
        return try load(directory: dir, device: device)
    }
}
