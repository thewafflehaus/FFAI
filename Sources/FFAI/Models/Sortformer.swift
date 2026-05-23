// Sortformer — NVIDIA's streaming multi-speaker diarization model.
//
// Architecture (diar_streaming_sortformer_4spk-v2.1):
//
//   Log-mel (128 bins, NeMo-style preemphasis) → FastConformer encoder
//   (ConvSubsampling /8 + 17 Conformer layers) → projection →
//   Transformer encoder (18 BART-style layers) → speaker sigmoid →
//   [time, numSpeakers] per-frame speech probabilities
//
// The model is ~30M parameters and runs on short clips, so the forward
// follows the same CPU (`VADCompute`) path as SileroVAD and SmartTurn.
// Weights are loaded from SafeTensors, copied to host once, and the
// forward is executed on `[Float]` arrays using Accelerate-backed
// primitives from VADCompute.swift.
//
// Checkpoint layout: `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16`.
// The MLX checkpoint stores Conv2d weights in `[outC, H, W, inC]` (MLX
// layout); we transpose to PyTorch `[outC, inC, H, W]` at load time.
// Conv1d weights are `[outC, K, inC]` in MLX → `[outC, inC, K]` in PyTorch.

import Foundation
import Accelerate

// MARK: - Config

/// FastConformer encoder configuration.
public struct SortformerFCConfig: Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let intermediateSize: Int
    public let numMelBins: Int
    public let convKernelSize: Int
    public let subsamplingFactor: Int
    public let subsamplingConvChannels: Int
    public let subsamplingConvKernelSize: Int
    public let subsamplingConvStride: Int
    public let attentionBias: Bool
    public let scaleInput: Bool

    public init(from raw: [String: Any]) {
        hiddenSize              = (raw["hidden_size"] as? Int) ?? 512
        numLayers               = (raw["num_hidden_layers"] as? Int) ?? 18
        numHeads                = (raw["num_attention_heads"] as? Int) ?? 8
        intermediateSize        = (raw["intermediate_size"] as? Int) ?? 2048
        numMelBins              = (raw["num_mel_bins"] as? Int) ?? 80
        convKernelSize          = (raw["conv_kernel_size"] as? Int) ?? 9
        subsamplingFactor       = (raw["subsampling_factor"] as? Int) ?? 8
        subsamplingConvChannels = (raw["subsampling_conv_channels"] as? Int) ?? 256
        subsamplingConvKernelSize = (raw["subsampling_conv_kernel_size"] as? Int) ?? 3
        subsamplingConvStride   = (raw["subsampling_conv_stride"] as? Int) ?? 2
        attentionBias           = (raw["attention_bias"] as? Bool) ?? true
        scaleInput              = (raw["scale_input"] as? Bool) ?? true
    }
}

/// Transformer (BART-style) encoder configuration.
public struct SortformerTFConfig: Sendable {
    public let dModel: Int
    public let numLayers: Int
    public let numHeads: Int
    public let ffnDim: Int
    public let layerNormEps: Float
    public let maxPositions: Int
    public let kProjBias: Bool

    public init(from raw: [String: Any]) {
        dModel       = (raw["d_model"] as? Int) ?? 192
        numLayers    = (raw["encoder_layers"] as? Int) ?? 18
        numHeads     = (raw["encoder_attention_heads"] as? Int) ?? 8
        ffnDim       = (raw["encoder_ffn_dim"] as? Int) ?? 768
        layerNormEps = (raw["layer_norm_eps"] as? Double).map { Float($0) } ?? 1e-5
        maxPositions = (raw["max_source_positions"] as? Int) ?? 1500
        kProjBias    = (raw["k_proj_bias"] as? Bool) ?? false
    }
}

/// Sortformer modules configuration (streaming / AOSC parameters).
public struct SortformerModulesConfig: Sendable {
    public let numSpeakers: Int
    public let fcDModel: Int
    public let tfDModel: Int
    public let subsamplingFactor: Int

    public init(from raw: [String: Any]) {
        numSpeakers      = (raw["num_speakers"] as? Int) ?? 4
        fcDModel         = (raw["fc_d_model"] as? Int) ?? 512
        tfDModel         = (raw["tf_d_model"] as? Int) ?? 192
        subsamplingFactor = (raw["subsampling_factor"] as? Int) ?? 8
    }
}

/// Audio processor / mel-feature configuration.
public struct SortformerProcessorConfig: Sendable {
    public let featureSize: Int
    public let sampleRate: Int
    public let hopLength: Int
    public let nFft: Int
    public let winLength: Int
    public let preemphasis: Float

    public init(from raw: [String: Any]) {
        featureSize  = (raw["feature_size"] as? Int) ?? 80
        sampleRate   = (raw["sampling_rate"] as? Int) ?? 16000
        hopLength    = (raw["hop_length"] as? Int) ?? 160
        nFft         = (raw["n_fft"] as? Int) ?? 512
        winLength    = (raw["win_length"] as? Int) ?? 400
        preemphasis  = (raw["preemphasis"] as? Double).map { Float($0) } ?? 0.97
    }
}

/// Top-level Sortformer configuration.
public struct SortformerConfig: Sendable {
    public let numSpeakers: Int
    public let fcEncoder: SortformerFCConfig
    public let tfEncoder: SortformerTFConfig
    public let modules: SortformerModulesConfig
    public let processor: SortformerProcessorConfig

    public init(from raw: [String: Any]) {
        numSpeakers = (raw["num_speakers"] as? Int) ?? 4
        fcEncoder  = SortformerFCConfig(from: (raw["fc_encoder_config"] as? [String: Any]) ?? [:])
        tfEncoder  = SortformerTFConfig(from: (raw["tf_encoder_config"] as? [String: Any]) ?? [:])
        modules    = SortformerModulesConfig(from: (raw["modules_config"] as? [String: Any]) ?? [:])
        processor  = SortformerProcessorConfig(from: (raw["processor_config"] as? [String: Any]) ?? [:])
    }
}

// MARK: - Error

public enum SortformerError: Error, CustomStringConvertible {
    case missingWeight(String)
    case configNotFound(URL)

    public var description: String {
        switch self {
        case .missingWeight(let k): return "SortformerModel: weight not found: \(k)"
        case .configNotFound(let u): return "SortformerModel: config.json not found at \(u.path)"
        }
    }
}

// MARK: - Audio front-end (NeMo log-mel, CPU)

/// Compute the NeMo-style log-mel spectrogram for Sortformer.
///
/// NeMo convention:
///  - preemphasis filter: y[n] = x[n] - coeff * x[n-1]
///  - STFT with zero-padding on the left/right by `nFft/2`, Hann window
///    of `winLength` samples center-padded to `nFft`
///  - Slaney mel filterbank (no HTK)
///  - log(mel + 1e-14) — no log10, no Whisper-style clamping
///  - per-feature normalization: (x - μ) / (σ + 1e-5) using Bessel's
///    correction for variance
///  - Output shape: `[featureSize, numFrames]` (channel-major)
func sortformerMelFeatures(
    waveform: [Float],
    proc: SortformerProcessorConfig
) -> (values: [Float], nMels: Int, nFrames: Int) {
    guard !waveform.isEmpty else { return ([], proc.featureSize, 0) }
    let sr = proc.sampleRate
    let nFft = proc.nFft
    let hop = proc.hopLength
    let winLen = proc.winLength
    let nMels = proc.featureSize

    // Preemphasis: y[0] = x[0], y[n] = x[n] - coeff * x[n-1]
    var wav = [Float](repeating: 0, count: waveform.count)
    wav[0] = waveform[0]
    for i in 1..<waveform.count {
        wav[i] = waveform[i] - proc.preemphasis * waveform[i - 1]
    }

    // Build Hann window of `winLen` samples, zero-padded to `nFft`.
    var window = VADAudioFrontend.hannWindow(size: winLen)
    if winLen < nFft {
        let left = (nFft - winLen) / 2
        let right = nFft - winLen - left
        window = [Float](repeating: 0, count: left) + window + [Float](repeating: 0, count: right)
    }

    // Zero-pad signal: NeMo pads `nFft/2` on each side with zeros (center=False in original
    // PyTorch stft, but NeMo's FilterbankFeatures uses center=True via `torch.stft`
    // which pads `nFft/2` on both sides by default in reflect mode).
    // The mlx-audio-swift reference uses constant (zero) padding.
    let pad = nFft / 2
    let padded = [Float](repeating: 0, count: pad) + wav + [Float](repeating: 0, count: pad)

    // STFT — manual DFT (small frames ≤ 512 samples, sufficient for inference).
    let nBins = nFft / 2 + 1
    let numFrames = padded.count >= nFft ? (padded.count - nFft) / hop + 1 : 0
    guard numFrames > 0 else { return ([], nMels, 0) }

    // Compute power spectrum for each frame.
    var power = [Float](repeating: 0, count: numFrames * nBins)
    var frame = [Float](repeating: 0, count: nFft)
    for f in 0..<numFrames {
        let start = f * hop
        for n in 0..<nFft { frame[n] = padded[start + n] * window[n] }
        // Real-to-complex DFT → power.
        for k in 0..<nBins {
            var re: Float = 0, im: Float = 0
            let w = -2 * Float.pi * Float(k) / Float(nFft)
            for n in 0..<nFft {
                let angle = w * Float(n)
                re += frame[n] * cosf(angle)
                im += frame[n] * sinf(angle)
            }
            power[f * nBins + k] = re * re + im * im
        }
    }

    // Slaney mel filterbank: [nBins, nMels].
    let fb = VADAudioFrontend.melFilterbank(sampleRate: sr, nFft: nFft, nMels: nMels)

    // Apply filterbank: mel[f, m] = Σ_k power[f, k] * fb[k, m]
    var mel = VADAudioFrontend.applyMelFilterbank(
        power: power, numFrames: numFrames, nBins: nBins,
        filterbank: fb, nMels: nMels)

    // Log with floor guard.
    let logGuard: Float = 1e-14
    for i in mel.indices { mel[i] = logf(mel[i] + logGuard) }

    // Per-feature normalization: z-score over time with Bessel's correction.
    // mel is [numFrames, nMels] row-major; normalize each mel bin over frames.
    let normEps: Float = 1e-5
    var normed = [Float](repeating: 0, count: numFrames * nMels)
    for m in 0..<nMels {
        var sum: Float = 0
        for f in 0..<numFrames { sum += mel[f * nMels + m] }
        let mean = sum / Float(numFrames)
        var varSum: Float = 0
        for f in 0..<numFrames { let d = mel[f * nMels + m] - mean; varSum += d * d }
        let std = (varSum / Float(max(numFrames - 1, 1))).squareRoot()
        for f in 0..<numFrames {
            normed[f * nMels + m] = (mel[f * nMels + m] - mean) / (std + normEps)
        }
    }

    // Transpose to channel-major [nMels, numFrames].
    var chMajor = [Float](repeating: 0, count: nMels * numFrames)
    for m in 0..<nMels {
        for f in 0..<numFrames { chMajor[m * numFrames + f] = normed[f * nMels + m] }
    }

    return (chMajor, nMels, numFrames)
}

// MARK: - FastConformer building blocks (CPU)

/// 2-D convolution (CPU) for the ConvSubsampling front-end.
/// `weight` is `[outC, inC, kH, kW]` (PyTorch convention, row-major).
private func conv2dForward(
    input: [Float], inC: Int, inH: Int, inW: Int,
    weight: [Float], bias: [Float]?,
    outC: Int, kH: Int, kW: Int,
    strideH: Int, strideW: Int,
    padH: Int, padW: Int,
    groups: Int = 1
) -> (values: [Float], outH: Int, outW: Int) {
    let outH = (inH + 2 * padH - kH) / strideH + 1
    let outW = (inW + 2 * padW - kW) / strideW + 1
    var out = [Float](repeating: 0, count: outC * outH * outW)
    let inCPerGroup = inC / groups
    let outCPerGroup = outC / groups
    for g in 0..<groups {
        for oc in 0..<outCPerGroup {
            let globalOC = g * outCPerGroup + oc
            let b = bias?[globalOC] ?? 0
            for oh in 0..<outH {
                for ow in 0..<outW {
                    var acc = b
                    for ic in 0..<inCPerGroup {
                        let globalIC = g * inCPerGroup + ic
                        let wBase = (globalOC * inCPerGroup + ic) * kH * kW
                        for kh in 0..<kH {
                            let ih = oh * strideH - padH + kh
                            if ih < 0 || ih >= inH { continue }
                            for kw in 0..<kW {
                                let iw = ow * strideW - padW + kw
                                if iw < 0 || iw >= inW { continue }
                                acc += input[(globalIC * inH + ih) * inW + iw]
                                     * weight[wBase + kh * kW + kw]
                            }
                        }
                    }
                    out[(globalOC * outH + oh) * outW + ow] = acc
                }
            }
        }
    }
    return (out, outH, outW)
}

/// Conv1d for depthwise and pointwise convolutions in the Conformer.
/// `weight` is `[outC, inC, K]` (PyTorch layout).
private func conv1dForward(
    input: [Float], inC: Int, length: Int,
    weight: [Float], bias: [Float]?,
    outC: Int, kernelSize: Int,
    stride: Int = 1, padding: Int = 0,
    groups: Int = 1
) -> (values: [Float], outLen: Int) {
    let outLen = (length + 2 * padding - kernelSize) / stride + 1
    var out = [Float](repeating: 0, count: outC * outLen)
    let inCPerGroup = inC / groups
    let outCPerGroup = outC / groups
    for g in 0..<groups {
        for oc in 0..<outCPerGroup {
            let goc = g * outCPerGroup + oc
            let b = bias?[goc] ?? 0
            let wBase = (goc * inCPerGroup) * kernelSize
            for t in 0..<outLen {
                var acc = b
                for ic in 0..<inCPerGroup {
                    let gic = g * inCPerGroup + ic
                    for k in 0..<kernelSize {
                        let idx = t * stride - padding + k
                        if idx >= 0 && idx < length {
                            acc += input[gic * length + idx] * weight[wBase + ic * kernelSize + k]
                        }
                    }
                }
                out[goc * outLen + t] = acc
            }
        }
    }
    return (out, outLen)
}

/// Batch normalization (inference) over the last dimension.
/// `input` is `[time, features]` row-major; mean/var/weight/bias are `[features]`.
private func batchNorm1dForward(
    input: [Float], rows: Int, features: Int,
    weight: [Float], bias: [Float],
    runningMean: [Float], runningVar: [Float],
    eps: Float = 1e-5
) -> [Float] {
    var out = [Float](repeating: 0, count: rows * features)
    for f in 0..<features {
        let inv = 1 / (runningVar[f] + eps).squareRoot()
        let w = weight[f]; let b = bias[f]; let mu = runningMean[f]
        for t in 0..<rows {
            out[t * features + f] = (input[t * features + f] - mu) * inv * w + b
        }
    }
    return out
}

/// SiLU activation.
private func silu(_ x: Float) -> Float { x / (1 + exp(-x)) }

/// ReLU activation.
private func relu(_ x: Float) -> Float { max(0, x) }

// MARK: - Conformer Relative Positional Encoding

/// Sinusoidal relative positional encoding (Transformer-XL style).
/// Returns positions `[seqLen-1, ..., -(seqLen-1)]` encoded as
/// `[2*seqLen-1, dModel]` interleaved sin/cos.
private func relPositionalEncoding(seqLen: Int, dModel: Int) -> [Float] {
    let posLen = 2 * seqLen - 1
    var pe = [Float](repeating: 0, count: posLen * dModel)
    let half = dModel / 2
    for i in 0..<posLen {
        let pos = Float(seqLen - 1 - i)
        for j in 0..<half {
            let divTerm = exp(Float(j) * (-log(10000.0) / Float(dModel)))
            pe[i * dModel + j] = sinf(pos * divTerm)
            pe[i * dModel + half + j] = cosf(pos * divTerm)
        }
    }
    return pe
}

// MARK: - Weight table

/// Flat weight lookup built from the SafeTensors bundle at load time.
public typealias WeightTable = [String: [Float]]

// MARK: - FastConformer encoder (CPU forward)

/// Full FastConformer forward: ConvSubsampling + N Conformer layers.
/// Returns `(output, outLen)` where output is `[outLen, hiddenSize]`.
private func fastConformerForward(
    input: [Float], nMels: Int, nFrames: Int,
    weights: WeightTable,
    cfg: SortformerFCConfig
) -> (values: [Float], outLen: Int) {
    let dModel = cfg.hiddenSize
    let convC = cfg.subsamplingConvChannels
    let ks = cfg.subsamplingConvKernelSize
    let stride = cfg.subsamplingConvStride
    let pad = (ks - 1) / 2

    // ConvSubsampling:
    // Input: [nMels, nFrames] channel-major. Reshape to [1, nMels, nFrames].
    // layers_0: Conv2d(1 → convC, ks, stride, pad), ReLU
    // layers_2: Conv2d(convC → convC, ks, stride, pad, groups=convC), …
    // layers_3: Conv2d(convC → convC, 1)
    // layers_5: Conv2d(convC → convC, ks, stride, pad, groups=convC)
    // layers_6: Conv2d(convC → convC, 1)
    // linear: (convC * ceil(nMels/8)) → dModel

    let l0w = weights["fc_encoder.subsampling.layers_0.weight"] ?? []
    let l0b = weights["fc_encoder.subsampling.layers_0.bias"]
    let l2w = weights["fc_encoder.subsampling.layers_2.weight"] ?? []
    let l2b = weights["fc_encoder.subsampling.layers_2.bias"]
    let l3w = weights["fc_encoder.subsampling.layers_3.weight"] ?? []
    let l3b = weights["fc_encoder.subsampling.layers_3.bias"]
    let l5w = weights["fc_encoder.subsampling.layers_5.weight"] ?? []
    let l5b = weights["fc_encoder.subsampling.layers_5.bias"]
    let l6w = weights["fc_encoder.subsampling.layers_6.weight"] ?? []
    let l6b = weights["fc_encoder.subsampling.layers_6.bias"]
    let linW = weights["fc_encoder.subsampling.linear.weight"] ?? []
    let linB = weights["fc_encoder.subsampling.linear.bias"]

    // layers_0: Conv2d(1 → convC, (ks,ks), stride=(stride,stride), pad=(pad,pad))
    var (h, curH, curW) = conv2dForward(
        input: input, inC: 1, inH: nMels, inW: nFrames,
        weight: l0w, bias: l0b,
        outC: convC, kH: ks, kW: ks,
        strideH: stride, strideW: stride, padH: pad, padW: pad)
    h = h.map { relu($0) }

    // layers_2: depthwise Conv2d(convC → convC, groups=convC)
    var (h2, curH2, curW2) = conv2dForward(
        input: h, inC: convC, inH: curH, inW: curW,
        weight: l2w, bias: l2b,
        outC: convC, kH: ks, kW: ks,
        strideH: stride, strideW: stride, padH: pad, padW: pad,
        groups: convC)
    // layers_3: pointwise Conv2d(convC → convC, 1x1)
    (h2, curH2, curW2) = conv2dForward(
        input: h2, inC: convC, inH: curH2, inW: curW2,
        weight: l3w, bias: l3b,
        outC: convC, kH: 1, kW: 1,
        strideH: 1, strideW: 1, padH: 0, padW: 0)
    h2 = h2.map { relu($0) }

    // layers_5: depthwise Conv2d(convC → convC, groups=convC)
    var (h3, curH3, curW3) = conv2dForward(
        input: h2, inC: convC, inH: curH2, inW: curW2,
        weight: l5w, bias: l5b,
        outC: convC, kH: ks, kW: ks,
        strideH: stride, strideW: stride, padH: pad, padW: pad,
        groups: convC)
    // layers_6: pointwise Conv2d(convC → convC, 1x1)
    (h3, curH3, curW3) = conv2dForward(
        input: h3, inC: convC, inH: curH3, inW: curW3,
        weight: l6w, bias: l6b,
        outC: convC, kH: 1, kW: 1,
        strideH: 1, strideW: 1, padH: 0, padW: 0)
    h3 = h3.map { relu($0) }

    // h3 shape: [convC, curH3, curW3]. Transpose to [curW3, convC*curH3].
    let outTime = curW3
    let flatFeat = convC * curH3
    var seq = [Float](repeating: 0, count: outTime * flatFeat)
    for t in 0..<outTime {
        for c in 0..<convC {
            for f in 0..<curH3 {
                // src: h3[(c * curH3 + f) * curW3 + t]
                // dst: seq[t * flatFeat + c * curH3 + f] — wait, PyTorch transposes as (b,t,c,f)
                // Reference: h.transposed(0,1,3,2).reshaped(b,t,c*f) so it's (c,f) order
                seq[t * flatFeat + c * curH3 + f] = h3[(c * curH3 + f) * curW3 + t]
            }
        }
    }

    // Linear projection: [flatFeat → dModel]
    let linLayer = VADLinear(weight: linW, bias: linB, inFeatures: flatFeat, outFeatures: dModel)
    var embeddings = linLayer.applyRows(seq, rows: outTime)  // [outTime, dModel]

    // Compute output lengths (3× floor((L-1)/2+1)).
    var outLen = nFrames
    for _ in 0..<3 { outLen = (outLen - 1) / 2 + 1 }
    let diarLen = min(outLen, outTime)

    // Scale input if configured.
    if cfg.scaleInput {
        let scale = Float(dModel).squareRoot()
        for i in embeddings.indices { embeddings[i] *= scale }
    }

    // Relative positional encoding: [2*diarLen-1, dModel].
    let posEmb = relPositionalEncoding(seqLen: diarLen, dModel: dModel)

    // Conformer layers.
    for layerIdx in 0..<cfg.numLayers {
        let prefix = "fc_encoder.layers.\(layerIdx)"
        embeddings = conformerLayerForward(
            x: embeddings, seqLen: diarLen, dModel: dModel,
            posEmb: posEmb, weights: weights, prefix: prefix,
            cfg: cfg)
    }

    // Return [diarLen, dModel].
    return (embeddings, diarLen)
}

/// One Conformer layer: FF1 → Self-Attn → Conv → FF2 → LN.
private func conformerLayerForward(
    x: [Float], seqLen: Int, dModel: Int,
    posEmb: [Float], weights: WeightTable, prefix: String,
    cfg: SortformerFCConfig
) -> [Float] {
    let ffFactor: Float = 0.5
    let nHeads = cfg.numHeads
    let headDim = dModel / nHeads
    let nFf = cfg.intermediateSize

    // ── FF1 sub-block ──────────────────────────────────────────────────
    let norm1 = vadLNFromWeights(weights, prefix: "\(prefix).norm_feed_forward1",
                                 dim: dModel)
    let ff1W1 = weights["\(prefix).feed_forward1.linear1.weight"] ?? []
    let ff1B1 = weights["\(prefix).feed_forward1.linear1.bias"]
    let ff1W2 = weights["\(prefix).feed_forward1.linear2.weight"] ?? []
    let ff1B2 = weights["\(prefix).feed_forward1.linear2.bias"]
    let lin1 = VADLinear(weight: ff1W1, bias: ff1B1, inFeatures: dModel, outFeatures: nFf)
    let lin2 = VADLinear(weight: ff1W2, bias: ff1B2, inFeatures: nFf, outFeatures: dModel)
    var residual = x
    let n1 = norm1.applyRows(x, rows: seqLen)
    let ff1mid = lin1.applyRows(n1, rows: seqLen).map { silu($0) }
    let ff1out = lin2.applyRows(ff1mid, rows: seqLen)
    for i in 0..<residual.count { residual[i] += ff1out[i] * ffFactor }

    // ── Self-attention sub-block ───────────────────────────────────────
    let normAttn = vadLNFromWeights(weights, prefix: "\(prefix).norm_self_att", dim: dModel)
    let qW = weights["\(prefix).self_attn.q_proj.weight"] ?? []
    let qB = cfg.attentionBias ? weights["\(prefix).self_attn.q_proj.bias"] : nil
    let kW = weights["\(prefix).self_attn.k_proj.weight"] ?? []
    let kB = cfg.attentionBias ? weights["\(prefix).self_attn.k_proj.bias"] : nil
    let vW = weights["\(prefix).self_attn.v_proj.weight"] ?? []
    let vB = cfg.attentionBias ? weights["\(prefix).self_attn.v_proj.bias"] : nil
    let oW = weights["\(prefix).self_attn.o_proj.weight"] ?? []
    let oB = cfg.attentionBias ? weights["\(prefix).self_attn.o_proj.bias"] : nil
    let relKW = weights["\(prefix).self_attn.relative_k_proj.weight"] ?? []
    let biasU = weights["\(prefix).self_attn.bias_u"] ?? []
    let biasV = weights["\(prefix).self_attn.bias_v"] ?? []

    let normedAttn = normAttn.applyRows(residual, rows: seqLen)
    let qL = VADLinear(weight: qW, bias: qB, inFeatures: dModel, outFeatures: dModel)
    let kL = VADLinear(weight: kW, bias: kB, inFeatures: dModel, outFeatures: dModel)
    let vL = VADLinear(weight: vW, bias: vB, inFeatures: dModel, outFeatures: dModel)
    let oL = VADLinear(weight: oW, bias: oB, inFeatures: dModel, outFeatures: dModel)
    let relKL = VADLinear(weight: relKW, bias: nil, inFeatures: dModel, outFeatures: dModel)

    let q = qL.applyRows(normedAttn, rows: seqLen)
    let k = kL.applyRows(normedAttn, rows: seqLen)
    let v = vL.applyRows(normedAttn, rows: seqLen)
    let p = relKL.applyRows(posEmb, rows: posEmb.count / dModel)

    let attnOut = relPositionMultiHeadAttention(
        q: q, k: k, v: v, p: p,
        biasU: biasU, biasV: biasV,
        seqLen: seqLen, posLen: posEmb.count / dModel,
        nHeads: nHeads, headDim: headDim)
    let attnProj = oL.applyRows(attnOut, rows: seqLen)
    for i in 0..<residual.count { residual[i] += attnProj[i] }

    // ── Conformer conv sub-block ───────────────────────────────────────
    let normConv = vadLNFromWeights(weights, prefix: "\(prefix).norm_conv", dim: dModel)
    let pw1W = weights["\(prefix).conv.pointwise_conv1.weight"] ?? []
    let pw1B = weights["\(prefix).conv.pointwise_conv1.bias"]
    let dwW  = weights["\(prefix).conv.depthwise_conv.weight"] ?? []
    let dwB  = weights["\(prefix).conv.depthwise_conv.bias"]
    let pw2W = weights["\(prefix).conv.pointwise_conv2.weight"] ?? []
    let pw2B = weights["\(prefix).conv.pointwise_conv2.bias"]
    let bnW  = weights["\(prefix).conv.norm.weight"] ?? []
    let bnB  = weights["\(prefix).conv.norm.bias"] ?? []
    let bnMu = weights["\(prefix).conv.norm.running_mean"] ?? []
    let bnVar = weights["\(prefix).conv.norm.running_var"] ?? []

    let convKern = cfg.convKernelSize
    let normedConv = normConv.applyRows(residual, rows: seqLen)
    // Transpose to channel-major [dModel, seqLen] for Conv1d.
    var chMajor = [Float](repeating: 0, count: dModel * seqLen)
    for t in 0..<seqLen { for d in 0..<dModel { chMajor[d * seqLen + t] = normedConv[t * dModel + d] } }

    // Pointwise conv1: [dModel → dModel*2], K=1.
    var (cv, cvLen) = conv1dForward(input: chMajor, inC: dModel, length: seqLen,
                                    weight: pw1W, bias: pw1B, outC: dModel * 2, kernelSize: 1)
    // GLU: split into two halves, apply sigmoid to second half.
    var gluOut = [Float](repeating: 0, count: dModel * cvLen)
    for c in 0..<dModel {
        for t in 0..<cvLen {
            let a = cv[c * cvLen + t]
            let b2 = cv[(c + dModel) * cvLen + t]
            gluOut[c * cvLen + t] = a * VADMath.sigmoid(b2)
        }
    }

    // Depthwise conv: [dModel → dModel], K=convKern, pad=(K-1)/2, groups=dModel.
    let dwPad = (convKern - 1) / 2
    var (dw, dwLen) = conv1dForward(input: gluOut, inC: dModel, length: cvLen,
                                    weight: dwW, bias: dwB,
                                    outC: dModel, kernelSize: convKern,
                                    stride: 1, padding: dwPad, groups: dModel)

    // Batch norm: input is [dModel, dwLen], norm operates per-feature over time.
    // Transpose to [dwLen, dModel] for batchNorm1dForward.
    var dwTime = [Float](repeating: 0, count: dwLen * dModel)
    for d in 0..<dModel { for t in 0..<dwLen { dwTime[t * dModel + d] = dw[d * dwLen + t] } }
    if !bnW.isEmpty {
        dwTime = batchNorm1dForward(input: dwTime, rows: dwLen, features: dModel,
                                    weight: bnW, bias: bnB, runningMean: bnMu, runningVar: bnVar)
    }
    // SiLU.
    dwTime = dwTime.map { silu($0) }
    // Transpose back to channel-major for pointwise conv2.
    for d in 0..<dModel { for t in 0..<dwLen { dw[d * dwLen + t] = dwTime[t * dModel + d] } }

    // Pointwise conv2: [dModel → dModel], K=1.
    var (pw2, _) = conv1dForward(input: dw, inC: dModel, length: dwLen,
                                 weight: pw2W, bias: pw2B, outC: dModel, kernelSize: 1)
    // Transpose back to [seqLen, dModel] and add residual.
    let pw2Len = min(dwLen, seqLen)
    for t in 0..<pw2Len {
        for d in 0..<dModel { residual[t * dModel + d] += pw2[d * pw2Len + t] }
    }

    // ── FF2 sub-block ──────────────────────────────────────────────────
    let norm3 = vadLNFromWeights(weights, prefix: "\(prefix).norm_feed_forward2", dim: dModel)
    let ff2W1 = weights["\(prefix).feed_forward2.linear1.weight"] ?? []
    let ff2B1 = weights["\(prefix).feed_forward2.linear1.bias"]
    let ff2W2 = weights["\(prefix).feed_forward2.linear2.weight"] ?? []
    let ff2B2 = weights["\(prefix).feed_forward2.linear2.bias"]
    let lin3 = VADLinear(weight: ff2W1, bias: ff2B1, inFeatures: dModel, outFeatures: nFf)
    let lin4 = VADLinear(weight: ff2W2, bias: ff2B2, inFeatures: nFf, outFeatures: dModel)
    let n3 = norm3.applyRows(residual, rows: seqLen)
    let ff2mid = lin3.applyRows(n3, rows: seqLen).map { silu($0) }
    let ff2out = lin4.applyRows(ff2mid, rows: seqLen)
    for i in 0..<residual.count { residual[i] += ff2out[i] * ffFactor }

    // ── Final layer norm ───────────────────────────────────────────────
    let normOut = vadLNFromWeights(weights, prefix: "\(prefix).norm_out", dim: dModel)
    return normOut.applyRows(residual, rows: seqLen)
}

/// Relative position multi-head attention (Transformer-XL style, CPU).
/// `q`, `k`, `v` are `[seqLen, dModel]`. `p` is `[posLen, dModel]`.
/// `biasU`, `biasV` are `[nHeads, headDim]`.
private func relPositionMultiHeadAttention(
    q: [Float], k: [Float], v: [Float], p: [Float],
    biasU: [Float], biasV: [Float],
    seqLen: Int, posLen: Int,
    nHeads: Int, headDim: Int
) -> [Float] {
    let dModel = nHeads * headDim
    let scale = Float(headDim).squareRoot()
    var out = [Float](repeating: 0, count: seqLen * dModel)

    // For each head: matrix_AC[i,j] = (q_i + bu_h) · k_j / scale
    //                matrix_BD[i,j] = relShift((q_i + bv_h) · p_j / scale)
    // Scores = matrix_AC + matrix_BD[:, :, :, :seqLen]
    for head in 0..<nHeads {
        let hOff = head * headDim
        let uOff = head * headDim  // biasU[head, :headDim]
        let vOff = head * headDim  // biasV[head, :headDim]

        // Compute score matrix.
        var scores = [Float](repeating: 0, count: seqLen * seqLen)
        for i in 0..<seqLen {
            let qBase = i * dModel + hOff
            // matrix_AC scores.
            for j in 0..<seqLen {
                let kBase = j * dModel + hOff
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += (q[qBase + d] + biasU[uOff + d]) * k[kBase + d]
                }
                scores[i * seqLen + j] = dot / scale
            }
            // matrix_BD scores (relative shift).
            // (q_i + bv) · p_j for j in 0..posLen-1, then relShift.
            var bdRow = [Float](repeating: 0, count: posLen)
            for j in 0..<posLen {
                let pBase = j * dModel + hOff
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += (q[qBase + d] + biasV[vOff + d]) * p[pBase + d]
                }
                bdRow[j] = dot / scale
            }
            // relShift: left-pad by 1, reshape [posLen+1, seqLen], drop row 0,
            // flatten back, keep first seqLen values.
            // Equiv: shifted[i,j] = bdRow[seqLen-1-i+j] for j in 0..seqLen
            for j in 0..<seqLen {
                let relIdx = seqLen - 1 - i + j
                if relIdx >= 0 && relIdx < posLen {
                    scores[i * seqLen + j] += bdRow[relIdx]
                }
            }
        }

        // Softmax over j (last dim).
        for i in 0..<seqLen {
            VADMath.softmaxInPlace(&scores, range: (i * seqLen)..<(i * seqLen + seqLen))
        }

        // Weighted sum of v.
        for i in 0..<seqLen {
            for j in 0..<seqLen {
                let w = scores[i * seqLen + j]
                if w == 0 { continue }
                let vBase = j * dModel + hOff
                let outBase = i * dModel + hOff
                for d in 0..<headDim { out[outBase + d] += w * v[vBase + d] }
            }
        }
    }
    return out
}

/// Build a `VADLayerNorm` from the weight table.
private func vadLNFromWeights(
    _ w: WeightTable, prefix: String, dim: Int, eps: Float = 1e-5
) -> VADLayerNorm {
    let weight = w["\(prefix).weight"] ?? [Float](repeating: 1, count: dim)
    let bias   = w["\(prefix).bias"]   ?? [Float](repeating: 0, count: dim)
    return VADLayerNorm(weight: weight, bias: bias, dim: dim, eps: eps)
}

// MARK: - Transformer (BART-style) encoder

/// Learned positional embedding lookup.
private func addLearnedPositionEmb(
    _ x: [Float], seqLen: Int, dModel: Int, embedTable: [Float]
) -> [Float] {
    var out = x
    for t in 0..<seqLen {
        let eBase = t * dModel
        let xBase = t * dModel
        guard eBase + dModel <= embedTable.count else { continue }
        for d in 0..<dModel { out[xBase + d] += embedTable[eBase + d] }
    }
    return out
}

/// BART-style transformer encoder (post-LN, ReLU FFN).
private func transformerEncoderForward(
    x: [Float], seqLen: Int, dModel: Int,
    weights: WeightTable,
    cfg: SortformerTFConfig
) -> [Float] {
    var h = x
    let headDim = dModel / cfg.numHeads

    for layerIdx in 0..<cfg.numLayers {
        let prefix = "tf_encoder.layers.\(layerIdx)"
        let attnLN = vadLNFromWeights(weights, prefix: "\(prefix).self_attn_layer_norm",
                                      dim: dModel, eps: cfg.layerNormEps)
        let finalLN = vadLNFromWeights(weights, prefix: "\(prefix).final_layer_norm",
                                       dim: dModel, eps: cfg.layerNormEps)

        let qW = weights["\(prefix).self_attn.q_proj.weight"] ?? []
        let qB = weights["\(prefix).self_attn.q_proj.bias"]
        let kW = weights["\(prefix).self_attn.k_proj.weight"] ?? []
        let kB = cfg.kProjBias ? weights["\(prefix).self_attn.k_proj.bias"] : nil
        let vW = weights["\(prefix).self_attn.v_proj.weight"] ?? []
        let vB = weights["\(prefix).self_attn.v_proj.bias"]
        let oW = weights["\(prefix).self_attn.out_proj.weight"] ?? []
        let oB = weights["\(prefix).self_attn.out_proj.bias"]

        let fc1W = weights["\(prefix).fc1.weight"] ?? []
        let fc1B = weights["\(prefix).fc1.bias"]
        let fc2W = weights["\(prefix).fc2.weight"] ?? []
        let fc2B = weights["\(prefix).fc2.bias"]

        // Post-LN self-attention sub-block: Attn(x) + x → LN.
        let qL = VADLinear(weight: qW, bias: qB, inFeatures: dModel, outFeatures: dModel)
        let kL = VADLinear(weight: kW, bias: kB, inFeatures: dModel, outFeatures: dModel)
        let vL = VADLinear(weight: vW, bias: vB, inFeatures: dModel, outFeatures: dModel)
        let oL = VADLinear(weight: oW, bias: oB, inFeatures: dModel, outFeatures: dModel)
        let q = qL.applyRows(h, rows: seqLen)
        let k = kL.applyRows(h, rows: seqLen)
        let v = vL.applyRows(h, rows: seqLen)
        let scale = Float(headDim).squareRoot()
        let attn = vadMultiHeadAttention(q: q, k: k, v: v, seqLen: seqLen,
                                         numHeads: cfg.numHeads, headDim: headDim, scale: scale)
        let attnProj = oL.applyRows(attn, rows: seqLen)
        var h2 = [Float](repeating: 0, count: seqLen * dModel)
        for i in 0..<h.count { h2[i] = h[i] + attnProj[i] }
        h2 = attnLN.applyRows(h2, rows: seqLen)

        // Post-LN FFN sub-block: FC2(ReLU(FC1(x))) + x → LN.
        let fc1L = VADLinear(weight: fc1W, bias: fc1B, inFeatures: dModel, outFeatures: cfg.ffnDim)
        let fc2L = VADLinear(weight: fc2W, bias: fc2B, inFeatures: cfg.ffnDim, outFeatures: dModel)
        let mid = fc1L.applyRows(h2, rows: seqLen).map { relu($0) }
        let ffOut = fc2L.applyRows(mid, rows: seqLen)
        var h3 = [Float](repeating: 0, count: seqLen * dModel)
        for i in 0..<h2.count { h3[i] = h2[i] + ffOut[i] }
        h = finalLN.applyRows(h3, rows: seqLen)
    }
    return h
}

// MARK: - SortformerModules (speaker sigmoid head)

/// Forward through the Sortformer output modules:
/// `ReLU(FC(h))` → `ReLU` → single_hidden_to_spks → sigmoid.
private func sortformerSpeakerSigmoids(
    h: [Float], seqLen: Int, tfDModel: Int, numSpeakers: Int,
    weights: WeightTable
) -> [Float] {
    let fhW = weights["sortformer_modules.first_hidden_to_hidden.weight"] ?? []
    let fhB = weights["sortformer_modules.first_hidden_to_hidden.bias"]
    let spkW = weights["sortformer_modules.single_hidden_to_spks.weight"] ?? []
    let spkB = weights["sortformer_modules.single_hidden_to_spks.bias"]

    // First ReLU (applied before the linear in forwardSpeakerSigmoids).
    var h2 = h.map { relu($0) }
    // First hidden → hidden linear.
    let fhL = VADLinear(weight: fhW, bias: fhB, inFeatures: tfDModel, outFeatures: tfDModel)
    h2 = fhL.applyRows(h2, rows: seqLen)
    // Second ReLU.
    h2 = h2.map { relu($0) }
    // Single hidden → speakers linear.
    let spkL = VADLinear(weight: spkW, bias: spkB, inFeatures: tfDModel, outFeatures: numSpeakers)
    let logits = spkL.applyRows(h2, rows: seqLen)
    // Sigmoid per element.
    return logits.map { VADMath.sigmoid($0) }
}

// MARK: - Main model

/// Loaded Sortformer diarization model.
///
/// Produces per-frame, per-speaker speech-activity probabilities from raw
/// audio. This is a VAD-family model: reached via `VADModelRegistry`, not
/// `ModelRegistry`. No `LanguageModel` protocol conformance.
public final class SortformerModel: @unchecked Sendable {
    public let config: SortformerConfig

    /// Host-resident weight table (all tensors copied to `[Float]` at load
    /// time for the CPU forward pass).
    private let weights: WeightTable

    init(config: SortformerConfig, weights: WeightTable) {
        self.config = config
        self.weights = weights
    }

    // MARK: - Forward pass

    /// Run the full model over a mono `audio` clip (16 kHz PCM).
    ///
    /// Returns `DiarizationOutput` with:
    ///  - `speakerProbabilities`: `[[Float]]` — one row per frame,
    ///    each row is `numSpeakers` sigmoid probabilities in `[0, 1]`.
    ///  - `frameStrideSamples`: hop length × subsampling factor.
    ///  - `segments`: threshold-segmented speaker runs (threshold=0.5).
    ///
    /// - Parameters:
    ///   - audio: mono PCM samples (any length).
    ///   - sampleRate: must be 16000 Hz.
    ///   - threshold: sigmoid threshold for segment extraction.
    public func detect(
        audio: [Float],
        sampleRate: Int = 16_000,
        threshold: Float = 0.5
    ) -> DiarizationOutput {
        guard !audio.isEmpty else {
            return DiarizationOutput(
                speakerProbabilities: [], frameStrideSamples: frameStride,
                sampleRate: sampleRate, segments: [], numSpeakers: config.numSpeakers)
        }

        // 1. Mel features: [nMels, nFrames] channel-major.
        let (melValues, nMels, nFrames) = sortformerMelFeatures(
            waveform: audio, proc: config.processor)
        guard nFrames > 0 else {
            return DiarizationOutput(
                speakerProbabilities: [], frameStrideSamples: frameStride,
                sampleRate: sampleRate, segments: [], numSpeakers: config.numSpeakers)
        }

        // 2. FastConformer encoder: [diarLen, fcDModel].
        let (fcOut, diarLen) = fastConformerForward(
            input: melValues, nMels: nMels, nFrames: nFrames,
            weights: weights, cfg: config.fcEncoder)

        // 3. Projection (encoder_proj): [diarLen, tfDModel].
        let tfDModel = config.modules.tfDModel
        let fcDModel = config.fcEncoder.hiddenSize
        let projW = weights["sortformer_modules.encoder_proj.weight"] ?? []
        let projB = weights["sortformer_modules.encoder_proj.bias"]
        let projL = VADLinear(weight: projW, bias: projB,
                              inFeatures: fcDModel, outFeatures: tfDModel)
        var projected = projL.applyRows(fcOut, rows: diarLen)  // [diarLen, tfDModel]

        // 4. Add learned positional embeddings.
        let posTable = weights["tf_encoder.embed_positions.weight"] ?? []
        projected = addLearnedPositionEmb(projected, seqLen: diarLen,
                                          dModel: tfDModel, embedTable: posTable)

        // 5. Transformer encoder: [diarLen, tfDModel].
        let transOut = transformerEncoderForward(
            x: projected, seqLen: diarLen, dModel: tfDModel,
            weights: weights, cfg: config.tfEncoder)

        // 6. Speaker sigmoid head: [diarLen, numSpeakers].
        let nSpk = config.numSpeakers
        let probs = sortformerSpeakerSigmoids(
            h: transOut, seqLen: diarLen, tfDModel: tfDModel,
            numSpeakers: nSpk, weights: weights)

        // 7. Build per-frame output rows.
        var speakerRows = [[Float]](repeating: [Float](repeating: 0, count: nSpk),
                                   count: diarLen)
        for f in 0..<diarLen {
            for s in 0..<nSpk { speakerRows[f][s] = probs[f * nSpk + s] }
        }

        // 8. Threshold to segments.
        let frameDur = Float(frameStride) / Float(sampleRate)
        let segments = Self.probsToSegments(
            speakerRows, frameDuration: frameDur, threshold: threshold)

        return DiarizationOutput(
            speakerProbabilities: speakerRows,
            frameStrideSamples: frameStride,
            sampleRate: sampleRate,
            segments: segments,
            numSpeakers: nSpk)
    }

    /// Audio samples per output diarization frame.
    public var frameStride: Int {
        config.processor.hopLength * config.modules.subsamplingFactor
    }

    // MARK: - Post-processing

    /// Convert per-frame speaker probabilities to contiguous speech
    /// segments using a simple per-speaker threshold.
    public static func probsToSegments(
        _ probs: [[Float]],
        frameDuration: Float,
        threshold: Float = 0.5
    ) -> [DiarizationSegment] {
        guard !probs.isEmpty else { return [] }
        let nSpk = probs[0].count
        var segments = [DiarizationSegment]()

        for spk in 0..<nSpk {
            var inSpeech = false
            var startFrame = 0

            for (f, row) in probs.enumerated() {
                let active = row[spk] > threshold
                if active && !inSpeech {
                    inSpeech = true
                    startFrame = f
                } else if !active && inSpeech {
                    inSpeech = false
                    let seg = DiarizationSegment(
                        startSeconds: Double(Float(startFrame) * frameDuration),
                        endSeconds: Double(Float(f) * frameDuration),
                        speaker: spk)
                    segments.append(seg)
                }
            }
            if inSpeech {
                let seg = DiarizationSegment(
                    startSeconds: Double(Float(startFrame) * frameDuration),
                    endSeconds: Double(Float(probs.count) * frameDuration),
                    speaker: spk)
                segments.append(seg)
            }
        }

        segments.sort { $0.startSeconds < $1.startSeconds }
        return segments
    }

    // MARK: - Weight remapping (checkpoint sanitization)

    /// Remap raw checkpoint keys to the flat names used by the weight
    /// table. Drops `num_batches_tracked` tensors and handles MLX ↔
    /// PyTorch layout transpositions:
    ///  - Conv2d weight: MLX `[outC, H, W, inC]` → PyTorch `[outC, inC, H, W]`
    ///  - Conv1d weight: MLX `[outC, K, inC]` → PyTorch `[outC, inC, K]`
    public static func remapAndTranspose(_ rawWeights: [String: Tensor]) -> WeightTable {
        var table = WeightTable()

        for (key, tensor) in rawWeights {
            if key.contains("num_batches_tracked") { continue }

            // Rename dot-indexed subsampling layers (layers.0 → layers_0 etc.)
            var k = key
            if k.contains("subsampling.layers.") {
                k = k.replacingOccurrences(of: "subsampling.layers.", with: "subsampling.layers_")
            }

            var floats = tensor.toFloatArray()

            // Conv2d weight transposition: MLX [outC, H, W, inC] → PyTorch [outC, inC, H, W].
            if k.contains("subsampling") && k.contains("weight") && !k.contains("linear") {
                if tensor.shape.count == 4 {
                    let (outC, H, W, inC) = (tensor.shape[0], tensor.shape[1],
                                             tensor.shape[2], tensor.shape[3])
                    var transposed = [Float](repeating: 0, count: floats.count)
                    for o in 0..<outC {
                        for ic in 0..<inC {
                            for h in 0..<H {
                                for w in 0..<W {
                                    // src: [o, H, W, inC] row-major → index (o*H*W + h*W + w)*inC + ic
                                    // dst: [o, ic, H, W] row-major → index ((o*inC + ic)*H + h)*W + w
                                    let src = (o * H * W + h * W + w) * inC + ic
                                    let dst = ((o * inC + ic) * H + h) * W + w
                                    transposed[dst] = floats[src]
                                }
                            }
                        }
                    }
                    floats = transposed
                }
            }

            // Conv1d weight transposition: MLX [outC, K, inC] → PyTorch [outC, inC, K].
            if (k.contains("pointwise_conv1") || k.contains("pointwise_conv2")
                    || k.contains("depthwise_conv")) && k.contains("weight") {
                if tensor.shape.count == 3 {
                    let (outC, K, inC) = (tensor.shape[0], tensor.shape[1], tensor.shape[2])
                    var transposed = [Float](repeating: 0, count: floats.count)
                    for o in 0..<outC {
                        for ic in 0..<inC {
                            for kk in 0..<K {
                                // src: [o, K, inC] → o*K*inC + kk*inC + ic
                                // dst: [o, inC, K] → o*inC*K + ic*K + kk
                                transposed[o * inC * K + ic * K + kk] = floats[o * K * inC + kk * inC + ic]
                            }
                        }
                    }
                    floats = transposed
                }
            }

            table[k] = floats
        }
        return table
    }

    // MARK: - Loading

    /// Load from a local snapshot directory containing `config.json` and
    /// one or more `.safetensors` weight files.
    public static func loadFromDirectory(
        _ directory: URL,
        device: Device = .shared
    ) throws -> SortformerModel {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { throw SortformerError.configNotFound(directory) }
        let config = SortformerConfig(from: raw)

        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        var rawWeights = [String: Tensor]()
        for key in bundle.allKeys {
            rawWeights[key] = try bundle.tensor(named: key)
        }
        let weights = remapAndTranspose(rawWeights)
        return SortformerModel(config: config, weights: weights)
    }

    /// Download (or hit cache) a Sortformer checkpoint from HuggingFace
    /// and load it.
    public static func fromPretrained(
        _ idOrPath: String,
        device: Device = .shared
    ) async throws -> SortformerModel {
        let dir = try await ModelLocator().resolve(idOrPath: idOrPath)
        return try loadFromDirectory(dir, device: device)
    }
}
