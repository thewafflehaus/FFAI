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
// FastVLM vision tower internals — FastViTHD architecture.
//
// This file holds the vision-tower implementation for Apple's FastVLM
// family: configuration decoding, all block parameter structs (ConvFFN,
// RepMixer, Attention, ConvStem, PatchEmbed, conv_exp, RepCPE), the full
// FastVLMVisionTower load + encode paths, the mlp2x_gelu projector, and
// the composed-tower facade that wires them into a VisionEncoder.
//
// The family orchestrator (load entrypoint, `FastVLMError`, `FastVLM`
// enum with `modelTypes` / `architectures` / `load()`) lives in
// `Models/FastVLM.swift`.

import Foundation
import Metal

// ─── Vision configuration ────────────────────────────────────────────

/// Static shape + hyper-parameters of the FastViTHD vision tower,
/// decoded from the checkpoint's `vision_config`.
public struct FastVLMVisionConfig {
    /// Channels at each stage: [96, 192, 384, 768, 1536].
    let embedDims: [Int]
    /// Number of transformer blocks per stage: [2, 12, 24, 4, 2].
    let layers: [Int]
    /// Token mixer type per stage: "repmixer" or "attention".
    let tokenMixers: [String]
    /// Whether each stage is followed by a PatchEmbed downsampler.
    let downSamples: [Bool]
    /// Positional embedding shapes per stage — nil means no CPE.
    /// Non-nil triggers a RepCPE before that stage.
    let posEmbShapes: [[Int]?]
    /// MLP expansion ratio per stage (all 4 in the 0.5B model).
    let mlpRatios: [Int]
    /// Square input image size (1024 for mlx-community/FastVLM-0.5B-bf16).
    let imageSize: Int
    /// Effective patch size = total downsampling factor = 64 (4x stem × 4x PEs).
    let patchSize: Int
    /// RepMixer depthwise kernel size (3 for FastViTHD-T).
    let repMixerKernelSize: Int
    /// PatchEmbed depthwise kernel size (7).
    let downPatchSize: Int
    /// PatchEmbed stride (2).
    let downStride: Int
    /// Channel expansion ratio for conv_exp: out_ch = embedDims.last * clsRatio.
    let clsRatio: Float
    /// Multi-modal hidden size = conv_exp output channels.
    let mmHiddenSize: Int
    /// Projection dim for the classification head (unused in VLM mode).
    let projectionDim: Int
    /// Layer-scale initializer value (1e-5).
    let layerScaleInitValue: Float
    /// Number of stages.
    var nStages: Int { layers.count }

    /// Derive the spatial resolution at each stage given input image size.
    /// Returns (H, W) pairs for stage outputs [0, nStages-1].
    func spatialResolutions() -> [(Int, Int)] {
        // ConvolutionalStem: 3 blocks with strides [2, 2, 1] = 4× total.
        var h = imageSize / 4
        var w = imageSize / 4
        var resolutions: [(Int, Int)] = []
        for i in 0 ..< nStages {
            resolutions.append((h, w))
            // After this stage, apply PatchEmbed downsampling if needed.
            if i < nStages - 1 && downSamples[i] {
                // LargeKernelBlock with stride=downStride and same padding:
                // out = ceil(in / stride) = (in + stride - 1) / stride.
                h = (h + downStride - 1) / downStride
                w = (w + downStride - 1) / downStride
            }
        }
        return resolutions
    }

    static func decode(_ c: ModelConfig) throws -> FastVLMVisionConfig {
        guard let embedDims = c.intArray("embed_dims"),
            let layers = c.intArray("layers"),
            let tokenMixers = c.raw["token_mixers"] as? [String],
            let downSamples = c.raw["downsamples"] as? [Bool],
            let mlpRatios = c.intArray("mlp_ratios"),
            let imageSize = c.int("image_size"),
            let patchSize = c.int("patch_size"),
            let downPatchSize = c.int("down_patch_size"),
            let downStride = c.int("down_stride")
        else { throw FastVLMError.missingConfig }

        // pos_embs_shapes is [[Int]?] — array of optional int arrays.
        // JSON represents this as `[null, null, null, [7,7], [7,7]]`.
        var posEmbShapes: [[Int]?] = Array(repeating: nil, count: layers.count)
        if let raw = c.raw["pos_embs_shapes"] as? [Any] {
            for (i, elem) in raw.enumerated() where i < layers.count {
                if let arr = elem as? [Int] {
                    posEmbShapes[i] = arr
                }
                // nil / NSNull → leave posEmbShapes[i] = nil
            }
        }

        let clsRatio = Float(c.float("cls_ratio") ?? 2.0)
        let mmHidden = c.int("intermediate_size") ?? c.int("hidden_size") ?? 3072
        // intermediate_size in vision_config = mm_hidden_size (3072 for 0.5B).
        // Alternatively, derivable as int(embedDims.last * clsRatio).
        let derivedMmHidden = Int(Float(embedDims.last!) * clsRatio)

        return FastVLMVisionConfig(
            embedDims: embedDims,
            layers: layers,
            tokenMixers: tokenMixers,
            downSamples: downSamples,
            posEmbShapes: posEmbShapes,
            mlpRatios: mlpRatios,
            imageSize: imageSize,
            patchSize: patchSize,
            repMixerKernelSize: c.int("repmixer_kernel_size") ?? 3,
            downPatchSize: downPatchSize,
            downStride: downStride,
            clsRatio: clsRatio,
            mmHiddenSize: derivedMmHidden,
            projectionDim: c.int("projection_dim") ?? 768,
            layerScaleInitValue: Float(c.float("layer_scale_init_value") ?? 1e-5))
    }
}

// ─── Batch-norm folding ───────────────────────────────────────────────

/// BatchNorm parameters folded into (scale, bias) for use in a conv.
/// At inference the ConvFFN DW conv stores BN running stats; we fold:
///   scale[c] = gamma[c] / sqrt(var[c] + eps)
///   bias[c]  = beta[c] - mean[c] * scale[c]
/// The folded (scale, bias) are applied per-channel after the convolution.
struct FoldedBN {
    let scale: [Float]  // [C]
    let bias: [Float]  // [C]

    init(
        weight: [Float], bias bnBias: [Float],
        runningMean: [Float], runningVar: [Float], eps: Float = 1e-5
    ) {
        let n = weight.count
        var s = [Float](repeating: 0, count: n)
        var b = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            s[i] = weight[i] / sqrtf(runningVar[i] + eps)
            b[i] = bnBias[i] - runningMean[i] * s[i]
        }
        self.scale = s
        self.bias = b
    }
}

// ─── NHWC convolution helpers ─────────────────────────────────────────

/// CPU depthwise convolution in NHWC format.
/// Weight layout: [outC, kH, kW, 1] OHWI (depthwise: inC_per_group = 1).
/// Applies `padding` on all sides (same-padding when padding = (kH-1)/2).
/// Parallelized over batch × channel using DispatchQueue.concurrentPerform.
///
/// Used for all depthwise (groups = channels) convolutions in FastViTHD,
/// since FFAI's Ops.conv2d is NCHW-only and doesn't support grouped conv.
private func depthwiseConvNHWC(
    _ input: [Float], w: [Float], bias: [Float]?,
    B: Int, H: Int, W: Int, C: Int,
    kH: Int, kW: Int, stride: Int, padding: Int
) -> ([Float], Int, Int) {
    let outH = (H + 2 * padding - kH) / stride + 1
    let outW = (W + 2 * padding - kW) / stride + 1
    var output = [Float](repeating: 0, count: B * outH * outW * C)
    let totalWork = B * C
    output.withUnsafeMutableBufferPointer { outBuf in
        let outPtr = outBuf.baseAddress!
        input.withUnsafeBufferPointer { inBuf in
            w.withUnsafeBufferPointer { wBuf in
                let inPtr = inBuf.baseAddress!
                let wPtr = wBuf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: totalWork) { work in
                    let b = work / C
                    let c = work % C
                    for oh in 0 ..< outH {
                        for ow in 0 ..< outW {
                            var acc: Float = 0
                            let ih0 = oh * stride - padding
                            let iw0 = ow * stride - padding
                            for ky in 0 ..< kH {
                                let ih = ih0 + ky
                                if ih < 0 || ih >= H { continue }
                                for kx in 0 ..< kW {
                                    let iw = iw0 + kx
                                    if iw < 0 || iw >= W { continue }
                                    let inIdx = ((b * H + ih) * W + iw) * C + c
                                    // Weight is OHWI depthwise: [c, ky, kx, 0]
                                    let wIdx = (c * kH + ky) * kW + kx
                                    acc += inPtr[inIdx] * wPtr[wIdx]
                                }
                            }
                            let outIdx = ((b * outH + oh) * outW + ow) * C + c
                            outPtr[outIdx] = acc + (bias?[c] ?? 0)
                        }
                    }
                }
            }
        }
    }
    return (output, outH, outW)
}

/// CPU pointwise (1×1) convolution as matrix-multiply, GPU-accelerated.
/// Input NHWC [B, H, W, inC] → output NHWC [B, H, W, outC].
/// Weight is OHWI [outC, 1, 1, inC]; we load it as a 2D matrix [outC, inC].
/// Flattens spatial to [B*H*W, inC], calls Ops.gemm, then adds bias CPU-side.
private func pointwiseConvNHWC(
    _ input: Tensor,  // flat [B*H*W*inC]
    weight2D: Tensor,  // [outC, inC]
    bias: [Float]?,
    BHW: Int, outC: Int, device: Device
) -> [Float] {
    let cmd = device.makeCommandBuffer()
    let projected = Ops.gemm(weight: weight2D, input: input, nRows: BHW, on: cmd)
    cmd.commit()
    cmd.waitUntilCompleted()
    var result = projected.toFloatArray()
    // Broadcast-add bias across all rows.
    if let b = bias {
        for row in 0 ..< BHW {
            for c in 0 ..< outC {
                result[row * outC + c] += b[c]
            }
        }
    }
    return result
}

/// Copy a [Float] array into a Tensor with matching shape.
private func floatToTensor(
    _ vals: [Float], shape: [Int],
    dtype: DType, device: Device
) -> Tensor {
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    ImagePreprocessing.copyFloats(vals, into: t)
    return t
}

/// Squeeze-Excite gate in NHWC:
///   reduce: [C] → [C_r] (pointwise, C_r = C / seRatio typically 16)
///   expand: [C_r] → [C] (pointwise)
///   gate = sigmoid(expand(relu(reduce(avg_pool(x)))))
///   output = x * gate  (channel-wise)
struct SEGateParams {
    let reduceWeight: [Float]  // [C_r, 1, 1, C] → use as [C_r, C]
    let reduceBias: [Float]  // [C_r]
    let expandWeight: [Float]  // [C, 1, 1, C_r] → use as [C, C_r]
    let expandBias: [Float]  // [C]

    /// Apply SE gate to NHWC input [B, H, W, C], return gated output.
    func apply(_ x: [Float], B: Int, H: Int, W: Int, C: Int) -> [Float] {
        let cR = reduceBias.count
        // Global average pool: [B, H, W, C] → [B, C]
        var pooled = [Float](repeating: 0, count: B * C)
        let area = Float(H * W)
        for b in 0 ..< B {
            for h in 0 ..< H {
                for w in 0 ..< W {
                    for c in 0 ..< C {
                        pooled[b * C + c] += x[((b * H + h) * W + w) * C + c]
                    }
                }
            }
            for c in 0 ..< C { pooled[b * C + c] /= area }
        }
        // reduce: [B, C] @ [C_r, C]^T → [B, C_r], then ReLU + bias
        var reduced = [Float](repeating: 0, count: B * cR)
        for b in 0 ..< B {
            for r in 0 ..< cR {
                var acc: Float = 0
                for c in 0 ..< C { acc += pooled[b * C + c] * reduceWeight[r * C + c] }
                reduced[b * cR + r] = max(0, acc + reduceBias[r])
            }
        }
        // expand: [B, C_r] @ [C, C_r]^T → [B, C], then sigmoid
        var gate = [Float](repeating: 0, count: B * C)
        for b in 0 ..< B {
            for c in 0 ..< C {
                var acc: Float = 0
                for r in 0 ..< cR { acc += reduced[b * cR + r] * expandWeight[c * cR + r] }
                gate[b * C + c] = 1.0 / (1.0 + expf(-(acc + expandBias[c])))
            }
        }
        // Broadcast gate [B, C] to [B, H, W, C] and multiply.
        var out = x
        for b in 0 ..< B {
            for h in 0 ..< H {
                for w in 0 ..< W {
                    for c in 0 ..< C {
                        out[((b * H + h) * W + w) * C + c] *= gate[b * C + c]
                    }
                }
            }
        }
        return out
    }
}

// ─── Block parameter containers ──────────────────────────────────────

/// Reparameterized ConvFFN used in both RepMixerBlock and AttentionBlock.
///
/// Forward (NHWC):
///   1. DW conv (7×7, same padding, with folded BN)
///   2. fc1 (pointwise 1×1, inC → intermediateC, GELU)
///   3. fc2 (pointwise 1×1, intermediateC → inC)
struct ConvFFNParams {
    let dwWeight: [Float]  // [C, kH, kW] (depthwise)
    let dwBN: FoldedBN  // folded BN for the DW conv
    let kH: Int
    let kW: Int
    let fc1Weight: Tensor  // [intermediateC, inC] 2D
    let fc1Bias: [Float]  // [intermediateC]
    let fc2Weight: Tensor  // [inC, intermediateC] 2D
    let fc2Bias: [Float]  // [inC]
    let inC: Int
    let intermediateC: Int

    /// Forward NHWC [B, H, W, C] → NHWC [B, H, W, C].
    func forward(
        _ x: [Float], B: Int, H: Int, W: Int,
        device: Device
    ) -> [Float] {
        let pad = (kH - 1) / 2
        // 1. DW conv with folded BN.
        var (dw, _, _) = depthwiseConvNHWC(
            x, w: dwWeight, bias: nil,
            B: B, H: H, W: W, C: inC,
            kH: kH, kW: kW, stride: 1, padding: pad)
        // Apply folded BN (scale + bias per channel).
        for b in 0 ..< B {
            for h in 0 ..< H {
                for w2 in 0 ..< W {
                    for c in 0 ..< inC {
                        let idx = ((b * H + h) * W + w2) * inC + c
                        dw[idx] = dw[idx] * dwBN.scale[c] + dwBN.bias[c]
                    }
                }
            }
        }
        // 2. fc1 (pointwise 1×1) + GELU.
        let dwT = floatToTensor(
            dw, shape: [B * H * W, inC],
            dtype: fc1Weight.dtype, device: device)
        var y = pointwiseConvNHWC(
            dwT, weight2D: fc1Weight, bias: fc1Bias,
            BHW: B * H * W, outC: intermediateC, device: device)
        // GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        for i in 0 ..< y.count {
            let v = y[i]
            y[i] = 0.5 * v * (1.0 + erff(v * Float(M_SQRT1_2)))
        }
        // 3. fc2 (pointwise 1×1).
        let yT = floatToTensor(
            y, shape: [B * H * W, intermediateC],
            dtype: fc2Weight.dtype, device: device)
        return pointwiseConvNHWC(
            yT, weight2D: fc2Weight, bias: fc2Bias,
            BHW: B * H * W, outC: inC, device: device)
    }
}

/// RepMixerBlock: metaformer with depthwise conv token mixer.
///   y = x + layerScale * convFFN(tokenMixer(x))
///   where tokenMixer(x) = x + reparam_conv(x)  (in the reference, mixer
///   is the fused reparam_conv branch applied as a depthwise conv)
struct RepMixerBlockParams {
    let mixerWeight: [Float]  // [C, kH, kW] depthwise
    let mixerBias: [Float]  // [C]
    let mixerKH: Int
    let mixerKW: Int
    let convFFN: ConvFFNParams
    let layerScale: [Float]  // [C]
    let inC: Int

    /// Forward NHWC [B, H, W, C] → NHWC [B, H, W, C].
    func forward(
        _ x: [Float], B: Int, H: Int, W: Int,
        device: Device
    ) -> [Float] {
        let pad = (mixerKH - 1) / 2
        // Token mixer: token_mixer output + residual x.
        let (mixed, _, _) = depthwiseConvNHWC(
            x, w: mixerWeight, bias: mixerBias,
            B: B, H: H, W: W, C: inC,
            kH: mixerKH, kW: mixerKW, stride: 1, padding: pad)
        // ConvFFN on the mixed tokens.
        let ffnOut = convFFN.forward(mixed, B: B, H: H, W: W, device: device)
        // Residual + layerScale.
        var out = x
        for b in 0 ..< B {
            for h in 0 ..< H {
                for w in 0 ..< W {
                    for c in 0 ..< inC {
                        let idx = ((b * H + h) * W + w) * inC + c
                        out[idx] += layerScale[c] * ffnOut[idx]
                    }
                }
            }
        }
        return out
    }
}

/// Multi-head self-attention block for vision (AttentionBlock).
/// QKV are fused: weight [3*C, C], no bias. proj weight [C, C] + bias [C].
/// headDim = 32 (hardcoded in the FastViTHD architecture) — attention is
/// now GPU-resident via `Ops.sdpaBidirectional(headDim: 32)`.
///
/// Operates on NHWC [B, H, W, C]; flattens to [B, N, C] for attention.
struct VisionMHSAParams {
    let qkvWeight: Tensor  // [3*C, C]
    let projWeight: Tensor  // [C, C]
    let projBias: [Float]  // [C]
    let inC: Int
    let numHeads: Int
    static let headDim = 32

    /// Forward NHWC [B, H, W, C] → NHWC [B, H, W, C].
    func forward(
        _ x: [Float], B: Int, H: Int, W: Int,
        device: Device
    ) -> [Float] {
        let N = H * W
        let headDim = Self.headDim
        let scale = 1.0 / sqrtf(Float(headDim))

        // Flatten to [B*N, C] for QKV projection.
        let xT = floatToTensor(
            x, shape: [B * N, inC],
            dtype: qkvWeight.dtype, device: device)
        let cmd = device.makeCommandBuffer()
        let qkvOut = Ops.gemm(weight: qkvWeight, input: xT, nRows: B * N, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        let qkvArr = qkvOut.toFloatArray()  // [B*N, 3*C]

        // Split into Q/K/V and run GPU bidirectional attention via
        // `Ops.sdpaBidirectional(headDim: 32)` — FastViT-HD uses the d32
        // variant. Each batch element is an independent attention problem
        // (no cross-batch attention), so we issue one kernel per batch.
        //
        // qkvArr layout per token row: [Q[0..C], K[0..C], V[0..C]] where
        // each C-wide block is [numHeads, headDim] row-major.
        //   → Q[bi]: extract to [N, numHeads, headDim] (same layout)
        //   → K/V[bi]: extract + transpose to [numHeads, N, headDim]
        var attnOut = [Float](repeating: 0, count: B * N * inC)
        let cLocal = inC
        let stride3C = 3 * cLocal
        for bi in 0 ..< B {
            var qFlat = [Float](repeating: 0, count: N * cLocal)
            var kFlat = [Float](repeating: 0, count: N * cLocal)
            var vFlat = [Float](repeating: 0, count: N * cLocal)
            for i in 0 ..< N {
                let srcRow = (bi * N + i) * stride3C
                // Q: same [numHeads, headDim] layout per token → flat copy.
                for d in 0 ..< cLocal { qFlat[i * cLocal + d] = qkvArr[srcRow + d] }
                // K/V: transpose to [numHeads, N, headDim] for the kernel.
                for h in 0 ..< numHeads {
                    let hOff = h * headDim
                    let kSrc = srcRow + cLocal + hOff
                    let vSrc = srcRow + 2 * cLocal + hOff
                    let dst = (h * N + i) * headDim
                    for d in 0 ..< headDim {
                        kFlat[dst + d] = qkvArr[kSrc + d]
                        vFlat[dst + d] = qkvArr[vSrc + d]
                    }
                }
            }
            let qT = floatToTensor(
                qFlat, shape: [N, numHeads, headDim],
                dtype: .f32, device: device)
            let kT = floatToTensor(
                kFlat, shape: [numHeads, N, headDim],
                dtype: .f32, device: device)
            let vT = floatToTensor(
                vFlat, shape: [numHeads, N, headDim],
                dtype: .f32, device: device)
            let attnCmd = device.makeCommandBuffer()
            let outT = Ops.sdpaBidirectional(
                q: qT, k: kT, v: vT,
                nQHeads: numHeads, nKVHeads: numHeads, headDim: headDim,
                baseKV: 0, nQuery: N, kvStride: N,
                scale: scale, on: attnCmd)
            attnCmd.commit()
            attnCmd.waitUntilCompleted()
            let outFlat = outT.toFloatArray()  // [N, numHeads, headDim] = [N, C]
            let dstBase = bi * N * cLocal
            for i in 0 ..< (N * cLocal) { attnOut[dstBase + i] = outFlat[i] }
        }

        // Output projection: [B*N, C] → [B*N, C] + bias.
        let attnT = floatToTensor(
            attnOut, shape: [B * N, inC],
            dtype: projWeight.dtype, device: device)
        return pointwiseConvNHWC(
            attnT, weight2D: projWeight, bias: projBias,
            BHW: B * N, outC: inC, device: device)
    }
}

/// LayerNormChannel: standard per-channel LayerNorm for NHWC feature maps.
/// Applied as: normalize each [C] vector over channel dim.
struct LayerNormChannelParams {
    let weight: [Float]  // [C]
    let bias: [Float]  // [C]
    let eps: Float

    func forward(_ x: [Float], N: Int, C: Int) -> [Float] {
        var out = [Float](repeating: 0, count: N * C)
        for n in 0 ..< N {
            // Compute mean and variance over C channels.
            var mean: Float = 0
            for c in 0 ..< C { mean += x[n * C + c] }
            mean /= Float(C)
            var variance: Float = 0
            for c in 0 ..< C {
                let d = x[n * C + c] - mean
                variance += d * d
            }
            variance /= Float(C)
            let invStd = 1.0 / sqrtf(variance + eps)
            for c in 0 ..< C {
                out[n * C + c] = (x[n * C + c] - mean) * invStd * weight[c] + bias[c]
            }
        }
        return out
    }
}

/// AttentionBlock: pre-norm attention + ConvFFN with layer-scale residuals.
///   y = x + layerScale1 * MHSA(LayerNormChannel(x))
///   y = y + layerScale2 * ConvFFN(y)
struct AttentionBlockParams {
    let norm: LayerNormChannelParams
    let mhsa: VisionMHSAParams
    let convFFN: ConvFFNParams
    let layerScale1: [Float]  // [C]
    let layerScale2: [Float]  // [C]
    let inC: Int

    func forward(
        _ x: [Float], B: Int, H: Int, W: Int,
        device: Device
    ) -> [Float] {
        let N = B * H * W
        // Attention sub-block.
        let normed = norm.forward(x, N: N, C: inC)
        let attnOut = mhsa.forward(normed, B: B, H: H, W: W, device: device)
        var y = x
        for i in 0 ..< N * inC {
            let c = i % inC
            y[i] += layerScale1[c] * attnOut[i]
        }
        // ConvFFN sub-block.
        let ffnOut = convFFN.forward(y, B: B, H: H, W: W, device: device)
        for i in 0 ..< N * inC {
            let c = i % inC
            y[i] += layerScale2[c] * ffnOut[i]
        }
        return y
    }
}

// ─── Vision tower blocks ──────────────────────────────────────────────

/// A stage in the FastViTHD network: array of RepMixerBlock or AttentionBlock.
enum FastVLMStageBlock {
    case repMixer(RepMixerBlockParams)
    case attention(AttentionBlockParams)

    func forward(
        _ x: [Float], B: Int, H: Int, W: Int,
        device: Device
    ) -> [Float] {
        switch self {
        case .repMixer(let p): return p.forward(x, B: B, H: H, W: W, device: device)
        case .attention(let p): return p.forward(x, B: B, H: H, W: W, device: device)
        }
    }
}

/// Reparameterized Conditional Positional Encoding:
/// a depthwise conv (7×7, stride=1, same padding) acting as a spatial
/// position bias added to the feature map.
struct RepCPEParams {
    let weight: [Float]  // [C, kH, kW] depthwise
    let bias: [Float]  // [C]
    let kH: Int
    let kW: Int

    func forward(_ x: [Float], B: Int, H: Int, W: Int, C: Int) -> [Float] {
        let pad = (kH - 1) / 2
        let (out, _, _) = depthwiseConvNHWC(
            x, w: weight, bias: bias,
            B: B, H: H, W: W, C: C,
            kH: kH, kW: kW, stride: 1, padding: pad)
        return out
    }
}

/// PatchEmbed between stages: LargeKernelBlock (LKB, depthwise 7×7 stride-2)
/// + pointwise 1×1 for channel expansion.
/// The two branches in the LKB are fused at inference:
///   lkb_reparam: depthwise [out_ch, 7, 7, 1] with stride 2
///   reparam_conv (1×1): pointwise [out_ch, 1, 1, out_ch] -> channel mix
struct PatchEmbedParams {
    let lkbWeight: [Float]  // [outC, kH, kW] depthwise
    let lkbBias: [Float]  // [outC]
    let pwWeight: Tensor  // [outC, outC] pointwise
    let pwBias: [Float]  // [outC]
    let kH: Int
    let kW: Int
    let stride: Int
    let inC: Int
    let outC: Int

    func forward(
        _ x: [Float], B: Int, H: Int, W: Int,
        device: Device
    ) -> ([Float], Int, Int) {
        let pad = (kH - 1) / 2
        let (dw, outH, outW) = depthwiseConvNHWC(
            x, w: lkbWeight, bias: lkbBias,
            B: B, H: H, W: W, C: inC,
            kH: kH, kW: kW, stride: stride, padding: pad)
        // Pointwise 1×1 over the downsampled map.
        let dwT = floatToTensor(
            dw, shape: [B * outH * outW, inC],
            dtype: pwWeight.dtype, device: device)
        let pw = pointwiseConvNHWC(
            dwT, weight2D: pwWeight, bias: pwBias,
            BHW: B * outH * outW, outC: outC, device: device)
        return (pw, outH, outW)
    }
}

/// ConvolutionalStem: 3 MobileOne blocks.
///   block 0: regular conv [96, 3, 3, 3] stride-2
///   block 1: depthwise [96, 3, 3, 1] stride-2
///   block 2: pointwise [96, 1, 1, 96] stride-1
/// After the stem, spatial resolution is image_size / 4.
struct ConvStemParams {
    let block0Weight: [Float]  // [C, 3, 3, 3] regular conv -> [C, 3, 3, 3] OIHW
    let block0Bias: [Float]
    let block1Weight: [Float]  // [C, 3, 3] depthwise
    let block1Bias: [Float]
    let block2Weight: Tensor  // [C, C] pointwise
    let block2Bias: [Float]
    let outC: Int  // = embedDims[0]

    func forward(_ x: [Float], H: Int, W: Int) -> ([Float], Int, Int, Int) {
        // block 0: regular 3×3 stride-2 conv [outC, inC=3, 3, 3] OIHW.
        // Input is NHWC [1, H, W, 3]; we need to handle the regular conv
        // (not depthwise). Use explicit nested loop for the 3-channel case.
        let B = 1
        let kH = 3
        let kW = 3
        let stride0 = 2
        let pad0 = 1
        let inC0 = 3
        let outH0 = (H + 2 * pad0 - kH) / stride0 + 1
        let outW0 = (W + 2 * pad0 - kW) / stride0 + 1
        var y0 = [Float](repeating: 0, count: B * outH0 * outW0 * outC)
        for oc in 0 ..< outC {
            for oh in 0 ..< outH0 {
                for ow in 0 ..< outW0 {
                    var acc: Float = 0
                    for ic in 0 ..< inC0 {
                        for ky in 0 ..< kH {
                            let ih = oh * stride0 - pad0 + ky
                            if ih < 0 || ih >= H { continue }
                            for kx in 0 ..< kW {
                                let iw = ow * stride0 - pad0 + kx
                                if iw < 0 || iw >= W { continue }
                                // Weight is OIHW: [oc, ic, ky, kx]
                                let wIdx = ((oc * inC0 + ic) * kH + ky) * kW + kx
                                let inIdx = (ih * W + iw) * inC0 + ic
                                acc += x[inIdx] * block0Weight[wIdx]
                            }
                        }
                    }
                    let outIdx = (oh * outW0 + ow) * outC + oc
                    y0[outIdx] = acc + block0Bias[oc]
                }
            }
        }
        // block 1: depthwise 3×3 stride-2.
        let pad1 = 1
        let (y1, outH1, outW1) = depthwiseConvNHWC(
            y0, w: block1Weight, bias: block1Bias,
            B: B, H: outH0, W: outW0, C: outC,
            kH: kH, kW: kW, stride: 2, padding: pad1)
        // block 2: pointwise 1×1.
        let y1T = floatToTensor(
            y1, shape: [B * outH1 * outW1, outC],
            dtype: block2Weight.dtype, device: .shared)
        let y2 = pointwiseConvNHWC(
            y1T, weight2D: block2Weight, bias: block2Bias,
            BHW: B * outH1 * outW1, outC: outC, device: .shared)
        return (y2, outH1, outW1, outC)
    }
}

/// MobileOne block used in conv_exp (depthwise + SE gate).
/// Forward: DW conv (stride=1) → SE gate → output.
/// No BN: the reparam_conv weight already folds BN at save time.
struct ConvExpParams {
    let dwWeight: [Float]  // [outC, kH, kW] depthwise
    let dwBias: [Float]  // [outC]
    let se: SEGateParams
    let kH: Int
    let kW: Int
    let inC: Int
    let outC: Int

    func forward(_ x: [Float], B: Int, H: Int, W: Int) -> [Float] {
        let pad = (kH - 1) / 2
        let (dw, _, _) = depthwiseConvNHWC(
            x, w: dwWeight, bias: dwBias,
            B: B, H: H, W: W, C: inC,
            kH: kH, kW: kW, stride: 1, padding: pad)
        return se.apply(dw, B: B, H: H, W: W, C: outC)
    }
}

// ─── Full tower ───────────────────────────────────────────────────────

/// The complete FastViTHD vision tower: ConvStem → stages → conv_exp.
/// Encapsulates all loaded weight parameters.
///
/// The tower's `encode` produces `[patchH * patchW, mmHiddenSize]` in
/// token-major layout, ready for the projector.
final class FastVLMVisionTower {
    let cfg: FastVLMVisionConfig
    let stem: ConvStemParams
    let stages: [[FastVLMStageBlock]]  // [nStages][nBlocks]
    let patchEmbeds: [PatchEmbedParams]  // between stages, len = nStages-1
    let cpes: [RepCPEParams?]  // one per stage, nil if no CPE
    let convExp: ConvExpParams
    let patchH: Int  // spatial H after all downsampling
    let patchW: Int  // spatial W after all downsampling
    let dtype: DType

    init(
        cfg: FastVLMVisionConfig,
        stem: ConvStemParams,
        stages: [[FastVLMStageBlock]],
        patchEmbeds: [PatchEmbedParams],
        cpes: [RepCPEParams?],
        convExp: ConvExpParams,
        patchH: Int, patchW: Int, dtype: DType
    ) {
        self.cfg = cfg
        self.stem = stem
        self.stages = stages
        self.patchEmbeds = patchEmbeds
        self.cpes = cpes
        self.convExp = convExp
        self.patchH = patchH
        self.patchW = patchW
        self.dtype = dtype
    }

    /// Encode NCHW image [1, 3, H, W] → token-major [patchH*patchW, mmHiddenSize].
    ///
    /// Internally all activations are NHWC [B, H, W, C]. The input NCHW
    /// image is transposed to NHWC before the stem, and the final token
    /// array is returned as a Tensor.
    func encode(image: Tensor, device: Device) -> Tensor {
        // image: NCHW [1, 3, H, W] → NHWC [1, H, W, 3].
        let imgArr = image.toFloatArray()
        let H = image.shape[2]
        let W = image.shape[3]
        var nhwc = [Float](repeating: 0, count: H * W * 3)
        for c in 0 ..< 3 {
            for h in 0 ..< H {
                for w in 0 ..< W {
                    nhwc[(h * W + w) * 3 + c] = imgArr[c * H * W + h * W + w]
                }
            }
        }

        // ConvolutionalStem: NHWC [1, H, W, 3] → NHWC [1, H', W', C0].
        var (act, curH, curW, curC) = stem.forward(nhwc, H: H, W: W)
        let B = 1

        for i in 0 ..< cfg.nStages {
            // Optional CPE before this stage.
            if let cpe = cpes[i] {
                act = cpe.forward(act, B: B, H: curH, W: curW, C: curC)
            }
            // Stage blocks.
            for block in stages[i] {
                act = block.forward(act, B: B, H: curH, W: curW, device: device)
            }
            // PatchEmbed downsampling between stages.
            if i < cfg.nStages - 1 && i < patchEmbeds.count {
                let pe = patchEmbeds[i]
                var newH: Int
                var newW: Int
                (act, newH, newW) = pe.forward(act, B: B, H: curH, W: curW, device: device)
                curH = newH
                curW = newW
                curC = pe.outC
            }
        }

        // conv_exp (MobileOne block, stride=1): NHWC [B, H, W, C] → [B, H, W, mmHidden].
        act = convExp.forward(act, B: B, H: curH, W: curW)
        let mmHidden = cfg.mmHiddenSize

        // Return token-major [patchH*patchW, mmHiddenSize] Tensor.
        let nTokens = curH * curW
        let result = Tensor.empty(shape: [nTokens, mmHidden], dtype: dtype, device: device)
        ImagePreprocessing.copyFloats(act, into: result)
        return result
    }

    /// Load the tower from the vision sub-tree of the checkpoint.
    static func load(
        cfg: FastVLMVisionConfig,
        weights: SafeTensorsBundle,
        dtype: DType,
        device: Device
    ) throws -> FastVLMVisionTower {

        // Helper: load OHWI weight as [C, kH, kW] float array for depthwise.
        // OHWI [outC, kH, kW, inC=1] → strip inC dim → [outC, kH, kW].
        func loadDWOHWI(_ key: String) throws -> [Float] {
            let t = try weights.tensor(named: key)
            return t.toFloatArray()  // [outC*kH*kW*1] = [outC*kH*kW]
        }

        // Helper: load pointwise weight as 2D [outC, inC].
        // Handles two storage layouts emitted by different FastViTHD layers:
        //   • OHWI 4D [outC, 1, 1, inC] — ConvStem block 2, PatchEmbed reparam_conv,
        //     and ConvFFN fc1/fc2. Strip the two unit spatial dims.
        //   • Plain 2D [outC, inC] — attention token_mixer.proj.weight.
        // Accessing shape[3] on a 2D tensor crashes with "Index out of range"
        // (ContiguousArrayBuffer fatal error), which was the load-time crash.
        func loadPW2D(_ key: String) throws -> Tensor {
            let t = try weights.tensor(named: key)
            if t.shape.count == 2 {
                // Already [outC, inC] — return as-is (no copy needed).
                return t
            }
            // Shape [outC, 1, 1, inC] — strip spatial dims to [outC, inC].
            let outC = t.shape[0]
            let inC = t.shape[3]
            let flat = t.toFloatArray()
            let result = Tensor.empty(shape: [outC, inC], dtype: t.dtype, device: device)
            ImagePreprocessing.copyFloats(flat, into: result)
            return result
        }

        // Helper: load bias as float array.
        func loadBias(_ key: String) throws -> [Float] {
            try weights.tensor(named: key).toFloatArray()
        }

        // Helper: load ConvFFN for a block at prefix `p`.
        func loadConvFFN(_ p: String, inC: Int, intermediateC: Int) throws -> ConvFFNParams {
            let dwW = try loadDWOHWI("\(p).convffn.conv.conv.weight")
            let kH = try weights.tensor(named: "\(p).convffn.conv.conv.weight").shape[1]
            let kW = try weights.tensor(named: "\(p).convffn.conv.conv.weight").shape[2]
            let bnWeight = try loadBias("\(p).convffn.conv.bn.weight")
            let bnBias = try loadBias("\(p).convffn.conv.bn.bias")
            let bnMean = try loadBias("\(p).convffn.conv.bn.running_mean")
            let bnVar = try loadBias("\(p).convffn.conv.bn.running_var")
            let bn = FoldedBN(
                weight: bnWeight, bias: bnBias,
                runningMean: bnMean, runningVar: bnVar)
            let fc1W = try loadPW2D("\(p).convffn.fc1.weight")
            let fc1B = try loadBias("\(p).convffn.fc1.bias")
            let fc2W = try loadPW2D("\(p).convffn.fc2.weight")
            let fc2B = try loadBias("\(p).convffn.fc2.bias")
            return ConvFFNParams(
                dwWeight: dwW, dwBN: bn, kH: kH, kW: kW,
                fc1Weight: fc1W, fc1Bias: fc1B,
                fc2Weight: fc2W, fc2Bias: fc2B,
                inC: inC, intermediateC: intermediateC)
        }

        // ── ConvolutionalStem ──────────────────────────────────────────
        // block 0: regular conv [outC, kH, kW, inC=3] OHWI → OIHW.
        let stemB0Raw = try weights.tensor(named: "patch_embed.blocks.0.reparam_conv.weight")
        let stemB0W = transposeOHWItoOIHW(stemB0Raw).toFloatArray()
        let stemB0Bias = try loadBias("patch_embed.blocks.0.reparam_conv.bias")
        // block 1: depthwise [outC, kH, kW, 1] OHWI.
        let stemB1W = try loadDWOHWI("patch_embed.blocks.1.reparam_conv.weight")
        let stemB1Bias = try loadBias("patch_embed.blocks.1.reparam_conv.bias")
        // block 2: pointwise [outC, 1, 1, outC] OHWI → [outC, outC] 2D.
        let stemB2W = try loadPW2D("patch_embed.blocks.2.reparam_conv.weight")
        let stemB2Bias = try loadBias("patch_embed.blocks.2.reparam_conv.bias")
        let stem = ConvStemParams(
            block0Weight: stemB0W, block0Bias: stemB0Bias,
            block1Weight: stemB1W, block1Bias: stemB1Bias,
            block2Weight: stemB2W, block2Bias: stemB2Bias,
            outC: cfg.embedDims[0])

        // ── Stages, PatchEmbeds, CPEs ──────────────────────────────────
        // Network layout: stages and PatchEmbeds are interleaved in a flat
        // sequence (network.0, network.1, ..., network.10). The mapping is:
        //   network.0  → stage 0 blocks
        //   network.1  → PatchEmbed 0 (between stage 0 and 1)
        //   network.2  → stage 1 blocks
        //   network.3  → PatchEmbed 1
        //   network.4  → stage 2 blocks
        //   network.5  → PatchEmbed 2
        //   network.6  → CPE for stage 3 (reparam_conv)
        //   network.7  → stage 3 blocks
        //   network.8  → PatchEmbed 3
        //   network.9  → CPE for stage 4
        //   network.10 → stage 4 blocks
        //
        // This structure is derived from cfg.downSamples + cfg.posEmbShapes:
        //   flatIdx tracks the running network.N index.
        var stages: [[FastVLMStageBlock]] = []
        var patchEmbeds: [PatchEmbedParams] = []
        var cpes: [RepCPEParams?] = []
        var flatIdx = 0

        for si in 0 ..< cfg.nStages {
            let dim = cfg.embedDims[si]
            let mlpIntermediate = dim * cfg.mlpRatios[si]
            let mixerType = cfg.tokenMixers[si]

            // CPE before this stage (if posEmbShapes[si] != nil).
            if cfg.posEmbShapes[si] != nil {
                let cpePrefix = "network.\(flatIdx)"
                let cpeW = try loadDWOHWI("\(cpePrefix).reparam_conv.weight")
                let cpeKH = try weights.tensor(named: "\(cpePrefix).reparam_conv.weight").shape[1]
                let cpeKW = try weights.tensor(named: "\(cpePrefix).reparam_conv.weight").shape[2]
                let cpeBias = try loadBias("\(cpePrefix).reparam_conv.bias")
                cpes.append(
                    RepCPEParams(
                        weight: cpeW, bias: cpeBias,
                        kH: cpeKH, kW: cpeKW))
                flatIdx += 1
            } else {
                cpes.append(nil)
            }

            // Stage blocks.
            let stageBase = flatIdx
            var stageBlocks: [FastVLMStageBlock] = []
            stageBlocks.reserveCapacity(cfg.layers[si])
            for bi in 0 ..< cfg.layers[si] {
                let p = "network.\(stageBase).\(bi)"
                let ffn = try loadConvFFN(p, inC: dim, intermediateC: mlpIntermediate)
                if mixerType == "attention" {
                    let qkvW = try weights.tensor(named: "\(p).token_mixer.qkv.weight")
                    let projW = try loadPW2D("\(p).token_mixer.proj.weight")
                    let projBias = try loadBias("\(p).token_mixer.proj.bias")
                    let normW = try loadBias("\(p).norm.weight")
                    let normBias = try loadBias("\(p).norm.bias")
                    let ls1 = try loadBias("\(p).layer_scale_1")
                    let ls2 = try loadBias("\(p).layer_scale_2")
                    // layer_scale tensors are [1, 1, C] in checkpoint — flatten.
                    let numHeads = dim / VisionMHSAParams.headDim
                    let attn = AttentionBlockParams(
                        norm: LayerNormChannelParams(weight: normW, bias: normBias, eps: 1e-6),
                        mhsa: VisionMHSAParams(
                            qkvWeight: qkvW,
                            projWeight: projW, projBias: projBias,
                            inC: dim, numHeads: numHeads),
                        convFFN: ffn,
                        layerScale1: ls1, layerScale2: ls2, inC: dim)
                    stageBlocks.append(.attention(attn))
                } else {
                    // repMixer
                    let mixW = try loadDWOHWI("\(p).token_mixer.reparam_conv.weight")
                    let mixKH = try weights.tensor(named: "\(p).token_mixer.reparam_conv.weight")
                        .shape[1]
                    let mixKW = try weights.tensor(named: "\(p).token_mixer.reparam_conv.weight")
                        .shape[2]
                    let mixBias = try loadBias("\(p).token_mixer.reparam_conv.bias")
                    let ls = try loadBias("\(p).layer_scale")
                    let mixer = RepMixerBlockParams(
                        mixerWeight: mixW, mixerBias: mixBias,
                        mixerKH: mixKH, mixerKW: mixKW,
                        convFFN: ffn, layerScale: ls, inC: dim)
                    stageBlocks.append(.repMixer(mixer))
                }
            }
            stages.append(stageBlocks)
            flatIdx += 1

            // PatchEmbed after this stage (between stage si and si+1).
            if si < cfg.nStages - 1 {
                let pePrefix = "network.\(flatIdx).proj"
                let lkbRaw = try weights.tensor(named: "\(pePrefix).0.lkb_reparam.weight")
                let lkbKH = lkbRaw.shape[1]
                let lkbKW = lkbRaw.shape[2]
                let lkbW = lkbRaw.toFloatArray()
                let lkbBias = try loadBias("\(pePrefix).0.lkb_reparam.bias")
                let outC = cfg.embedDims[si + 1]
                let pwW = try loadPW2D("\(pePrefix).1.reparam_conv.weight")
                let pwBias = try loadBias("\(pePrefix).1.reparam_conv.bias")
                patchEmbeds.append(
                    PatchEmbedParams(
                        lkbWeight: lkbW, lkbBias: lkbBias,
                        pwWeight: pwW, pwBias: pwBias,
                        kH: lkbKH, kW: lkbKW,
                        stride: cfg.downStride,
                        inC: cfg.embedDims[si], outC: outC))
                flatIdx += 1
            }
        }

        // ── conv_exp ───────────────────────────────────────────────────
        let lastDim = cfg.embedDims.last!
        let expandedDim = cfg.mmHiddenSize
        let convExpDW = try loadDWOHWI("conv_exp.reparam_conv.weight")
        let convExpDWBias = try loadBias("conv_exp.reparam_conv.bias")
        let seReduceW = try weights.tensor(named: "conv_exp.se.reduce.weight")
        let seReduceBias = try loadBias("conv_exp.se.reduce.bias")
        let seExpandW = try weights.tensor(named: "conv_exp.se.expand.weight")
        let seExpandBias = try loadBias("conv_exp.se.expand.bias")
        // SE weights are OHWI pointwise [outC, 1, 1, inC] → flatten to [outC, inC].
        let seCR = seReduceBias.count  // squeezed channel count
        let convExpKH = try weights.tensor(named: "conv_exp.reparam_conv.weight").shape[1]
        let convExpKW = try weights.tensor(named: "conv_exp.reparam_conv.weight").shape[2]
        // SE reduce weight: [C_r, 1, 1, C_exp] → [C_r, C_exp].
        let seReduceFlat = seReduceW.toFloatArray()
        // SE expand weight: [C_exp, 1, 1, C_r] → [C_exp, C_r].
        let seExpandFlat = seExpandW.toFloatArray()
        _ = lastDim  // suppress unused warning — seCR derived from checkpoint shape
        _ = expandedDim
        let se = SEGateParams(
            reduceWeight: seReduceFlat, reduceBias: seReduceBias,
            expandWeight: seExpandFlat, expandBias: seExpandBias)
        let convExpParams = ConvExpParams(
            dwWeight: convExpDW, dwBias: convExpDWBias,
            se: se, kH: convExpKH, kW: convExpKW,
            inC: cfg.embedDims.last!, outC: cfg.mmHiddenSize)

        // ── Compute output spatial resolution ──────────────────────────
        let resolutions = cfg.spatialResolutions()
        let (finalH, finalW) = resolutions.last!

        return FastVLMVisionTower(
            cfg: cfg, stem: stem, stages: stages,
            patchEmbeds: patchEmbeds, cpes: cpes,
            convExp: convExpParams,
            patchH: finalH, patchW: finalW,
            dtype: dtype)
    }
}

// ─── Multi-modal projector ────────────────────────────────────────────

/// FastVLM mlp2x_gelu projector: two linear layers with a GELU between.
///   linear1 [mmHidden → textHidden] → GELU → linear2 [textHidden → textHidden]
///
/// The projector runs on the GPU (Ops.gemm for each Linear).
final class FastVLMProjector {
    let linear1Weight: Tensor  // [textHidden, mmHidden]
    let linear1Bias: [Float]  // [textHidden]
    let linear2Weight: Tensor  // [textHidden, textHidden]
    let linear2Bias: [Float]  // [textHidden]
    let mmHidden: Int
    let textHidden: Int

    init(
        linear1Weight: Tensor, linear1Bias: [Float],
        linear2Weight: Tensor, linear2Bias: [Float],
        mmHidden: Int, textHidden: Int
    ) {
        self.linear1Weight = linear1Weight
        self.linear1Bias = linear1Bias
        self.linear2Weight = linear2Weight
        self.linear2Bias = linear2Bias
        self.mmHidden = mmHidden
        self.textHidden = textHidden
    }

    static func load(
        mmHidden: Int, textHidden: Int,
        weights: SafeTensorsBundle, device: Device
    ) throws -> FastVLMProjector {
        let w1 = try weights.tensor(named: "mm_projector.0.weight")
        let b1 = try weights.tensor(named: "mm_projector.0.bias").toFloatArray()
        let w2 = try weights.tensor(named: "mm_projector.2.weight")
        let b2 = try weights.tensor(named: "mm_projector.2.bias").toFloatArray()
        return FastVLMProjector(
            linear1Weight: w1, linear1Bias: b1,
            linear2Weight: w2, linear2Bias: b2,
            mmHidden: mmHidden, textHidden: textHidden)
    }

    /// Project `[nTokens, mmHidden]` vision tokens → `[nTokens, textHidden]`.
    func project(tokens: Tensor, nTokens: Int, device: Device) -> Tensor {
        // linear1 + GELU.
        let cmd1 = device.makeCommandBuffer()
        var h = Ops.gemm(
            weight: linear1Weight, input: tokens,
            nRows: nTokens, on: cmd1)
        h = Ops.gelu(h, on: cmd1)
        cmd1.commit()
        cmd1.waitUntilCompleted()
        // Add bias for linear1.
        var hArr = h.toFloatArray()
        for row in 0 ..< nTokens {
            for c in 0 ..< textHidden { hArr[row * textHidden + c] += linear1Bias[c] }
        }
        let hBiased = floatToTensor(
            hArr, shape: [nTokens, textHidden],
            dtype: linear1Weight.dtype, device: device)
        // linear2.
        let cmd2 = device.makeCommandBuffer()
        let out = Ops.gemm(
            weight: linear2Weight, input: hBiased,
            nRows: nTokens, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        var outArr = out.toFloatArray()
        for row in 0 ..< nTokens {
            for c in 0 ..< textHidden { outArr[row * textHidden + c] += linear2Bias[c] }
        }
        let result = floatToTensor(
            outArr, shape: [nTokens, textHidden],
            dtype: linear2Weight.dtype, device: device)
        return result
    }
}

// ─── Composed vision tower ────────────────────────────────────────────

/// Couples the FastViTHD tower with the mlp2x_gelu projector so the pair
/// presents a single `VisionEncoder`-shaped surface to `VisionModel`.
/// The composed tower's `encode` produces `[imageTokenCount, textHidden]`.
final class FastVLMComposedTower {
    let tower: FastVLMVisionTower
    let projector: FastVLMProjector
    let imageTokenCount: Int
    let textHidden: Int
    let dtype: DType

    init(
        tower: FastVLMVisionTower, projector: FastVLMProjector,
        imageTokenCount: Int, textHidden: Int, dtype: DType
    ) {
        self.tower = tower
        self.projector = projector
        self.imageTokenCount = imageTokenCount
        self.textHidden = textHidden
        self.dtype = dtype
    }

    /// Wrap this composed tower as a `VisionEncoder` whose `encode`
    /// runs the FastViTHD forward pass + the projector.
    func asVisionEncoder() -> VisionEncoder {
        FastVLMComposedEncoder(tower: self)
    }
}

/// A `VisionEncoder` subclass whose `encode` runs the FastViTHD tower
/// then the mlp2x_gelu projector — so `VisionModel` sees a single tower
/// producing `[imageTokenCount, textHidden]` tokens.
final class FastVLMComposedEncoder: VisionEncoder {
    let composedTower: FastVLMComposedTower

    init(tower: FastVLMComposedTower) {
        self.composedTower = tower
        // Build a facade VisionEncoderConfig that reports the correct
        // `numPatches` (imageTokenCount) so `VisionModel.imageTokenCount` works.
        // The actual image encode path is overridden below — the
        // parent's conv2d / block stack is never called.
        let imageSize = tower.tower.cfg.imageSize
        let patchSize = tower.tower.cfg.patchSize
        let facadeConfig = VisionEncoderConfig(
            imageSize: imageSize, patchSize: patchSize,
            hidden: tower.textHidden, intermediate: tower.textHidden,
            nLayers: 0, nHeads: 1,
            layerNormEps: 1e-6, textHidden: tower.textHidden)
        // Placeholder tensors — required by VisionEncoder.init but never used.
        let placeholderW = Tensor.empty(
            shape: [tower.textHidden, 3, patchSize, patchSize], dtype: tower.dtype)
        let placeholderB = Tensor.empty(shape: [tower.textHidden], dtype: tower.dtype)
        let placeholderPos = Tensor.empty(
            shape: [tower.imageTokenCount, tower.textHidden], dtype: tower.dtype)
        let placeholderLN = LayerNorm(
            weight: Tensor.empty(shape: [tower.textHidden], dtype: tower.dtype),
            bias: Tensor.empty(shape: [tower.textHidden], dtype: tower.dtype),
            eps: 1e-6)
        super.init(
            config: facadeConfig,
            patchEmbedWeight: placeholderW, patchEmbedBias: placeholderB,
            positionEmbedding: placeholderPos, layers: [],
            postLayerNorm: placeholderLN, projection: nil,
            dtype: tower.dtype)
    }

    /// Run the FastViTHD tower then the projector. Returns
    /// `[imageTokenCount, textHidden]`.
    override func encode(image: Tensor, device: Device = .shared) -> Tensor {
        let featureTokens = composedTower.tower.encode(image: image, device: device)
        return composedTower.projector.project(
            tokens: featureTokens,
            nTokens: composedTower.imageTokenCount,
            device: device)
    }
}
