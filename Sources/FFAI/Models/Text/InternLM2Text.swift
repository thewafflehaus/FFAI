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
// InternLM 2 loader — Shanghai AI Lab's InternLM v2 dense decoder.
//
// InternLM2 is architecturally Llama-3-shaped (pre-norm, RoPE, GQA,
// SwiGLU MLP), so it reuses `LlamaLayer` / `LlamaModel` for the forward
// path. What it can't share is the *loader*: InternLM2 ships
// non-standard tensor names and a single FUSED q/k/v projection.
//
//   InternLM2 name                      → FFAI / Llama name
//   ─────────────────────────────────────────────────────────────
//   model.tok_embeddings                → model.embed_tokens
//   model.layers.N.attention.wqkv       → q_proj + k_proj + v_proj (split)
//   model.layers.N.attention.wo         → self_attn.o_proj
//   model.layers.N.feed_forward.w1      → mlp.gate_proj
//   model.layers.N.feed_forward.w3      → mlp.up_proj
//   model.layers.N.feed_forward.w2      → mlp.down_proj
//   model.layers.N.attention_norm       → input_layernorm
//   model.layers.N.ffn_norm             → post_attention_layernorm
//   model.norm                          → model.norm
//   output                              → lm_head
//
// The fused `wqkv` packs, per KV group, `[q_per_kv query heads, 1 key
// head, 1 value head]` along the output dimension:
//
//   wqkv[out] = concat over g in 0..<num_kv of
//                 [ g's q_per_kv q-heads | g's k-head | g's v-head ]   (× head_dim rows each)
//
// Splitting by output row preserves quantization (rows == output
// channels; affine scales/biases are per-row), and the grouped ordering
// lines up with FFAI's GQA fan-out (consecutive query heads share a KV
// head), so no re-permutation is needed.

import Foundation
import Metal

public enum InternLM2Dense {

    /// InternLM2 dense defaults — Llama-shaped, greedy-friendly.
    public static let defaultGenerationParameters = LlamaDense.defaultGenerationParameters

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LlamaModel {
        guard let hidden = config.hiddenSize,
            let nLayers = config.numLayers,
            let nHeads = config.numAttentionHeads,
            let vocab = config.vocabSize,
            let intermediate = config.intermediateSize,
            let eps = config.rmsNormEps
        else {
            throw LlamaError.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let headDim = config.headDim ?? (hidden / nHeads)
        let theta = Float(config.ropeTheta ?? 1_000_000)
        let maxSeq = config.int("max_position_embeddings") ?? 32768
        let quant = config.quantization
        let tieEmbed = config.tieWordEmbeddings

        let embedTokens = try loadEmbedding(
            base: "model.tok_embeddings", in: weights,
            hidden: hidden, quantization: quant)

        var layers: [LlamaLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"
            let (qProj, kProj, vProj) = try splitWQKV(
                base: "\(p).attention.wqkv", in: weights, quantization: quant,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim, device: device)
            let oProj = try loadLinear(
                base: "\(p).attention.wo", in: weights, quantization: quant)

            // InternLM2 MLP: w1 = gate, w3 = up, w2 = down (SwiGLU).
            let gateProj = try loadLinear(
                base: "\(p).feed_forward.w1", in: weights, quantization: quant)
            let upProj = try loadLinear(
                base: "\(p).feed_forward.w3", in: weights, quantization: quant)
            let downProj = try loadLinear(
                base: "\(p).feed_forward.w2", in: weights, quantization: quant)

            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).attention_norm.weight"),
                eps: Float(eps))
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).ffn_norm.weight"),
                eps: Float(eps))

            layers.append(
                LlamaLayer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                    headDim: headDim, intermediate: intermediate,
                    ropeTheta: theta, ropeScaling: .none))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"),
            eps: Float(eps))

        // LM head — InternLM2 uses `output.weight` (untied by default).
        let lmHead: AnyLinear
        if !tieEmbed, weights.has("output.weight") {
            lmHead = try loadLinear(base: "output", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("model.tok_embeddings") {
            let t = try weights.quantizedTriplet("model.tok_embeddings")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: bits, groupSize: q.groupSize))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        let activationDtype: DType
        if weights.isQuantized("model.tok_embeddings"),
            let scales = try? weights.tensor(named: "model.tok_embeddings.scales")
        {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        return LlamaModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxContextWindow: maxSeq, ropeTheta: theta, dtype: activationDtype,
            kvCacheKind: options.kvCache,
            kvEviction: options.kvEviction,
            auraDecodePath: options.auraDecodePath)
    }

    /// Split a fused `wqkv` into separate q/k/v `AnyLinear`s. Handles raw
    /// and affine-quantized checkpoints alike — both slice by output row
    /// (the packed u32 weight and the per-row scales/biases share the
    /// same output-channel ordering).
    private static func splitWQKV(
        base: String, in weights: SafeTensorsBundle,
        quantization q: ModelConfig.QuantizationConfig?,
        nHeads: Int, nKVHeads: Int, headDim: Int, device: Device
    ) throws -> (q: AnyLinear, k: AnyLinear, v: AnyLinear) {
        let (qRanges, kRanges, vRanges) = wqkvRowRanges(
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim)

        if weights.isQuantized(base), let q {
            let t = try weights.quantizedTriplet(base)
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            func lin(_ ranges: [(Int, Int)]) -> AnyLinear {
                AnyLinear(
                    QuantizedLinear(
                        weight: gatherRows(t.weight, ranges, device: device),
                        scales: gatherRows(t.scales, ranges, device: device),
                        biases: gatherRows(t.biases, ranges, device: device),
                        bits: bits, groupSize: q.groupSize))
            }
            return (lin(qRanges), lin(kRanges), lin(vRanges))
        }

        let w = try weights.tensor(named: "\(base).weight")
        func lin(_ ranges: [(Int, Int)]) -> AnyLinear {
            AnyLinear(Linear(weight: gatherRows(w, ranges, device: device)))
        }
        return (lin(qRanges), lin(kRanges), lin(vRanges))
    }

    /// Output-row ranges of the fused `wqkv` for the q, k, and v
    /// projections. Per KV group the rows are laid out
    /// `[q_per_kv query heads | 1 key head | 1 value head]`, each block
    /// `head_dim` rows tall. The grouped ordering matches FFAI's GQA
    /// fan-out, so the gathered q/k/v need no further permutation.
    /// Exposed (internal) so the layout math is unit-testable without a
    /// real checkpoint.
    static func wqkvRowRanges(
        nHeads: Int, nKVHeads: Int, headDim: Int
    ) -> (q: [(start: Int, count: Int)], k: [(start: Int, count: Int)],
        v: [(start: Int, count: Int)])
    {
        let qPerKV = nHeads / nKVHeads
        let groupRows = (qPerKV + 2) * headDim
        var qRanges: [(start: Int, count: Int)] = []
        var kRanges: [(start: Int, count: Int)] = []
        var vRanges: [(start: Int, count: Int)] = []
        for g in 0 ..< nKVHeads {
            let b = g * groupRows
            qRanges.append((b, qPerKV * headDim))
            kRanges.append((b + qPerKV * headDim, headDim))
            vRanges.append((b + (qPerKV + 1) * headDim, headDim))
        }
        return (qRanges, kRanges, vRanges)
    }

    /// Gather output-row ranges from `src` (`[outDim, inner…]`) into a
    /// fresh contiguous tensor by raw byte copy — dtype-agnostic, so it
    /// works on bf16/f32 weights and packed u32 / per-row scale tensors
    /// alike.
    private static func gatherRows(
        _ src: Tensor, _ ranges: [(start: Int, count: Int)], device: Device
    ) -> Tensor {
        let inner = src.shape.dropFirst().reduce(1, *)
        let rowBytes = inner * src.dtype.byteSize
        let totalRows = ranges.reduce(0) { $0 + $1.count }
        var newShape = src.shape
        newShape[0] = totalRows
        let dst = Tensor.empty(shape: newShape, dtype: src.dtype, device: device)
        let dstBase = dst.buffer.contents().advanced(by: dst.offset)
        let srcBase = src.buffer.contents().advanced(by: src.offset)
        var dstRow = 0
        for (start, count) in ranges {
            memcpy(
                dstBase.advanced(by: dstRow * rowBytes),
                srcBase.advanced(by: start * rowBytes),
                count * rowBytes)
            dstRow += count
        }
        return dst
    }
}
