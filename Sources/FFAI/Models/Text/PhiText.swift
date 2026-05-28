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
// Phi text — concrete variants + the dense decoder support for the
// Phi-3 / Phi-3.5 family. The family enum (`enum Phi`), variant
// protocol (`PhiVariant`), and error type (`PhiError`) live in
// `Models/Phi.swift` (the family root / main interface).
//
// This file holds:
//   • `Phi3Dense` — `PhiVariant` conformance + the per-variant
//     `loadModel` entry. Slices the fused `qkv_proj` and
//     `gate_up_proj` weights into row-views that drop straight into
//     `LlamaLayer`, then returns a `LlamaModel` engine.

import Foundation

public struct Phi3Dense: PhiVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Phi-3's reference generation defaults. Slightly cooler than
    /// Llama (temperature 0.0 by default in HF examples), but we
    /// mirror Llama's family default so the user-facing surface is
    /// consistent across families.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.6,
        topP: 1.0,
        topK: 0,
        repetitionPenalty: 1.0
    )

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
            throw PhiError.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // Phi-3 puts head_dim in the config explicitly on later
        // revisions, but the canonical Phi-3-mini config does not —
        // it implies head_dim = hidden / nHeads.
        let headDim = config.headDim ?? (hidden / nHeads)
        let theta = Float(config.ropeTheta ?? 10_000)
        let maxSeq = config.int("max_position_embeddings") ?? 4096
        let tieEmbed = config.tieWordEmbeddings

        // SuScaledRoPE / longrope is queued as a follow-up. For now
        // refuse those variants explicitly so the user gets a clear
        // error instead of a silently-wrong RoPE.
        if let rs = config.nested("rope_scaling"),
            let t = rs["type"] as? String ?? rs["rope_type"] as? String,
            t != "default" && t != "linear"
        {
            throw PhiError.unsupportedRopeScaling(t)
        }
        // Linear rope_scaling is just a frequency divisor — propagate
        // via the Llama machinery's ropeScaling.scaleFactor.
        let ropeScaling: Ops.RoPEScaling
        if let rs = config.nested("rope_scaling"),
            let factor = rs["factor"] as? Double
        {
            // Linear scaling reduces frequencies uniformly; Llama's
            // Ops.RoPEScaling expresses this as `scaleFactor`.
            ropeScaling = Ops.RoPEScaling(
                scaleFactor: Float(factor),
                lowFreqFactor: 1.0,
                highFreqFactor: 1.0,
                originalMaxPosition: Float(maxSeq)
            )
        } else {
            ropeScaling = .none
        }

        let quant = config.quantization
        // Embedding (Phi-3 fp16/bf16 keep this dense; some 4-bit
        // packs do quantize it — same path as Llama).
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        // Layers
        var layers: [LlamaLayer] = []
        layers.reserveCapacity(nLayers)

        let qSize = nHeads * headDim
        let kvSize = nKVHeads * headDim

        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"

            // Fused qkv split. Phi-3 ships q/k/v fused along dim 0
            // ([(Q + 2·KV) * head_dim, hidden]). Affine quantization
            // groups the *hidden* dim (dim 1) — so axis-0 row slicing
            // preserves group boundaries and is safe for both the
            // raw weight and the packed u32 triplet (weight + scales
            // + biases all share dim 0). The 4-bit conversion of
            // Phi-3-mini-4k matches this contract: qSize = kvSize =
            // 3072, both multiples of group_size and pack_factor.
            let qProj: AnyLinear
            let kProj: AnyLinear
            let vProj: AnyLinear
            if let q = quant, weights.isQuantized("\(p).self_attn.qkv_proj") {
                let triplet = try weights.quantizedTriplet("\(p).self_attn.qkv_proj")
                let qkvParts = splitQuantizedTripletRowsPhi(
                    triplet, counts: [qSize, kvSize, kvSize], expectedTotalRows: qSize + 2 * kvSize,
                    quantization: q, label: "qkv_proj")
                qProj = qkvParts[0]
                kProj = qkvParts[1]
                vProj = qkvParts[2]
            } else {
                let qkvFused = try weights.tensor(named: "\(p).self_attn.qkv_proj.weight")
                precondition(
                    qkvFused.shape == [qSize + 2 * kvSize, hidden],
                    "Phi3 qkv_proj shape mismatch: got \(qkvFused.shape), expected [\(qSize + 2 * kvSize), \(hidden)]"
                )
                let qWeight = qkvFused.slicedRows(start: 0, count: qSize)
                let kWeight = qkvFused.slicedRows(start: qSize, count: kvSize)
                let vWeight = qkvFused.slicedRows(start: qSize + kvSize, count: kvSize)
                qProj = AnyLinear(Linear(weight: qWeight))
                kProj = AnyLinear(Linear(weight: kWeight))
                vProj = AnyLinear(Linear(weight: vWeight))
            }

            let oProj = try loadLinear(
                base: "\(p).self_attn.o_proj",
                in: weights, quantization: quant)

            // Fused gate_up split — same axis-0 slicing argument.
            let gateProj: AnyLinear
            let upProj: AnyLinear
            if let q = quant, weights.isQuantized("\(p).mlp.gate_up_proj") {
                let triplet = try weights.quantizedTriplet("\(p).mlp.gate_up_proj")
                let parts = splitQuantizedTripletRowsPhi(
                    triplet, counts: [intermediate, intermediate],
                    expectedTotalRows: 2 * intermediate,
                    quantization: q, label: "gate_up_proj")
                gateProj = parts[0]
                upProj = parts[1]
            } else {
                let gateUpFused = try weights.tensor(named: "\(p).mlp.gate_up_proj.weight")
                precondition(
                    gateUpFused.shape == [2 * intermediate, hidden],
                    "Phi3 gate_up_proj shape mismatch: got \(gateUpFused.shape), expected [\(2 * intermediate), \(hidden)]"
                )
                let gateWeight = gateUpFused.slicedRows(start: 0, count: intermediate)
                let upWeight = gateUpFused.slicedRows(start: intermediate, count: intermediate)
                gateProj = AnyLinear(Linear(weight: gateWeight))
                upProj = AnyLinear(Linear(weight: upWeight))
            }

            let downProj = try loadLinear(
                base: "\(p).mlp.down_proj",
                in: weights, quantization: quant)

            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: Float(eps))
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: Float(eps))

            layers.append(
                LlamaLayer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                    headDim: headDim, intermediate: intermediate,
                    ropeTheta: theta, ropeScaling: ropeScaling
                ))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"),
            eps: Float(eps))

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("model.embed_tokens") {
            let t = try weights.quantizedTriplet("model.embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: bits, groupSize: q.groupSize
                ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        let activationDtype: DType
        if weights.isQuantized("model.embed_tokens"),
            let scales = try? weights.tensor(named: "model.embed_tokens.scales")
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
            kvEviction: options.kvEviction
        )
    }
}

// ─── Quantized fused-projection split helpers ────────────────────────
//
// Phi-3 ships `qkv_proj` and `gate_up_proj` as single fused tensors
// along dim 0 (rows). The 4-bit (affine) conversion preserves the
// fusion: the packed u32 `weight`, the bf16 `scales`, and the bf16
// `biases` all share the same dim-0 layout, and the group_size /
// pack_factor packing happens along dim 1. Row-wise slicing is
// therefore safe for the quantized triplet — each slice is a
// self-contained `QuantizedLinear` with its own row range and the
// full hidden-side column range. This is the same axis-0-fused
// pattern Llama-style models use for unfused QKV.

/// Split a fused quantized triplet (`[fusedRows, hidden/{pack,group}]`)
/// into N `AnyLinear`s by row counts. Preconditions: row counts must
/// sum to `expectedTotalRows`; the triplet's three tensors must share
/// dim 0 = expectedTotalRows.
private func splitQuantizedTripletRowsPhi(
    _ triplet: SafeTensorsBundle.QuantizedTriplet,
    counts: [Int],
    expectedTotalRows: Int,
    quantization q: ModelConfig.QuantizationConfig,
    label: String
) -> [AnyLinear] {
    precondition(
        counts.reduce(0, +) == expectedTotalRows,
        "Phi3 \(label): row counts \(counts) sum to "
            + "\(counts.reduce(0, +)) but expected \(expectedTotalRows)")
    precondition(
        triplet.weight.shape.count == 2
            && triplet.weight.shape[0] == expectedTotalRows,
        "Phi3 \(label): weight rows \(triplet.weight.shape) "
            + "≠ expected \(expectedTotalRows)")
    precondition(
        triplet.scales.shape.count == 2
            && triplet.scales.shape[0] == expectedTotalRows,
        "Phi3 \(label): scales rows \(triplet.scales.shape) "
            + "≠ expected \(expectedTotalRows)")
    precondition(
        triplet.biases.shape.count == 2
            && triplet.biases.shape[0] == expectedTotalRows,
        "Phi3 \(label): biases rows \(triplet.biases.shape) "
            + "≠ expected \(expectedTotalRows)")
    // bits derives from the packed_cols / scale_cols ratio — same
    // for every slice (they all share the hidden-side column range).
    let bits = deriveAffineQuantBits(
        weightPackedCols: triplet.weight.shape[triplet.weight.shape.count - 1],
        scaleCols: triplet.scales.shape[triplet.scales.shape.count - 1],
        groupSize: q.groupSize)
    var out: [AnyLinear] = []
    out.reserveCapacity(counts.count)
    var rowStart = 0
    for cnt in counts {
        let w = triplet.weight.slicedRows(start: rowStart, count: cnt)
        let s = triplet.scales.slicedRows(start: rowStart, count: cnt)
        let b = triplet.biases.slicedRows(start: rowStart, count: cnt)
        out.append(
            AnyLinear(
                QuantizedLinear(
                    weight: w, scales: s, biases: b,
                    bits: bits, groupSize: q.groupSize)))
        rowStart += cnt
    }
    return out
}
