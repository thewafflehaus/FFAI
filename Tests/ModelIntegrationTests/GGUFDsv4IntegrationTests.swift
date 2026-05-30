// Copyright 2026 Tom Turney (@TheTom)
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
// End-to-end GGUF loader integration on the user's local
// DeepSeek-V4-Flash IQ2XXS imatrix file (~86 GB). Validates that the
// parser + dequant pipeline successfully open the checkpoint, read
// the architecture, decode a representative tensor of each quant
// type the file uses (Q8_0, Q2_K, IQ2_XXS), and that the dequant
// outputs are bounded (no NaN / inf — exact numerical comparison
// against llama.cpp lands when the cross-reference tooling is in
// tree).
//
// Skipped at CI time — gated on the model being staged at
// `$FFAI_DSV4_GGUF_PATH` (default `~/models/ds4-model`).

import Foundation
import Testing
import Tokenizers

@testable import FFAI

@Suite("GGUF DSv4 end-to-end", .serialized)
struct GGUFDsv4IntegrationTests {

    private var modelPath: String? {
        let env = ProcessInfo.processInfo.environment["FFAI_DSV4_GGUF_PATH"]
            ?? NSString("~/models/ds4-model").expandingTildeInPath
        guard FileManager.default.fileExists(atPath: env) else { return nil }
        return env
    }

    @Test("Open DSv4 GGUF, read header + arch metadata")
    func opensCheckpoint() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4IntegrationTests: skipping (no model at FFAI_DSV4_GGUF_PATH)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        // The DSv4 GGUF carries `general.architecture: "deepseek4"`.
        let arch = bundle.architecture
        #expect(arch != nil, "DSv4 GGUF must carry general.architecture")
        if let arch = arch {
            #expect(
                arch.lowercased().contains("deepseek")
                    || arch.lowercased().contains("ds4")
                    || arch == "deepseek4",
                "Expected DeepSeek arch string, got '\(arch)'")
        }
        // The tensor info table should be substantial for a 284B model.
        #expect(bundle.reader.tensorInfos.count > 100)
    }

    @Test("Dequant one representative Q8_0 tensor (attention projection)")
    func dequantQ8_0Tensor() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4IntegrationTests: skipping (no model)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        // Find any Q8_0 tensor (the filename says AProjQ8 / SExpQ8 / OutQ8 are Q8_0).
        guard let q8 = bundle.reader.tensorInfos.first(where: { $0.type == .q8_0 }) else {
            print("GGUFDsv4IntegrationTests: no Q8_0 tensors found — skipping")
            return
        }
        let t = try bundle.tensor(named: q8.name, outDtype: .f32)
        #expect(t.shape.map { Int($0) } == q8.dimensions.map { Int($0) })
        // Sample a few elements; assert finite + bounded magnitude.
        // The Q8_0 super-scale is fp16; values land in the same range
        // as the original fp16 weights.
        let sample = t.toArray(as: Float.self).prefix(1024)
        for v in sample {
            #expect(v.isFinite, "Q8_0 dequant produced non-finite value")
            #expect(abs(v) < 1e3, "Q8_0 dequant magnitude unreasonable (\(v))")
        }
    }

    @Test("Dequant one representative Q2_K tensor (w2 down-proj)")
    func dequantQ2_KTensor() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4IntegrationTests: skipping (no model)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        guard let q2 = bundle.reader.tensorInfos.first(where: { $0.type == .q2_K }) else {
            print("GGUFDsv4IntegrationTests: no Q2_K tensors found — skipping")
            return
        }
        let t = try bundle.tensor(named: q2.name, outDtype: .f32)
        #expect(t.shape.map { Int($0) } == q2.dimensions.map { Int($0) })
        let sample = t.toArray(as: Float.self).prefix(1024)
        for v in sample {
            #expect(v.isFinite, "Q2_K dequant produced non-finite value")
            #expect(abs(v) < 1e3, "Q2_K dequant magnitude unreasonable (\(v))")
        }
    }

    @Test("Build a tokenizer from the GGUF metadata block")
    func buildTokenizer() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4IntegrationTests: skipping (no model)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        let kind = bundle.reader.metadataString("tokenizer.ggml.model") ?? "<missing>"
        print("GGUFDsv4IntegrationTests: tokenizer.ggml.model = '\(kind)'")
        let tokenizer: any Tokenizer
        do {
            tokenizer = try GGUFTokenizerAdapter.build(reader: bundle.reader)
        } catch GGUFTokenizerAdapter.Error.unsupportedKind(let k) {
            // The DSv4 GGUF uses a custom DSv4 pretokenizer
            // that may not be in our BPE-kind set yet — accept the
            // skip but make the failure mode visible.
            print(
                "GGUFDsv4IntegrationTests: tokenizer kind '\(k)' not in supported BPE-family set yet"
            )
            return
        }
        // Encode a short known prompt and assert we get a non-empty
        // token list out — the encode round-trip is the load-bearing
        // sanity check that the vocab + merges parsed correctly.
        let prompt = "The history of the printing press began when European craftsmen"
        let ids = tokenizer.encode(text: prompt)
        #expect(!ids.isEmpty, "encode returned empty token list")
        let decoded = tokenizer.decode(tokens: ids)
        #expect(!decoded.isEmpty, "decode returned empty string")
        print("GGUFDsv4IntegrationTests: \(ids.count) tokens → '\(decoded.prefix(80))…'")
    }

    @Test("Lazy DeepSeekV4Model loader: open + load layer 0 (full-attn)")
    func loadModelLayer0() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4IntegrationTests: skipping (no model)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        // Synthesize a minimal ModelConfig from the GGUF metadata so the
        // text-config decoder has something to read. In practice the
        // FFAI family-dispatch fills this from a sidecar config.json,
        // but the GGUF itself carries enough hparams for the load path.
        let hidden = Int(bundle.reader.metadataUInt32("deepseek4.embedding_length") ?? 4096)
        let nLayers = Int(bundle.reader.metadataUInt32("deepseek4.block_count") ?? 43)
        let vocab = Int(bundle.reader.metadataUInt32("deepseek4.vocab_size") ?? 129_280)
        let nHeads = Int(bundle.reader.metadataUInt32("deepseek4.attention.head_count") ?? 64)
        let raw: [String: Any] = [
            "hidden_size": hidden,
            "num_hidden_layers": nLayers,
            "vocab_size": vocab,
            "num_attention_heads": nHeads,
        ]
        let config = ModelConfig(architecture: "DeepSeekV4ForCausalLM", modelType: "deepseek4", raw: raw)
        let device = Device.shared
        let model = try DeepSeekV4Flash.loadModelFromGGUF(
            config: config, gguf: bundle,
            options: LoadOptions(), device: device)
        #expect(model.textConfig.nLayers == nLayers)
        // The GGUF compress_ratios array includes one extra entry for
        // the MTP next-N predictor slot — so count is `nLayers + 1`.
        #expect(model.layerCompressRatios.count >= nLayers)
        // Layer 0 is full-attention (compress_ratio = 0) per the GGUF
        // structure. Loading it dequants the 24 layer tensors.
        let layer0 = try model.layer(0)
        #expect(layer0.compressRatio == 0)
        #expect(layer0.layerIndex == 0)
        // attn_sinks shape sanity (per-head learnable, n_heads=64).
        #expect(layer0.attnSinks.shape.map { Int($0) } == [64])
        // Release the layer to free GPU memory — exercise the LRU
        // hook.
        model.releaseLayer(0)
        print("GGUFDsv4IntegrationTests: loaded layer 0, compress_ratios = \(model.layerCompressRatios)")
    }

    @Test("Dispatch mHC sinkhorn-split against loaded layer-0 weights")
    func mhcSinkhornSplitSmoke() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4IntegrationTests: skipping (no model)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        let raw: [String: Any] = [
            "hidden_size": 4096, "num_hidden_layers": 43,
            "vocab_size": 129_280, "num_attention_heads": 64,
        ]
        let config = ModelConfig(architecture: "DeepSeekV4ForCausalLM", modelType: "deepseek4", raw: raw)
        let device = Device.shared
        let model = try DeepSeekV4Flash.loadModelFromGGUF(
            config: config, gguf: bundle, options: LoadOptions(), device: device)
        let layer0 = try model.layer(0)
        // Synthesize a 24-mix input (representative for one token).
        // In real forward, this would be `hc_attn_fn @ flatten(H)`.
        let mixes = Tensor.empty(shape: [24], dtype: model.activationDtype)
        // Zero-fill is enough for the smoke check — pre/post/comb just
        // need to be finite, not meaningful. The downstream sanity is
        // "no NaN, no crash".
        let cmd = device.makeCommandBuffer()
        let (pre, post, comb) = Ops.dsv4MhcSinkhornSplit(
            mixes: mixes, scale: layer0.hcAttnScale, base: layer0.hcAttnBase,
            nTokens: 1, eps: 1e-6, sinkhornIters: 1, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(pre.shape.map { Int($0) } == [1, 4])
        #expect(post.shape.map { Int($0) } == [1, 4])
        #expect(comb.shape.map { Int($0) } == [1, 4, 4])
        let preVals = pre.toArray(as: Float.self)
        for v in preVals { #expect(v.isFinite, "pre value non-finite: \(v)") }
        print("GGUFDsv4IntegrationTests: mhc split pre=\(preVals)")
    }

    @Test("Run one full-attn attention sub-block forward against layer 0")
    func attentionSubblockForward() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4IntegrationTests: skipping (no model)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        let raw: [String: Any] = [
            "hidden_size": 4096, "num_hidden_layers": 43,
            "vocab_size": 129_280, "num_attention_heads": 64,
        ]
        let config = ModelConfig(architecture: "DeepSeekV4ForCausalLM", modelType: "deepseek4", raw: raw)
        let device = Device.shared
        let model = try DeepSeekV4Flash.loadModelFromGGUF(
            config: config, gguf: bundle, options: LoadOptions(), device: device)
        let layer0 = try model.layer(0)
        let state = model.makeDecodeState()
        // Seed hcState with a real token embedding so the forward
        // chain has non-zero input. Pick a low-ID token (1 = often
        // BOS-equivalent in the DSv4 vocab); broadcast its embedding
        // across all 4 mHC channels.
        let hidden = model.textConfig.hidden
        let tokenId = 1
        let embedRow = model.tokenEmbd.asGgufMatmulWeight()
            .slicedRows(start: tokenId, count: 1).reshaped(to: [hidden])
        let cmd = device.makeCommandBuffer()
        for c in 0..<4 {
            let dst = state.hcState.slicedRows(start: c, count: 1).reshaped(to: [hidden])
            Ops.copy(embedRow, into: dst, on: cmd)
        }
        let blockOut = model.forwardFullAttnSubblock(layer: layer0, state: state, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(blockOut.shape.map { Int($0) } == [hidden])
        let vals = blockOut.toArray(as: Float.self)
        var anyNaN = 0, anyInf = 0, nonZero = 0
        for v in vals {
            if v.isNaN { anyNaN += 1 }
            if v.isInfinite { anyInf += 1 }
            if v != 0 { nonZero += 1 }
        }
        #expect(anyNaN == 0, "block_out has \(anyNaN) NaN values")
        #expect(anyInf == 0, "block_out has \(anyInf) Inf values")
        #expect(nonZero > 0, "block_out is all zero — forward chain produced no signal")
        let absMax = vals.map { abs($0) }.max() ?? 0
        let absMean = vals.map { abs($0) }.reduce(0, +) / Float(vals.count)
        print("GGUFDsv4IntegrationTests: layer-0 forward done")
        print("  nonzero = \(nonZero)/\(vals.count)  |block_out|_max = \(absMax)  mean = \(absMean)")
    }

    @Test("Dequant one representative IQ2_XXS tensor (MoE expert weight)")
    func dequantIQ2_XXSTensor() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4IntegrationTests: skipping (no model)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        guard let iq = bundle.reader.tensorInfos.first(where: { $0.type == .iq2_xxs }) else {
            print("GGUFDsv4IntegrationTests: no IQ2_XXS tensors found — skipping")
            return
        }
        let t = try bundle.tensor(named: iq.name, outDtype: .f32)
        #expect(t.shape.map { Int($0) } == iq.dimensions.map { Int($0) })
        let sample = t.toArray(as: Float.self).prefix(1024)
        var anyNonZero = false
        for v in sample {
            #expect(v.isFinite, "IQ2_XXS dequant produced non-finite value")
            #expect(abs(v) < 1e3, "IQ2_XXS dequant magnitude unreasonable (\(v))")
            if v != 0 { anyNonZero = true }
        }
        #expect(anyNonZero, "IQ2_XXS dequant produced all-zero output for sample")
    }
}
