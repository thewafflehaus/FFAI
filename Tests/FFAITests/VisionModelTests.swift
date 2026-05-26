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
import Foundation
import Metal
import Testing
@testable import FFAI

// VisionModel — cross-modal token splice + VL generate composition.
//
// The splice is the load-bearing VL-specific logic: it interleaves
// vision-encoder patch tokens into the text embedding stream at the
// image-placeholder positions. These tests drive it with a minimal stub
// engine so the splice contract is exercised without a multi-GB
// checkpoint (real-checkpoint coherence is the ModelTests integration
// suites' job).
@Suite("VisionModel")
struct VisionModelTests {

    /// Minimal `LanguageModel` stub — supports the embedding-input
    /// surface a `VisionModel` needs. `textEmbedding` returns a tagged
    /// constant row per token id so the splice ordering is checkable.
    final class StubEngine: LanguageModel {
        let hidden: Int
        let nLayers = 1, nHeads = 1, nKVHeads = 1, headDim = 8
        let vocab = 1000, maxSeq = 4096
        let dtype: DType = .f32

        init(hidden: Int) { self.hidden = hidden }

        func parameters() -> [(String, Tensor)] { [] }

        func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
            [KVCache(nKVHeads: nKVHeads, headDim: headDim,
                     maxSeq: maxSeq ?? self.maxSeq, dtype: dtype, device: device)]
        }

        func forward(tokenId: Int, position: Int,
                     caches: [any LayerCacheProtocol],
                     on cmd: MTLCommandBuffer, device: Device) -> Tensor {
            Tensor.empty(shape: [vocab], dtype: dtype)
        }

        func forwardSample(tokenId: Int, position: Int,
                            caches: [any LayerCacheProtocol], device: Device) -> Int {
            (tokenId + 1) % vocab
        }

        var supportsEmbeddingInput: Bool { true }

        func forward(inputEmbedding: Tensor, position: Int,
                     caches: [any LayerCacheProtocol],
                     on cmd: MTLCommandBuffer, device: Device) -> Tensor {
            Tensor.empty(shape: [vocab], dtype: dtype)
        }

        /// Tag every element of the row with `tokenId` so the splice
        /// output is identity-checkable.
        func textEmbedding(tokenId: Int, device: Device) -> Tensor {
            let t = Tensor.empty(shape: [hidden], dtype: dtype, device: device)
            t.copyIn(from: [Float](repeating: Float(tokenId), count: hidden))
            return t
        }
    }

    /// A tiny VisionEncoder whose `encode` is never called by these
    /// tests (they pass image tokens directly into `splice`).
    private func makeStubVisionEncoder(hidden: Int) -> VisionEncoder {
        let config = VisionEncoderConfig(
            imageSize: 16, patchSize: 16, hidden: hidden,
            intermediate: hidden * 2, nLayers: 1, nHeads: 1, textHidden: hidden)
        func t(_ shape: [Int]) -> Tensor {
            let n = shape.reduce(1, *)
            let tensor = Tensor.empty(shape: shape, dtype: .f32)
            tensor.copyIn(from: [Float](repeating: 0, count: n))
            return tensor
        }
        let ln = LayerNorm(weight: t([hidden]), bias: t([hidden]), eps: 1e-6)
        let layer = VisionEncoderLayer(
            layerNorm1: ln,
            qProj: Linear(weight: t([hidden, hidden]), bias: t([hidden])),
            kProj: Linear(weight: t([hidden, hidden]), bias: t([hidden])),
            vProj: Linear(weight: t([hidden, hidden]), bias: t([hidden])),
            oProj: Linear(weight: t([hidden, hidden]), bias: t([hidden])),
            layerNorm2: ln,
            fc1: Linear(weight: t([hidden * 2, hidden]), bias: t([hidden * 2])),
            fc2: Linear(weight: t([hidden, hidden * 2]), bias: t([hidden])),
            hidden: hidden, nHeads: 1, intermediate: hidden * 2)
        return VisionEncoder(
            config: config, patchEmbedWeight: t([hidden, 3, 16, 16]),
            patchEmbedBias: t([hidden]),
            positionEmbedding: t([config.numPatches, hidden]),
            layers: [layer], postLayerNorm: ln, projection: nil, dtype: .f32)
    }

    @Test("VisionModel — rejects an engine without embedding-input support")
    func rejectsUnsupportedEngine() {
        // A text-only engine (supportsEmbeddingInput == false) can't VLM.
        final class TextOnly: LanguageModel {
            let hidden = 8, nLayers = 1, nHeads = 1, nKVHeads = 1
            let headDim = 8, vocab = 100, maxSeq = 64
            let dtype: DType = .f32
            func parameters() -> [(String, Tensor)] { [] }
            func makeLayerCaches(maxSeq: Int?, device: Device)
                -> [any LayerCacheProtocol] { [] }
            func forward(tokenId: Int, position: Int,
                         caches: [any LayerCacheProtocol],
                         on cmd: MTLCommandBuffer, device: Device) -> Tensor {
                Tensor.empty(shape: [vocab], dtype: dtype)
            }
            func forwardSample(tokenId: Int, position: Int,
                               caches: [any LayerCacheProtocol],
                               device: Device) -> Int { 0 }
        }
        let enc = makeStubVisionEncoder(hidden: 8)
        #expect(throws: VisionModelError.self) {
            _ = try VisionModel(visionEncoder: enc, engine: TextOnly(),
                            imageTokenId: 42, normalization: .siglip)
        }
    }

    @Test("splice — image tokens land at placeholder positions")
    func spliceInjectsImageTokens() throws {
        let hidden = 8
        let enc = makeStubVisionEncoder(hidden: hidden)
        let vlm = try VisionModel(visionEncoder: enc, engine: StubEngine(hidden: hidden),
                              imageTokenId: 99, normalization: .siglip)

        // Prompt: text, text, <image>, <image>, text.
        let prompt = [10, 11, 99, 99, 12]
        // Two vision rows, tagged 1000.0 and 2000.0.
        let imageTokens = Tensor.empty(shape: [2, hidden], dtype: .f32)
        var data = [Float](repeating: 0, count: 2 * hidden)
        for c in 0..<hidden { data[c] = 1000; data[hidden + c] = 2000 }
        imageTokens.copyIn(from: data)

        let stream = try vlm.splice(promptTokens: prompt, imageTokens: imageTokens)
        #expect(stream.count == 5)
        // Text positions carry the token-id tag; image positions carry
        // the vision row tag.
        #expect(stream[0].toArray(as: Float.self)[0] == 10)
        #expect(stream[1].toArray(as: Float.self)[0] == 11)
        #expect(stream[2].toArray(as: Float.self)[0] == 1000)
        #expect(stream[3].toArray(as: Float.self)[0] == 2000)
        #expect(stream[4].toArray(as: Float.self)[0] == 12)
    }

    @Test("splice — placeholder count must match vision row count")
    func splicePlaceholderMismatch() throws {
        let hidden = 8
        let enc = makeStubVisionEncoder(hidden: hidden)
        let vlm = try VisionModel(visionEncoder: enc, engine: StubEngine(hidden: hidden),
                              imageTokenId: 99, normalization: .siglip)
        // One placeholder, but two vision rows.
        let prompt = [10, 99, 11]
        let imageTokens = Tensor.empty(shape: [2, hidden], dtype: .f32)
        imageTokens.copyIn(from: [Float](repeating: 0, count: 2 * hidden))
        #expect(throws: VisionModelError.self) {
            _ = try vlm.splice(promptTokens: prompt, imageTokens: imageTokens)
        }
    }

    @Test("splice — single image token at the head of the prompt")
    func spliceImageAtHead() throws {
        let hidden = 8
        let enc = makeStubVisionEncoder(hidden: hidden)
        let vlm = try VisionModel(visionEncoder: enc, engine: StubEngine(hidden: hidden),
                              imageTokenId: 99, normalization: .siglip)
        let prompt = [99, 5, 6, 7]
        let imageTokens = Tensor.empty(shape: [1, hidden], dtype: .f32)
        imageTokens.copyIn(from: [Float](repeating: 777, count: hidden))
        let stream = try vlm.splice(promptTokens: prompt, imageTokens: imageTokens)
        #expect(stream.count == 4)
        #expect(stream.map { $0.toArray(as: Float.self)[0] } == [777, 5, 6, 7])
    }

    @Test("imageTokenCount — equals the vision encoder patch count")
    func imageTokenCount() throws {
        let enc = makeStubVisionEncoder(hidden: 8)
        let vlm = try VisionModel(visionEncoder: enc, engine: StubEngine(hidden: 8),
                              imageTokenId: 99, normalization: .siglip)
        // 16x16 image, 16x16 patch → 1 patch.
        #expect(vlm.imageTokenCount == 1)
    }
}
