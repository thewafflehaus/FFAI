// LanguageModel — common surface every text-generating model conforms
// to. Lets Generate.swift and the CLI work against any model family
// without knowing the concrete type.

import Foundation
import Metal

public protocol LanguageModel: Module {
    var hidden: Int { get }
    var nLayers: Int { get }
    var nHeads: Int { get }
    var nKVHeads: Int { get }
    var headDim: Int { get }
    var vocab: Int { get }
    var maxSeq: Int { get }
    var dtype: DType { get }

    /// One per-layer KV cache, sized for the model's defaults.
    func makeKVCache(maxSeq: Int?, device: Device) -> [KVCache]

    /// Single-token forward pass. Returns logits [vocab].
    func forward(tokenId: Int, position: Int, caches: [KVCache], device: Device) -> Tensor

    /// Forward + GPU argmax in one command buffer. Returns just the
    /// chosen token id (4-byte readback) — no full logits transfer.
    func forwardSample(tokenId: Int, position: Int,
                       caches: [KVCache], device: Device) -> Int
}

public extension LanguageModel {
    func makeKVCache(maxSeq: Int? = nil, device: Device = .shared) -> [KVCache] {
        makeKVCache(maxSeq: maxSeq, device: device)
    }

    func forward(tokenId: Int, position: Int, caches: [KVCache]) -> Tensor {
        forward(tokenId: tokenId, position: position, caches: caches, device: .shared)
    }

    func forwardSample(tokenId: Int, position: Int, caches: [KVCache]) -> Int {
        forwardSample(tokenId: tokenId, position: position, caches: caches, device: .shared)
    }
}
