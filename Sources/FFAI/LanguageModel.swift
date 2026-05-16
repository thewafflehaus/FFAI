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

    /// Forward + GPU softmax-categorical-sample for the pure-temperature
    /// sampling path (T > 0, no top-K / top-P / min-P / rep-penalty).
    /// Logits never cross to CPU; only the chosen token id (4 bytes)
    /// flows back.
    ///
    /// Default impl runs forward() then queues
    /// `Ops.softmaxCategoricalSample` on a separate command buffer.
    /// Family files can override to fuse into a single cmdbuf (TODO
    /// follow-up for Llama / Qwen 3).
    func forwardSampleCategorical(
        tokenId: Int, position: Int, caches: [KVCache],
        temperature: Float, uniformDraw: Float,
        device: Device = .shared
    ) -> Int {
        let logits = forward(tokenId: tokenId, position: position,
                             caches: caches, device: device)
        let tBuf = device.makeBuffer(length: 4)
        var tVal = temperature
        memcpy(tBuf.contents(), &tVal, 4)
        let temperatureT = Tensor(buffer: tBuf, offset: 0, shape: [1], dtype: .f32)

        let uBuf = device.makeBuffer(length: 4)
        var uVal = uniformDraw
        memcpy(uBuf.contents(), &uVal, 4)
        let uniformT = Tensor(buffer: uBuf, offset: 0, shape: [1], dtype: .f32)

        let outBuf = device.makeBuffer(length: 4)
        let outT = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)

        let cmd = device.makeCommandBuffer()
        Ops.softmaxCategoricalSample(logits, into: outT,
                                     temperature: temperatureT,
                                     uniform: uniformT, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }
}
