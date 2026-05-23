// Standard transformer layer building blocks: Linear, Embedding,
// RMSNorm. Each holds its weight tensors as fields and exposes
// `parameters()` for SafeTensors weight binding.

import Foundation
import Metal

// ─── Linear (no bias for now — Llama doesn't use biases) ─────────────

public final class Linear: Module {
    /// weight shape [out_features, in_features], row-major.
    public let weight: Tensor
    /// Optional bias shape [out_features]. `nil` means no bias (the
    /// 3-series + Mistral7B default). Qwen 2.x, BLOOM, some Falcon and
    /// older variants ship biases on the QKV projections.
    public let bias: Tensor?

    public init(weight: Tensor, bias: Tensor? = nil) {
        precondition(weight.shape.count == 2, "Linear: weight must be 2D")
        if let b = bias {
            precondition(b.shape == [weight.shape[0]],
                         "Linear: bias shape \(b.shape) must match output features \([weight.shape[0]])")
            precondition(b.dtype == weight.dtype,
                         "Linear: bias dtype must match weight dtype")
        }
        self.weight = weight
        self.bias = bias
    }

    public func parameters() -> [(String, Tensor)] {
        if let b = bias {
            return [("weight", weight), ("bias", b)]
        }
        return [("weight", weight)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        let y = Ops.gemv(weight: weight, input: x, on: cmd)
        if let b = bias {
            return Ops.add(y, b, on: cmd)
        }
        return y
    }

    /// T-batched. `x` is `[T, inDim]` flat row-major, returns
    /// `[T, outDim]` flat. Dispatches `Ops.gemm` (one kernel, no per-row
    /// host overhead). Bias path is unimplemented — the families that
    /// currently call this (Qwen3-series) do not use linear biases.
    ///
    /// (Rebase 2026-05-23: keep Tom's `callMany` API, drop my parallel
    /// `batched(_:nRows:)` from `3a87b78`. Both did the same job;
    /// Tom's QuantizedLinear.callMany variant actually wraps the
    /// `mt_qmm_mma`-backed `Ops.dequantGemmDynamicM` kernel — what my
    /// doc comment had flagged as a TODO. Caller updates in Llama /
    /// Qwen3 forwardMulti come in this same rebase commit.)
    public func callMany(_ x: Tensor, t: Int, on cmd: MTLCommandBuffer) -> Tensor {
        precondition(bias == nil,
                     "Linear.callMany: bias broadcast over T rows not implemented; no Qwen3-series caller needs it today")
        return Ops.gemm(weight: weight, input: x, nRows: t, on: cmd)
    }
}

// ─── QuantizedLinear (mlx int4 format) ────────────────────────────────

/// Linear layer backed by mlx-format quantized weights (int4 or int8).
/// Storage is the (weight, scales, biases) triplet plus (bits, group_size).
///
///   weight   [out_features, in_features / pack_factor]  uint32
///            pack_factor = 32 / bits (8 for int4, 4 for int8)
///   scales   [out_features, in_features / group_size]
///   biases   [out_features, in_features / group_size]
///
/// callAsFunction dispatches Ops.dequantGemv — fused dequant + gemv.
public final class QuantizedLinear: Module {
    public let weight: Tensor
    public let scales: Tensor
    public let biases: Tensor
    public let bits: Int
    public let groupSize: Int

    public init(weight: Tensor, scales: Tensor, biases: Tensor,
                bits: Int, groupSize: Int) {
        precondition(weight.dtype == .u32, "QuantizedLinear: weight must be u32 packed")
        precondition(weight.shape.count == 2, "QuantizedLinear: weight must be 2D")
        precondition(bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
                     "QuantizedLinear: bits must be one of 3, 4, 5, 6, or 8")
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.bits = bits
        self.groupSize = groupSize
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales), ("biases", biases)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.dequantGemv(
            weight: weight, scales: scales, biases: biases,
            input: x, bits: bits, groupSize: groupSize, on: cmd
        )
    }

    /// T-batched. `x` is `[T, inDim]` flat row-major, returns
    /// `[T, outDim]` flat. Dispatches `Ops.dequantGemmDynamicM` — one
    /// `mt_qmm_mma` kernel for T multiple of 32, or one padded
    /// dispatch + host slice for ragged T. Matches the batched-prefill
    /// path (BP2).
    public func callMany(_ x: Tensor, t: Int,
                         on cmd: MTLCommandBuffer,
                         device: Device) -> Tensor {
        let outDim = weight.shape[0]
        let inDim = scales.shape[scales.shape.count - 1] * groupSize
        // 4-bit goes through the fast mt_qmm_mma path. Other bit-widths
        // (commonly 8-bit on smaller projections like Qwen3.5's
        // shared_expert_gate at hidden→1) fall back to T sequential
        // `dequantGemv` calls on the same `cmd`. Slower than a true
        // batched kernel but bit-identical to the per-token path, and
        // the projections that hit this branch are tiny so the per-token
        // launch overhead is in the noise.
        if bits == 4 {
            let out = Tensor.empty(shape: [t, outDim], dtype: x.dtype, device: device)
            Ops.dequantGemmDynamicM(
                input: x, weight: weight, scales: scales, biases: biases,
                t: t, nOut: outDim, kIn: inDim, groupSize: groupSize,
                on: cmd, device: device, into: out)
            return out
        }
        let dtBytes = x.dtype.byteSize
        let out = Tensor.empty(shape: [t, outDim], dtype: x.dtype, device: device)
        for r in 0..<t {
            let xRow = Tensor(buffer: x.buffer,
                              offset: x.offset + r * inDim * dtBytes,
                              shape: [inDim], dtype: x.dtype)
            let rowOut = Ops.dequantGemv(
                weight: weight, scales: scales, biases: biases,
                input: xRow, bits: bits, groupSize: groupSize, on: cmd)
            let outRow = Tensor(buffer: out.buffer,
                                offset: out.offset + r * outDim * dtBytes,
                                shape: [outDim], dtype: x.dtype)
            Ops.copy(rowOut, into: outRow, on: cmd)
        }
        return out
    }
}

/// Type-erasing wrapper so layers can hold either a regular Linear or a
/// QuantizedLinear without templating every call site.
public final class AnyLinear: Module {
    public let inner: any Module
    private let forward: (Tensor, MTLCommandBuffer) -> Tensor
    private let forwardMany: (Tensor, Int, MTLCommandBuffer, Device) -> Tensor

    public init(_ linear: Linear) {
        self.inner = linear
        self.forward = { linear($0, on: $1) }
        self.forwardMany = { x, t, cmd, _ in linear.callMany(x, t: t, on: cmd) }
    }

    public init(_ linear: QuantizedLinear) {
        self.inner = linear
        self.forward = { linear($0, on: $1) }
        self.forwardMany = { x, t, cmd, dev in
            linear.callMany(x, t: t, on: cmd, device: dev)
        }
    }

    public func parameters() -> [(String, Tensor)] { inner.parameters() }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        forward(x, cmd)
    }

    /// T-batched. `x` is `[T, inDim]` flat row-major, returns
    /// `[T, outDim]` flat. Dispatches one `gemm` (dense) or one
    /// `dequantGemmDynamicM` (quantized). Unblocks batched-prefill.
    public func callMany(_ x: Tensor, t: Int,
                         on cmd: MTLCommandBuffer,
                         device: Device) -> Tensor {
        forwardMany(x, t, cmd, device)
    }
}

/// Derive the affine-quantization bit-width of one mlx-quantized tensor
/// from its packed shapes, instead of trusting the global `bits` in
/// `config.json`.
///
/// MLX affine quantization packs `32 / bits` weight values into each
/// uint32 lane, so a linear with `inFeatures` inputs stores its packed
/// `weight` with `inFeatures * bits / 32` columns and its `scales` with
/// `inFeatures / groupSize` columns. Eliminating `inFeatures` gives
///
///     bits = 32 * weightPackedCols / (scaleCols * groupSize)
///
/// Mixed-precision checkpoints (e.g. `mlx-community/gemma-4-26b-a4b-it-4bit`)
/// carry a single global `bits` plus per-tensor overrides in `config.json`;
/// the global value is wrong for every overridden tensor, which would
/// dequantize as garbage. Deriving from the shapes is exact and
/// per-tensor, so the load path needs no per-key config plumbing.
/// `groupSize` is assumed uniform across the checkpoint — the global
/// `quantization.group_size` — which holds for every published mlx
/// affine conversion.
public func deriveAffineQuantBits(
    weightPackedCols: Int, scaleCols: Int, groupSize: Int
) -> Int {
    let inFeatures = scaleCols * groupSize
    precondition(inFeatures > 0,
                 "deriveAffineQuantBits: non-positive in-features "
                 + "(scaleCols=\(scaleCols), groupSize=\(groupSize))")
    precondition((32 * weightPackedCols) % inFeatures == 0,
                 "deriveAffineQuantBits: \(weightPackedCols) packed weight "
                 + "columns are inconsistent with \(inFeatures) in-features "
                 + "at group size \(groupSize) — shapes do not describe an "
                 + "mlx affine-quantized tensor")
    return (32 * weightPackedCols) / inFeatures
}

/// Build the right Linear variant for a weight at `<base>.weight` —
/// QuantizedLinear if the bundle has matching `.scales`/`.biases` and
/// the per-tensor bit-width derived from the shapes is supported
/// (3/4/5/6/8), regular Linear otherwise. The bit-width is derived per
/// tensor — not read from the global config — so mixed-precision
/// checkpoints load correctly (see `deriveAffineQuantBits`).
public func loadLinear(
    base: String, in bundle: SafeTensorsBundle,
    quantization: ModelConfig.QuantizationConfig?
) throws -> AnyLinear {
    if let q = quantization, bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        let bits = deriveAffineQuantBits(
            weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
            scaleCols: t.scales.shape[t.scales.shape.count - 1],
            groupSize: q.groupSize)
        if [3, 4, 5, 6, 8].contains(bits) {
            return AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: bits, groupSize: q.groupSize
            ))
        }
    }
    let weight = try bundle.tensor(named: "\(base).weight")
    // Bias is optional — present on Qwen 2.x QKV projections, BLOOM,
    // older Falcon, etc. Absent on the 3-series + Mistral7B + Phi-3.
    let bias = bundle.has("\(base).bias")
        ? try bundle.tensor(named: "\(base).bias")
        : nil
    return AnyLinear(Linear(weight: weight, bias: bias))
}

// ─── Embedding ───────────────────────────────────────────────────────

public final class Embedding: Module {
    /// weight shape [vocab_size, hidden_size]
    public let weight: Tensor

    public init(weight: Tensor) {
        precondition(weight.shape.count == 2, "Embedding: weight must be 2D")
        self.weight = weight
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    /// Look up `tokenIds` (one-element u32 tensor for decode) and return
    /// [n_tokens, hidden] in the table's dtype.
    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.gather(table: weight, tokenIds: tokenIds, on: cmd)
    }
}

// ─── QuantizedEmbedding (mlx int4 format) ─────────────────────────────

public final class QuantizedEmbedding: Module {
    public let weight: Tensor   // [vocab, hidden/pack_factor] uint32
    public let scales: Tensor
    public let biases: Tensor
    public let hidden: Int
    public let bits: Int
    public let groupSize: Int

    public init(weight: Tensor, scales: Tensor, biases: Tensor,
                hidden: Int, bits: Int, groupSize: Int) {
        precondition(bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
                     "QuantizedEmbedding: bits must be one of 3, 4, 5, 6, or 8")
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.hidden = hidden
        self.bits = bits
        self.groupSize = groupSize
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("scales", scales), ("biases", biases)]
    }

    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.dequantGather(
            weight: weight, scales: scales, biases: biases,
            tokenIds: tokenIds, hidden: hidden, bits: bits, groupSize: groupSize,
            on: cmd
        )
    }
}

/// Type-erasing wrapper over Embedding / QuantizedEmbedding so model
/// loaders don't need to template every call site.
public final class AnyEmbedding: Module {
    public let inner: any Module
    public let weight: Tensor   // expose for tying with lm_head
    private let forward: (Tensor, MTLCommandBuffer) -> Tensor

    public init(_ embed: Embedding) {
        self.inner = embed
        self.weight = embed.weight
        self.forward = { embed($0, on: $1) }
    }

    public init(_ embed: QuantizedEmbedding) {
        self.inner = embed
        self.weight = embed.weight
        self.forward = { embed($0, on: $1) }
    }

    public func parameters() -> [(String, Tensor)] { inner.parameters() }

    public func callAsFunction(_ tokenIds: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        forward(tokenIds, cmd)
    }
}

/// Build the right Embedding variant depending on quantization presence.
public func loadEmbedding(
    base: String, in bundle: SafeTensorsBundle,
    hidden: Int, quantization: ModelConfig.QuantizationConfig?
) throws -> AnyEmbedding {
    if let q = quantization, bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        let bits = deriveAffineQuantBits(
            weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
            scaleCols: t.scales.shape[t.scales.shape.count - 1],
            groupSize: q.groupSize)
        if [3, 4, 5, 6, 8].contains(bits) {
            return AnyEmbedding(QuantizedEmbedding(
                weight: t.weight, scales: t.scales, biases: t.biases,
                hidden: hidden, bits: bits, groupSize: q.groupSize
            ))
        }
    }
    return AnyEmbedding(Embedding(weight: try bundle.tensor(named: "\(base).weight")))
}

// ─── RMSNorm ─────────────────────────────────────────────────────────

public final class RMSNorm: Module {
    /// weight shape [n] — per-channel scale.
    public let weight: Tensor
    public let eps: Float

    public init(weight: Tensor, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.rmsNorm(x, weight: weight, eps: eps, on: cmd)
    }
}

// ─── LayerNorm ───────────────────────────────────────────────────────

/// Standard LayerNorm with learned scale + shift — the normalization
/// vision-transformer encoders (SigLIP / CLIP) use instead of RMSNorm.
/// `callAsFunction` treats `x` as a single `[rowSize]` row; multi-row
/// callers (the vision encoder, processing all patch tokens at once)
/// call `Ops.layerNorm` directly with an explicit `nRows`.
public final class LayerNorm: Module {
    /// weight shape [n] — per-channel scale (`gamma`).
    public let weight: Tensor
    /// bias shape [n] — per-channel shift (`beta`).
    public let bias: Tensor
    public let eps: Float

    public init(weight: Tensor, bias: Tensor, eps: Float) {
        precondition(weight.shape == bias.shape,
                     "LayerNorm: weight/bias shape mismatch")
        self.weight = weight
        self.bias = bias
        self.eps = eps
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("bias", bias)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.layerNorm(x, weight: weight, bias: bias, eps: eps,
                      nRows: 1, rowSize: x.elementCount, on: cmd)
    }
}
