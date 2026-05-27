// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
            precondition(
                b.shape == [weight.shape[0]],
                "Linear: bias shape \(b.shape) must match output features \([weight.shape[0]])")
            // Some upstream checkpoints (Qwen2-VL, Voxtral, mlx-vlm-
            // converted Qwen 2.5 / 3 audio variants) ship the QKV bias
            // in f32 even when the weight is bf16/f16. Auto-cast to
            // weight dtype here so the Linear contract stays "matching
            // dtype" without forcing every loader to convert.
            if b.dtype == weight.dtype {
                self.bias = b
            } else {
                let casted = Tensor.empty(shape: b.shape, dtype: weight.dtype)
                let floats = b.toFloatArray()
                switch weight.dtype {
                case .f32: casted.copyIn(from: floats)
                case .f16: casted.copyIn(from: floats.map { Float16($0) })
                case .bf16:
                    casted.copyIn(
                        from: floats.map { v -> UInt16 in
                            let bits = v.bitPattern
                            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                            return UInt16(rounded >> 16)
                        })
                default:
                    preconditionFailure(
                        "Linear: cannot auto-cast bias \(b.dtype) → \(weight.dtype) "
                            + "(unsupported activation dtype)")
                }
                self.bias = casted
            }
        } else {
            self.bias = nil
        }
        self.weight = weight
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
    /// host overhead). When the Linear carries a bias (Qwen 2.x QKV
    /// projections, GLM-ASR encoder, etc.) the `[outDim]` bias is tiled
    /// to `[T, outDim]` and folded in via `Ops.add` on the same command
    /// buffer — one extra dispatch, no host roundtrip. Without this,
    /// every biased multi-row caller (Qwen 2.5 1.5B/7B forwardMulti,
    /// GLM-ASR encoder, etc.) would crash on the old "bias broadcast
    /// not implemented" precondition.
    ///
    /// (Rebase 2026-05-23: keep Tom's `callMany` API, drop my parallel
    /// `batched(_:nRows:)` from `3a87b78`. Both did the same job;
    /// Tom's QuantizedLinear.callMany variant actually wraps the
    /// `mt_qmm_mma`-backed `Ops.dequantGemmDynamicM` kernel — what my
    /// doc comment had flagged as a TODO.)
    public func callMany(_ x: Tensor, t: Int, on cmd: MTLCommandBuffer) -> Tensor {
        let y = Ops.gemm(weight: weight, input: x, nRows: t, on: cmd)
        guard let b = bias else { return y }
        // Tile [outDim] → [T, outDim] on the CPU then add. Bias tensors
        // are tiny (a few KB at most), so the host-side replication is
        // in the noise relative to the GEMM that just ran.
        let outDim = weight.shape[0]
        let biasVals = b.toFloatArray()
        var tiledFlat = [Float](repeating: 0, count: t * outDim)
        for r in 0 ..< t {
            let base = r * outDim
            for c in 0 ..< outDim { tiledFlat[base + c] = biasVals[c] }
        }
        let tiled = Tensor.empty(shape: [t, outDim], dtype: y.dtype)
        switch y.dtype {
        case .f32: tiled.copyIn(from: tiledFlat)
        case .f16: tiled.copyIn(from: tiledFlat.map { Float16($0) })
        case .bf16:
            tiled.copyIn(
                from: tiledFlat.map { v -> UInt16 in
                    let bits = v.bitPattern
                    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                    return UInt16(rounded >> 16)
                })
        default:
            preconditionFailure(
                "Linear.callMany: bias broadcast unsupported for dtype \(y.dtype)")
        }
        return Ops.add(y, tiled, on: cmd)
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
    /// Optional additive output bias `[out_features]`. This is *not* the
    /// per-group dequant offset (which is `biases` plural) — it's the
    /// classic Linear additive bias `y = Wx + b`. mlx-community's 4-bit
    /// Qwen 2.x checkpoints ship this alongside the quantization triplet
    /// (`q_proj.bias` next to `q_proj.{weight,scales,biases}`); silently
    /// dropping it makes the model degenerate to one repeated token
    /// (DeepSeek-R1-Distill-Qwen-1.5B emitted "000000…" before this).
    public let additiveBias: Tensor?

    public init(
        weight: Tensor, scales: Tensor, biases: Tensor,
        bits: Int, groupSize: Int,
        additiveBias: Tensor? = nil
    ) {
        precondition(weight.dtype == .u32, "QuantizedLinear: weight must be u32 packed")
        precondition(weight.shape.count == 2, "QuantizedLinear: weight must be 2D")
        precondition(
            bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
            "QuantizedLinear: bits must be one of 2, 3, 4, 5, 6, or 8")
        if let ab = additiveBias {
            precondition(
                ab.shape.count == 1,
                "QuantizedLinear: additiveBias must be 1D [out_features]")
            // Cast bias to the activation dtype so the post-dequant add
            // works without dtype gymnastics. We don't know the
            // activation dtype here, but the bias dtype recorded on
            // disk matches the original Linear's weight dtype (bf16/f16
            // for mlx-community conversions), which is exactly what the
            // dequantized gemv output uses. If a future caller hits a
            // dtype mismatch, mirror the Linear.init auto-cast pattern.
            self.additiveBias = ab
        } else {
            self.additiveBias = nil
        }
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.bits = bits
        self.groupSize = groupSize
    }

    public func parameters() -> [(String, Tensor)] {
        if let ab = additiveBias {
            return [
                ("weight", weight), ("scales", scales),
                ("biases", biases), ("bias", ab),
            ]
        }
        return [("weight", weight), ("scales", scales), ("biases", biases)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        let y = Ops.dequantGemv(
            weight: weight, scales: scales, biases: biases,
            input: x, bits: bits, groupSize: groupSize, on: cmd
        )
        if let ab = additiveBias {
            return Ops.add(y, ab, on: cmd)
        }
        return y
    }

    /// T-batched. `x` is `[T, inDim]` flat row-major, returns
    /// `[T, outDim]` flat. Dispatches `Ops.dequantGemmDynamicM` — one
    /// `mt_qmm_mma` kernel for T multiple of 32, or one padded
    /// dispatch + host slice for ragged T. Matches the batched-prefill
    /// path (BP2).
    public func callMany(
        _ x: Tensor, t: Int,
        on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
        let outDim = weight.shape[0]
        let inDim = scales.shape[scales.shape.count - 1] * groupSize
        // 4-bit and 2-bit go through the fast mt_qmm_mma path when the
        // output dimension is a multiple of 32 (the BN tile of the
        // underlying `dequantGemmDynamicM` kernel). Anything narrower —
        // including 4-bit shared_expert_gate at outDim=1 — falls through
        // to the per-row `dequantGemv` loop. Other bit-widths (commonly
        // 8-bit on the same narrow projections, plus 3/5/6) take the
        // same fallback. Both fallbacks are bit-identical to the
        // per-token path; their per-token launch overhead is in the
        // noise on hidden→1 shapes.
        let out: Tensor
        if (bits == 4 || bits == 2) && outDim % 32 == 0 {
            out = Tensor.empty(shape: [t, outDim], dtype: x.dtype, device: device)
            Ops.dequantGemmDynamicM(
                input: x, weight: weight, scales: scales, biases: biases,
                t: t, nOut: outDim, kIn: inDim, groupSize: groupSize,
                on: cmd, device: device, into: out, bits: bits)
        } else {
            let dtBytes = x.dtype.byteSize
            out = Tensor.empty(shape: [t, outDim], dtype: x.dtype, device: device)
            for r in 0 ..< t {
                let xRow = Tensor(
                    buffer: x.buffer,
                    offset: x.offset + r * inDim * dtBytes,
                    shape: [inDim], dtype: x.dtype)
                let rowOut = Ops.dequantGemv(
                    weight: weight, scales: scales, biases: biases,
                    input: xRow, bits: bits, groupSize: groupSize, on: cmd)
                let outRow = Tensor(
                    buffer: out.buffer,
                    offset: out.offset + r * outDim * dtBytes,
                    shape: [outDim], dtype: x.dtype)
                Ops.copy(rowOut, into: outRow, on: cmd)
            }
        }
        // Fold the additive bias on top of the [T, outDim] batched output
        // by tiling the [outDim] bias to [T, outDim] on the host then
        // adding on the same command buffer. Same pattern as
        // Linear.callMany. Without this, QuantizedLinear.callMany silently
        // omitted the bias on the prefill (chunked) path while the
        // single-token path applied it, producing inconsistent state.
        guard let ab = additiveBias else { return out }
        let biasVals = ab.toFloatArray()
        var tiledFlat = [Float](repeating: 0, count: t * outDim)
        for r in 0 ..< t {
            let base = r * outDim
            for c in 0 ..< outDim { tiledFlat[base + c] = biasVals[c] }
        }
        let tiled = Tensor.empty(shape: [t, outDim], dtype: out.dtype)
        switch out.dtype {
        case .f32: tiled.copyIn(from: tiledFlat)
        case .f16: tiled.copyIn(from: tiledFlat.map { Float16($0) })
        case .bf16:
            tiled.copyIn(
                from: tiledFlat.map { v -> UInt16 in
                    let bits = v.bitPattern
                    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                    return UInt16(rounded >> 16)
                })
        default:
            preconditionFailure(
                "QuantizedLinear.callMany: bias broadcast unsupported for dtype \(out.dtype)")
        }
        return Ops.add(out, tiled, on: cmd)
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
    public func callMany(
        _ x: Tensor, t: Int,
        on cmd: MTLCommandBuffer,
        device: Device
    ) -> Tensor {
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
    precondition(
        inFeatures > 0,
        "deriveAffineQuantBits: non-positive in-features "
            + "(scaleCols=\(scaleCols), groupSize=\(groupSize))")
    precondition(
        (32 * weightPackedCols) % inFeatures == 0,
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
        if [2, 3, 4, 5, 6, 8].contains(bits) {
            // Pick up the additive output bias (e.g. Qwen 2.x QKV) if
            // the checkpoint ships it alongside the quantization triplet.
            // The mlx-community 4-bit Qwen 2.x conversions DO retain it
            // — older comments here incorrectly assumed otherwise, which
            // gave silent degenerate output (DeepSeek-R1-Distill-Qwen-1.5B
            // emitted token-15 "0" forever).
            let additiveBias: Tensor? =
                bundle.has("\(base).bias")
                ? try bundle.tensor(named: "\(base).bias")
                : nil
            return AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: bits, groupSize: q.groupSize,
                    additiveBias: additiveBias
                ))
        }
    }
    let weight = try bundle.tensor(named: "\(base).weight")
    // Bias is optional — present on Qwen 2.x QKV projections, BLOOM,
    // older Falcon, etc. Absent on the 3-series + Mistral7B + Phi-3.
    let bias =
        bundle.has("\(base).bias")
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
    public let weight: Tensor  // [vocab, hidden/pack_factor] uint32
    public let scales: Tensor
    public let biases: Tensor
    public let hidden: Int
    public let bits: Int
    public let groupSize: Int

    public init(
        weight: Tensor, scales: Tensor, biases: Tensor,
        hidden: Int, bits: Int, groupSize: Int
    ) {
        precondition(
            bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
            "QuantizedEmbedding: bits must be one of 2, 3, 4, 5, 6, or 8")
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
    public let weight: Tensor  // expose for tying with lm_head
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
        if [2, 3, 4, 5, 6, 8].contains(bits) {
            return AnyEmbedding(
                QuantizedEmbedding(
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
        precondition(
            weight.shape == bias.shape,
            "LayerNorm: weight/bias shape mismatch")
        self.weight = weight
        self.bias = bias
        self.eps = eps
    }

    public func parameters() -> [(String, Tensor)] {
        [("weight", weight), ("bias", bias)]
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        Ops.layerNorm(
            x, weight: weight, bias: bias, eps: eps,
            nRows: 1, rowSize: x.elementCount, on: cmd)
    }
}
