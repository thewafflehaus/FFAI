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
// WeightFolding — fold scalar multipliers into weights at load time so
// the forward path stays multiplier-agnostic.
//
// Several families (Granite 3 / 4, FalconH1) use maximal-update-
// parametrization (µP) multipliers that scale embeddings, attention
// scores, residual branches, and final logits. Folding the constant
// branch scalars (`embedding_multiplier`, `residual_multiplier`,
// `logits_scaling`) into the relevant weights — or, for quantized
// checkpoints, into the affine `scales`/`biases` — means the runtime
// kernels never need to know the multiplier exists. Only
// `attention_multiplier` can't be folded (it's a softmax temperature),
// so it rides through as the attention `scale`.
//
// NOTE: `Granite4Text` and `FalconH1Text` predate this file and carry
// their own private copies of the host-float + scale helpers; they
// should migrate here (tracked separately) to drop the duplication.

import Foundation

/// µP multipliers parsed from a checkpoint config. All identity by
/// default, so passing `.identity` (or omitting fields) makes the
/// folding a no-op — the path stays byte-identical to a vanilla model.
public struct MuPMultipliers: Sendable {
    /// Scales the token embeddings (folded into the embedding weight,
    /// or the embedding's quantized scales/biases). The tied LM-head, if
    /// any, must use the *unscaled* embedding.
    public var embedding: Float
    /// Replaces the usual `1/sqrt(head_dim)` attention score scale when
    /// present. `nil` keeps the default scale.
    public var attention: Float?
    /// Scales each residual branch output (folded into `o_proj` and
    /// `down_proj` so the residual add stays a plain add).
    public var residual: Float
    /// Final logits are divided by this in the model forward.
    public var logits: Float

    public init(
        embedding: Float = 1, attention: Float? = nil,
        residual: Float = 1, logits: Float = 1
    ) {
        self.embedding = embedding
        self.attention = attention
        self.residual = residual
        self.logits = logits
    }

    public static let identity = MuPMultipliers()

    /// `true` when every multiplier is identity — nothing to fold.
    public var isIdentity: Bool {
        embedding == 1 && attention == nil && residual == 1 && logits == 1
    }
}

/// Read a tensor's elements as host `[Float]`, converting from the
/// stored dtype (f32 / bf16 / f16).
func hostFloats(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .bf16:
        return t.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
    case .f16:
        return t.toArray(as: Float16.self).map { Float($0) }
    default:
        fatalError("WeightFolding: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Build a tensor of `shape`/`dtype` from host `[Float]`.
func tensorFromFloats(
    _ values: [Float], shape: [Int], dtype: DType, device: Device
) -> Tensor {
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    switch dtype {
    case .f32:
        t.copyIn(from: values)
    case .bf16:
        t.copyIn(from: values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    default:
        fatalError("WeightFolding: unsupported dtype for host conversion: \(dtype)")
    }
    return t
}

/// Multiply every element of `t` by `m`, returning a fresh tensor in
/// `t`'s dtype. Identity fast-path: returns `t` unchanged at `m == 1`.
func scaleTensorElements(_ t: Tensor, by m: Float, device: Device) -> Tensor {
    if m == 1.0 { return t }
    return tensorFromFloats(
        hostFloats(t).map { $0 * m },
        shape: t.shape, dtype: t.dtype, device: device)
}

/// Load a Linear with a scalar `m` folded into its output:
///
///     dequant      = nibble * scale + bias
///     dequant * m  = nibble * (m·scale) + (m·bias)
///
/// Raw checkpoints scale the weight directly; quantized checkpoints
/// scale the affine `scales`/`biases` (the packed u32 weight is
/// untouched). At `m == 1` this is equivalent to a plain `loadLinear`.
func loadLinearScaled(
    base: String, in weights: SafeTensorsBundle,
    quantization q: ModelConfig.QuantizationConfig?, by m: Float, device: Device
) throws -> AnyLinear {
    guard weights.isQuantized(base), let q else {
        let w = scaleTensorElements(
            try weights.tensor(named: "\(base).weight"), by: m, device: device)
        return AnyLinear(Linear(weight: w))
    }
    let t = try weights.quantizedTriplet(base)
    let scaledScales = scaleTensorElements(t.scales, by: m, device: device)
    let scaledBiases = scaleTensorElements(t.biases, by: m, device: device)
    let bits = deriveAffineQuantBits(
        weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
        scaleCols: t.scales.shape[t.scales.shape.count - 1],
        groupSize: q.groupSize)
    return AnyLinear(
        QuantizedLinear(
            weight: t.weight, scales: scaledScales, biases: scaledBiases,
            bits: bits, groupSize: q.groupSize))
}
