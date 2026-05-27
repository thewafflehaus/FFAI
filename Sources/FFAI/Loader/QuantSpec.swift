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
// `QuantSpec` — per-tensor-class quantization choice for `ffai convert`.
//
// Each role in a checkpoint (main linear projections, embedding,
// lm_head, vision tower) can independently select either an affine
// bit-width (2 / 3 / 4 / 5 / 6 / 8) or a no-quant downcast (fp16 /
// bf16). The `ConvertDriver` switches on the chosen `QuantSpec` per
// tensor — `.bits(n)` runs `QuantizedOps.quantizeAffine` to produce
// the standard MLX `(weight, scales, biases)` triplet; `.fp16` /
// `.bf16` skip quantization and just emit the source weight cast to
// the target dtype.
//
// The CLI parser lives in `FFAICLI/ConvertCommand.swift` via an
// `ExpressibleByArgument` conformance defined alongside the flag —
// keeping the FFAI library free of an `ArgumentParser` dependency.

import Foundation

/// Per-tensor quantization choice. Identifies either:
///   * an affine bit-width that `QuantizedOps.quantizeAffine` can
///     emit (`.bits(2 / 3 / 4 / 5 / 6 / 8)`), or
///   * a no-quant downcast that just writes the tensor in the
///     target floating-point dtype (`.fp16` / `.bf16`).
///
/// Used by `ConvertOptions` to drive `ffai convert` and by future
/// callers that want a single "what should this tensor look like in
/// the output" knob without splitting bit-width and dtype across
/// two parameters.
public enum QuantSpec: Sendable, Equatable {
    /// Affine-quantize this tensor to `n` bits per code via
    /// `QuantizedOps.quantizeAffine`. Supported widths: 2 / 3 / 4 /
    /// 5 / 6 / 8 — matches what `mt_affine_quantize_int{N}_*` ships.
    case bits(Int)

    /// Downcast this tensor to IEEE-754 fp16 (no quantization). Used
    /// when a checkpoint is published as bf16 but you want fp16
    /// inputs at inference time, or vice versa.
    case fp16

    /// Downcast this tensor to bfloat16 (no quantization). Same
    /// semantics as `.fp16`, just the alternate 16-bit float layout.
    case bf16

    /// Affine bit-widths the convert driver can emit. Centralised so
    /// the CLI parser and validators agree on what's accepted.
    public static let supportedBits: [Int] = [2, 3, 4, 5, 6, 8]

    /// True if the spec triggers `QuantizedOps.quantizeAffine`
    /// (versus a plain dtype downcast).
    public var isQuantized: Bool {
        if case .bits = self { return true }
        return false
    }

    /// The chosen bit-width, or `nil` for a downcast spec.
    public var bits: Int? {
        if case .bits(let n) = self { return n }
        return nil
    }

    /// The target dtype for a downcast spec, or `nil` for a quantized
    /// spec. Quantized triplets always carry the source weight's dtype
    /// on their scales/biases; the choice of target dtype only applies
    /// when the tensor is written unquantized.
    public var downcastDtype: DType? {
        switch self {
        case .fp16: return .f16
        case .bf16: return .bf16
        case .bits: return nil
        }
    }

    /// Parse a CLI / config string into a `QuantSpec`. Returns `nil`
    /// for anything that isn't a supported bit-width literal or one of
    /// `fp16` / `f16` / `float16` / `bf16` / `bfloat16` (case-
    /// insensitive). The CLI's `ExpressibleByArgument` conformance
    /// wraps this parser.
    public init?(parsing raw: String) {
        let s = raw.lowercased()
        if let n = Int(s), QuantSpec.supportedBits.contains(n) {
            self = .bits(n)
            return
        }
        switch s {
        case "fp16", "f16", "float16", "half":
            self = .fp16
        case "bf16", "bfloat16":
            self = .bf16
        default:
            return nil
        }
    }

    /// Human-readable rendering used for log output (`"quantizing X @
    /// 4bit"`, `"copying  Y @ fp16"`).
    public var label: String {
        switch self {
        case .bits(let n): return "\(n)bit"
        case .fp16: return "fp16"
        case .bf16: return "bf16"
        }
    }
}
