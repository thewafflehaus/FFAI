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
// QuantizedOpsValidation — wrapper preconditions for `QuantizedOps.*`.
//
// Pure functions returning `nil` on accept, `String` on reject. Same
// pattern as `OpsValidation.swift`; lives in its own file so the test
// suite can exercise both surfaces without coupling.

import Foundation

public enum QuantizedOpsValidation {

    // ─── shared helper ─────────────────────────────────────────────

    /// Pack-factor for bit-width `bits`. Only defined for the clean
    /// power-of-two bit widths that fit a `u32` evenly (2/4/8).
    public static func packFactor(forBits bits: Int) -> Int? {
        switch bits {
        case 2: return 16
        case 4: return 8
        case 8: return 4
        default: return nil
        }
    }

    // ─── affineDequantize ──────────────────────────────────────────

    /// Validate `QuantizedOps.dequantizeAffine` inputs.
    /// `numel = n_groups * group_size`; `packedCount = numel / pack_factor`;
    /// `scalesCount = biasesCount = n_groups`.
    public static func validateAffineDequantize(
        numel: Int, packedCount: Int,
        scalesCount: Int, biasesCount: Int,
        bits: Int, groupSize: Int
    ) -> String? {
        guard let pf = packFactor(forBits: bits) else {
            return "bits=\(bits) unsupported — must be one of 2, 4, or 8 (clean pack-factor path)"
        }
        if numel <= 0 {
            return "numel must be positive (got \(numel))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        if !numel.isMultiple(of: groupSize) {
            return "numel=\(numel) must be a multiple of groupSize=\(groupSize)"
        }
        if !groupSize.isMultiple(of: pf) {
            return "groupSize=\(groupSize) must be a multiple of pack_factor=\(pf) for bits=\(bits)"
        }
        let nGroups = numel / groupSize
        let expectedPacks = numel / pf
        if packedCount != expectedPacks {
            return
                "packed buffer must have numel/pack_factor = \(numel)/\(pf) = \(expectedPacks) entries, got \(packedCount)"
        }
        if scalesCount != nGroups {
            return "scales must have n_groups=\(nGroups) entries, got \(scalesCount)"
        }
        if biasesCount != nGroups {
            return "biases must have n_groups=\(nGroups) entries, got \(biasesCount)"
        }
        return nil
    }

    // ─── affineQuantize ────────────────────────────────────────────

    /// Validate `QuantizedOps.quantizeAffine` inputs. Same contract as
    /// dequantize — caller-supplied output buffers must be pre-sized
    /// to `n_groups * group_size / pack_factor` packed entries +
    /// `n_groups` scale + bias entries each.
    public static func validateAffineQuantize(
        numel: Int, packedCount: Int,
        scalesCount: Int, biasesCount: Int,
        bits: Int, groupSize: Int
    ) -> String? {
        guard let pf = packFactor(forBits: bits) else {
            return "bits=\(bits) unsupported — must be one of 2, 4, or 8 (clean pack-factor path)"
        }
        if numel <= 0 {
            return "numel must be positive (got \(numel))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        // The metaltile quantize kernels (`mt_affine_quantize_int{2,4,8}`
        // in `crates/metaltile-std/src/mlx/quantized.rs`) hardcode the
        // min/max reduction shape at one simdgroup × 2 elements/lane =
        // 64 elements/group. Each lane reads `w[in_base + lane * 2]`
        // and `w[in_base + lane * 2 + 1]`, then a simdgroup-wide
        // `simd_min` / `simd_max` reduces across all 32 lanes. Passing
        // `groupSize != 64` makes lanes either read past the group
        // boundary (groupSize < 64 → lanes 16..31 spill into the next
        // group) or skip elements (groupSize > 64 → lanes don't cover
        // the tail). Only group_size=64 is emitted at this layer.
        if groupSize != 64 {
            return
                "groupSize=\(groupSize) unsupported for affine quantize — only group_size=64 is emitted (kernel reduces over one simdgroup × 2 elements/lane = 64 elements; see metaltile mt_affine_quantize_int\(bits))"
        }
        if !numel.isMultiple(of: groupSize) {
            return
                "numel=\(numel) must be a multiple of groupSize=\(groupSize) — partial trailing group would be silently dropped"
        }
        if !groupSize.isMultiple(of: pf) {
            return "groupSize=\(groupSize) must be a multiple of pack_factor=\(pf) for bits=\(bits)"
        }
        let nGroups = numel / groupSize
        let expectedPacks = numel / pf
        if packedCount != expectedPacks {
            return
                "packed buffer must have numel/pack_factor = \(numel)/\(pf) = \(expectedPacks) entries, got \(packedCount)"
        }
        if scalesCount != nGroups {
            return "scales must have n_groups=\(nGroups) entries, got \(scalesCount)"
        }
        if biasesCount != nGroups {
            return "biases must have n_groups=\(nGroups) entries, got \(biasesCount)"
        }
        return nil
    }
}
