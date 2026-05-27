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

    // ─── shared helpers ────────────────────────────────────────────

    /// Pack-factor for bit-width `bits`. Only defined for the clean
    /// power-of-two bit widths that fit a `u32` evenly (2/4/8).
    /// Returns `nil` for odd widths (3/5/6) — those use a byte-stream
    /// packing where the storage size is `numel * bits / 32` u32s
    /// (no integer pack factor exists). Use `packedUInt32Count` instead
    /// when you need the storage size for an arbitrary affine-packed
    /// bit-width.
    public static func packFactor(forBits bits: Int) -> Int? {
        switch bits {
        case 2: return 16
        case 4: return 8
        case 8: return 4
        default: return nil
        }
    }

    /// Storage size in u32 words for `numel` weights at affine `bits`
    /// packing. Generalises `numel / packFactor` to the odd-width
    /// bit-stream packings (3/5/6) where there's no clean integer
    /// pack factor — those store `numel * bits` bits flat, sized up
    /// to a u32 boundary.
    ///
    /// Returns nil if `bits` is unsupported or if the storage would
    /// not be u32-aligned at this `numel` (which would corrupt the
    /// bit-stream decode at the row boundary).
    public static func packedUInt32Count(numel: Int, bits: Int) -> Int? {
        guard [2, 3, 4, 5, 6, 8].contains(bits) else { return nil }
        let bitCount = numel * bits
        guard bitCount % 32 == 0 else { return nil }
        return bitCount / 32
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
        if numel <= 0 {
            return "numel must be positive (got \(numel))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        if !numel.isMultiple(of: groupSize) {
            return "numel=\(numel) must be a multiple of groupSize=\(groupSize)"
        }
        // For 2/4/8 (clean pack-factor) enforce the group-size alignment;
        // for 3/5/6 the byte-stream packing tolerates any groupSize that
        // makes `groupSize * bits` a multiple of 32.
        if let pf = packFactor(forBits: bits) {
            if !groupSize.isMultiple(of: pf) {
                return
                    "groupSize=\(groupSize) must be a multiple of pack_factor=\(pf) for bits=\(bits)"
            }
        } else if [3, 5, 6].contains(bits) {
            if (groupSize * bits) % 32 != 0 {
                return
                    "groupSize=\(groupSize) × bits=\(bits) must be a multiple of 32 (u32 bit-stream alignment)"
            }
        } else {
            return "bits=\(bits) unsupported — must be 2 / 3 / 4 / 5 / 6 / 8"
        }
        let nGroups = numel / groupSize
        guard let expectedPacks = packedUInt32Count(numel: numel, bits: bits) else {
            return
                "numel=\(numel) × bits=\(bits) must be a multiple of 32 (u32 bit-stream alignment)"
        }
        if packedCount != expectedPacks {
            return
                "packed buffer must have numel*bits/32 = \(numel)*\(bits)/32 = \(expectedPacks) entries, got \(packedCount)"
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
        guard [2, 3, 4, 5, 6, 8].contains(bits) else {
            return "bits=\(bits) unsupported — must be 2 / 3 / 4 / 5 / 6 / 8"
        }
        if numel <= 0 {
            return "numel must be positive (got \(numel))"
        }
        if groupSize <= 0 {
            return "groupSize must be positive (got \(groupSize))"
        }
        // The metaltile quantize kernels (`mt_affine_quantize_int{2,3,4,5,6,8}`
        // in `crates/metaltile-std/src/mlx/quantized.rs`) all hardcode
        // the min/max reduction shape at one simdgroup × 2 elements/lane
        // = 64 elements/group. Each lane reads `w[in_base + lane * 2]`
        // and `w[in_base + lane * 2 + 1]`, then a simdgroup-wide
        // `simd_min` / `simd_max` reduces across all 32 lanes. Passing
        // `groupSize != 64` makes lanes either read past the group
        // boundary (groupSize < 64 → lanes 16..31 spill into the next
        // group) or skip elements (groupSize > 64 → lanes don't cover
        // the tail). Only group_size=64 is emitted at this layer for
        // every bit-width.
        if groupSize != 64 {
            return
                "groupSize=\(groupSize) unsupported for affine quantize — only group_size=64 is emitted (kernel reduces over one simdgroup × 2 elements/lane = 64 elements; see metaltile mt_affine_quantize_int\(bits))"
        }
        if !numel.isMultiple(of: groupSize) {
            return
                "numel=\(numel) must be a multiple of groupSize=\(groupSize) — partial trailing group would be silently dropped"
        }
        // Pack-factor alignment for the clean widths (2/4/8); byte-
        // stream u32 alignment for the odd widths (3/5/6).
        if let pf = packFactor(forBits: bits) {
            if !groupSize.isMultiple(of: pf) {
                return
                    "groupSize=\(groupSize) must be a multiple of pack_factor=\(pf) for bits=\(bits)"
            }
        } else if (groupSize * bits) % 32 != 0 {
            return
                "groupSize=\(groupSize) × bits=\(bits) must be a multiple of 32 (u32 bit-stream alignment)"
        }
        let nGroups = numel / groupSize
        guard let expectedPacks = packedUInt32Count(numel: numel, bits: bits) else {
            return
                "numel=\(numel) × bits=\(bits) must be a multiple of 32 (u32 bit-stream alignment)"
        }
        if packedCount != expectedPacks {
            return
                "packed buffer must have numel*bits/32 = \(numel)*\(bits)/32 = \(expectedPacks) entries, got \(packedCount)"
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
