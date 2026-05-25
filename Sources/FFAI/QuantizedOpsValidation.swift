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
            return "packed buffer must have numel/pack_factor = \(numel)/\(pf) = \(expectedPacks) entries, got \(packedCount)"
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
        if !numel.isMultiple(of: groupSize) {
            return "numel=\(numel) must be a multiple of groupSize=\(groupSize) — partial trailing group would be silently dropped"
        }
        if !groupSize.isMultiple(of: pf) {
            return "groupSize=\(groupSize) must be a multiple of pack_factor=\(pf) for bits=\(bits)"
        }
        // The quantize kernel uses TPG = 32 (one simdgroup). The lane
        // count writes packs_per_group = groupSize / pack_factor packed
        // words. packs_per_group must be ≤ 32 for the single-simdgroup
        // path; otherwise the kernel silently drops the tail packs.
        let packsPerGroup = groupSize / pf
        if packsPerGroup > 32 {
            return "groupSize=\(groupSize) too large for bits=\(bits) — packs_per_group=\(packsPerGroup) exceeds the simdgroup width of 32 (only group_size ≤ \(32 * pf) supported)"
        }
        let nGroups = numel / groupSize
        let expectedPacks = numel / pf
        if packedCount != expectedPacks {
            return "packed buffer must have numel/pack_factor = \(numel)/\(pf) = \(expectedPacks) entries, got \(packedCount)"
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
