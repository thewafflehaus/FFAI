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
// AURACodebook — precomputed Lloyd-Max scalar codebooks for AURA's
// compressed KV cache.
//
// AURA quantises each *rotated* coordinate against a global 1D scalar
// codebook of `2^bits` centroids. After a random orthogonal rotation
// the coordinate distribution of unit-sphere vectors converges to a
// near-Gaussian, so a fixed Lloyd-Max table is near-optimal.
//
// The reference values here are mined from llama.cpp's `k_quants`
// tables (empirically optimal for unit-norm Gaussian data at d=128)
// and scaled to other head dims by √(128 / dim) — a heuristic that
// approximates the analytic 1/√d Beta-variance scaling from the
// TurboQuant paper (arXiv:2504.19874 §3.1). Ported from
// `mlx-swift-lm` (`Libraries/MLXLMCommon/TurboQuantKVCache.swift`
// `TurboQuantCodebook`), which itself sources from
// `ggml/src/ggml-metal/ggml-metal.metal` (`turbo_centroids_*bit`).
//
// See `papers/aura-compression-algorithm.md` §2.3 for the design
// rationale and §5.1 / §5.2 for the open question of per-coordinate
// vs global codebooks and the HIGGS-style multivariate Gaussian-MSE
// grid as a future direction.

import Foundation

/// Precomputed Lloyd-Max codebooks + midpoint boundaries for AURA's
/// scalar quantizer. The 2/3/4-bit tables are short enough to inline;
/// the 8-bit table holds all 256 centroids (plus 255 midpoints)
/// pre-baked to avoid a ~1–4s `generateCentroids` cliff during the
/// first prefill window.
public enum AURACodebook {

    // MARK: - Reference tables (d=128, Lloyd-Max optimal for unit-norm Gaussian)

    /// 2-bit centroids (4 levels).
    private static let centroids128_2bit: [Float] = [
        -0.133462, -0.039994, 0.039994, 0.133462,
    ]

    /// 2-bit midpoints (3 boundaries between adjacent centroids).
    private static let midpoints128_2bit: [Float] = [
        -0.086728, 0.0, 0.086728,
    ]

    /// 3-bit centroids (8 levels).
    private static let centroids128_3bit: [Float] = [
        -0.190685, -0.117832, -0.065717, -0.021460,
        0.021460, 0.065717, 0.117832, 0.190685,
    ]

    /// 3-bit midpoints (7 boundaries).
    private static let midpoints128_3bit: [Float] = [
        -0.154259, -0.091775, -0.043589, 0.0,
        0.043589, 0.091775, 0.154259,
    ]

    /// 4-bit centroids (16 levels).
    private static let centroids128_4bit: [Float] = [
        -0.173926, -0.117195, -0.089527, -0.068756,
        -0.051262, -0.035597, -0.020989, -0.006938,
        0.006938, 0.020989, 0.035597, 0.051262,
        0.068756, 0.089527, 0.117195, 0.173926,
    ]

    /// 4-bit midpoints (15 boundaries).
    private static let midpoints128_4bit: [Float] = [
        -0.145560, -0.103361, -0.079142, -0.060009,
        -0.043430, -0.028293, -0.013963, 0.000000,
        0.013963, 0.028293, 0.043430, 0.060009,
        0.079142, 0.103361, 0.145560,
    ]

    /// 8-bit centroids (256 levels). Computed offline via weighted
    /// k-means on Beta(d=128); pre-baked to avoid the runtime cliff.
    private static let centroids128_8bit: [Float] = [
        -0.321782, -0.280567, -0.254225, -0.234551,
        -0.218845, -0.205896, -0.194947, -0.185487,
        -0.177261, -0.170089, -0.163792, -0.158191,
        -0.153197, -0.148718, -0.144665, -0.140977,
        -0.137562, -0.134328, -0.131279, -0.128411,
        -0.125697, -0.123103, -0.120632, -0.118283,
        -0.116025, -0.113890, -0.111876, -0.109954,
        -0.108093, -0.106292, -0.104584, -0.102905,
        -0.101196, -0.099518, -0.097901, -0.096345,
        -0.094819, -0.093324, -0.091828, -0.090333,
        -0.088899, -0.087495, -0.086091, -0.084687,
        -0.083345, -0.082032, -0.080689, -0.079377,
        -0.078096, -0.076844, -0.075624, -0.074372,
        -0.073121, -0.071901, -0.070710, -0.069551,
        -0.068391, -0.067231, -0.066072, -0.064912,
        -0.063783, -0.062654, -0.061525, -0.060426,
        -0.059358, -0.058290, -0.057191, -0.056123,
        -0.055085, -0.054048, -0.053010, -0.051972,
        -0.050935, -0.049928, -0.048921, -0.047914,
        -0.046906, -0.045899, -0.044892, -0.043885,
        -0.042909, -0.041932, -0.040955, -0.040009,
        -0.039063, -0.038117, -0.037171, -0.036225,
        -0.035279, -0.034333, -0.033417, -0.032502,
        -0.031556, -0.030610, -0.029694, -0.028779,
        -0.027863, -0.026948, -0.026032, -0.025117,
        -0.024232, -0.023346, -0.022431, -0.021515,
        -0.020630, -0.019745, -0.018860, -0.017975,
        -0.017060, -0.016175, -0.015290, -0.014405,
        -0.013520, -0.012635, -0.011780, -0.010926,
        -0.010040, -0.009155, -0.008270, -0.007385,
        -0.006531, -0.005646, -0.004761, -0.003906,
        -0.003052, -0.002167, -0.001312, -0.000458,
        0.000458, 0.001343, 0.002197, 0.003052,
        0.003937, 0.004822, 0.005676, 0.006561,
        0.007446, 0.008301, 0.009186, 0.010071,
        0.010926, 0.011811, 0.012696, 0.013550,
        0.014435, 0.015320, 0.016205, 0.017090,
        0.017975, 0.018860, 0.019745, 0.020661,
        0.021546, 0.022431, 0.023346, 0.024262,
        0.025147, 0.026032, 0.026948, 0.027863,
        0.028779, 0.029694, 0.030610, 0.031556,
        0.032502, 0.033418, 0.034364, 0.035310,
        0.036225, 0.037171, 0.038148, 0.039094,
        0.040040, 0.040986, 0.041963, 0.042970,
        0.043946, 0.044923, 0.045930, 0.046937,
        0.047914, 0.048921, 0.049928, 0.050935,
        0.051972, 0.053010, 0.054048, 0.055085,
        0.056123, 0.057191, 0.058290, 0.059388,
        0.060456, 0.061555, 0.062684, 0.063813,
        0.064943, 0.066072, 0.067231, 0.068391,
        0.069551, 0.070741, 0.071931, 0.073152,
        0.074403, 0.075593, 0.076814, 0.078096,
        0.079408, 0.080720, 0.082032, 0.083345,
        0.084687, 0.086091, 0.087495, 0.088929,
        0.090394, 0.091859, 0.093323, 0.094819,
        0.096375, 0.097962, 0.099549, 0.101196,
        0.102875, 0.104584, 0.106353, 0.108154,
        0.109985, 0.111876, 0.113890, 0.116025,
        0.118283, 0.120632, 0.123103, 0.125697,
        0.128411, 0.131279, 0.134328, 0.137562,
        0.140977, 0.144665, 0.148718, 0.153197,
        0.158191, 0.163792, 0.170089, 0.177261,
        0.185487, 0.194947, 0.205897, 0.218845,
        0.234551, 0.254225, 0.280567, 0.321781,
    ]

    /// 8-bit midpoints (255 boundaries). Pre-baked arithmetic
    /// midpoints between adjacent centroids.
    private static let midpoints128_8bit: [Float] = [
        -0.301175, -0.267396, -0.244388, -0.226698,
        -0.212371, -0.200422, -0.190217, -0.181374,
        -0.173675, -0.166941, -0.160992, -0.155694,
        -0.150958, -0.146692, -0.142821, -0.139269,
        -0.135945, -0.132803, -0.129845, -0.127054,
        -0.124400, -0.121868, -0.119458, -0.117154,
        -0.114958, -0.112883, -0.110915, -0.109023,
        -0.107193, -0.105438, -0.103745, -0.102051,
        -0.100357, -0.098709, -0.097123, -0.095582,
        -0.094071, -0.092576, -0.091080, -0.089616,
        -0.088197, -0.086793, -0.085389, -0.084016,
        -0.082688, -0.081361, -0.080033, -0.078736,
        -0.077470, -0.076234, -0.074998, -0.073747,
        -0.072511, -0.071305, -0.070131, -0.068971,
        -0.067811, -0.066651, -0.065492, -0.064347,
        -0.063218, -0.062089, -0.060975, -0.059892,
        -0.058824, -0.057740, -0.056657, -0.055604,
        -0.054566, -0.053529, -0.052491, -0.051454,
        -0.050431, -0.049424, -0.048417, -0.047410,
        -0.046403, -0.045396, -0.044389, -0.043397,
        -0.042420, -0.041444, -0.040482, -0.039536,
        -0.038590, -0.037644, -0.036698, -0.035752,
        -0.034806, -0.033875, -0.032960, -0.032029,
        -0.031083, -0.030152, -0.029236, -0.028321,
        -0.027405, -0.026490, -0.025574, -0.024674,
        -0.023789, -0.022889, -0.021973, -0.021073,
        -0.020188, -0.019303, -0.018418, -0.017517,
        -0.016617, -0.015732, -0.014847, -0.013962,
        -0.013077, -0.012207, -0.011353, -0.010483,
        -0.009598, -0.008713, -0.007828, -0.006958,
        -0.006088, -0.005203, -0.004334, -0.003479,
        -0.002609, -0.001740, -0.000885, 0.000000,
        0.000900, 0.001770, 0.002625, 0.003494,
        0.004379, 0.005249, 0.006119, 0.007004,
        0.007874, 0.008743, 0.009629, 0.010498,
        0.011368, 0.012253, 0.013123, 0.013993,
        0.014878, 0.015763, 0.016648, 0.017533,
        0.018418, 0.019303, 0.020203, 0.021103,
        0.021988, 0.022889, 0.023804, 0.024705,
        0.025590, 0.026490, 0.027405, 0.028321,
        0.029236, 0.030152, 0.031083, 0.032029,
        0.032960, 0.033891, 0.034837, 0.035767,
        0.036698, 0.037659, 0.038621, 0.039567,
        0.040513, 0.041474, 0.042466, 0.043458,
        0.044434, 0.045426, 0.046433, 0.047425,
        0.048417, 0.049424, 0.050431, 0.051454,
        0.052491, 0.053529, 0.054566, 0.055604,
        0.056657, 0.057740, 0.058839, 0.059922,
        0.061006, 0.062120, 0.063249, 0.064378,
        0.065507, 0.066651, 0.067811, 0.068971,
        0.070146, 0.071336, 0.072541, 0.073777,
        0.074998, 0.076204, 0.077455, 0.078752,
        0.080064, 0.081376, 0.082688, 0.084016,
        0.085389, 0.086793, 0.088212, 0.089661,
        0.091126, 0.092591, 0.094071, 0.095597,
        0.097168, 0.098755, 0.100373, 0.102036,
        0.103729, 0.105468, 0.107254, 0.109069,
        0.110931, 0.112883, 0.114958, 0.117154,
        0.119458, 0.121868, 0.124400, 0.127054,
        0.129845, 0.132803, 0.135945, 0.139269,
        0.142821, 0.146692, 0.150958, 0.155694,
        0.160992, 0.166941, 0.173675, 0.181374,
        0.190217, 0.200422, 0.212371, 0.226698,
        0.244388, 0.267396, 0.301174,
    ]

    /// Supported bit widths with pre-baked tables.
    public static let supportedBits: Set<Int> = [2, 3, 4, 8]

    // MARK: - Public API

    /// Lloyd-Max centroids scaled to the given head dim. Returns
    /// `2^bits` Float values. The `dim != 128` case applies the
    /// √(128 / dim) heuristic that approximates the 1/√d Beta-variance
    /// scaling — exact for d=128, approximate elsewhere.
    public static func centroids(dim: Int, bits: Int) -> [Float] {
        let base = referenceCentroids(bits: bits)
        if dim == 128 { return base }
        let scale = Float((128.0 / Double(dim)).squareRoot())
        return base.map { $0 * scale }
    }

    /// Midpoint boundaries between adjacent centroids. Returns
    /// `2^bits - 1` Float values, scaled to `dim` analogously.
    public static func boundaries(dim: Int, bits: Int) -> [Float] {
        let base = referenceMidpoints(bits: bits)
        if dim == 128 { return base }
        let scale = Float((128.0 / Double(dim)).squareRoot())
        return base.map { $0 * scale }
    }

    /// Bytes-per-token after AURA packing at this bit width and dim.
    /// `ceil(dim * bits / 32) * 4` for the packed u32 array, plus 4
    /// bytes for the f32 per-token norm. Excludes any per-vector DC
    /// bias (off by default).
    public static func bytesPerToken(dim: Int, bits: Int) -> Int {
        let packedWidthU32 = (dim * bits + 31) / 32
        return packedWidthU32 * 4 + 4
    }

    /// `ceil(dim * bits / 32)` — number of u32 words required to bit-
    /// pack `dim` codebook indices at `bits` each.
    public static func packedWidth(dim: Int, bits: Int) -> Int {
        (dim * bits + 31) / 32
    }

    // MARK: - Internals

    private static func referenceCentroids(bits: Int) -> [Float] {
        switch bits {
        case 2: return centroids128_2bit
        case 3: return centroids128_3bit
        case 4: return centroids128_4bit
        case 8: return centroids128_8bit
        default:
            fatalError(
                "AURACodebook: unsupported bits=\(bits); use one of \(supportedBits.sorted())")
        }
    }

    private static func referenceMidpoints(bits: Int) -> [Float] {
        switch bits {
        case 2: return midpoints128_2bit
        case 3: return midpoints128_3bit
        case 4: return midpoints128_4bit
        case 8: return midpoints128_8bit
        default:
            fatalError(
                "AURACodebook: unsupported bits=\(bits); use one of \(supportedBits.sorted())")
        }
    }
}
