// AURARotation — generate the random orthogonal rotation matrix that
// AURA's encode + decode kernels expect.
//
// The rotation Π is fixed at codec init and never updated post-load
// (QuaRot-style, not SpinQuant-style learned). Two construction paths:
//
//   - **Hadamard / SRHT**. For power-of-2 dims up to 1024, build the
//     Sylvester Hadamard matrix H, multiply each column by a fixed
//     random ±1 sign vector s, scale by 1/√d. Cheap to construct, fast
//     to apply, and provides the JL concentration property that
//     flattens activation distributions (QuaRot Sec. 4).
//
//   - **Identity (Π = I)**. Degenerate "no rotation" path that still
//     exercises the encode/decode kernels end-to-end. Codebook quality
//     degrades because the Lloyd-Max levels assume rotated coordinates
//     are Beta-distributed, but the runtime pipeline is identical —
//     useful as a first-light integration before the SRHT path is
//     wired through Qwen3's attention (W_o pre-multiply by Π).
//
// See `papers/aura-compression-algorithm.md` §2.2 for the rotation
// design and §3.3 for the QuaRot relationship.

import Foundation

public enum AURARotation {

    /// Build the Sylvester Hadamard matrix `H` of size `dim × dim`,
    /// row-major. `dim` must be a power of 2. The matrix satisfies
    /// `H · H^T = dim · I`, i.e. `H / √dim` is orthogonal.
    ///
    /// Constructed recursively: `H_1 = [[1]]`, `H_{2n} = [[H_n, H_n],
    /// [H_n, -H_n]]`.
    public static func hadamardMatrix(dim: Int) -> [Float] {
        precondition(dim > 0 && (dim & (dim - 1)) == 0,
                     "AURARotation.hadamardMatrix: dim=\(dim) must be a power of 2")
        var h = [Float](repeating: 1, count: 1)
        var size = 1
        while size < dim {
            let newSize = size * 2
            var next = [Float](repeating: 0, count: newSize * newSize)
            for i in 0..<size {
                for j in 0..<size {
                    let v = h[i * size + j]
                    next[i * newSize + j] = v
                    next[i * newSize + (j + size)] = v
                    next[(i + size) * newSize + j] = v
                    next[(i + size) * newSize + (j + size)] = -v
                }
            }
            h = next
            size = newSize
        }
        return h
    }

    /// Deterministic random ±1 sign vector of length `dim`, seeded by
    /// `seed`. SplitMix64-warms the seed once so close seeds produce
    /// very different streams, then drives an xorshift64 RNG.
    public static func whtSigns(dim: Int, seed: UInt64) -> [Float] {
        precondition(dim > 0, "AURARotation.whtSigns: dim must be positive")
        // SplitMix64 mix; output is non-zero and well-distributed for
        // any input including 0.
        var z = seed &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z &>> 31)
        var state = z | 1   // xorshift64 requires non-zero seed

        var out = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            state ^= state &<< 13
            state ^= state &>> 7
            state ^= state &<< 17
            out[i] = (state & 1) == 0 ? -1.0 : 1.0
        }
        return out
    }

    /// SRHT rotation `Π = H · diag(s) / √dim`, row-major
    /// `[dim × dim]`. Power-of-2 `dim` only. Orthogonal — preserves
    /// norms exactly, so the AURA encoder's norm-correction factor is
    /// trivially 1.0 along this path.
    public static func srhtMatrix(dim: Int, seed: UInt64) -> [Float] {
        let h = hadamardMatrix(dim: dim)
        let s = whtSigns(dim: dim, seed: seed)
        let inv = Float(1.0 / Double(dim).squareRoot())
        var rot = [Float](repeating: 0, count: dim * dim)
        for i in 0..<dim {
            for j in 0..<dim {
                rot[i * dim + j] = h[i * dim + j] * s[j] * inv
            }
        }
        return rot
    }

    /// Row-major `[dim × dim]` identity matrix. Used by the "Π = I"
    /// integration path that exercises the AURA encode/decode kernels
    /// without modifying the host model's attention path. Lower
    /// codec quality than the SRHT path; see file header.
    public static func identityMatrix(dim: Int) -> [Float] {
        var out = [Float](repeating: 0, count: dim * dim)
        for i in 0..<dim { out[i * dim + i] = 1.0 }
        return out
    }
}
