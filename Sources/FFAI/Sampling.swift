// Sampling — Phase 2 implementation does CPU argmax over logits.
// The user reads logits back to CPU after the forward pass and picks the
// next token here. Move to GPU sampling kernels in a later phase.

import Foundation

public enum Sampling {
    /// Debug: top-N highest logits as (index, value) pairs.
    public static func topN(_ logits: Tensor, n: Int) -> [(Int, Float)] {
        let count = logits.elementCount
        var values = [Float](repeating: 0, count: count)
        switch logits.dtype {
        case .f32:
            let arr = logits.toArray(as: Float.self)
            for i in 0..<count { values[i] = arr[i] }
        case .f16:
            let arr = logits.toArray(as: Float16.self)
            for i in 0..<count { values[i] = Float(arr[i]) }
        case .bf16:
            let bits = logits.toArray(as: UInt16.self)
            for i in 0..<count { values[i] = Float(bitPattern: UInt32(bits[i]) << 16) }
        default:
            fatalError("Sampling.topN: unsupported dtype \(logits.dtype)")
        }
        let indexed = values.enumerated().sorted { $0.element > $1.element }
        return indexed.prefix(n).map { ($0.offset, $0.element) }
    }

    /// Greedy: argmax over a 1D logits tensor.
    public static func argmax(_ logits: Tensor) -> Int {
        let n = logits.elementCount
        switch logits.dtype {
        case .f32:
            let values = logits.toArray(as: Float.self)
            var best = 0
            var bestVal = values[0]
            for i in 1..<n {
                if values[i] > bestVal {
                    bestVal = values[i]
                    best = i
                }
            }
            return best
        case .f16:
            let values = logits.toArray(as: Float16.self)
            var best = 0
            var bestVal = values[0]
            for i in 1..<n {
                if values[i] > bestVal {
                    bestVal = values[i]
                    best = i
                }
            }
            return best
        case .bf16:
            // bf16 → f32: shift bf16's 16 bits into the upper half of an f32.
            let bits = logits.toArray(as: UInt16.self)
            var best = 0
            var bestVal = Float(bitPattern: UInt32(bits[0]) << 16)
            for i in 1..<n {
                let v = Float(bitPattern: UInt32(bits[i]) << 16)
                if v > bestVal {
                    bestVal = v
                    best = i
                }
            }
            return best
        default:
            fatalError("Sampling.argmax: unsupported dtype \(logits.dtype)")
        }
    }
}
