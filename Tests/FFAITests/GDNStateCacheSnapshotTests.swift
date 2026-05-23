// Round-trip test for `GDNStateCache.snapshot()` / `.restore(from:)`.
//
// Foundation for speculative-decode work — GDN's recurrent state is
// not reversible (the kernel folds the new token's delta into `current`
// non-invertibly), so spec decode at γ ≥ 1 needs a snapshot/restore
// pair to undo the effect of a wrong drafted token.
//
// This test verifies: snapshot → mutate state → restore → state matches
// snapshot bit-for-bit. Required correctness pin before wiring spec
// decode through the GDN layers.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("GDNStateCache snapshot/restore round-trip")
struct GDNStateCacheSnapshotTests {

    @Test("snapshot + restore exactly recovers pre-mutation state")
    func snapshotRestoreRoundTrip() throws {
        let cache = GDNStateCache(
            numValueHeads: 4, valueHeadDim: 16, keyHeadDim: 32,
            device: .shared)

        // Seed `current` with deterministic non-zero values.
        let nElems = 4 * 16 * 32
        let seeded: [Float] = (0..<nElems).map { Float($0) * 0.001 - 1.0 }
        seeded.withUnsafeBytes { src in
            cache.current.buffer.contents()
                .advanced(by: cache.current.offset)
                .copyMemory(from: src.baseAddress!,
                            byteCount: nElems * MemoryLayout<Float>.stride)
        }

        // Snapshot pre-mutation.
        let snap = cache.snapshot()

        // Mutate `current` so it diverges from the snapshot.
        let mutated: [Float] = (0..<nElems).map { _ in 42.5 }
        mutated.withUnsafeBytes { src in
            cache.current.buffer.contents()
                .advanced(by: cache.current.offset)
                .copyMemory(from: src.baseAddress!,
                            byteCount: nElems * MemoryLayout<Float>.stride)
        }
        // Sanity: current differs from snapshot.
        let currentPtr = cache.current.buffer.contents()
            .advanced(by: cache.current.offset)
            .bindMemory(to: Float.self, capacity: nElems)
        #expect(currentPtr[0] == 42.5,
                "pre-restore: current[0]=\(currentPtr[0]) should be 42.5")

        // Restore from snapshot.
        cache.restore(from: snap)

        // Verify current matches the seeded values bit-for-bit.
        let restoredPtr = cache.current.buffer.contents()
            .advanced(by: cache.current.offset)
            .bindMemory(to: Float.self, capacity: nElems)
        var maxAbsDiff: Float = 0
        for i in 0..<nElems {
            maxAbsDiff = max(maxAbsDiff, abs(restoredPtr[i] - seeded[i]))
        }
        #expect(maxAbsDiff == 0,
                "snapshot+restore round-trip max |Δ| = \(maxAbsDiff) (expected 0)")
        print("GDNStateCache snapshot/restore round-trip: max |Δ| = \(maxAbsDiff)")
    }
}
