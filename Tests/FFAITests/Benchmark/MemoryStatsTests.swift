// MemoryStatsTests — MemorySnapshot construction + PhaseMemoryTracker
// state machine. Real memory growth attribution is exercised end-to-end
// via the integration tests.

import Foundation
import Testing
@testable import FFAI

@Suite("MemoryStats")
struct MemoryStatsTests {

    @Test("MemorySnapshot.capture returns plausible non-zero values")
    func captureSnapshot() {
        let snap = MemorySnapshot.capture()
        #expect(snap.gpuBytes >= 0)
        #expect(snap.wiredTicketBytes > 0)
        #expect(snap.timestamp.timeIntervalSinceNow < 1.0)
    }

    @Test("PhaseMemoryTracker initial state matches baseline")
    func trackerInitial() {
        let t = PhaseMemoryTracker()
        #expect(t.prefillPeakBytes == t.baseline.gpuBytes)
        #expect(t.decodePeakBytes == t.baseline.gpuBytes)
        #expect(t.postPrefill == nil)
        #expect(t.postDecode == nil)
    }

    @Test("endPrefill captures postPrefill snapshot + transitions to decode phase")
    func endPrefill() {
        let t = PhaseMemoryTracker()
        t.endPrefill()
        #expect(t.postPrefill != nil)
        // Subsequent samples should track decodePeak, not prefillPeak.
        let priorPrefillPeak = t.prefillPeakBytes
        t.sample()
        // prefillPeak doesn't grow after endPrefill.
        #expect(t.prefillPeakBytes == priorPrefillPeak)
    }

    @Test("endDecode captures postDecode snapshot")
    func endDecode() {
        let t = PhaseMemoryTracker()
        t.endPrefill()
        t.endDecode()
        #expect(t.postDecode != nil)
    }

    @Test("Growth helpers compute relative deltas")
    func growthHelpers() {
        // We can't force the OS to allocate a known amount of memory in
        // a test, so just assert the helpers don't crash and return
        // arithmetically consistent values.
        let t = PhaseMemoryTracker()
        t.endPrefill()
        t.endDecode()
        let prefillGrowth = t.prefillGrowthBytes
        let decodeGrowth = t.decodeGrowthBytes
        // Both should be defined (zero is fine if no allocations happened).
        #expect(prefillGrowth + decodeGrowth >= -t.baseline.gpuBytes)   // sanity bound
        #expect(t.peakGPUBytes >= t.baseline.gpuBytes)
    }
}
