// KVEvictionStateTests — pure-CPU coverage of the slot-allocator
// powering every KVCache implementation's sliding-window behaviour.
//
// Doesn't touch Metal — the slot math is the part that has to be
// right, and the cache classes are tested separately in the GPU
// integration tests.

import Foundation
import Testing
@testable import FFAI

@Suite("KVEvictionState — slot allocation")
struct KVEvictionStateTests {

    @Test("unbounded mode hands out 0..<bufferCapacity then panics")
    func unboundedSequential() {
        var s = KVEvictionState(policy: .unbounded, bufferCapacity: 8)
        for i in 0..<8 {
            #expect(s.length == i)
            #expect(s.absolutePosition == i)
            #expect(s.reserveNextSlot() == i)
        }
        #expect(s.length == 8)
    }

    @Test("unbounded length saturates at absolutePosition")
    func unboundedLengthMatchesPosition() {
        var s = KVEvictionState(policy: .unbounded, bufferCapacity: 16)
        for i in 0..<10 { _ = s.reserveNextSlot(); _ = i }
        #expect(s.length == 10)
        #expect(s.absolutePosition == 10)
    }

    @Test("window(maxSize: 4, keep: 0) rotates after maxSize appends")
    func windowNoKeepRotates() {
        var s = KVEvictionState(policy: .window(maxSize: 4, keep: 0),
                                bufferCapacity: 16)
        let expected = [0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3]
        for i in expected {
            #expect(s.reserveNextSlot() == i)
        }
        // length saturates at maxSize, absolutePosition keeps growing.
        #expect(s.length == 4)
        #expect(s.absolutePosition == 12)
    }

    @Test("window(maxSize: 6, keep: 2) pins first 2 slots, rotates the rest")
    func windowWithKeepPreservesSinks() {
        var s = KVEvictionState(policy: .window(maxSize: 6, keep: 2),
                                bufferCapacity: 16)
        // First 6 appends: linear fill.
        for i in 0..<6 {
            #expect(s.reserveNextSlot() == i)
        }
        // Past maxSize: ring within [keep, maxSize) = [2, 6).
        // Next slots: 2, 3, 4, 5, 2, 3, 4, 5, …
        let postFill = [2, 3, 4, 5, 2, 3, 4, 5]
        for expected in postFill {
            #expect(s.reserveNextSlot() == expected)
        }
        #expect(s.length == 6)
        #expect(s.absolutePosition == 14)
    }

    @Test("length grows up to maxSize then sticks; absolutePosition keeps climbing")
    func lengthSaturatesAtWindow() {
        var s = KVEvictionState(policy: .window(maxSize: 3, keep: 0),
                                bufferCapacity: 8)
        for step in 0..<8 {
            _ = s.reserveNextSlot()
            #expect(s.length == min(step + 1, 3))
            #expect(s.absolutePosition == step + 1)
        }
    }

    @Test("reset zeros both length and absolutePosition")
    func resetClearsState() {
        var s = KVEvictionState(policy: .window(maxSize: 4),
                                bufferCapacity: 8)
        for _ in 0..<10 { _ = s.reserveNextSlot() }
        #expect(s.length == 4)
        #expect(s.absolutePosition == 10)
        s.reset()
        #expect(s.length == 0)
        #expect(s.absolutePosition == 0)
        // Restart from slot 0 after reset.
        #expect(s.reserveNextSlot() == 0)
    }

    @Test("keep = maxSize - 1 produces a degenerate 1-slot rotation")
    func keepLeavesOnlyOneRotatingSlot() {
        var s = KVEvictionState(policy: .window(maxSize: 4, keep: 3),
                                bufferCapacity: 8)
        // 0, 1, 2 are pinned. After that, slot 3 is the only one
        // that rotates — every subsequent append targets slot 3.
        let expected = [0, 1, 2, 3, 3, 3, 3]
        for i in expected {
            #expect(s.reserveNextSlot() == i)
        }
    }

    @Test("truncate rolls unbounded state back and resumes from that slot")
    func truncateUnboundedRollsBack() {
        var s = KVEvictionState(policy: .unbounded, bufferCapacity: 16)
        for _ in 0..<10 { _ = s.reserveNextSlot() }
        #expect(s.length == 10)
        s.truncate(toLength: 6)
        #expect(s.length == 6)
        #expect(s.absolutePosition == 6)
        // Next append continues from slot 6, overwriting the discarded tail.
        #expect(s.reserveNextSlot() == 6)
        #expect(s.reserveNextSlot() == 7)
    }

    @Test("truncate to current length and to zero are both valid")
    func truncateBoundaries() {
        var s = KVEvictionState(policy: .unbounded, bufferCapacity: 8)
        for _ in 0..<5 { _ = s.reserveNextSlot() }
        s.truncate(toLength: 5)        // no-op
        #expect(s.length == 5)
        s.truncate(toLength: 0)        // full rollback
        #expect(s.length == 0)
        #expect(s.reserveNextSlot() == 0)
    }

    @Test("truncate on a window cache before rotation is allowed")
    func truncateWindowPreRotation() {
        var s = KVEvictionState(policy: .window(maxSize: 8, keep: 0),
                                bufferCapacity: 8)
        for _ in 0..<5 { _ = s.reserveNextSlot() }   // absoluteCount 5 ≤ maxSize 8
        s.truncate(toLength: 3)
        #expect(s.length == 3)
        #expect(s.reserveNextSlot() == 3)
    }
}

@Suite("KVEviction — KVCache (raw) integration")
struct KVCacheSlidingWindowTests {

    @Test("raw KVCache with window policy reports length = min(appended, maxSize)")
    func rawCacheLengthSaturates() throws {
        let cache = KVCache(
            nKVHeads: 2, headDim: 4, maxSeq: 8, dtype: .f32,
            eviction: .window(maxSize: 4, keep: 0)
        )
        // No appends yet.
        #expect(cache.length == 0)
        #expect(cache.eviction == .window(maxSize: 4, keep: 0))
        #expect(cache.effectiveMaxSize == 4)
        #expect(cache.maxSeq == 8)
    }

    @Test("raw KVCache append rotates physical positions in ring order")
    func rawCacheAppendRotates() throws {
        let cache = KVCache(
            nKVHeads: 1, headDim: 2, maxSeq: 8, dtype: .f32,
            eviction: .window(maxSize: 3, keep: 0)
        )
        let device = Device.shared
        let buf = device.makeBuffer(length: 8)
        // Stamp slot ID + 1.0 into K and V at each append so we can
        // tell which physical row holds which absolute token.
        for absPos in 0..<7 {
            let f = Float(absPos) + 1.0
            buf.contents().assumingMemoryBound(to: Float.self)[0] = f
            buf.contents().assumingMemoryBound(to: Float.self)[1] = f
            let kFlat = Tensor(buffer: buf, offset: 0, shape: [1, 2], dtype: .f32)
            let vFlat = Tensor(buffer: buf, offset: 0, shape: [1, 2], dtype: .f32)
            cache.append(kFlat: kFlat, vFlat: vFlat)
            // length should saturate at 3.
            #expect(cache.length == min(absPos + 1, 3))
            #expect(cache.absolutePosition == absPos + 1)
        }
        // After 7 absolute appends with maxSize=3, the ring has done
        // 2 full laps. The three slots hold absolute tokens 4, 5, 6
        // (counting from 0): slot 0 ← token 6, slot 1 ← token 4,
        // slot 2 ← token 5  (since 6 % 3 = 0, 4 % 3 = 1, 5 % 3 = 2).
        let kPtr = cache.kBuffer.buffer.contents()
            .advanced(by: cache.kBuffer.offset)
            .assumingMemoryBound(to: Float.self)
        // Slot strides: cache.kBuffer is [nKVHeads=1, maxSeq=8, headDim=2].
        // Row r starts at index r * 2.
        #expect(kPtr[0 * 2] == 7.0)  // slot 0 ← token 6 (value 7.0)
        #expect(kPtr[1 * 2] == 5.0)  // slot 1 ← token 4 (value 5.0)
        #expect(kPtr[2 * 2] == 6.0)  // slot 2 ← token 5 (value 6.0)
    }

    @Test("raw KVCache.reset returns to a fresh empty state")
    func rawCacheReset() throws {
        let cache = KVCache(
            nKVHeads: 1, headDim: 1, maxSeq: 4, dtype: .f32,
            eviction: .window(maxSize: 2)
        )
        let device = Device.shared
        let buf = device.makeBuffer(length: 4)
        buf.contents().assumingMemoryBound(to: Float.self)[0] = 1.0
        let f = Tensor(buffer: buf, offset: 0, shape: [1, 1], dtype: .f32)
        cache.append(kFlat: f, vFlat: f)
        cache.append(kFlat: f, vFlat: f)
        cache.append(kFlat: f, vFlat: f)
        #expect(cache.length == 2)
        #expect(cache.absolutePosition == 3)
        cache.reset()
        #expect(cache.length == 0)
        #expect(cache.absolutePosition == 0)
    }

    @Test("legacy KVCache init defaults to unbounded eviction")
    func rawCacheDefaultIsUnbounded() throws {
        let cache = KVCache(nKVHeads: 1, headDim: 4, maxSeq: 8, dtype: .f32)
        #expect(cache.eviction == .unbounded)
        #expect(cache.effectiveMaxSize == 8)
    }
}
