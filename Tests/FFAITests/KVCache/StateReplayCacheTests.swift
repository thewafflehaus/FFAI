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
// StateReplayCacheTests — pin the Phase 5e cache protocol's contract.
// The full record + rollback surface isn't wired yet (waiting on the
// gated_delta / ssm_replay kernel ports), but every recurrent cache
// conforms to the protocol so speculative decoding has a uniform
// surface to call against.

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("StateReplayCache protocol")
struct StateReplayCacheTests {

    @Test("SSMStateCache conforms to StateReplayCache with canStateReplay=false")
    func ssmStateCacheConforms() {
        let cache = SSMStateCache(nHeads: 4, stateDim: 16, headDim: 64)
        let asReplay: any StateReplayCache = cache
        #expect(
            asReplay.canStateReplay == false,
            "SSMStateCache should declare no replay support until ssm_replay kernel lands")
        #expect(asReplay.length == 0)
        #expect(asReplay.maxSeq == .max)
        #expect(asReplay.bytesInUse == asReplay.bytesAllocated)
    }

    @Test("rollback(acceptedPrefix: 0) is equivalent to reset() — fills state with zeros")
    func rollbackZerosState() {
        let cache = SSMStateCache(nHeads: 2, stateDim: 4, headDim: 8)
        // Poison the state with non-zero data; rollback should clear it.
        let bytes = cache.h.byteCount
        memset(
            cache.h.buffer.contents().advanced(by: cache.h.offset),
            0xFF, bytes)
        let beforeFirstByte = cache.h.buffer.contents()
            .advanced(by: cache.h.offset)
            .load(as: UInt8.self)
        #expect(beforeFirstByte == 0xFF, "test setup should poison state")

        let cmd = Device.shared.makeCommandBuffer()
        cache.rollback(acceptedPrefix: 0, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // First byte (and every byte) of state should now be zero.
        let afterFirstByte = cache.h.buffer.contents()
            .advanced(by: cache.h.offset)
            .load(as: UInt8.self)
        #expect(afterFirstByte == 0x00)
    }

    @Test("rollback with non-zero acceptedPrefix still resets (degraded until tape lands)")
    func rollbackWithPrefixIsDegradedReset() {
        let cache = SSMStateCache(nHeads: 1, stateDim: 2, headDim: 4)
        memset(
            cache.h.buffer.contents().advanced(by: cache.h.offset),
            0x42, cache.h.byteCount)
        let cmd = Device.shared.makeCommandBuffer()
        cache.rollback(acceptedPrefix: 3, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // Until ssm_replay kernel ships, the non-zero-prefix path
        // degrades to reset. Pin the behaviour so the regression
        // shows up the day the kernel actually lands.
        let firstByte = cache.h.buffer.contents()
            .advanced(by: cache.h.offset)
            .load(as: UInt8.self)
        #expect(
            firstByte == 0x00,
            "rollback with acceptedPrefix > 0 should reset until ssm_replay kernel exists")
    }

    @Test("beginRecord + commit are no-ops on the non-replay path")
    func recordCommitAreNoOps() {
        let cache = SSMStateCache(nHeads: 1, stateDim: 2, headDim: 4)
        let cmd = Device.shared.makeCommandBuffer()
        cache.beginRecord(on: cmd)
        cache.commit(on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        // No crash, no state mutation expected — that's all we pin.
    }
}
