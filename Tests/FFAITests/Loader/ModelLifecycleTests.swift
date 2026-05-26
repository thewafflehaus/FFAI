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
import Foundation
import Testing
@testable import FFAI

@Suite("ModelLifecycle")
struct ModelLifecycleTests {
    @Test("LoadProgress.fraction handles total > 0 and total == 0")
    func loadProgressFraction() {
        let p = LoadProgress(stage: "weights", completed: 42, total: 84)
        #expect(p.fraction == 0.5)

        let zero = LoadProgress(stage: "noop", completed: 0, total: 0)
        #expect(zero.fraction == 0)
    }

    @Test("LoadProgress carries stage, completed, total")
    func loadProgressFields() {
        let p = LoadProgress(stage: "config", completed: 3, total: 10)
        #expect(p.stage == "config")
        #expect(p.completed == 3)
        #expect(p.total == 10)
    }

    @Test("ModelLifecycleError wraps Error and preserves message")
    func wrapError() {
        struct Boom: Error, CustomStringConvertible {
            var description: String { "boom" }
        }
        let wrapped = ModelLifecycleError(Boom())
        #expect(wrapped.message.contains("boom"))
        #expect(String(describing: wrapped).contains("boom"))
    }

    @Test("ModelLifecycleError accepts a raw message")
    func messageInit() {
        let e = ModelLifecycleError(message: "something")
        #expect(e.message == "something")
        #expect(e.description == "something")
    }

    @Test("ModelLifecycleEvent default capability is nil")
    func eventDefaults() {
        let e = ModelLifecycleEvent(state: .ready)
        #expect(e.capability == nil)
        if case .ready = e.state { /* ok */ } else {
            Issue.record("expected .ready")
        }
    }

    @Test("ModelLifecycleEvent can target a specific capability")
    func eventCapability() {
        let e = ModelLifecycleEvent(capability: .imageIn,
                                    state: .loading(LoadProgress(stage: "vision", completed: 0, total: 1)))
        #expect(e.capability == .imageIn)
        if case .loading(let p) = e.state {
            #expect(p.stage == "vision")
        } else {
            Issue.record("expected .loading state")
        }
    }

    // MARK: - Model.events buffering policy

    @Test("Model.eventsBufferCapacity is positive and bounded")
    func eventsBufferCapacityIsBounded() {
        // Sanity: not zero (would drop every event); not enormous
        // (would defeat the whole point of bounding the buffer).
        #expect(Model.eventsBufferCapacity > 0)
        #expect(Model.eventsBufferCapacity <= 1024)
    }

    @Test("AsyncStream.bufferingNewest drops oldest events when buffer is full")
    func bufferingNewestSemantic() async {
        // Pins the AsyncStream contract that `Model.events` relies on.
        // `Model` previously used the default `.unbounded` policy, which
        // leaks unconsumed events forever. The fix routes through
        // `.bufferingNewest(Model.eventsBufferCapacity)`; this test
        // verifies the policy actually drops older items when full.
        let cap = 4
        let (stream, cont) = AsyncStream<Int>.makeStream(
            bufferingPolicy: .bufferingNewest(cap)
        )

        // Yield 10 items with no consumer attached. Older 6 should be
        // dropped; only the last 4 (6..9) retained.
        for i in 0..<10 {
            cont.yield(i)
        }
        cont.finish()

        var received: [Int] = []
        for await item in stream {
            received.append(item)
        }
        #expect(received == [6, 7, 8, 9],
                "bufferingNewest(\(cap)) should keep newest 4; got \(received)")
    }
}
