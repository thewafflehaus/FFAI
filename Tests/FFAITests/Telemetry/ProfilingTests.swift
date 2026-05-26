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
// ProfilingTests — ProfileLevel ordering, time(...) recording,
// signpost(...) passthrough, PhaseTimings formatting.

import Foundation
import Testing

@testable import FFAI

// .serialized — most tests in this suite mutate `Profile.shared`
// (set `.level`, call `resetPhases()`). Parallel within-suite tests
// race on that shared state: one test's `resetPhases()` wipes another
// test's recorded phases mid-assertion.
//
// `Profile` is now injectable (`init()` is public) — see
// `independentInstancesDoNotShareState` for the per-instance pattern.
// New tests should prefer a fresh `Profile()` over `Profile.shared`;
// the suite stays `.serialized` for the legacy singleton-mutating
// tests until they're all migrated.
@Suite("Profiling", .serialized)
struct ProfilingTests {

    @Test("ProfileLevel rawValue + ordering")
    func levelOrdering() {
        #expect(ProfileLevel.off.rawValue == 0)
        #expect(ProfileLevel.wallclock.rawValue == 1)
        #expect(ProfileLevel.signposts.rawValue == 2)
        #expect(ProfileLevel.off < ProfileLevel.wallclock)
        #expect(ProfileLevel.wallclock < ProfileLevel.signposts)
        #expect(ProfileLevel.allCases.count == 3)
    }

    @Test("time(...) is a passthrough at level off")
    func timePassthroughOff() {
        Profile.shared.level = .off
        Profile.shared.resetPhases()
        let result = Profile.time("phase-a") { 42 }
        #expect(result == 42)
        #expect(Profile.shared.phases.entries.isEmpty)
    }

    @Test("time(...) records duration at level wallclock")
    func timeRecords() {
        Profile.shared.level = .wallclock
        Profile.shared.resetPhases()
        let result = Profile.time("phase-x") {
            // Tiny work.
            (0 ..< 100).reduce(0, +)
        }
        #expect(result == 4950)
        let phases = Profile.shared.phases
        #expect(phases.entries.count == 1)
        #expect(phases.entries.first?.name == "phase-x")
        #expect((phases.entries.first?.durationS ?? -1) >= 0)
        Profile.shared.level = .off
    }

    @Test("signpost(...) is a passthrough below level signposts")
    func signpostPassthrough() {
        Profile.shared.level = .wallclock
        let result = Profile.signpost("noop") { 7 }
        #expect(result == 7)
    }

    @Test("recordPhase respects level gate")
    func recordPhaseGated() {
        Profile.shared.level = .off
        Profile.shared.resetPhases()
        Profile.shared.recordPhase("ignored", durationS: 1.23)
        #expect(Profile.shared.phases.entries.isEmpty)

        Profile.shared.level = .wallclock
        Profile.shared.recordPhase("kept", durationS: 1.23)
        #expect(Profile.shared.phases.entries.count == 1)
        #expect(Profile.shared.phases.entries.first?.name == "kept")
        #expect(Profile.shared.phases.entries.first?.durationS == 1.23)
        Profile.shared.level = .off
        Profile.shared.resetPhases()
    }

    @Test("PhaseTimings.formatted renders header + each entry")
    func phaseTimingsFormatted() {
        var t = PhaseTimings()
        t.record(name: "model_load", durationS: 2.34)
        t.record(name: "prefill", durationS: 0.082)
        t.record(name: "decode", durationS: 1.0)
        let out = t.formatted()
        #expect(out.hasPrefix("[PROFILE]"))
        #expect(out.contains("model_load"))
        #expect(out.contains("prefill"))
        #expect(out.contains("decode"))
        // Sub-second values render as ms; second+ as s.
        #expect(out.contains("ms"))
        #expect(out.contains(" s"))
    }

    @Test("PhaseTimings.isEmpty matches reality")
    func emptyState() {
        var t = PhaseTimings()
        #expect(t.isEmpty)
        t.record(name: "x", durationS: 0.1)
        #expect(!t.isEmpty)
    }

    @Test("Async time + signpost passthroughs return body value")
    func asyncPassthroughs() async {
        Profile.shared.level = .off
        let r1 = await Profile.timeAsync("a") { 11 }
        let r2 = await Profile.signpostAsync("b") { 12 }
        #expect(r1 == 11)
        #expect(r2 == 12)
    }

    @Test("Independent Profile instances accumulate phases separately")
    func independentInstancesDoNotShareState() {
        // Two freshly-constructed instances — the injectable surface
        // for Phase 8 per-sequence telemetry. Each owns its own level
        // and phase accumulator; neither touches Profile.shared.
        let a = Profile()
        let b = Profile()
        a.level = .wallclock
        b.level = .wallclock

        // Record disjoint phases on each instance.
        a.recordPhase("a-prefill", durationS: 0.5)
        a.recordPhase("a-decode", durationS: 1.5)
        b.recordPhase("b-decode", durationS: 2.0)

        // a sees only its own two phases.
        #expect(a.phases.entries.count == 2)
        #expect(a.phases.entries.map(\.name) == ["a-prefill", "a-decode"])
        // b sees only its own one phase — a's records did not leak in.
        #expect(b.phases.entries.count == 1)
        #expect(b.phases.entries.first?.name == "b-decode")
        #expect(b.phases.entries.first?.durationS == 2.0)

        // Resetting one instance leaves the other untouched.
        a.resetPhases()
        #expect(a.phases.entries.isEmpty)
        #expect(b.phases.entries.count == 1)

        // Levels are independent too — flipping b's level off does not
        // gate a, and neither instance disturbs Profile.shared.
        b.level = .off
        b.recordPhase("b-ignored", durationS: 9.9)
        #expect(b.phases.entries.count == 1)  // gated, not recorded
        a.recordPhase("a-kept", durationS: 0.1)
        #expect(a.phases.entries.count == 1)  // a still at .wallclock

        // The instance time(...) / signpost(...) helpers run against
        // the instance, not the singleton.
        let timed = a.time("a-timed") { 99 }
        #expect(timed == 99)
        #expect(a.phases.entries.contains { $0.name == "a-timed" })
        #expect(!b.phases.entries.contains { $0.name == "a-timed" })
    }
}
