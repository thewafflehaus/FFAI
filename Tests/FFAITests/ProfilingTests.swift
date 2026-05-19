// ProfilingTests — ProfileLevel ordering, time(...) recording,
// signpost(...) passthrough, PhaseTimings formatting.

import Foundation
import Testing
@testable import FFAI

// .serialized — every test in this suite mutates `Profile.shared`
// (sets `.level`, calls `resetPhases()`). Parallel within-suite tests
// race on that shared state: one test's `resetPhases()` wipes another
// test's recorded phases mid-assertion.
//
// The right long-term fix is to make Profile non-singleton (inject
// a local instance per test). Until that lands, .serialized is the
// cheap correct option.
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
            (0..<100).reduce(0, +)
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
}
