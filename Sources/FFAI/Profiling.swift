// Profiling — wallclock + os_signpost instrumentation.
//
// Three levels, mirroring mlx-swift-lm's `MLX_BENCH_PROFILE`:
//
//   • 0 (off, default) — `Profile.signpost(...)` is a passthrough,
//     `Profile.recordPhase(...)` is a no-op. Zero overhead anywhere.
//   • 1 (wallclock)    — phase-boundary timestamps captured into a
//     `PhaseTimings` struct. Printed at the end of a CLI run via
//     `--profiling 1`. Adds a few `Date()` reads per generation
//     (sub-µs each, well under the noise floor of single-token decode).
//   • 2 (signposts)    — level 1 plus `os_signpost` intervals at
//     every wrapped call site. Captured by Instruments / `xctrace`
//     under subsystem `ai.ffai`. **Zero overhead when no tracer is
//     attached** because `OSSignposter` checks a flag and bails — the
//     interval state isn't built. Aligns on the same Instruments
//     timeline as Apple's Metal subsystem signposts (`com.apple.Metal`)
//     so kernel GPU spans attribute under the right phase.
//
// ─── Pattern for new call sites ─────────────────────────────────────
//
//     Profile.signpost("MyOp.compute") {
//         // existing impl
//     }
//
// `signpost(...)` is a generic passthrough — wraps both throwing and
// non-throwing closures, returns the body's value untouched. When
// `Profile.shared.level < 2` it's a single `body()` call with no
// signpost machinery; the wrapped op pays nothing. When level 2 is
// set, we begin/end an interval named `"MyOp.compute"`.
//
// For phase-boundary events that don't bracket a closure (e.g. "first
// token sampled"), use `Profile.event("ttft")` — emits an
// `OSSignpostType.event` marker.
//
// ─── Apple Metal kernel signposts ──────────────────────────────────
//
// Metal System Trace in Instruments captures every `MTLComputeCommand
// Encoder.dispatchThreadgroups(...)` automatically — you don't need to
// wrap kernel dispatches yourself to see them on the timeline.
// FFAI's `Profile.signpost(...)` calls add the *phase* spans
// (prefill / decode / per-layer) so the Metal kernel events nest
// under the right context.

import Foundation
import os.log
import os.signpost

public enum ProfileLevel: Int, Sendable, Comparable, CaseIterable {
    case off = 0
    case wallclock = 1
    case signposts = 2

    public static func < (lhs: ProfileLevel, rhs: ProfileLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public final class Profile: @unchecked Sendable {
    /// Process-wide default sink. `Model.generate(..., profile:)` and
    /// the `Generate.swift` entry points default to this; pass a fresh
    /// `Profile()` to accumulate telemetry independently (per-sequence
    /// telemetry under Phase 8 batched decode, isolated test state).
    public static let shared = Profile()

    /// Atomically-set level. CLI sets this once at startup; callers
    /// read freely without a lock (writes are infrequent / startup-only
    /// in practice).
    private let _level = OSAllocatedUnfairLock<ProfileLevel>(initialState: .off)
    public var level: ProfileLevel {
        get { _level.withLock { $0 } }
        set { _level.withLock { $0 = newValue } }
    }

    public let log = OSLog(subsystem: "ai.ffai", category: .pointsOfInterest)
    public let signposter: OSSignposter

    /// Create an independent profiling sink. Each instance owns its own
    /// level + phase-timing accumulator; instances do not share state.
    public init() {
        self.signposter = OSSignposter(logHandle: log)
    }

    // ─── Wallclock phase timings (level ≥ 1) ─────────────────────────

    private let _phases = OSAllocatedUnfairLock<PhaseTimings>(initialState: PhaseTimings())
    public var phases: PhaseTimings {
        _phases.withLock { $0 }
    }

    /// Record a wallclock duration for a named phase. No-op below
    /// level 1.
    public func recordPhase(_ name: String, durationS: Double) {
        guard level >= .wallclock else { return }
        _phases.withLock { $0.record(name: name, durationS: durationS) }
    }

    public func resetPhases() {
        _phases.withLock { $0 = PhaseTimings() }
    }
}

// MARK: - Instance helpers (injectable profiling)

// Phase 8's batched / speculative decode needs per-sequence telemetry —
// several `Profile` instances accumulating independently rather than one
// process-wide singleton. These instance methods are the injectable
// surface: `Model.generate(..., profile:)` and `Generate.swift`'s
// generate entry points take a `Profile` and call `profile.signpost(…)`
// instead of reaching for `Profile.shared`. The `static` variants below
// are thin `shared`-bound wrappers kept for call sites (Ops wrappers,
// model layers) that have no injected instance to thread.
public extension Profile {
    /// Wrap a closure in an `os_signpost` interval. Below level 2
    /// this is a single `body()` call with no instrumentation cost.
    @inlinable
    func signpost<T>(_ name: StaticString,
                     _ body: () throws -> T) rethrows -> T {
        guard level >= .signposts else { return try body() }
        let id = signposter.makeSignpostID()
        let state = signposter.beginInterval(name, id: id)
        defer { signposter.endInterval(name, state) }
        return try body()
    }

    /// Async variant.
    @inlinable
    func signpostAsync<T>(_ name: StaticString,
                          _ body: () async throws -> T) async rethrows -> T {
        guard level >= .signposts else { return try await body() }
        let id = signposter.makeSignpostID()
        let state = signposter.beginInterval(name, id: id)
        defer { signposter.endInterval(name, state) }
        return try await body()
    }

    /// Emit a single point-in-time signpost event. Use for instants
    /// without duration (e.g. "first token sampled", "EOS hit").
    @inlinable
    func event(_ name: StaticString) {
        guard level >= .signposts else { return }
        signposter.emitEvent(name)
    }

    /// Time `body()` and record it under `phaseName` if level ≥ 1.
    /// Always returns the body's value.
    @inlinable
    func time<T>(_ phaseName: String,
                 _ body: () throws -> T) rethrows -> T {
        guard level >= .wallclock else { return try body() }
        let start = Date()
        defer { recordPhase(phaseName, durationS: Date().timeIntervalSince(start)) }
        return try body()
    }

    /// Async variant.
    @inlinable
    func timeAsync<T>(_ phaseName: String,
                      _ body: () async throws -> T) async rethrows -> T {
        guard level >= .wallclock else { return try await body() }
        let start = Date()
        defer { recordPhase(phaseName, durationS: Date().timeIntervalSince(start)) }
        return try await body()
    }
}

// MARK: - Free helpers (call site ergonomics)

// `shared`-bound forwarding wrappers. Call sites with an injected
// `Profile` should prefer the instance methods above; these exist for
// the many fixed call sites (Ops wrappers, model layers) that have no
// instance to thread and always profile against the process singleton.
public extension Profile {
    /// Wrap a closure in an `os_signpost` interval. Below level 2
    /// this is a single `body()` call with no instrumentation cost.
    @inlinable
    static func signpost<T>(_ name: StaticString,
                            _ body: () throws -> T) rethrows -> T {
        try shared.signpost(name, body)
    }

    /// Async variant.
    @inlinable
    static func signpostAsync<T>(_ name: StaticString,
                                 _ body: () async throws -> T) async rethrows -> T {
        try await shared.signpostAsync(name, body)
    }

    /// Emit a single point-in-time signpost event. Use for instants
    /// without duration (e.g. "first token sampled", "EOS hit").
    @inlinable
    static func event(_ name: StaticString) {
        shared.event(name)
    }

    /// Time `body()` and record it under `phaseName` if level ≥ 1.
    /// Always returns the body's value.
    @inlinable
    static func time<T>(_ phaseName: String,
                        _ body: () throws -> T) rethrows -> T {
        try shared.time(phaseName, body)
    }

    /// Async variant.
    @inlinable
    static func timeAsync<T>(_ phaseName: String,
                             _ body: () async throws -> T) async rethrows -> T {
        try await shared.timeAsync(phaseName, body)
    }
}

// MARK: - Phase timings printer (level 1 output)

public struct PhaseTimings: Sendable {
    public private(set) var entries: [(name: String, durationS: Double)] = []

    public mutating func record(name: String, durationS: Double) {
        entries.append((name, durationS))
    }

    public var isEmpty: Bool { entries.isEmpty }

    public func formatted() -> String {
        var out = "[PROFILE]\n"
        let nameWidth = max(18, (entries.map { $0.name.count }.max() ?? 0) + 2)
        for (name, dur) in entries {
            let padded = name.padding(toLength: nameWidth, withPad: " ", startingAt: 0)
            if dur < 1.0 {
                out += String(format: "  %@ %6.2f ms\n", padded, dur * 1000)
            } else {
                out += String(format: "  %@ %6.2f s\n", padded, dur)
            }
        }
        return out
    }
}
