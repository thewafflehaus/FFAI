// Debug — gated, subsystem-tagged log helpers.
//
// FFAI subsystems opt into noisy debug output via env vars. The
// global `FFAI_DEBUG=1` enables every subsystem; per-subsystem
// `FFAI_DEBUG_<NAME>=1` enables just that one. The CLI's `--debug`
// flag sets `FFAI_DEBUG=1` early in main before any model load runs.
//
// Output goes to stderr so it doesn't pollute stdout (where the model
// text + stats live). When no subsystem is enabled, `Debug.log(...)`
// is a guarded no-op — the message string isn't even constructed
// because the parameter is `@autoclosure`.
//
// ─── Thread safety ──────────────────────────────────────────────────
//
// POSIX `setenv(3)` is **not thread-safe** on Darwin (or most
// platforms): it mutates a global `environ` array that may be
// realloc'd, and concurrent `getenv(3)` can dereference a freed
// pointer. We serialize all *Debug-internal* setenv/getenv calls
// through `Debug.lock` so reads + writes from within FFAI cooperate.
//
// External callers that mutate the env directly (`setenv` in a test
// fixture, the CLI's `--debug` handler, OS launch context) bypass
// this lock — they're inherently racy w.r.t. our reads and remain
// the caller's responsibility. The conventional safe pattern is to
// set every env var at process startup BEFORE any background thread
// is created (Swift Testing's `.serialized` suite trait on
// `DebugTests` enforces this for our own test fixtures).

import Foundation

public enum DebugSubsystem: String, Sendable, CaseIterable {
    case loader      // ModelLocator / ModelDownloader
    case load        // Model.load + family loaders
    case kernels     // Per-kernel dispatch (very chatty; opt-in)
    case sampling    // Sampling.swift
    case kvcache     // KVCache append + slice
    case generate    // Generate loop + per-token decisions
    case dispatch    // Per-MTLCommandBuffer commit/wait
    case bench       // Bench harness internals

    /// `true` when the global `FFAI_DEBUG=1` is set OR
    /// `FFAI_DEBUG_<RAWVALUE_UPPERCASED>=1`. Reads via `Debug.envIsSet`
    /// rather than `ProcessInfo.processInfo.environment` so callers
    /// that mutate the env at runtime (CLI `--debug`, tests) see the
    /// change immediately — `ProcessInfo` snapshots once and never
    /// updates.
    public var isEnabled: Bool {
        if Debug.envIsSet("FFAI_DEBUG") { return true }
        return Debug.envIsSet("FFAI_DEBUG_\(rawValue.uppercased())")
    }
}

public enum Debug {
    /// Serializes all setenv/getenv calls made by Debug itself.
    /// See file-level note on thread safety for the external-call
    /// boundary.
    private static let lock = NSLock()

    /// Thread-safe check whether an env var is set. Used internally
    /// + by `DebugSubsystem.isEnabled`. Returns `true` for any
    /// non-nil value (we treat presence-of-var as enabled, regardless
    /// of value — matches the historical contract).
    fileprivate static func envIsSet(_ key: String) -> Bool {
        lock.withLock { getenv(key) != nil }
    }

    /// Emit a debug line for `subsystem`. The message closure is only
    /// evaluated when the subsystem is enabled.
    public static func log(_ subsystem: DebugSubsystem,
                           _ message: @autoclosure () -> String) {
        guard subsystem.isEnabled else { return }
        let line = "[ffai:\(subsystem.rawValue)] \(message())\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    /// Programmatically enable the global gate from Swift code (e.g. a
    /// CLI `--debug` flag handler). Equivalent to setting
    /// `FFAI_DEBUG=1` in the process env.
    public static func enableAll() {
        lock.withLock { _ = setenv("FFAI_DEBUG", "1", 1) }
    }

    /// Programmatically enable just one subsystem.
    public static func enable(_ subsystem: DebugSubsystem) {
        lock.withLock {
            _ = setenv("FFAI_DEBUG_\(subsystem.rawValue.uppercased())", "1", 1)
        }
    }

    /// `true` when *any* subsystem (or the global gate) is enabled.
    /// Useful for callers that want to opt out of expensive
    /// instrumentation paths when nobody's listening.
    public static var isAnyEnabled: Bool {
        if envIsSet("FFAI_DEBUG") { return true }
        return DebugSubsystem.allCases.contains(where: { $0.isEnabled })
    }
}
