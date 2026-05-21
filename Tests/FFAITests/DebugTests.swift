// DebugTests — env-var gating + per-subsystem opt-in.
//
// Tests mutate the process env via setenv/unsetenv. Restored on exit
// of each test so the suite stays hermetic.

import Foundation
import Testing
@testable import FFAI

@Suite("Debug", .serialized)
struct DebugTests {

    // MARK: helpers

    private func clearAllDebugEnv() {
        unsetenv("FFAI_DEBUG")
        for sub in DebugSubsystem.allCases {
            unsetenv("FFAI_DEBUG_\(sub.rawValue.uppercased())")
        }
    }

    // MARK: tests

    @Test("All subsystems off by default")
    func defaultOff() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        for sub in DebugSubsystem.allCases {
            #expect(sub.isEnabled == false)
        }
        #expect(Debug.isAnyEnabled == false)
    }

    @Test("FFAI_DEBUG=1 enables every subsystem")
    func globalGate() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        setenv("FFAI_DEBUG", "1", 1)
        for sub in DebugSubsystem.allCases {
            #expect(sub.isEnabled, "\(sub.rawValue) should be enabled by FFAI_DEBUG")
        }
        #expect(Debug.isAnyEnabled)
    }

    @Test("FFAI_DEBUG_<NAME>=1 enables just that subsystem")
    func perSubsystemGate() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        setenv("FFAI_DEBUG_LOADER", "1", 1)
        #expect(DebugSubsystem.loader.isEnabled)
        #expect(DebugSubsystem.kernels.isEnabled == false)
        #expect(Debug.isAnyEnabled)
    }

    @Test("Debug.enableAll() flips the global gate")
    func enableAll() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        Debug.enableAll()
        #expect(DebugSubsystem.generate.isEnabled)
    }

    @Test("Debug.enable(_:) flips one subsystem")
    func enableOne() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        Debug.enable(.bench)
        #expect(DebugSubsystem.bench.isEnabled)
        #expect(DebugSubsystem.kvcache.isEnabled == false)
    }

    @Test("Debug.log evaluates the closure only when subsystem enabled")
    func lazyEvaluation() {
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }
        // Off — closure must NOT fire.
        var called = false
        Debug.log(.kvcache, { () -> String in called = true; return "x" }())
        #expect(called == false)

        // On — closure DOES fire.
        setenv("FFAI_DEBUG_KVCACHE", "1", 1)
        Debug.log(.kvcache, { () -> String in called = true; return "x" }())
        #expect(called == true)
    }

    @Test("DebugSubsystem rawValues stable")
    func rawValues() {
        let expected: [DebugSubsystem: String] = [
            .loader: "loader", .load: "load", .kernels: "kernels",
            .sampling: "sampling", .kvcache: "kvcache",
            .generate: "generate", .dispatch: "dispatch", .bench: "bench",
        ]
        for (sub, raw) in expected {
            #expect(sub.rawValue == raw)
        }
    }

    @Test("Concurrent enable and isEnabled access survives without crash")
    func concurrentAccessDoesNotCrash() {
        // POSIX setenv/getenv are not thread-safe. Debug serializes its
        // own access through an NSLock so concurrent FFAI-internal
        // reads/writes can't dereference a freed `environ` pointer.
        // This test exercises that lock contract: hammer Debug.enable
        // and Debug.isAnyEnabled from many threads at once and assert
        // no crash plus a deterministic final state.
        //
        // External setenv (test fixtures via the C API directly) is
        // still racy against this lock and remains the caller's
        // problem; that's not what this test verifies.
        clearAllDebugEnv()
        defer { clearAllDebugEnv() }

        let iterations = 200
        let workerCount = 8

        // DispatchQueue.concurrentPerform is the simplest way to fan
        // work across cores synchronously. The closure body is invoked
        // once per `i ∈ [0, workerCount)`, in parallel, with the call
        // blocking until every iteration completes — so the assertions
        // below run on a stable post-state.
        DispatchQueue.concurrentPerform(iterations: workerCount) { i in
            for _ in 0..<iterations {
                // Half the workers flip subsystems; half read.
                // Both paths take Debug.lock.
                if i % 2 == 0 {
                    Debug.enable(.kernels)
                    Debug.enable(.sampling)
                } else {
                    _ = Debug.isAnyEnabled
                    _ = DebugSubsystem.kernels.isEnabled
                    _ = DebugSubsystem.sampling.isEnabled
                }
            }
        }

        // After all workers finish, the writes from the even-indexed
        // tasks have settled. .kernels + .sampling should be enabled;
        // other subsystems should still be off (no other writers).
        #expect(DebugSubsystem.kernels.isEnabled)
        #expect(DebugSubsystem.sampling.isEnabled)
        #expect(DebugSubsystem.loader.isEnabled == false)
        #expect(DebugSubsystem.bench.isEnabled == false)
    }
}
