// Global mutex around Model.load(...) for the integration test bundle.
//
// Why: Swift Testing parallelizes across @Suite by default. Per-suite
// `.serialized` traits keep tests *within* a suite sequential, but
// don't prevent two different ModelTests suites from running
// concurrently. Without this lock, two suites can race their
// Model.load(...) calls — each downloading + mmap-ing a multi-GB
// checkpoint, allocating GPU buffers for weights, and JIT-compiling
// dozens of PSOs. The RAM / disk-IO / GPU-memory spike is enough
// to OOM 8-16 GB boxes and stalls even larger ones.
//
// This lock makes Model.load(...) a global critical section across
// the whole bundle, so the suites still benefit from parallel test
// discovery + setup, but the heavyweight load step happens
// one-at-a-time. Once a model is loaded, the rest of its suite
// runs serially (via `.serialized`) and the lock is free for the
// next suite.
//
// Implementation note on actor reentrancy: a naive actor method
// would NOT serialize, because Swift's actor isolation releases at
// every `await`. We track an explicit `busy` flag and use a queue
// of continuations so concurrent callers genuinely wait their turn.

import Foundation
@testable import FFAI

actor ModelLoadLock {
    static let shared = ModelLoadLock()

    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Run `work` while holding the global model-load lock. Concurrent
    /// callers block until this caller returns or throws. Errors from
    /// `work` are re-thrown after the lock is released.
    ///
    /// Typical use at a test call-site:
    ///
    ///     let m = try await ModelLoadLock.shared.loadSerially {
    ///         try await Model.load(modelId)
    ///     }
    ///
    func loadSerially<T: Sendable>(
        _ work: @Sendable () async throws -> T
    ) async throws -> T {
        await acquire()
        do {
            let result = try await work()
            release()
            return result
        } catch {
            release()
            throw error
        }
    }

    private func acquire() async {
        if !busy {
            busy = true
            return
        }
        await withCheckedContinuation { cont in
            waiters.append(cont)
        }
        // When resumed, the releaser has already left `busy = true`
        // for us, so this caller now owns the lock.
    }

    private func release() {
        if let next = waiters.first {
            waiters.removeFirst()
            // Don't toggle `busy`; ownership transfers directly to
            // the next waiter so a third caller seeing `busy == false`
            // can't sneak in between.
            next.resume()
        } else {
            busy = false
        }
    }
}
