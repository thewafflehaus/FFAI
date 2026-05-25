// Test-bundle-wide helpers shared by FFAITests + ModelTests.
//
// `Tests/Helpers/` is its own SwiftPM target (see Package.swift) so it
// can be a dependency of both test targets without source duplication.
// Helpers only touch FFAI's public API — no `@testable` import.

import Foundation
import FFAI

// MARK: - ModelLoadLock

/// Global mutex around `Model.load(...)` for integration tests.
///
/// Swift Testing parallelises across `@Suite` by default. Per-suite
/// `.serialized` keeps tests *within* a suite sequential but doesn't
/// prevent two different suites racing their `Model.load(...)` calls
/// — each downloading + mmap-ing multi-GB checkpoints, allocating GPU
/// buffers, and JIT-compiling PSOs. The RAM / disk-IO / GPU-memory
/// spike OOMs 8–16 GB boxes.
///
/// This actor makes `Model.load(...)` a global critical section across
/// the whole bundle so suites still benefit from parallel test
/// discovery + setup, but the heavyweight load step happens
/// one-at-a-time.
public actor ModelLoadLock {
    public static let shared = ModelLoadLock()

    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Run `work` while holding the global model-load lock. Concurrent
    /// callers block until this caller returns or throws. Errors from
    /// `work` are re-thrown after the lock is released.
    public func loadSerially<T: Sendable>(
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

// MARK: - Shared model loading

/// Load a model from the local HuggingFace cache, falling back to a
/// fresh download if it isn't there. Defaults to the 4-bit MLX
/// quantisation if `repoId` already points to one (callers pass the
/// 4-bit repo id directly — this helper doesn't translate names).
///
/// The load is serialised through `ModelLoadLock.shared` so concurrent
/// suites don't double-download or trample each other's GPU memory.
///
/// **Fails the test if the load fails** — caller intent in tests is
/// "I need this model to run my assertions", so a load failure surfaces
/// as a thrown error rather than being silently skipped.
public func loadModel(
    _ repoId: String,
    options: LoadOptions = LoadOptions()
) async throws -> Model {
    try await ModelLoadLock.shared.loadSerially {
        try await Model.load(repoId, options: options)
    }
}

// MARK: - Resource fixtures

/// Absolute URL to `Tests/Resources/<name>`. Resources are looked up
/// via `#filePath` rather than a SwiftPM bundle so they work from any
/// working directory the test runner uses.
public func resourceURL(_ name: String, file: StaticString = #filePath) -> URL {
    URL(fileURLWithPath: "\(file)")
        .deletingLastPathComponent()      // Tests/Helpers/
        .deletingLastPathComponent()      // Tests/
        .appendingPathComponent("Resources")
        .appendingPathComponent(name)
}
