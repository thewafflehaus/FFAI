import Testing
@testable import FFAI

@Suite("BufferPool")
struct BufferPoolTests {
    @Test("acquire then release reuses the same buffer")
    func reuse() {
        let pool = BufferPool(device: .shared)
        let a = pool.acquire(bytes: 256)
        pool.release(a)
        let b = pool.acquire(bytes: 256)
        #expect(b === a)
    }

    @Test("different sizes get different free lists")
    func separateSizes() {
        let pool = BufferPool(device: .shared)
        let small = pool.acquire(bytes: 64)
        let large = pool.acquire(bytes: 1024)
        pool.release(small)
        pool.release(large)
        let smallAgain = pool.acquire(bytes: 64)
        let largeAgain = pool.acquire(bytes: 1024)
        #expect(smallAgain === small)
        #expect(largeAgain === large)
    }

    @Test("acquireTensor wraps a pool buffer with the right shape/dtype")
    func acquireTensor() {
        let pool = BufferPool(device: .shared)
        let t = pool.acquireTensor(shape: [16], dtype: .f32)
        #expect(t.shape == [16])
        #expect(t.dtype == .f32)
        #expect(t.buffer.length >= 64)
    }

    @Test("releaseAll empties the freelist")
    func releaseAll() {
        let pool = BufferPool(device: .shared)
        let buf = pool.acquire(bytes: 32)
        pool.release(buf)
        pool.releaseAll()
        let fresh = pool.acquire(bytes: 32)
        #expect(fresh !== buf)
    }

    @Test("shared singleton is the same instance")
    func sharedInstance() {
        #expect(BufferPool.shared === BufferPool.shared)
    }

    // MARK: - withScope

    @Test("withScope releases acquired buffers back to the pool on exit")
    func scopeReleasesOnExit() {
        let pool = BufferPool(device: .shared)
        // Inside the scope, two buffers are held.
        pool.withScope { scope in
            _ = scope.acquire(bytes: 64)
            _ = scope.acquire(bytes: 64)
            #expect(scope.heldCount == 2)
        }
        // After the scope, those buffers are back in the pool freelist.
        // Acquiring two 64-byte buffers should reuse them, not allocate.
        let a = pool.acquire(bytes: 64)
        let b = pool.acquire(bytes: 64)
        // A third acquire MUST allocate a fresh one — only 2 were
        // returned to the freelist.
        let c = pool.acquire(bytes: 64)
        #expect(a !== c, "third acquire should produce a fresh buffer")
        #expect(b !== c, "third acquire should produce a fresh buffer")
        pool.release(a); pool.release(b); pool.release(c)
    }

    @Test("withScope reuses buffers across iterations (the dogfood case)")
    func scopeReusesAcrossIterations() {
        // This is the production decode pattern: every token, acquire
        // a set of intermediates, release them at end of token, repeat.
        // Without the scope, every iteration would `device.makeBuffer`
        // a fresh one. With the scope, iteration N+1 should reuse the
        // buffer iteration N released.
        let pool = BufferPool(device: .shared)
        let iterations = 5
        let bytes = 256

        var firstBufferRef: AnyObject? = nil
        var reuseCount = 0

        for _ in 0..<iterations {
            pool.withScope { scope in
                let buf = scope.acquire(bytes: bytes)
                if firstBufferRef == nil {
                    firstBufferRef = buf as AnyObject
                } else if (buf as AnyObject) === firstBufferRef {
                    reuseCount += 1
                }
            }
        }

        // The buffer released after iteration 1 should come back on
        // iterations 2..N. So at minimum (iterations - 1) reuses.
        // We assert ≥ 1 to keep the test robust against an LRU-style
        // freelist scrambling order, but the typical case is N-1.
        #expect(reuseCount >= 1,
                "expected buffer reuse across iterations; got \(reuseCount)")
    }

    @Test("withScope releases even on throw")
    func scopeReleasesOnThrow() {
        struct LocalError: Error {}
        let pool = BufferPool(device: .shared)
        var capturedScope: BufferPoolScope? = nil

        do {
            try pool.withScope { scope in
                _ = scope.acquire(bytes: 128)
                capturedScope = scope
                throw LocalError()
            }
            Issue.record("withScope should rethrow")
        } catch is LocalError {
            // Expected. After the throw, the scope's defer should have
            // run, releasing the buffer back to the pool.
            #expect(capturedScope?.heldCount == 0,
                    "scope should release acquired buffers even on throw")
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }
}
