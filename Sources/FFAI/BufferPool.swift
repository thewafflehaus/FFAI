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
// BufferPool — simple reusable MTLBuffer allocator for activation
// tensors. implementation: per-byte-size LIFO. No fancy
// fragmentation handling, no sub-allocation. Activations are small and
// short-lived; this is good enough until profiles say otherwise.

import Foundation
import Metal

public final class BufferPool: @unchecked Sendable {
    public let device: Device
    private let lock = NSLock()
    private var freelists: [Int: [MTLBuffer]] = [:]  // bytes → free buffers

    public static let shared = BufferPool(device: .shared)

    public init(device: Device) {
        self.device = device
    }

    /// Borrow a buffer of at least `bytes`. Caller MUST return it via
    /// `release(_:)` when done, or memory accumulates.
    public func acquire(bytes: Int) -> MTLBuffer {
        lock.lock()
        if var pool = freelists[bytes], let buf = pool.popLast() {
            freelists[bytes] = pool
            lock.unlock()
            return buf
        }
        lock.unlock()
        return device.makeBuffer(length: bytes)
    }

    /// Return a buffer to the pool. Don't use it after this call.
    public func release(_ buffer: MTLBuffer) {
        let bytes = buffer.length
        lock.lock()
        freelists[bytes, default: []].append(buffer)
        lock.unlock()
    }

    /// Allocate a Tensor from the pool. NOTE: caller is responsible for
    /// returning `tensor.buffer` to the pool when done. For typical use
    /// inside a single forward pass, drop the entire pool's buffers at
    /// end of forward pass with `releaseAll()`.
    public func acquireTensor(shape: [Int], dtype: DType) -> Tensor {
        let count = shape.reduce(1, *)
        let bytes = count * dtype.byteSize
        let buf = acquire(bytes: bytes)
        return Tensor(buffer: buf, offset: 0, shape: shape, dtype: dtype)
    }

    /// Drop everything. For we just reallocate freely; caller
    /// invokes this between forward passes if memory is a concern.
    public func releaseAll() {
        lock.lock()
        freelists.removeAll()
        lock.unlock()
    }

    /// Run `body` inside a scope. Buffers acquired via the supplied
    /// `BufferPoolScope` get released back to the pool automatically
    /// when the scope ends (success or throw). This is the right
    /// shape for per-token / per-forward-pass allocation arenas.
    ///
    /// `Tensor` is a value type that holds an `MTLBuffer` reference;
    /// Metal's own refcount frees the buffer when the last Tensor
    /// drops, which means a naive `acquire` without scope or manual
    /// `release` just leaks one new MTLBuffer per call (or worse,
    /// returns to Metal's pool while we think it's in ours). The
    /// scope makes the acquire/release pair impossible to forget.
    public func withScope<R>(_ body: (BufferPoolScope) throws -> R) rethrows -> R {
        let scope = BufferPoolScope(pool: self)
        defer { scope.releaseAcquired() }
        return try body(scope)
    }
}

/// Per-scope acquire-and-release tracker. Created by `BufferPool.withScope`.
/// All buffers acquired through this object are returned to the parent pool
/// when the scope ends.
public final class BufferPoolScope: @unchecked Sendable {
    private let pool: BufferPool
    private let lock = NSLock()
    private var acquired: [MTLBuffer] = []

    fileprivate init(pool: BufferPool) {
        self.pool = pool
    }

    /// Borrow a buffer of at least `bytes`. Returned to the pool when
    /// the surrounding `withScope` ends.
    public func acquire(bytes: Int) -> MTLBuffer {
        let buf = pool.acquire(bytes: bytes)
        lock.lock()
        acquired.append(buf)
        lock.unlock()
        return buf
    }

    /// Borrow a Tensor backed by a pool buffer. Returned to the pool
    /// when the surrounding `withScope` ends.
    public func acquireTensor(shape: [Int], dtype: DType) -> Tensor {
        let count = shape.reduce(1, *)
        let bytes = count * dtype.byteSize
        let buf = acquire(bytes: bytes)
        return Tensor(buffer: buf, offset: 0, shape: shape, dtype: dtype)
    }

    fileprivate func releaseAcquired() {
        lock.lock()
        let toRelease = acquired
        acquired.removeAll()
        lock.unlock()
        for buf in toRelease {
            pool.release(buf)
        }
    }

    /// Test-visible: number of buffers currently held by this scope.
    /// Goes to 0 after the scope's `body` returns.
    internal var heldCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return acquired.count
    }
}
