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
// KVCacheEviction — maximum-size + FIFO eviction policy shared across
// every concrete KV cache implementation.
//
// mlx-swift-lm calls this "rotating" / "sliding window" cache; the
// semantics are identical:
//   - Cache stores up to `maxSize` positions.
//   - First `keep` positions are pinned (attention sinks).
//   - When full, the oldest non-sink slot is overwritten by the next
//     append. The buffer rotates through positions
//     `[keep, keep+1, …, maxSize-1, keep, …]`.
//   - SDPA reads the full buffer up to `length = min(absolute, maxSize)`.
//
// RoPE is baked into K at insertion time, so a rotated buffer still
// produces correct softmax attention — temporal order in the buffer
// does not matter (softmax is permutation-invariant over the key axis).
// What does matter is that each K row carries the rotation for the
// *absolute* sequence position when it was inserted, not the slot
// index.
//
// This file defines the policy + a tiny helper for computing the
// physical write slot. Each KV cache implementation embeds a
// `KVCacheEviction` and uses `nextWriteSlot()` from `appendOnGPU`.

import Foundation

/// FIFO eviction policy. `.unbounded` (default) preserves the legacy
/// behaviour: append fails once `length == maxSeq`. `.window(maxSize:
/// keep:)` enables ring-buffer rotation past `maxSize`.
public enum KVEviction: Sendable, Equatable {
    /// Legacy: cache grows up to `maxSeq` and then `appendOnGPU` panics.
    /// Equivalent to setting `maxSize = nil` on every caller.
    case unbounded

    /// Sliding-window with optional attention sinks.
    /// - `maxSize`: maximum positions retained. Must satisfy
    ///   `keep ≤ maxSize ≤ maxSeq`.
    /// - `keep`: number of initial positions that are pinned and
    ///   never evicted (the "attention sinks" of Xiao et al. 2023).
    ///   Default 0 — pure FIFO with no sinks.
    case window(maxSize: Int, keep: Int = 0)
}

/// Stateful slot-allocator + length tracker. Lives inside every
/// KVCache implementation; the cache calls `reserveNextSlot()` from
/// inside its length-lock critical section in `appendOnGPU`.
///
/// Not `Sendable` — callers (KVCache, AffineQuantizedKVCache,
/// AURAQuantizedKVCache) hold this as a private member and serialise
/// access via their existing lengthLock NSLock.
public struct KVEvictionState {
    /// The configured eviction policy.
    public let policy: KVEviction
    /// Physical capacity of the backing buffer. Even in `.window`
    /// mode the buffer may be sized larger than `maxSize` (e.g. when
    /// the model file declared `maxSeq=4096` but the caller passes
    /// `maxSize=512`) — the unused tail is just wasted memory.
    ///
    /// Mutable to support an incrementally-grown `.unbounded` cache:
    /// the backing buffer starts small (KVCache's initial capacity) and
    /// `KVCache.ensureCapacity` reallocs + bumps this via `grow(to:)`
    /// when the live length approaches the current capacity. Windows
    /// never grow (their buffer is sized to `maxSize` up front).
    public private(set) var bufferCapacity: Int

    /// Monotonic count of tokens appended (never resets except via
    /// the parent cache's `reset()`). Used to derive the rotation
    /// offset.
    private var absoluteCount: Int = 0

    public init(policy: KVEviction, bufferCapacity: Int) {
        switch policy {
        case .unbounded:
            break
        case .window(let maxSize, let keep):
            precondition(
                maxSize > 0, "KVEviction.window: maxSize must be positive (got \(maxSize))")
            precondition(keep >= 0, "KVEviction.window: keep must be non-negative (got \(keep))")
            precondition(
                keep < maxSize,
                "KVEviction.window: keep (\(keep)) must be < maxSize (\(maxSize))")
            precondition(
                maxSize <= bufferCapacity,
                "KVEviction.window: maxSize (\(maxSize)) must be ≤ bufferCapacity (\(bufferCapacity))"
            )
        }
        self.policy = policy
        self.bufferCapacity = bufferCapacity
    }

    /// The effective live length of the cache — what `KVCacheProtocol.length`
    /// returns. Saturates at the window's `maxSize` once filled.
    public var length: Int {
        switch policy {
        case .unbounded:
            return absoluteCount
        case .window(let maxSize, _):
            return min(absoluteCount, maxSize)
        }
    }

    /// Total tokens appended to the cache since the last `reset()`.
    /// Used for RoPE position computation (Q rotates against this).
    public var absolutePosition: Int { absoluteCount }

    /// Reserve and return the next physical slot index. Panics if
    /// `.unbounded` and `absoluteCount == bufferCapacity`.
    ///
    /// Side effect: bumps `absoluteCount`.
    public mutating func reserveNextSlot() -> Int {
        let slot: Int
        switch policy {
        case .unbounded:
            precondition(
                absoluteCount < bufferCapacity,
                "KVCache: capacity exhausted (\(bufferCapacity)) — pass `.window(maxSize:)` to enable FIFO eviction"
            )
            slot = absoluteCount
        case .window(let maxSize, let keep):
            if absoluteCount < keep {
                slot = absoluteCount
            } else {
                // Ring within [keep, maxSize). After `maxSize - keep`
                // post-sink appends, slot returns to `keep`.
                slot = keep + ((absoluteCount - keep) % (maxSize - keep))
            }
        }
        absoluteCount += 1
        return slot
    }

    public mutating func reset() {
        absoluteCount = 0
    }

    /// Raise the recorded buffer capacity after the parent cache
    /// reallocates its backing buffers to a larger size. Only valid
    /// for `.unbounded` policies (windows are sized to `maxSize` at
    /// construction and never grow). `newCapacity` must be strictly
    /// larger than the current capacity.
    public mutating func grow(to newCapacity: Int) {
        precondition(
            { if case .unbounded = policy { return true } else { return false } }(),
            "KVEvictionState.grow: only .unbounded caches grow (window is sized to maxSize)")
        precondition(
            newCapacity > bufferCapacity,
            "KVEvictionState.grow: newCapacity \(newCapacity) must exceed current \(bufferCapacity)")
        bufferCapacity = newCapacity
    }

    /// Roll the cache state back to `length` appended tokens, discarding
    /// the tail. Used by speculative decoding to drop rejected draft
    /// tokens after a verify pass. The physical buffer slots are left
    /// untouched — the next `reserveNextSlot()` simply overwrites them.
    ///
    /// `length` must be in `0...self.length`. For `.window` policies the
    /// rollback must stay within the pre-rotation region
    /// (`absoluteCount ≤ maxSize`) so slot indices remain identity-mapped
    /// with their absolute positions; rolling back a rotated ring buffer
    /// is rejected.
    public mutating func truncate(toLength length: Int) {
        precondition(
            length >= 0 && length <= self.length,
            "KVEvictionState.truncate: length \(length) out of range 0...\(self.length)")
        if case .window(let maxSize, _) = policy {
            precondition(
                absoluteCount <= maxSize,
                "KVEvictionState.truncate: unsupported after window rotation "
                    + "(absoluteCount \(absoluteCount) > maxSize \(maxSize))")
        }
        absoluteCount = length
    }
}
