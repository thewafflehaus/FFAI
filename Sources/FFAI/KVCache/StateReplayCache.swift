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
// StateReplayCache — protocol shared by every "recurrent + replay"
// per-layer cache. Speculative decoding commits N tokens
// optimistically, then rolls back to the longest accepted prefix if
// any failed. For attention layers, rollback is trivial (drop K/V
// rows past the accepted index). For recurrent layers (Mamba 2,
// GatedDeltaNet, RWKV, ...), the state has already absorbed the
// optimistic tokens — we need either:
//
//   (a) a *delta tape* that records each step's update so we can
//       replay the accepted prefix from a saved snapshot, OR
//   (b) a *snapshot* taken before the speculative burst, restored on
//       rollback.
//
// The mlx-swift-lm reference picks (a) — it amortises better across
// large speculation budgets and avoids the snapshot allocation when
// the accept rate is high. We follow the same shape.
//
// First-light scope: protocol + a non-replay default conformance for
// Mamba 2 (which keeps state in a fixed [nHeads, stateDim, headDim]
// tensor and just zeroes on rollback). The full delta-tape
// implementation lands with the gated_delta / ssm_replay kernel ports
// in plannede.

import Foundation
import Metal

/// Per-layer cache with a `record + rollback` surface for speculative
/// decoding. Every concrete recurrent cache (`SSMStateCache`,
/// `GDNStateCache`, ...) conforms; attention caches (`KVCache`,
/// `AffineQuantizedKVCache`, `AURAQuantizedKVCache`) don't — they
/// achieve rollback by re-setting the sliding-window slot allocator
/// to the accepted prefix length, which is a separate codepath in
/// the speculative decoder.
public protocol StateReplayCache: LayerCacheProtocol {
    /// True when this cache implementation supports recording +
    /// rollback. `false` means rollback degrades to `reset()` and the
    /// speculative driver MUST restart from a clean prompt prefix —
    /// expensive but correct.
    var canStateReplay: Bool { get }

    /// Begin recording a speculative burst. Subsequent `appendOnGPU`-
    /// equivalent updates queue deltas onto the per-cache tape until
    /// `commit(...)` or `rollback(...)` is called.
    ///
    /// Idempotent within a single burst.
    func beginRecord(on cmd: MTLCommandBuffer)

    /// Commit the burst — drop the tape (or merge it into state, if
    /// the implementation stages updates that way).
    func commit(on cmd: MTLCommandBuffer)

    /// Roll back to the snapshot recorded at `beginRecord(...)` time
    /// plus `acceptedPrefix` confirmed extensions. `acceptedPrefix=0`
    /// is equivalent to restoring the pre-burst state. Implementations
    /// without a tape must `reset()`.
    func rollback(acceptedPrefix: Int, on cmd: MTLCommandBuffer)
}

// MARK: - Default conformance for SSMStateCache

/// Mamba 2 + future SSM variants opt into the protocol with the
/// non-replay default — rollback resets the recurrent state, the
/// speculative driver handles re-priming.
extension SSMStateCache: StateReplayCache {
    public var canStateReplay: Bool { false }

    public var length: Int { 0 }  // Recurrent state, no notion of length.
    public var maxSeq: Int { .max }
    public var bytesInUse: Int { bytesAllocated }

    public func beginRecord(on cmd: MTLCommandBuffer) {
        // No-op until the tape is wired in (state_replay
        // kernel port).
    }
    public func commit(on cmd: MTLCommandBuffer) {
        // No-op until tape is wired.
    }
    public func rollback(acceptedPrefix: Int, on cmd: MTLCommandBuffer) {
        // Without a delta tape we can't reconstruct intermediate
        // states. acceptedPrefix > 0 is therefore unrecoverable — the
        // speculative driver re-primes from the prompt.
        if acceptedPrefix == 0 {
            reset()
        } else {
            // Degraded path until ssm_replay kernel lands.
            reset()
        }
    }
}
