// Copyright 2026 Tom Turney (@TheTom)
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
// CacheSnapshot — composite snapshot/restore over `[any LayerCacheProtocol]`.
//
// Speculative decode needs to roll back ALL of a model's per-layer
// caches on draft rejection: KVCache (attention), GDNStateCache +
// ConvStateCache (GDN). This file gives the spec-decode driver one
// call to snapshot every layer + one call to restore.
//
// Per-cache-kind snapshot payload:
//   * Attention caches     → `length` + `absolutePosition` integers
//     (KVCache /             only (the slots beyond `length` are unused;
//      Affine- /             appending after restore overwrites them —
//      AURAQuantizedKVCache) affine overwrites the packed bits, AURA
//                            zeros a re-written slot's stale atomic_or
//                            bits via its high-water mark). No buffer
//                            copy. Window eviction is the exception —
//                            see plan.md Phase 8.25.
//   * GDNStateCache        → Tensor snapshot of `current` +
//                            `length` integer.
//   * ConvStateCache       → Tensor snapshot of `state`.
//   * GDNLayerCache        → both of the above (conv + GDN +
//                            shared `length`).
//
// Cost on Qwen3.6-A3B: 60 MiB GDN state snapshots + 900 KiB conv
// snapshots + ~zero attention metadata = ~61 MiB per spec step. On
// Apple silicon unified memory at ~400 GB/s that's ~150 µs in
// aggregate blit time — negligible vs ~60 ms decode step.

import Foundation

/// Per-layer snapshot. The enum cases match the concrete cache types
/// the model layer slots take; the spec-decode driver routes per
/// layer.
public enum LayerCacheSnapshot {
    case kv(length: Int, absolutePosition: Int)
    case gdn(currentState: Tensor, length: Int)
    case conv(state: Tensor)
    case gdnLayer(conv: Tensor, gdnState: Tensor, length: Int)
}

extension Array where Element == any LayerCacheProtocol {
    /// Snapshot every layer cache. Cost is ~150 µs aggregate blit time
    /// on Qwen3.6-A3B (60 MiB GDN + 900 KB conv).
    public func snapshotAll(device: Device = .shared) -> [LayerCacheSnapshot] {
        return self.map { cache -> LayerCacheSnapshot in
            if let composite = cache as? GDNLayerCache {
                return .gdnLayer(
                    conv: composite.conv.snapshot(device: device),
                    gdnState: composite.gdn.snapshot(device: device),
                    length: composite.length)
            }
            if let gdn = cache as? GDNStateCache {
                return .gdn(
                    currentState: gdn.snapshot(device: device),
                    length: gdn.length)
            }
            // All attention caches — raw `KVCache`, `AffineQuantizedKVCache`,
            // `AURAQuantizedKVCache` — roll back with a pure length rewind:
            // the physical slots beyond `length` are unused and get
            // overwritten on the next append. Affine quantize overwrites a
            // slot's packed bits directly; AURA zeros a re-written slot's
            // stale `atomic_or` bits via its high-water mark. So no buffer
            // copy is needed — `length` + `absolutePosition` is the whole
            // snapshot. (Rotating-window eviction is the one case this
            // length-only rewind can't undo — see plan.md Phase 8.25.)
            if let kv = cache as? (any KVCacheProtocol) {
                return .kv(
                    length: kv.length,
                    absolutePosition: kv.absolutePosition)
            }
            preconditionFailure(
                "CacheSnapshot: unhandled LayerCacheProtocol subtype \(type(of: cache)). Add a case if a new cache kind ships."
            )
        }
    }

    /// Restore every layer cache from a snapshot taken via `snapshotAll`.
    /// Caller must pass the same `[any LayerCacheProtocol]` instance
    /// the snapshot came from — order + identity must match.
    public func restoreAll(
        from snapshots: [LayerCacheSnapshot],
        device: Device = .shared
    ) {
        precondition(
            self.count == snapshots.count,
            "CacheSnapshot.restoreAll: layer count mismatch (caches=\(self.count), snapshots=\(snapshots.count))"
        )
        for (cache, snap) in zip(self, snapshots) {
            switch (cache, snap) {
            case let (composite as GDNLayerCache, .gdnLayer(convT, gdnT, length)):
                composite.conv.restore(from: convT, device: device)
                composite.gdn.restore(from: gdnT, device: device)
                composite.setLength(length)
            case let (gdn as GDNStateCache, .gdn(currentT, length)):
                gdn.restore(from: currentT, device: device)
                gdn.setLength(length)
            case let (kv as any KVCacheProtocol, .kv(length, _)):
                kv.truncate(toLength: length)
            default:
                preconditionFailure(
                    "CacheSnapshot.restoreAll: cache/snapshot mismatch at layer — cache=\(type(of: cache)), snapshot=\(snap)"
                )
            }
        }
    }
}
