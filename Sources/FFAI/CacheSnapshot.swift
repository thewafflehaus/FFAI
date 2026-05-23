// CacheSnapshot — composite snapshot/restore over `[any LayerCacheProtocol]`.
//
// Speculative decode needs to roll back ALL of a model's per-layer
// caches on draft rejection: KVCache (attention), GDNStateCache +
// ConvStateCache (GDN). This file gives the spec-decode driver one
// call to snapshot every layer + one call to restore.
//
// Per-cache-kind snapshot payload:
//   * KVCache              → `length` integer only (the KV slots
//                            beyond `length` are unused; appending
//                            after restore overwrites them).
//   * GDNStateCache        → Tensor snapshot of `current` +
//                            `length` integer.
//   * ConvStateCache       → Tensor snapshot of `state`.
//   * Qwen35GDNLayerCache  → both of the above (conv + GDN +
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

public extension Array where Element == any LayerCacheProtocol {
    /// Snapshot every layer cache. Cost is ~150 µs aggregate blit time
    /// on Qwen3.6-A3B (60 MiB GDN + 900 KB conv).
    func snapshotAll(device: Device = .shared) -> [LayerCacheSnapshot] {
        return self.map { cache -> LayerCacheSnapshot in
            if let composite = cache as? Qwen35GDNLayerCache {
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
            if let conv = cache as? ConvStateCache {
                return .conv(state: conv.snapshot(device: device))
            }
            if let kv = cache as? KVCache {
                return .kv(length: kv.length, absolutePosition: kv.absolutePosition)
            }
            preconditionFailure(
                "CacheSnapshot: unhandled LayerCacheProtocol subtype \(type(of: cache)). Add a case if a new cache kind ships."
            )
        }
    }

    /// Restore every layer cache from a snapshot taken via `snapshotAll`.
    /// Caller must pass the same `[any LayerCacheProtocol]` instance
    /// the snapshot came from — order + identity must match.
    func restoreAll(from snapshots: [LayerCacheSnapshot],
                    device: Device = .shared) {
        precondition(self.count == snapshots.count,
                     "CacheSnapshot.restoreAll: layer count mismatch (caches=\(self.count), snapshots=\(snapshots.count))")
        for (cache, snap) in zip(self, snapshots) {
            switch (cache, snap) {
            case let (composite as Qwen35GDNLayerCache, .gdnLayer(convT, gdnT, length)):
                composite.conv.restore(from: convT, device: device)
                composite.gdn.restore(from: gdnT, device: device)
                // Truncate `length` back to the snapshotted value.
                // Qwen35GDNLayerCache only exposes `advance()`; rewind
                // by re-creating the counter via internal state — see
                // the type's `restoreLength` accessor below.
                composite.restoreLength(to: length)
            case let (gdn as GDNStateCache, .gdn(currentT, length)):
                gdn.restore(from: currentT, device: device)
                gdn.restoreLength(to: length)
            case let (conv as ConvStateCache, .conv(stateT)):
                conv.restore(from: stateT, device: device)
            case let (kv as KVCache, .kv(length, _)):
                kv.truncate(toLength: length)
            default:
                preconditionFailure(
                    "CacheSnapshot.restoreAll: cache/snapshot mismatch at layer — cache=\(type(of: cache)), snapshot=\(snap)"
                )
            }
        }
    }
}

// `length` rewind helpers — `swap()` / `advance()` only move forward.
// Spec decode needs to move backward on reject.

public extension GDNStateCache {
    func restoreLength(to length: Int) {
        precondition(length >= 0, "GDNStateCache.restoreLength: length must be ≥ 0")
        // `length` is `public private(set)` — go through the snapshot's
        // value by resetting and re-incrementing. Cheap (counter only;
        // state buffers already restored separately).
        self.reset()
        for _ in 0..<length { self.swap() }
    }
}

public extension Qwen35GDNLayerCache {
    func restoreLength(to length: Int) {
        precondition(length >= 0, "Qwen35GDNLayerCache.restoreLength: length must be ≥ 0")
        // Mirror GDN's reset-then-advance pattern at the composite
        // level. The cache's `length` is bumped via `advance()` on
        // each token; rewinding by re-running advance from a known
        // baseline is the cleanest path without changing the
        // public API surface.
        self.reset()
        for _ in 0..<length { self.advance() }
    }
}
