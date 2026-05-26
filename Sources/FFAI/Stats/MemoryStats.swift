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
// MemoryStats — GPU memory accounting captured at phase boundaries.
//
// On Apple Silicon `MTLDevice.currentAllocatedSize` is the canonical
// "how much GPU memory am I using right now" reading. It's a single
// property access (the framework caches the value behind a fast lock,
// not a per-call kernel syscall) so it's safe to sample at every token
// boundary without inflating the inference pipeline. For sub-token
// peak tracking, --profiling 2's per-op signposts are the right tool.
//
// Phase split (prefill vs decode) is what the user wants from --stats:
// "where is the growth coming from?" Sampling at every token boundary
// inside each phase gives a real per-phase peak rather than a single
// global max — KV cache growth during prefill vs decode shows up as
// separate numbers.

import Foundation
import Metal

/// One snapshot of GPU + process memory at a moment in time.
public struct MemorySnapshot: Sendable, Equatable {
    /// `MTLDevice.currentAllocatedSize` — bytes the Metal framework has
    /// allocated to this process's GPU heap.
    public let gpuBytes: Int
    /// `MTLDevice.recommendedMaxWorkingSetSize` — the wired-memory
    /// "ticket size" the OS will try to keep resident before paging.
    public let wiredTicketBytes: Int
    /// Wall-clock when the snapshot was captured.
    public let timestamp: Date

    public static func capture(device: Device = .shared) -> MemorySnapshot {
        MemorySnapshot(
            gpuBytes: device.mtlDevice.currentAllocatedSize,
            wiredTicketBytes: Int(device.mtlDevice.recommendedMaxWorkingSetSize),
            timestamp: Date()
        )
    }
}

/// Aggregates phase-boundary snapshots + per-token peak samples for one
/// `generate(...)` run. Cheap to construct + sample (one `Int` read per
/// `sample()`); `--stats` always uses one of these.
public final class PhaseMemoryTracker: @unchecked Sendable {
    public enum Phase: Sendable { case prefill, decode }

    public let device: Device
    public let baseline: MemorySnapshot
    public private(set) var prefillPeakBytes: Int
    public private(set) var decodePeakBytes: Int
    public private(set) var postPrefill: MemorySnapshot?
    public private(set) var postDecode: MemorySnapshot?

    private var phase: Phase = .prefill

    public init(device: Device = .shared) {
        self.device = device
        let snap = MemorySnapshot.capture(device: device)
        self.baseline = snap
        // Seed the per-phase peaks with the baseline so a phase that
        // takes zero tokens still reports a sensible value.
        self.prefillPeakBytes = snap.gpuBytes
        self.decodePeakBytes = snap.gpuBytes
    }

    /// One-property GPU-allocated read. Call at each token boundary.
    public func sample() {
        let now = device.mtlDevice.currentAllocatedSize
        switch phase {
        case .prefill: if now > prefillPeakBytes { prefillPeakBytes = now }
        case .decode: if now > decodePeakBytes { decodePeakBytes = now }
        }
    }

    /// Mark the prefill→decode transition. Captures `postPrefill`.
    public func endPrefill() {
        postPrefill = MemorySnapshot.capture(device: device)
        // Carry the latest reading into decode so decodePeak doesn't
        // start below what prefill already grew to.
        if let post = postPrefill, post.gpuBytes > decodePeakBytes {
            decodePeakBytes = post.gpuBytes
        }
        phase = .decode
    }

    /// Mark the end of decode. Captures `postDecode`.
    public func endDecode() {
        postDecode = MemorySnapshot.capture(device: device)
        if let post = postDecode, post.gpuBytes > decodePeakBytes {
            decodePeakBytes = post.gpuBytes
        }
    }

    /// Bytes allocated *during* prefill, attributable to the prompt
    /// (KV cache fill, intermediate activations).
    public var prefillGrowthBytes: Int {
        (postPrefill?.gpuBytes ?? baseline.gpuBytes) - baseline.gpuBytes
    }

    /// Bytes allocated *during* decode, on top of what prefill left.
    public var decodeGrowthBytes: Int {
        (postDecode?.gpuBytes ?? 0) - (postPrefill?.gpuBytes ?? 0)
    }

    /// Max gpuBytes seen across both phases.
    public var peakGPUBytes: Int { max(prefillPeakBytes, decodePeakBytes) }
}
