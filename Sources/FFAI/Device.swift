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
// Device — singleton wrapper over the system MTLDevice + a default
// command queue. Exposes a single Sendable handle that all of FFAI uses
// to allocate buffers and submit work.

import Foundation
import Metal
import MetalTileSwift

public final class Device: @unchecked Sendable {
    public let mtlDevice: MTLDevice
    public let commandQueue: MTLCommandQueue

    /// Lazy MTLResidencySet that pins the model's weight buffers so
    /// every command buffer skips per-allocation residency tracking.
    /// Populated after `Model.load` finishes. Typed as `Any?` so the
    /// deployment target stays below macOS 15; the cast back to
    /// `MTLResidencySet` lives inside the `@available` block in
    /// `markWeightsResident`. Initialised under `residencyLock` to
    /// single-flight the descriptor build.
    private var weightResidencySet: Any?
    private let residencyLock = NSLock()

    public static let shared: Device = {
        // Reuse the same MTLDevice + queue MetalTileSwift uses, so PSOs
        // and buffers are guaranteed compatible.
        let lib = MetalTileLibrary.shared
        return Device(mtlDevice: lib.device, commandQueue: lib.commandQueue)
    }()

    public init(mtlDevice: MTLDevice, commandQueue: MTLCommandQueue) {
        self.mtlDevice = mtlDevice
        self.commandQueue = commandQueue
    }

    /// Allocate a fresh shared-storage MTLBuffer of the given byte length.
    public func makeBuffer(length: Int) -> MTLBuffer {
        guard let buf = mtlDevice.makeBuffer(length: length, options: .storageModeShared) else {
            fatalError("Device.makeBuffer(length: \(length)) returned nil")
        }
        return buf
    }

    /// Make a new MTLCommandBuffer.
    public func makeCommandBuffer() -> MTLCommandBuffer {
        guard let cb = commandQueue.makeCommandBuffer() else {
            fatalError("Device.makeCommandBuffer() returned nil")
        }
        return cb
    }

    /// Add `buffers` to a persistent MTLResidencySet attached to the
    /// command queue. Without this, Apple's Metal driver re-validates
    /// per-allocation residency on every command-buffer encode — at
    /// model sizes with tens of thousands of dispatches per prefill,
    /// the per-dispatch overhead dominates wall time. One residency
    /// set is shared across all weight buffers; repeated calls add
    /// to it. Requires macOS 15+ / iOS 18+; older OSes silently
    /// no-op. Set `FFAI_NO_RESIDENCY_SET=1` to disable for A/B.
    public func markWeightsResident(_ buffers: [MTLBuffer]) {
        if ProcessInfo.processInfo.environment["FFAI_NO_RESIDENCY_SET"] != nil { return }
        guard #available(macOS 15.0, iOS 18.0, *) else { return }
        residencyLock.lock()
        defer { residencyLock.unlock() }
        if weightResidencySet == nil {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "FFAI weights"
            descriptor.initialCapacity = max(buffers.count, 1024)
            do {
                let set = try mtlDevice.makeResidencySet(descriptor: descriptor)
                commandQueue.addResidencySet(set)
                weightResidencySet = set
            } catch {
                // Driver refused to create the set; fall back to default
                // residency tracking. Not fatal — just slower.
                weightResidencySet = nil
                return
            }
        }
        guard let set = weightResidencySet as? MTLResidencySet else { return }
        for buf in buffers {
            set.addAllocation(buf)
        }
        set.commit()
        set.requestResidency()
    }
}
