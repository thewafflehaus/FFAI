// Device — singleton wrapper over the system MTLDevice + a default
// command queue. Exposes a single Sendable handle that all of FFAI uses
// to allocate buffers and submit work.

import Foundation
import Metal
import MetalTileSwift

public final class Device: @unchecked Sendable {
    public let mtlDevice: MTLDevice
    public let commandQueue: MTLCommandQueue

    /// Lazy MTLResidencySet holding the model's weight buffers. Marked
    /// resident after `Model.load` finishes; saves per-command-buffer
    /// residency tracking on every prefill / decode dispatch. Disable
    /// via `FFAI_NO_RESIDENCY_SET=1`. Stored as `Any?` so the deployment
    /// target stays < macOS 15; the actual cast lives inside the
    /// `@available` block below.
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

    /// Mark `buffers` as permanently resident so every command buffer
    /// the queue runs skips per-allocation residency tracking. Apple's
    /// Metal driver otherwise re-validates a buffer's residency state at
    /// each encode (the cost shows as host gap when there are tens of
    /// thousands of small dispatches per prefill). One residency set
    /// is shared across all weight buffers; subsequent calls add to it.
    /// Requires macOS 15+ / iOS 18+. Opt-out via `FFAI_NO_RESIDENCY_SET=1`.
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
                // Driver couldn't create the set — fall back silently.
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
