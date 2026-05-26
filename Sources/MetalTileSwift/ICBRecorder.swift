// ICBRecorder — high-level helper for recording sequences of FFAI
// kernel dispatches into an MTLIndirectCommandBuffer.
//
// Use case: collapse the per-token CPU dispatch overhead that
// dominates decode T=1 on Qwen3.6-A3B (~48 ms of 60 ms total for ~600
// dispatches). Recording the forward path once + replaying per token
// via executeCommandsInBuffer projects decode tps from 16.8 → 40-80.
//
// Architecture:
//
//   ICBRecorder owns:
//     * the MTLIndirectCommandBuffer (capacity = max recorded commands)
//     * a paramsBuffer holding packed scalar args for every recorded
//       command (each generated `<kernel>_record` wrapper writes its
//       scalars at a caller-allocated cursor offset)
//     * the set of MTLResource references used by recorded commands
//       (so executeCommandsInBuffer's encoder can call useResource on
//       each — Metal can't see through ICB bindings for residency
//       tracking)
//
//   Lifecycle:
//     1. Allocate: `let rec = ICBRecorder(device: …, maxCommands: 1000,
//        paramsBytes: paramsBudget)`
//     2. Record dispatches: for each, call `rec.next()` to get the
//        (command, paramsOffset) pair, pass them to the generated
//        `mt_<kernel>_record(...)` wrapper, register each touched
//        MTLBuffer via `rec.use(resource:, usage:)`.
//     3. Execute: `rec.execute(on: commandBuffer)` — internally creates
//        a compute encoder, calls useResource for every registered
//        buffer, then executeCommandsInBuffer.
//     4. Per-token replay loop: mutate paramsBuffer / input buffer
//        contents in place for the per-token-varying scalars, call
//        `rec.execute(on: nextCommandBuffer)` again. The ICB itself is
//        reused — only buffer CONTENTS change.

import Foundation
import Metal

/// Recorder for ICB-based dispatch graphs. Thin wrapper over Apple's
/// `MTLIndirectCommandBuffer` API plus a paramsBuffer cursor for the
/// scalar packing every generated `_record` wrapper expects.
public final class ICBRecorder {
    public let device: MTLDevice
    public let icb: MTLIndirectCommandBuffer
    public let paramsBuffer: MTLBuffer
    public let maxCommands: Int

    private var nextIndex: Int = 0
    private var paramsCursor: Int = 0
    // Track every resource referenced by recorded commands so
    // executeCommandsInBuffer's enclosing encoder can call useResource
    // on each — Metal cannot infer this from ICB bindings.
    private var usedResources: [(MTLResource, MTLResourceUsage)] = []
    // Group boundaries: command indices where the recorder splits
    // execution into dependency groups separated by buffer-memory
    // barriers. `MTLIndirectCommandBuffer.concurrentDispatch` makes
    // commands within a single `executeCommandsInBuffer` call run in
    // parallel, so dependent producer→consumer chains MUST be split
    // across boundary calls with `enc.memoryBarrier(scope: .buffers)`
    // between them. Empty = single concurrent batch (no barriers).
    private var groupBoundaries: [Int] = []

    public init(device: MTLDevice,
                maxCommands: Int,
                paramsBytes: Int,
                maxKernelBufferBindCount: Int = 16) {
        self.device = device
        self.maxCommands = maxCommands

        let icbDescriptor = MTLIndirectCommandBufferDescriptor()
        icbDescriptor.commandTypes = .concurrentDispatch
        // ICB has its own resource bindings — not inherited from the
        // enclosing encoder. We bake all bindings at recording time.
        icbDescriptor.inheritBuffers = false
        icbDescriptor.inheritPipelineState = false
        icbDescriptor.maxKernelBufferBindCount = maxKernelBufferBindCount
        guard let icb = device.makeIndirectCommandBuffer(
            descriptor: icbDescriptor, maxCommandCount: maxCommands,
            options: .storageModeShared) else {
            preconditionFailure("ICBRecorder: failed to allocate MTLIndirectCommandBuffer (maxCommands=\(maxCommands))")
        }
        self.icb = icb

        // Shared-storage paramsBuffer so the host can mutate scalars
        // per token without round-tripping through .makeBuffer.
        guard let params = device.makeBuffer(length: max(paramsBytes, 16),
                                             options: .storageModeShared) else {
            preconditionFailure("ICBRecorder: failed to allocate paramsBuffer (bytes=\(paramsBytes))")
        }
        self.paramsBuffer = params
    }

    /// Reserve the next command slot. Returns:
    ///   - `command`: the `MTLIndirectComputeCommand` for the caller to
    ///     pass into a generated `_record` wrapper.
    ///   - `paramsOffset`: where the wrapper should pack its scalars
    ///     inside the recorder's paramsBuffer.
    ///
    /// Caller must immediately consume the returned slot via a
    /// `<kernel>_record(...)` call before calling `next` again.
    public func next(paramsSize: Int) -> (command: MTLIndirectComputeCommand, paramsOffset: Int) {
        precondition(nextIndex < maxCommands,
                     "ICBRecorder: command capacity exceeded (max=\(maxCommands)). Increase maxCommands at init.")
        precondition(paramsCursor + paramsSize <= paramsBuffer.length,
                     "ICBRecorder: paramsBuffer exhausted (have=\(paramsBuffer.length), need=\(paramsCursor + paramsSize)). Increase paramsBytes at init.")
        let cmd = icb.indirectComputeCommandAt(nextIndex)
        let off = paramsCursor
        nextIndex += 1
        paramsCursor += paramsSize
        return (cmd, off)
    }

    /// Register a resource the recorded ICB will touch. Required for
    /// `executeCommandsInBuffer` residency tracking — Metal cannot see
    /// through ICB bindings, so the enclosing compute encoder must
    /// `useResource` each one before executing the ICB.
    public func use(_ resource: MTLResource, usage: MTLResourceUsage) {
        usedResources.append((resource, usage))
    }

    /// Mark a dependency boundary at the CURRENT recorded command
    /// position. `execute(on:)` will split the ICB into sub-ranges at
    /// these boundaries and call `memoryBarrier(scope: .buffers)`
    /// between them. Use AFTER recording a producer dispatch (or a
    /// group of independent producers) whose output the next group
    /// depends on. Boundaries with zero commands between them are
    /// silently coalesced.
    public func groupBoundary() {
        if let last = groupBoundaries.last, last == nextIndex { return }
        groupBoundaries.append(nextIndex)
    }

    /// Execute the recorded commands on `commandBuffer`. Creates a
    /// compute encoder, calls useResource for every registered
    /// resource (plus paramsBuffer), executes the ICB, ends the
    /// encoder. Does NOT commit the command buffer — caller decides
    /// when to commit.
    ///
    /// If `groupBoundary()` has been called between dependent
    /// dispatches, the encoder issues `memoryBarrier(scope: .buffers)`
    /// between each `executeCommandsInBuffer` sub-range. Without group
    /// boundaries the entire ICB runs as one concurrent batch — only
    /// safe when no command reads another's output.
    public func execute(on commandBuffer: MTLCommandBuffer) {
        guard nextIndex > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            preconditionFailure("ICBRecorder.execute: makeComputeCommandEncoder failed")
        }
        enc.useResource(paramsBuffer, usage: .read)
        for (res, usage) in usedResources {
            enc.useResource(res, usage: usage)
        }
        if groupBoundaries.isEmpty {
            enc.executeCommandsInBuffer(icb, range: 0..<nextIndex)
        } else {
            var prev = 0
            for b in groupBoundaries where b > prev {
                enc.executeCommandsInBuffer(icb, range: prev..<b)
                enc.memoryBarrier(scope: .buffers)
                prev = b
            }
            if prev < nextIndex {
                enc.executeCommandsInBuffer(icb, range: prev..<nextIndex)
            }
        }
        enc.endEncoding()
    }

    /// Number of commands recorded so far.
    public var recordedCount: Int { nextIndex }

    /// Bytes consumed in paramsBuffer.
    public var paramsBytesUsed: Int { paramsCursor }
}
