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
// InspectTap — shared per-op intermediate-value dump helper that
// every model's `forward(...)` uses for first-light debugging.
//
// Replaces bespoke per-model env knobs (the original
// `GEMMA3_DEBUG_TAPS=1` pattern that found the Gemma 3 GELU NaN)
// with a single uniform surface:
//
//   FFAI_INSPECT=1                 — turn on dumps
//   FFAI_INSPECT_LAYERS=0,1,5      — optional layer filter
//
// Every new model implementation calls `tap.dumpLayerBoundary(...)`
// at the layer-input + layer-output boundaries inside
// `<Family>Model.forward(...)`. That's enough granularity to
// localise the failing layer in two `ffai inspect --layer-trace`
// runs (one to find which layer's output goes non-finite, one to
// confirm the fix). For inside-layer triage (which op produced
// the NaN) drop temporary fine-grained calls — see the Gemma 3
// 2026-05-19 post-mortem for the pattern.
//
// Wired into `ffai inspect --layer-trace` so the diagnostic
// surface is reachable from the CLI without setting env vars
// manually. See documentation/using-the-cli.md and
// documentation/developing/adding-a-model.md.

import Foundation
import Metal

/// Toggle + filter for layer-boundary intermediate dumps. Value
/// type — every `forward(...)` captures it from the cached env
/// snapshot once and threads it through the layer loop.
///
/// ## Performance contract
///
/// Production paths (`FFAI_INSPECT` unset, the default) pay
/// **zero** overhead per layer beyond a single bool compare:
///
///   1. `fromEnvironment` returns a process-wide cached snapshot;
///      reading the env happens exactly once per process, on first
///      access. Subsequent `forward()` calls are a static-load.
///   2. `dumpLayerBoundary` is annotated `@inline(__always)` and
///      forwards to a slow-path function only when `active` is
///      true; in the inactive case the optimizer collapses the call
///      to `if false { ... }` and removes it entirely. The `inout
///      cmd` parameter is observed but never written on the inactive
///      branch, so Swift's exclusivity tracking doesn't add cost.
///   3. `makeWorkCmd` returns the caller's `cmd` unchanged in
///      inactive mode (single branch + identity passthrough).
public struct InspectTap: Sendable {
    public let active: Bool
    public let layerFilter: Set<Int>?

    public init(active: Bool, layerFilter: Set<Int>? = nil) {
        self.active = active
        self.layerFilter = layerFilter
    }

    /// Read `FFAI_INSPECT` + `FFAI_INSPECT_LAYERS` from the
    /// environment **once** at first call, then cache the result
    /// for the lifetime of the process. Avoids paying the
    /// `ProcessInfo.processInfo.environment` dict-allocation cost
    /// on every `forward()` invocation.
    ///
    /// Setting these env vars after the cache is filled has no
    /// effect — production callers shouldn't be toggling debug
    /// state mid-process. Use the CLI's `--layer-trace` flag,
    /// which calls `setenv(...)` *before* the model loads.
    private static let cachedFromEnvironment: InspectTap = {
        let env = ProcessInfo.processInfo.environment
        let active = env["FFAI_INSPECT"] == "1"
        let filter: Set<Int>? = env["FFAI_INSPECT_LAYERS"]
            .map { Set($0.split(separator: ",").compactMap { Int($0) }) }
        return InspectTap(active: active, layerFilter: filter)
    }()

    /// Process-wide cached tap state. See `cachedFromEnvironment`
    /// for the lifetime contract.
    public static var fromEnvironment: InspectTap { cachedFromEnvironment }

    /// True when the caller should pay the tap overhead for this
    /// layer. Inlined into the hot path so non-active taps fold
    /// into a single load + compare.
    @inline(__always)
    public func shouldDump(layer: Int) -> Bool {
        guard active else { return false }
        guard let filter = layerFilter else { return true }
        return filter.contains(layer)
    }

    /// Synchronously read a tensor's contents to fp32 and print
    /// shape + min/max/nan/inf/first-4. Commits the supplied
    /// cmdbuf, waits, and returns a fresh cmdbuf the caller should
    /// continue queueing on. When the tap is inactive, returns the
    /// supplied `cmd` unchanged — `workCmd = tap.dumpLayerBoundary(...)`
    /// collapses to `workCmd = workCmd`, which the compiler
    /// optimizes away entirely.
    ///
    /// **Hot-path-cheap.** Annotated `@inline(__always)`; when the
    /// tap is inactive, the call folds to a no-op identity return.
    /// The cmdbuf is passed by value (MTLCommandBuffer is a class —
    /// "by value" means by-reference under the hood), so there's no
    /// `inout` exclusivity slot allocated on the caller's stack
    /// frame.
    ///
    /// `layer` is the layer index for the printout (use `-1` for
    /// outside-layer dumps like the embed or final-norm tap).
    /// `label` describes what the tensor is — keep short for
    /// readability (`"h_in"`, `"layer_out"`, `"logits"`).
    @inline(__always)
    public func dumpLayerBoundary(
        _ t: Tensor, label: String, layer: Int,
        cmd: MTLCommandBuffer, device: Device
    ) -> MTLCommandBuffer {
        if !shouldDump(layer: layer) { return cmd }
        return dumpSlow(t, label: label, layer: layer, cmd: cmd, device: device)
    }

    /// Slow path — only called when a dump is actually requested.
    /// `@inline(never)` to keep the inactive caller's instruction
    /// footprint tiny: formatting + buffer-readback code lives in
    /// this function alone and only runs during a debug session.
    @inline(never)
    private func dumpSlow(
        _ t: Tensor, label: String, layer: Int,
        cmd: MTLCommandBuffer, device: Device
    ) -> MTLCommandBuffer {
        cmd.commit()
        cmd.waitUntilCompleted()

        let n = t.elementCount
        let basePtr = t.buffer.contents().advanced(by: t.offset)
        var floats: [Float] = []
        floats.reserveCapacity(n)
        switch t.dtype {
        case .f32:
            let p = basePtr.bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { floats.append(p[i]) }
        case .f16:
            let p = basePtr.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n { floats.append(halfBitsToFloatForTest(p[i])) }
        case .bf16:
            let p = basePtr.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n { floats.append(bf16BitsToFloatForTest(p[i])) }
        default:
            print("[L\(layer) \(label)] (unsupported dtype \(t.dtype))")
            return device.makeCommandBuffer()
        }

        var nanCount = 0, infCount = 0
        var mn: Float = .infinity, mx: Float = -.infinity
        for v in floats {
            if v.isNaN { nanCount += 1 }
            else if !v.isFinite { infCount += 1 }
            else {
                if v < mn { mn = v }
                if v > mx { mx = v }
            }
        }
        let head = floats.prefix(4)
            .map { String(format: "%.4f", $0) }
            .joined(separator: ", ")
        let mnStr = mn.isFinite ? String(format: "%+.4f", mn) : "—"
        let mxStr = mx.isFinite ? String(format: "%+.4f", mx) : "—"
        let prefix = layer < 0 ? "[\(label)]" : "[L\(layer) \(label)]"
        print("\(prefix) n=\(n) min=\(mnStr) max=\(mxStr) nan=\(nanCount) inf=\(infCount) first=[\(head)]")

        return device.makeCommandBuffer()
    }
}

public extension InspectTap {
    /// Build the cmdbuf that callers should queue work on. When the
    /// tap is active, this is a *private* cmdbuf (separate from the
    /// caller's). The tap commits + waits + replaces it at each
    /// dump; the caller's original `cmd` is never touched, so the
    /// caller's downstream commit / waitUntilCompleted stays a fast
    /// no-op when taps are active.
    ///
    /// When the tap is inactive (production path), returns the
    /// caller's cmd unchanged — single-branch identity passthrough.
    @inline(__always)
    func makeWorkCmd(from callerCmd: MTLCommandBuffer, device: Device) -> MTLCommandBuffer {
        active ? device.makeCommandBuffer() : callerCmd
    }
}
