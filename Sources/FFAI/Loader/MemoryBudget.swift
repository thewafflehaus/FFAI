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
// MemoryBudget — the over-allocation guard for KV-cache sizing.
//
// Apple Silicon defaults the GPU's wired-memory "ticket"
// (`recommendedMaxWorkingSetSize`) to ~75% of total unified memory.
// Allocating past it doesn't hard-fail — the OS just starts paging,
// which tanks decode throughput and, at the extreme (e.g. a 256K-context
// KV cache on a 27B model), can wedge the machine. This guard computes a
// model's worst-case footprint at a requested context and clamps the
// context ceiling so weights + max-KV + a working-memory margin stay
// within the budget. The budget itself is `recommendedMaxWorkingSetSize`
// by default, or a caller-supplied `wiredLimitBytes` (the "raise the
// ticket" override) clamped to a hard fraction of physical RAM so a load
// can never request more than the box can physically back.
//
// The KV estimate is deliberately CONSERVATIVE: it assumes every layer
// is a full attention layer holding a raw fp16/bf16 K+V cache. Hybrid
// (SSM/GDN) models have fewer attention layers, and quantized / AURA KV
// caches store less — both make the real footprint SMALLER than this
// estimate, so the guard errs toward clamping context slightly early
// rather than under-budgeting and paging. Better a few hundred fewer
// tokens of headroom than a wedged machine.

import Foundation
import Metal

public enum MemoryBudget {
    /// Bytes of physical RAM always left for the OS (and other
    /// processes) — the GPU working set may never claim memory that
    /// would push the free pool below this. A *fixed* reserve, not a
    /// fraction: a percentage starves small machines (92% of 16 GB
    /// leaves the OS only 1.3 GB) while over-reserving on large ones.
    /// 8 GB keeps WindowServer, the file cache, and background daemons
    /// healthy across the Apple-Silicon range. Tunable for future
    /// per-hardware refinement.
    public static let osReserveBytes = 8 * 1_073_741_824  // 8 GiB

    /// Working-memory margin reserved on top of weights + KV for
    /// per-token scratch tensors, activations, and command-buffer
    /// overhead. Taken as a fraction of the budget.
    public static let workingMemoryFraction = 0.05

    /// The largest wired-memory budget that still leaves `osReserveBytes`
    /// free for the OS. On machines smaller than the reserve (≤ 8 GB,
    /// where a full reserve would leave nothing) it degrades to half of
    /// physical RAM so the box can still attempt to run something.
    public static func safeMaxBudget(device _: Device = .shared) -> Int {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        return physical > osReserveBytes ? physical - osReserveBytes : physical / 2
    }

    /// The wired-memory budget (bytes) the guard works against. Uses the
    /// caller's `wiredLimitBytes` override when set, otherwise the
    /// device's `recommendedMaxWorkingSetSize` (Apple's ~75% ticket).
    /// Either way the result is clamped to `safeMaxBudget` so the OS
    /// reserve is never violated. (An *explicit* over-request is also
    /// rejected up front by `validateWiredLimit` at load, so a user who
    /// sets too high a value is told rather than silently clamped.)
    public static func budgetBytes(
        options: LoadOptions, device: Device = .shared
    ) -> Int {
        let safeMax = safeMaxBudget(device: device)
        if let override = options.wiredLimitBytes {
            return Swift.max(1, Swift.min(override, safeMax))
        }
        let recommended = Int(device.mtlDevice.recommendedMaxWorkingSetSize)
        // `recommendedMaxWorkingSetSize` can read as 0 on some
        // configurations; fall back to safeMax so the guard still has a
        // sane budget. Clamp the recommendation too — on small machines
        // Apple's ~75% ticket can exceed the OS reserve.
        let base = recommended > 0 ? recommended : safeMax
        return Swift.min(base, safeMax)
    }

    /// Reject an explicit `wiredLimitBytes` that would starve the OS.
    /// Called at load so a caller who deliberately raised the ticket
    /// past the safe maximum gets an actionable error instead of a
    /// silent clamp. No-op when no override is set.
    public static func validateWiredLimit(
        options: LoadOptions, device: Device = .shared
    ) throws {
        guard let override = options.wiredLimitBytes else { return }
        let safeMax = safeMaxBudget(device: device)
        if override > safeMax {
            throw ModelError.wiredLimitTooHigh(
                requestedBytes: override, safeMaxBytes: safeMax,
                osReserveBytes: osReserveBytes)
        }
    }

    /// Total bytes of the model's resident weights (sum of every
    /// parameter tensor's byte count — packed u32 for quantized).
    public static func weightBytes(engine: any LanguageModel) -> Int {
        engine.parameters().reduce(0) { $0 + $1.1.byteCount }
    }

    /// Conservative per-context-token KV cost (bytes) for one decode
    /// position across the whole model: assumes EVERY layer is a raw
    /// fp16/bf16 attention layer holding K + V. Real footprints (hybrid
    /// models, quantized / AURA caches) are smaller.
    public static func kvBytesPerContextToken(engine: any LanguageModel) -> Int {
        kvBytesPerContextToken(
            nLayers: engine.nLayers, nKVHeads: engine.nKVHeads,
            headDim: engine.headDim, bytesPerElement: engine.dtype.byteSize)
    }

    /// Primitive-input KV-per-token cost — `nLayers × nKVHeads × headDim
    /// × 2 (K+V) × bytesPerElement`. Exposed for testing the arithmetic
    /// without a live engine.
    public static func kvBytesPerContextToken(
        nLayers: Int, nKVHeads: Int, headDim: Int, bytesPerElement: Int
    ) -> Int {
        nLayers * nKVHeads * headDim * 2 * bytesPerElement
    }

    /// Largest context (tokens) whose worst-case footprint —
    /// weights + KV(context) + working margin — fits `budget`. Returns 0
    /// when weights + margin already exhaust the budget. Primitive-input
    /// core; the engine overload below delegates here.
    public static func maxFittingContext(
        budget: Int, weightBytes: Int, kvBytesPerToken: Int
    ) -> Int {
        let usable = Int(Double(budget) * (1.0 - workingMemoryFraction))
        guard kvBytesPerToken > 0 else { return Int.max }
        let availableForKV = usable - weightBytes
        guard availableForKV >= kvBytesPerToken else { return 0 }
        return availableForKV / kvBytesPerToken
    }

    /// Largest context (in tokens) whose worst-case footprint —
    /// weights + KV(context) + working margin — fits the budget. Returns
    /// 0 when weights + margin already exhaust the budget (the model
    /// can't run at any context on this machine / budget).
    public static func maxFittingContext(
        engine: any LanguageModel, options: LoadOptions, device: Device = .shared
    ) -> Int {
        maxFittingContext(
            budget: budgetBytes(options: options, device: device),
            weightBytes: weightBytes(engine: engine),
            kvBytesPerToken: kvBytesPerContextToken(engine: engine))
    }

    /// Clamp a requested context ceiling so the worst-case footprint fits
    /// the budget. Throws `ModelError.insufficientMemory` when the model's
    /// weights + a minimal KV cache already exceed the budget (it can't
    /// run at any usable context here).
    public static func clampContext(
        requestedCeiling: Int,
        engine: any LanguageModel,
        options: LoadOptions,
        device: Device = .shared
    ) throws -> Int {
        let fitting = maxFittingContext(engine: engine, options: options, device: device)
        if fitting <= 0 {
            let weights = weightBytes(engine: engine)
            let budget = budgetBytes(options: options, device: device)
            throw ModelError.insufficientMemory(
                weightBytes: weights, budgetBytes: budget,
                detail:
                    "model weights (\(formatGB(weights))) + working margin exceed the "
                    + "wired-memory budget (\(formatGB(budget))). Free memory, quantize "
                    + "the model further, or raise the budget via "
                    + "LoadOptions.wiredLimitBytes.")
        }
        return Swift.min(requestedCeiling, fitting)
    }

    static func formatGB(_ bytes: Int) -> String {
        String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
    }
}
