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
    /// Fraction of total physical unified memory a manual
    /// `wiredLimitBytes` override may not exceed. Even when the caller
    /// asks to "raise the ticket," the guard never lets the working set
    /// request more than this share of the machine — the remainder is
    /// left for the OS, other processes, and non-GPU allocations.
    public static let hardMachineFraction = 0.92

    /// Working-memory margin reserved on top of weights + KV for
    /// per-token scratch tensors, activations, and command-buffer
    /// overhead. Taken as a fraction of the budget.
    public static let workingMemoryFraction = 0.05

    /// The wired-memory budget (bytes) the guard works against. Uses the
    /// caller's `wiredLimitBytes` override when set (clamped to
    /// `hardMachineFraction` of physical RAM), otherwise the device's
    /// `recommendedMaxWorkingSetSize` (Apple's ~75% ticket).
    public static func budgetBytes(
        options: LoadOptions, device: Device = .shared
    ) -> Int {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        let hardCeiling = Int(Double(physical) * hardMachineFraction)
        if let override = options.wiredLimitBytes {
            return Swift.max(1, Swift.min(override, hardCeiling))
        }
        let recommended = Int(device.mtlDevice.recommendedMaxWorkingSetSize)
        // `recommendedMaxWorkingSetSize` can read as 0 on some
        // configurations; fall back to the hard machine ceiling so the
        // guard still has a sane budget to work against.
        return recommended > 0 ? recommended : hardCeiling
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
