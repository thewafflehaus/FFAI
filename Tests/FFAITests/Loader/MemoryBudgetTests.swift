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
// MemoryBudgetTests — the over-allocation guard arithmetic. Exercises
// the primitive-input core (KV-per-token, max-fitting-context, the
// working-margin reservation) + the wired-limit budget resolution
// (override clamping to the hard machine fraction) without needing a
// live model.

import Foundation
import Testing

@testable import FFAI

@Suite("MemoryBudget")
struct MemoryBudgetTests {

    // ─── KV-per-token arithmetic ─────────────────────────────────────

    @Test("kvBytesPerContextToken = nLayers × nKVHeads × headDim × 2 × bytes")
    func kvPerToken() {
        // 32 layers, 8 KV heads, headDim 128, bf16 (2 bytes).
        // 32 × 8 × 128 × 2 × 2 = 131072 bytes/token.
        let v = MemoryBudget.kvBytesPerContextToken(
            nLayers: 32, nKVHeads: 8, headDim: 128, bytesPerElement: 2)
        #expect(v == 32 * 8 * 128 * 2 * 2)
        #expect(v == 131_072)
    }

    @Test("kvBytesPerContextToken scales with each factor")
    func kvPerTokenScaling() {
        let base = MemoryBudget.kvBytesPerContextToken(
            nLayers: 10, nKVHeads: 4, headDim: 64, bytesPerElement: 2)
        let doubleLayers = MemoryBudget.kvBytesPerContextToken(
            nLayers: 20, nKVHeads: 4, headDim: 64, bytesPerElement: 2)
        let fp32 = MemoryBudget.kvBytesPerContextToken(
            nLayers: 10, nKVHeads: 4, headDim: 64, bytesPerElement: 4)
        #expect(doubleLayers == base * 2)
        #expect(fp32 == base * 2)
    }

    // ─── maxFittingContext — fits / clamps / refuses ─────────────────

    @Test("maxFittingContext returns the budget-limited token count")
    func maxFittingBasic() {
        // budget 1000, 5% margin → usable 950. weights 150 → 800 for KV.
        // kvPerToken 8 → 800 / 8 = 100 tokens.
        let n = MemoryBudget.maxFittingContext(
            budget: 1000, weightBytes: 150, kvBytesPerToken: 8)
        #expect(n == 100)
    }

    @Test("maxFittingContext returns 0 when weights alone exhaust the budget")
    func maxFittingWeightsTooBig() {
        // usable = 950; weights 960 > usable → 0 (can't fit even 1 token).
        let n = MemoryBudget.maxFittingContext(
            budget: 1000, weightBytes: 960, kvBytesPerToken: 8)
        #expect(n == 0)
    }

    @Test("maxFittingContext returns 0 when weights + one KV token won't fit")
    func maxFittingOneTokenOverBudget() {
        // usable 950, weights 945 → 5 left, but kvPerToken 8 > 5 → 0.
        let n = MemoryBudget.maxFittingContext(
            budget: 1000, weightBytes: 945, kvBytesPerToken: 8)
        #expect(n == 0)
    }

    @Test("maxFittingContext is unbounded when there are no attention layers")
    func maxFittingNoKV() {
        // A pure-SSM model with kvBytesPerToken == 0 isn't context-bound
        // by attention KV — return Int.max (the cap comes from elsewhere).
        let n = MemoryBudget.maxFittingContext(
            budget: 1000, weightBytes: 100, kvBytesPerToken: 0)
        #expect(n == Int.max)
    }

    @Test("working-memory margin is reserved before KV")
    func workingMarginReserved() {
        // budget 1_000_000, margin 5% → usable 950_000. weights 0.
        // kvPerToken 1000 → 950 tokens (not 1000 — the 5% is withheld).
        let n = MemoryBudget.maxFittingContext(
            budget: 1_000_000, weightBytes: 0, kvBytesPerToken: 1000)
        #expect(n == 950)
    }

    // ─── Budget resolution — OS reserve + wiredLimitBytes override ───

    @Test("safeMaxBudget leaves the OS reserve free on a normal machine")
    func safeMaxLeavesOSReserve() {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        let safeMax = MemoryBudget.safeMaxBudget()
        // FFAI targets Apple Silicon ≥ 16 GB; the reserve applies.
        if physical > MemoryBudget.osReserveBytes {
            #expect(safeMax == physical - MemoryBudget.osReserveBytes)
            // At least 8 GB stays free for the OS.
            #expect(physical - safeMax == MemoryBudget.osReserveBytes)
        } else {
            #expect(safeMax == physical / 2)
        }
    }

    @Test("wiredLimitBytes override is honored when within the safe maximum")
    func budgetOverrideHonored() {
        // 1 GB is well under (physical − 8 GB) on any supported machine.
        let oneGB = 1_073_741_824
        var opts = LoadOptions()
        opts.wiredLimitBytes = oneGB
        #expect(MemoryBudget.budgetBytes(options: opts) == oneGB)
    }

    @Test("budgetBytes clamps an over-high override to the safe maximum")
    func budgetOverrideClampedToSafeMax() {
        let safeMax = MemoryBudget.safeMaxBudget()
        var opts = LoadOptions()
        opts.wiredLimitBytes = Int(ProcessInfo.processInfo.physicalMemory) * 10
        #expect(MemoryBudget.budgetBytes(options: opts) == safeMax)
    }

    @Test("default budget (no override) is positive and leaves the OS reserve")
    func budgetDefaultSane() {
        let safeMax = MemoryBudget.safeMaxBudget()
        let budget = MemoryBudget.budgetBytes(options: LoadOptions())
        #expect(budget > 0)
        #expect(budget <= safeMax)
    }

    // ─── validateWiredLimit — explicit over-request throws ───────────

    @Test("validateWiredLimit throws when an explicit override starves the OS")
    func validateWiredLimitThrows() {
        var opts = LoadOptions()
        opts.wiredLimitBytes = Int(ProcessInfo.processInfo.physicalMemory) * 2
        #expect(throws: ModelError.self) {
            try MemoryBudget.validateWiredLimit(options: opts)
        }
    }

    @Test("validateWiredLimit accepts an override within the safe maximum")
    func validateWiredLimitAccepts() throws {
        var opts = LoadOptions()
        opts.wiredLimitBytes = 1_073_741_824  // 1 GB — safe everywhere
        try MemoryBudget.validateWiredLimit(options: opts)  // must not throw
    }

    @Test("validateWiredLimit is a no-op when no override is set")
    func validateWiredLimitNoOp() throws {
        try MemoryBudget.validateWiredLimit(options: LoadOptions())  // must not throw
    }

    // ─── End-to-end clamp scenarios via the budget number ────────────

    @Test("a fitting request passes through unclamped")
    func requestFits() {
        // usable 950, weights 150 → 800 for KV, kvPerToken 8 → 100 max.
        // Requesting 50 (≤ 100) returns 50 unchanged.
        let fitting = MemoryBudget.maxFittingContext(
            budget: 1000, weightBytes: 150, kvBytesPerToken: 8)
        #expect(min(50, fitting) == 50)
    }

    @Test("an over-budget request is clamped to the fitting count")
    func requestClamped() {
        let fitting = MemoryBudget.maxFittingContext(
            budget: 1000, weightBytes: 150, kvBytesPerToken: 8)
        // Requesting 100_000 clamps to the 100-token fit.
        #expect(min(100_000, fitting) == 100)
    }
}
