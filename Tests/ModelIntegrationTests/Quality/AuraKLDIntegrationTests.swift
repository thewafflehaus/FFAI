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
// AURA KL-divergence regression gate. Loads a small local checkpoint
// twice — once with the fp16 baseline KV cache and once with an AURA
// codec — emits per-position logits over a fixed prompt, and reports
// the aggregate KLD vs the baseline + same-top-token rate. Mirrors the
// canonical TQ+ harness output from
// `bench-tq+/harness/kld_vs_baseline.py` (in
// /Users/tom/local_llms/llama.cpp).
//
// This test is the regression gate for every subsequent TQ+ port —
// matched-norm L2 correction, InnerQ equalization, per-group FP8 scale.
// Each port should improve or hold the recorded `same_top_fraction`
// and lower the `mean_kld` vs the baseline AURA scheme.
//
// Currently gated on the local Qwen3-0.6B-4bit checkpoint — the
// smallest AURA-compatible model that runs end-to-end on a dev box.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

private let qwen3LocalPath = "/Users/tom/models/Qwen3-0.6B-4bit"

@Suite("AURA KL-divergence regression gate", .serialized)
struct AuraKLDIntegrationTests {

    /// Fixed sample prompt. Chosen for diversity (en-prose + rare
    /// tokens + a code-like fragment) so the codec's tail behaviour
    /// gets exercised, not just the head of the distribution.
    private static let samplePrompt =
        "The history of the printing press began when European craftsmen "
        + "combined movable metal type with oil-based ink and a wooden screw "
        + "press. The first printed book was the Gutenberg Bible in 1455. "
        + "Compute the next item in this sequence: 2, 4, 8, 16, "

    @Test("model loads + single-step forward smoke")
    func loadSmoke() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("loadSmoke skipped: \(qwen3LocalPath) not found")
            return
        }
        print("loadSmoke: loading...")
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(qwen3LocalPath, options: opts)
        }
        print("loadSmoke: loaded. engine type=\(type(of: m.engine))")
        let caches = m.engine.makeLayerCaches(maxSeq: 256)
        print("loadSmoke: caches=\(caches.count)")
        let logits = m.engine.forward(tokenId: 0, position: 0, caches: caches)
        print("loadSmoke: vocab=\(logits.elementCount)")
        let firstFew = Array(logits.toFloatArray().prefix(5))
        print("loadSmoke: first-5-logits=\(firstFew)")
    }

    @Test("baseline-only smoke — emit logits + measure KLD vs self == 0")
    func baselineSmoke() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("baselineSmoke skipped: \(qwen3LocalPath) not found")
            return
        }
        let trace = try await emitTrace(kvCache: .raw, scheme: nil)
        // Self-KLD should be ~0 across positions (same logits both sides).
        let metrics = LogitsEmitter.compare(
            baseline: trace.logits, codec: trace.logits, tokenIds: trace.tokenIds)
        print(KLDivergence.summaryLine(label: "self vs self", metrics: metrics))
        #expect(
            metrics.meanKld < 1e-6,
            "self-KLD should be ≈0, got \(String(format: "%.6e", metrics.meanKld))")
        #expect(metrics.sameTopFraction == 1.0)
    }

    @Test("AURA aura4v4 vs fp16 baseline — KLD + same-top regression gate")
    func aura4v4VsBaseline() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("auraKLD skipped: \(qwen3LocalPath) not found")
            return
        }
        let metrics = try await runKLDComparison(scheme: .default)
        print(
            KLDivergence.summaryLine(label: "aura4v4 (default)", metrics: metrics))
        // Baseline gate, pinned at the current state of AURA on
        // Qwen3-0.6B-4bit so a regression on subsequent TQ+ ports
        // (matched-norm L2, InnerQ, per-group fp8 scale) gets caught
        // by CI. The measured floor today is mean_kld≈1.24 + same_top
        // ≈47.5%. P1 ports should drive these down/up; tighten the
        // thresholds as each port lands.
        let sameTopStr = String(format: "%.4f", metrics.sameTopFraction)
        let meanKldStr = String(format: "%.4f", metrics.meanKld)
        let sameTopMsg =
            "same_top \(sameTopStr) below current-state floor 0.40 — "
            + "AURA quality regressed below the 2026-05-26 baseline of 0.475"
        let meanKldMsg =
            "mean_kld \(meanKldStr) above current-state ceiling 1.5 — "
            + "AURA quality regressed above the 2026-05-26 baseline of 1.24"
        #expect(metrics.sameTopFraction > 0.40, "\(sameTopMsg)")
        #expect(metrics.meanKld < 1.5, "\(meanKldMsg)")
    }

    @Test("AURA aura8v4 — TQ+ production recipe (high-bit K, aggressive V)")
    func aura8v4() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("aura8v4 skipped: \(qwen3LocalPath) not found")
            return
        }
        let scheme = AURAScheme(keyBits: 8, valueBits: 4)
        let metrics = try await runKLDComparison(scheme: scheme)
        print(KLDivergence.summaryLine(label: "aura8v4 (TQ+ recipe)", metrics: metrics))
    }

    @Test("AURA aura3v3 — mid-bit data point in the curve")
    func aura3v3() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("aura3v3 skipped: \(qwen3LocalPath) not found")
            return
        }
        let scheme = AURAScheme(keyBits: 3, valueBits: 3)
        let metrics = try await runKLDComparison(scheme: scheme)
        print(KLDivergence.summaryLine(label: "aura3v3 (3-bit sym)", metrics: metrics))
    }

    @Test("AURA aura2v2 — low-bit data point in the curve")
    func aura2v2() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("aura2v2 skipped: \(qwen3LocalPath) not found")
            return
        }
        let scheme = AURAScheme(keyBits: 2, valueBits: 2)
        let metrics = try await runKLDComparison(scheme: scheme)
        print(KLDivergence.summaryLine(label: "aura2v2 (2-bit sym)", metrics: metrics))
    }

    @Test("AURA aura8v8 — high-bit sanity check (should be near-baseline)")
    func aura8v8SanityCheck() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("auraKLD aura8v8 skipped: \(qwen3LocalPath) not found")
            return
        }
        let scheme = AURAScheme(keyBits: 8, valueBits: 8)
        let metrics = try await runKLDComparison(scheme: scheme)
        print(
            KLDivergence.summaryLine(
                label: "aura8v8 (high-bit sanity)", metrics: metrics))
        // 8-bit Lloyd-Max codebook should be very near baseline. If
        // this is far from baseline the issue is the pipeline
        // (rotation, encode→decode round-trip, dispatch), not the
        // codebook — guides where to look for the quality gap.
    }

    @Test("AURA aura4v2 (asymmetric 4-bit K / 2-bit V) vs fp16 baseline")
    func aura4v2VsBaseline() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("auraKLD aura4v2 skipped: \(qwen3LocalPath) not found")
            return
        }
        let scheme = AURAScheme(keyBits: 4, valueBits: 2)
        let metrics = try await runKLDComparison(scheme: scheme)
        print(
            KLDivergence.summaryLine(
                label: "aura4v2 (production)", metrics: metrics))
        // aura4v2 is more aggressive — looser floors than aura4v4.
        // Specifically the 2-bit V codebook chops a lot of fidelity;
        // we mostly want to know it doesn't go catastrophic.
        let sameTopStr = String(format: "%.4f", metrics.sameTopFraction)
        let meanKldStr = String(format: "%.4f", metrics.meanKld)
        #expect(
            metrics.sameTopFraction > 0.40,
            "same_top \(sameTopStr) below aggressive-V floor 0.40")
        #expect(
            metrics.meanKld < 3.0,
            "mean_kld \(meanKldStr) above aggressive-V ceiling 3.0")
    }

    // MARK: - Helpers

    /// Load the same checkpoint twice (raw fp16 baseline + AURA codec
    /// with the given scheme), emit per-position logits over the
    /// sample prompt for both, and aggregate the KLD.
    private func runKLDComparison(
        scheme: AURAScheme
    ) async throws -> KLDivergence.AggregateMetrics {
        let baselineTrace = try await emitTrace(kvCache: .raw, scheme: nil)
        let codecTrace = try await emitTrace(
            kvCache: .auraQuantized(scheme: scheme), scheme: scheme)
        let tokenIds = baselineTrace.tokenIds
        return LogitsEmitter.compare(
            baseline: baselineTrace.logits,
            codec: codecTrace.logits,
            tokenIds: tokenIds)
    }

    private struct Trace {
        let tokenIds: [Int]
        let logits: [[Float]]
    }

    /// Load + tokenize + emit logits with one specific KV-cache setting.
    /// Kept as its own helper so the model goes out of scope (along
    /// with its big weight buffers) between the baseline + codec
    /// runs — a 0.6B 4-bit model is small but two side-by-side
    /// instances would still cost ~600 MB.
    private func emitTrace(
        kvCache: KVCacheKind, scheme: AURAScheme?
    ) async throws -> Trace {
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        optsBuilder.kvCache = kvCache
        // dequantMirror = the only AURA path actually wired in main today;
        // the compressed flash kernels exist but the wrapper falls back.
        optsBuilder.auraDecodePath = .dequantMirror
        let opts = optsBuilder
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(qwen3LocalPath, options: opts)
        }
        let tokens = m.tokenizer.encode(text: Self.samplePrompt)
        guard let engine = m.qwen3 else {
            throw NSError(
                domain: "AuraKLDIntegrationTests", code: -1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "expected Qwen3Model engine for \(qwen3LocalPath)"
                ])
        }
        // Cap maxSeq tight against the prompt — Qwen3's default is
        // huge (256K), allocating that for ~64 tokens of bench data
        // wastes seconds + GBs of memory and was the cause of an
        // earlier SIGSEGV on this harness.
        let logits = LogitsEmitter.emit(
            model: engine, tokenIds: tokens, maxSeq: max(tokens.count + 32, 256))
        return Trace(tokenIds: tokens, logits: logits)
    }
}
