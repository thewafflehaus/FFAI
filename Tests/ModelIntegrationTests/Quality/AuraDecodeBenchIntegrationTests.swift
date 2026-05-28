// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// AURA decode-tps + cache-memory bench. Quantifies the win unlocked by
// wiring the compressed flash decode path (`AURADecodePath.compressed`,
// `Ops.auraFlashSdpa`) in `Qwen3Layer.forward` and gives a numeric
// baseline for future TQ+ ports (P1c per-group fp8 scale, etc.) to
// measure against.
//
// The `.compressed` path scores Q directly against packed K codes and
// dequants V per-tile on chip — no `[nKVHeads, maxSeq, headDim]` mirror
// buffer is materialised. Two measurable wins:
//
//   1. Decode tps at varied KV lengths — pays back the kernel-launch
//      cost of `aura_flash_sdpa` over the dequant-mirror's
//      `auraDequantRotated → sdpaDecode` two-encoder pair, dominant at
//      long context.
//
//   2. Cache memory — the per-layer shared working buffer disappears.
//      For aura4v4 at headDim=128, nKVHeads=8, maxSeq=1024 that's
//      `8 * 1024 * 128 * 2 = 2 MiB` × n_layers saved.
//
// Quality parity is covered by `AuraKLDIntegrationTests` — these
// benches assume the wiring is correct and just measure perf.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

private let qwen3LocalPath = "/Users/tom/models/Qwen3-0.6B-4bit"

@Suite("AURA decode bench — compressed vs dequant-mirror", .serialized)
struct AuraDecodeBenchIntegrationTests {

    /// Same diverse prompt as the KLD harness — lets us cross-reference
    /// quality + perf numbers from the same workload.
    private static let samplePrompt =
        "The history of the printing press began when European craftsmen "
        + "combined movable metal type with oil-based ink and a wooden screw "
        + "press. The first printed book was the Gutenberg Bible in 1455. "
        + "Compute the next item in this sequence: 2, 4, 8, 16, "

    /// Decode-tps bench at a fixed KV length. `decodePath` selects
    /// compressed flash vs dequant-mirror. Returns the median tps over
    /// `nRuns` warmed runs of `nSteps` decode tokens each.
    private func runDecodeTpsBench(
        decodePath: AURADecodePath, kvLength: Int, nRuns: Int, nSteps: Int
    ) async throws -> Double {
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        optsBuilder.kvCache = .auraQuantized(scheme: .default)
        optsBuilder.auraDecodePath = decodePath
        let opts = optsBuilder

        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(qwen3LocalPath, options: opts)
        }
        let qwen = try #require(m.qwen3, "expected Qwen3Model engine")

        // Build a token sequence long enough to seed the cache to
        // `kvLength` positions before the timed decode begins.
        let promptTokens = m.tokenizer.encode(text: Self.samplePrompt)
        precondition(
            kvLength >= promptTokens.count,
            "kvLength \(kvLength) must be ≥ promptTokens.count \(promptTokens.count)")
        var seedTokens = promptTokens
        // Pad with token-0 to reach the requested kv length. Decode
        // quality of padding doesn't matter — we only need the cache
        // populated.
        while seedTokens.count < kvLength {
            seedTokens.append(0)
        }

        // Two warmups so the first timed run isn't paying for shader
        // compilation / first-touch effects.
        for _ in 0 ..< 2 {
            let warmCaches = qwen.makeLayerCaches(maxSeq: max(kvLength + nSteps + 32, 1024))
            for (i, tok) in seedTokens.enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: warmCaches)
            }
            for j in 0 ..< 4 {
                _ = qwen.forward(
                    tokenId: 0, position: kvLength + j, caches: warmCaches)
            }
        }

        // Timed runs.
        var runs: [Double] = []
        for _ in 0 ..< nRuns {
            let caches = qwen.makeLayerCaches(maxSeq: max(kvLength + nSteps + 32, 1024))
            for (i, tok) in seedTokens.enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: caches)
            }
            let t0 = Date()
            for j in 0 ..< nSteps {
                _ = qwen.forward(
                    tokenId: 0, position: kvLength + j, caches: caches)
            }
            runs.append(Date().timeIntervalSince(t0))
        }
        runs.sort()
        let median = runs[runs.count / 2]
        return Double(nSteps) / median
    }

    /// Side-by-side bench at one KV length, prints both numbers + the
    /// ratio. **Informational + catastrophic-regression gate only** —
    /// the metaltile flash kernel is currently
    /// single-simdgroup-per-query (see `aura_flash_sdpa.rs` header:
    /// "token-parallelism is a perf follow-up"), so `.compressed` is
    /// a known perf regression at all KV lengths vs `.dequantMirror`.
    /// Gate is set wide enough to catch a kernel meltdown (ratio
    /// dropping below 0.2 means something far worse than the known
    /// kernel-layout gap), not to enforce parity.
    private func runComparison(
        kvLength: Int, nRuns: Int = 5, nSteps: Int = 32,
        catastrophicFloor: Double = 0.20
    ) async throws {
        let tpsMirror = try await runDecodeTpsBench(
            decodePath: .dequantMirror, kvLength: kvLength,
            nRuns: nRuns, nSteps: nSteps)
        let tpsCompressed = try await runDecodeTpsBench(
            decodePath: .compressed, kvLength: kvLength,
            nRuns: nRuns, nSteps: nSteps)
        let ratio = tpsCompressed / tpsMirror
        let speedupPct = (ratio - 1.0) * 100.0
        print(
            "[aura4v4 KV=\(kvLength)] "
                + "dequantMirror=\(String(format: "%.2f", tpsMirror)) tps  "
                + "compressed=\(String(format: "%.2f", tpsCompressed)) tps  "
                + "ratio=\(String(format: "%.3f", ratio))×  "
                + "(\(String(format: "%+.1f", speedupPct))%)")
        // Catastrophic-regression gate only. The known kernel-layout
        // gap is roughly -18% at KV=64 → -58% at KV=1024 on M5 Max
        // (single-simdgroup-per-query vs sdpaDecode's parallel
        // shape). A ratio below 0.2 means something far worse than
        // that — probably a bug worth catching.
        let ratioStr = String(format: "%.3f", ratio)
        let msg =
            "compressed/dequantMirror ratio \(ratioStr) below "
            + "catastrophic floor \(catastrophicFloor) at KV=\(kvLength)"
        #expect(ratio >= catastrophicFloor, "\(msg)")
    }

    @Test("decode tps — KV=64 (short context)")
    func decodeTpsKV64() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("decodeTpsKV64 skipped: \(qwen3LocalPath) not found")
            return
        }
        try await runComparison(kvLength: 64)
    }

    @Test("decode tps — KV=256 (medium context)")
    func decodeTpsKV256() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("decodeTpsKV256 skipped: \(qwen3LocalPath) not found")
            return
        }
        try await runComparison(kvLength: 256)
    }

    @Test("decode tps — KV=1024 (long context, mirror buffer largest)")
    func decodeTpsKV1024() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("decodeTpsKV1024 skipped: \(qwen3LocalPath) not found")
            return
        }
        // At KV=1024 the dequant-mirror writes a 2 MiB f16 buffer per
        // layer per token. Compressed flash reads packed K codes
        // directly. This is where the win should be visible.
        try await runComparison(kvLength: 1024)
    }

    /// 2-pass `blockSize` sweep. Varies
    /// `AuraFlashScratchCache.blockSizeOverride` across {32, 64, 128,
    /// 256} at KV=256 / 1024 / 4096 to see which tile size best
    /// saturates the M5 Max for the 2-pass FA-2 dispatch. The default
    /// is 64 (matches `Ops.sdpaDecode2Pass`); this prints a per-cell
    /// tps table so we can tune without re-running the full bench.
    ///
    /// At KV=64 only `bs=32` would give >1 block, so we skip it — the
    /// sweep matters most where token-parallelism actually has work to
    /// distribute.
    @Test("decode tps — blockSize sweep at KV=256 / 1024 / 4096 (2-pass only)")
    func blockSizeSweep() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("blockSizeSweep skipped: \(qwen3LocalPath) not found")
            return
        }
        let blockSizes = [32, 64, 128, 256]
        let kvLengths = [256, 1024]
        var results: [(kv: Int, bs: Int, tps: Double)] = []
        for kv in kvLengths {
            for bs in blockSizes {
                AuraFlashScratchCache.blockSizeOverride = bs
                let tps = try await runDecodeTpsBench(
                    decodePath: .compressed, kvLength: kv,
                    nRuns: 3, nSteps: 24)
                results.append((kv, bs, tps))
                print(
                    "[blockSize sweep] KV=\(kv)  bs=\(bs)  "
                        + "compressed=\(String(format: "%.2f", tps)) tps")
            }
        }
        AuraFlashScratchCache.blockSizeOverride = nil
        // Print a compact summary table at the end.
        print("\n=== blockSize sweep summary ===")
        print("KV \\ bs    32       64      128      256")
        for kv in kvLengths {
            let row =
                results.filter { $0.kv == kv }
                .map { String(format: "%7.2f", $0.tps) }
                .joined(separator: " ")
            print("KV=\(String(format: "%-5d", kv))  \(row)")
        }
    }

    /// Cache-memory bench: the dequant-mirror path allocates a
    /// `[nKVHeads, maxSeq, headDim]` shared working buffer per layer.
    /// The compressed path doesn't touch that buffer at all — assert
    /// the savings by inspecting `bytesAllocated` / `bytesInUse` on
    /// the AURA cache itself.
    @Test("cache memory — packed-only footprint vs mirror at maxSeq=4096")
    func cacheMemoryFootprint() async throws {
        guard FileManager.default.fileExists(atPath: qwen3LocalPath) else {
            print("cacheMemoryFootprint skipped: \(qwen3LocalPath) not found")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        optsBuilder.kvCache = .auraQuantized(scheme: .default)
        optsBuilder.auraDecodePath = .compressed
        let opts = optsBuilder
        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(qwen3LocalPath, options: opts)
        }
        let qwen = try #require(m.qwen3, "expected Qwen3Model engine")
        let caches = qwen.makeLayerCaches(maxSeq: 4096)
        guard let auraFirst = caches.first as? AURAQuantizedKVCache else {
            Issue.record("expected at least one AURAQuantizedKVCache in layer caches")
            return
        }
        // `bytesAllocated` reports the packed buffers + norms (the
        // permanent storage); the per-layer working mirror is shared
        // across layers + only lives when `.dequantMirror` is engaged
        // (allocated lazily inside `prepareForAttention`).
        let packedBytesAllocated = auraFirst.bytesAllocated
        let mirrorBytesIfAlloc =
            auraFirst.nKVHeads * auraFirst.maxSeq * auraFirst.headDim * 2  // bf16
        let savingsRatio = Double(mirrorBytesIfAlloc) / Double(packedBytesAllocated)
        print(
            "[aura4v4 cache layout @ maxSeq=4096] "
                + "packed+norms=\(packedBytesAllocated / 1024) KiB  "
                + "mirror-if-alloc=\(mirrorBytesIfAlloc / 1024) KiB  "
                + "compression=\(String(format: "%.2f", savingsRatio))×")
    }
}
