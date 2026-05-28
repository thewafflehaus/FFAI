// Copyright 2026 Tom Turney (@TheTom)
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
// SpecDecodeVerifyCostBench — answers: at γ+1 ≤ 4, does the optimised
// decode T=1 path (which benefits from scalarFMA + bm8 qmm + GDN
// fused prep) beat the batched `forwardManyAllLogits` path?
//
// Methodology: time N iterations of each shape on identical caches;
// `snapshotAll` / `restoreAll` between iters so we measure pure
// forward work without context drift. Reports median ms / iter and
// the batched-vs-single ratio for both T=3 (γ=2 verify) and T=2
// (γ=1 verify).

import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

/// Bench target — the cached `mlx-community/Qwen3.6-35B-A3B-4bit`
/// MoE checkpoint. Switched off Tom's prior `/Users/tom/models/...`
/// hard-coded local path; matches the same identifier the sibling
/// `Qwen36TextBenchTest` + `SpecDecodeBenchTests` use.
private let qwen36VerifyModelId = "mlx-community/Qwen3.6-35B-A3B-4bit"

/// Local-cache predicate — bench is also disabled if the checkpoint
/// isn't already cached.
private let qwen36VerifyCacheAvailable: Bool = {
    let cache =
        ("~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
            as NSString)
        .expandingTildeInPath
    return FileManager.default.fileExists(atPath: cache)
}()

@Suite(
    "SpecDecode verify-cost bench",
    .enabled(
        if: IntegrationGroupGating.enableBenchmarkSuites && qwen36VerifyCacheAvailable,
        IntegrationGroupGating.benchmarkSkipReason)
)
struct SpecDecodeVerifyCostBench {

    @Test("Compare forwardManyAllLogits(T=3) vs 3× forward() at decode shape")
    func compareBatchedVsSingleStep() async throws {
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(qwen36VerifyModelId, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine")
            return
        }

        let prompt = """
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n - 1) + fibonacci(n - 2)

            def
            """
        let promptTokens = m.tokenizer.encode(text: prompt)
        let promptLen = promptTokens.count
        let device = Device.shared

        let caches = qwen.makeLayerCaches()
        // Prefill — untimed.
        for (i, tok) in promptTokens.enumerated() {
            _ = qwen.forward(tokenId: tok, position: i, caches: caches)
        }

        let inputIds = [
            promptTokens[promptLen - 4],
            promptTokens[promptLen - 3],
            promptTokens[promptLen - 2],
        ]

        // Snapshot caches so iteration N starts in identical state.
        let snap0 = caches.snapshotAll(device: device)

        // Warm the single-step path.
        for tok in inputIds {
            _ = qwen.forward(tokenId: tok, position: promptLen, caches: caches)
        }
        caches.restoreAll(from: snap0, device: device)

        // Warm the batched path.
        let warmCmd = device.makeCommandBuffer()
        _ = qwen.forwardManyAllLogits(
            tokenIds: inputIds, startPosition: promptLen,
            caches: caches, on: warmCmd, device: device)
        warmCmd.commit()
        await warmCmd.awaitCompletion()
        caches.restoreAll(from: snap0, device: device)

        // 16 timed iters of each path, restoring caches between each.
        let nIters = 16

        var singleTimes: [Double] = []
        for _ in 0 ..< nIters {
            let t0 = Date()
            for (i, tok) in inputIds.enumerated() {
                _ = qwen.forward(
                    tokenId: tok, position: promptLen + i, caches: caches)
            }
            singleTimes.append(Date().timeIntervalSince(t0))
            caches.restoreAll(from: snap0, device: device)
        }

        var batchedTimes: [Double] = []
        for _ in 0 ..< nIters {
            let t0 = Date()
            let cmd = device.makeCommandBuffer()
            _ = qwen.forwardManyAllLogits(
                tokenIds: inputIds, startPosition: promptLen,
                caches: caches, on: cmd, device: device)
            cmd.commit()
            await cmd.awaitCompletion()
            batchedTimes.append(Date().timeIntervalSince(t0))
            caches.restoreAll(from: snap0, device: device)
        }

        let singleMedian = singleTimes.sorted()[nIters / 2]
        let batchedMedian = batchedTimes.sorted()[nIters / 2]
        let singleMs = singleMedian * 1000
        let batchedMs = batchedMedian * 1000
        let ratio = batchedMs / singleMs

        print("VerifyCostBench T=3 (median of \(nIters) iters):")
        print("  3× forward() loop:           \(String(format: "%.2f", singleMs)) ms")
        print("  1× forwardManyAllLogits:     \(String(format: "%.2f", batchedMs)) ms")
        print("  batched / single ratio:      \(String(format: "%.2fx", ratio))")
        if ratio > 1.0 {
            print(
                "  → single-step LOOP is faster by \(String(format: "%.1f", (ratio - 1.0) * 100))%"
            )
        } else {
            print(
                "  → batched is faster by \(String(format: "%.1f", (1.0 / ratio - 1.0) * 100))%"
            )
        }

        // T=2 case (γ=1 verify shape).
        let inputIds2 = [
            promptTokens[promptLen - 2], promptTokens[promptLen - 1],
        ]
        let snap2 = caches.snapshotAll(device: device)
        for tok in inputIds2 {
            _ = qwen.forward(tokenId: tok, position: promptLen, caches: caches)
        }
        caches.restoreAll(from: snap2, device: device)
        let warmCmd2 = device.makeCommandBuffer()
        _ = qwen.forwardManyAllLogits(
            tokenIds: inputIds2, startPosition: promptLen,
            caches: caches, on: warmCmd2, device: device)
        warmCmd2.commit()
        await warmCmd2.awaitCompletion()
        caches.restoreAll(from: snap2, device: device)

        var singleT2: [Double] = []
        for _ in 0 ..< nIters {
            let t0 = Date()
            for (i, tok) in inputIds2.enumerated() {
                _ = qwen.forward(
                    tokenId: tok, position: promptLen + i, caches: caches)
            }
            singleT2.append(Date().timeIntervalSince(t0))
            caches.restoreAll(from: snap2, device: device)
        }
        var batchedT2: [Double] = []
        for _ in 0 ..< nIters {
            let t0 = Date()
            let cmd = device.makeCommandBuffer()
            _ = qwen.forwardManyAllLogits(
                tokenIds: inputIds2, startPosition: promptLen,
                caches: caches, on: cmd, device: device)
            cmd.commit()
            await cmd.awaitCompletion()
            batchedT2.append(Date().timeIntervalSince(t0))
            caches.restoreAll(from: snap2, device: device)
        }
        let s2ms = singleT2.sorted()[nIters / 2] * 1000
        let b2ms = batchedT2.sorted()[nIters / 2] * 1000
        print("VerifyCostBench T=2 (γ=1 shape, median \(nIters) iters):")
        print("  2× forward() loop:           \(String(format: "%.2f", s2ms)) ms")
        print("  1× forwardManyAllLogits:     \(String(format: "%.2f", b2ms)) ms")
        print("  ratio:                       \(String(format: "%.2fx", b2ms / s2ms))")
    }
}
