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
// Qwen3.5-35B-A3B (MoE + GDN hybrid) forwardMany / decode perf bench.
//
// Same shape as `Qwen36TextIntegrationTests.forwardManyBench*` but
// pointed at the locally-available Qwen3.5-35B-A3B-4bit checkpoint.
// Identical engine path (`m.qwen35`), so this exercises every Bagel
// wiring landed on `tom/bagel-clean`: GPU MoE router, batched QKV
// fast paths, GDN chunked recurrence, rmsNormQgemvInt4Fast finalNorm,
// 2-pass Flash-Decoding, MTLResidencySet weight pinning, etc.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

private let qwen35MoELocalPath = "/Users/tom/models/Qwen3.5-35B-A3B-4bit"

@Suite("Qwen3.5-35B-A3B local-checkpoint bench", .serialized)
struct Qwen35MoEBenchIntegrationTests {

    @Test("Qwen3.5-35B-A3B decode T=1 tps — 5 runs, median over 32 steps")
    func decodeBenchT1() async throws {
        guard FileManager.default.fileExists(atPath: qwen35MoELocalPath) else {
            print("decodeBenchT1 skipped: \(qwen35MoELocalPath) not found")
            return
        }
        let m = try await loadModel()
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine")
            return
        }

        let prompt = "The history of the printing press began when"
        let promptTokens = m.tokenizer.encode(text: prompt)
        let promptLen = promptTokens.count

        // Warm.
        for _ in 0 ..< 2 {
            let warmCaches = qwen.makeLayerCaches()
            for (i, tok) in promptTokens.enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: warmCaches)
            }
            for j in 0 ..< 4 {
                _ = qwen.forward(
                    tokenId: 0, position: promptLen + j, caches: warmCaches)
            }
        }

        // 5 timed runs of decode-only, 32 steps each (median).
        let nSteps = 32
        var runs: [Double] = []
        for _ in 0 ..< 5 {
            let caches = qwen.makeLayerCaches()
            for (i, tok) in promptTokens.enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: caches)
            }
            let t0 = Date()
            for j in 0 ..< nSteps {
                _ = qwen.forward(
                    tokenId: 0, position: promptLen + j, caches: caches)
            }
            runs.append(Date().timeIntervalSince(t0))
        }
        runs.sort()
        let median = runs[runs.count / 2]
        let tps = Double(nSteps) / median
        print(
            "Qwen3.5-35B-A3B decode T=1: runs=\(runs.map { String(format: "%.3f", $0) })s "
                + "median=\(String(format: "%.3f", median))s → \(String(format: "%.2f", tps)) tps"
        )
    }

    @Test("Qwen3.5-35B-A3B forwardMany T=8 smoke")
    func forwardManyT8Smoke() async throws {
        guard FileManager.default.fileExists(atPath: qwen35MoELocalPath) else {
            print("forwardManyT8Smoke skipped: \(qwen35MoELocalPath) not found")
            return
        }
        let m = try await loadModel()
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")
        let seed = "The quick brown fox"
        var encoded = m.tokenizer.encode(text: seed)
        while encoded.count < 8 { encoded.append(0) }
        encoded = Array(encoded.prefix(8))
        let caches = qwen.makeLayerCaches()
        let cmd = Device.shared.makeCommandBuffer()
        let t0 = Date()
        _ = qwen.forwardMany(
            tokenIds: encoded, startPosition: 0,
            caches: caches, on: cmd, device: Device.shared)
        cmd.commit()
        await cmd.completed()
        let dt = Date().timeIntervalSince(t0)
        print("Qwen3.5-35B-A3B forwardMany T=8: \(String(format: "%.3f", dt))s")
    }

    @Test("Qwen3.5-35B-A3B per-token forward T=128 (no batched)")
    func perTokenT128() async throws {
        guard FileManager.default.fileExists(atPath: qwen35MoELocalPath) else {
            print("perTokenT128 skipped: \(qwen35MoELocalPath) not found")
            return
        }
        let m = try await loadModel()
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")
        let seed = "The quick brown fox jumps over the lazy dog. "
        let seedEncoded = m.tokenizer.encode(text: seed)
        var encoded = seedEncoded
        while encoded.count < 128 { encoded.append(contentsOf: seedEncoded) }
        encoded = Array(encoded.prefix(128))
        let caches = qwen.makeLayerCaches()
        let t0 = Date()
        for (i, tok) in encoded.enumerated() {
            _ = qwen.forward(tokenId: tok, position: i, caches: caches)
        }
        let dt = Date().timeIntervalSince(t0)
        print(
            "Qwen3.5-35B-A3B per-token T=128: \(String(format: "%.3f", dt))s → "
                + "\(String(format: "%.1f", 128.0 / dt)) tps")
    }

    @Test("Qwen3.5-35B-A3B forwardManyBench T=128")
    func forwardManyT128() async throws {
        try await runForwardManyBench(targetT: 128)
    }

    @Test("Qwen3.5-35B-A3B forwardManyBench T=512")
    func forwardManyT512() async throws {
        try await runForwardManyBench(targetT: 512)
    }

    @Test("Qwen3.5-35B-A3B forwardManyBench T=2048 (long-context)")
    func forwardManyT2K() async throws {
        try await runForwardManyBench(targetT: 2048)
    }

    @Test("Qwen3.5-35B-A3B decode after T=1024 prefill — long-KV decode tps")
    func decodeAfterLongPrefill() async throws {
        guard FileManager.default.fileExists(atPath: qwen35MoELocalPath) else {
            print("decodeAfterLongPrefill skipped: \(qwen35MoELocalPath) not found")
            return
        }
        let m = try await loadModel()
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")
        let seed =
            "The history of the printing press began when European craftsmen of the 15th century combined movable metal type with oil based ink screw presses paper to mass produce printed books pamphlets and broadsheets revolutionising communication"
        let seedEncoded = m.tokenizer.encode(text: seed)
        var encoded = seedEncoded
        while encoded.count < 1024 { encoded.append(contentsOf: seedEncoded) }
        encoded = Array(encoded.prefix(1024))

        // Warm.
        for _ in 0 ..< 2 {
            let warmCaches = qwen.makeLayerCaches()
            let warmCmd = Device.shared.makeCommandBuffer()
            _ = qwen.forwardMany(
                tokenIds: encoded, startPosition: 0,
                caches: warmCaches, on: warmCmd, device: Device.shared)
            warmCmd.commit()
            await warmCmd.completed()
        }

        // Prefill 1024 via forwardMany, then decode 32 steps and
        // measure decode-only tps. Verifies the wirings (sdpaDecode +
        // sdpaDecode2Pass routing, KV cache residency, etc.) hold up
        // at non-trivial KV.
        let nSteps = 32
        var runs: [Double] = []
        for _ in 0 ..< 5 {
            let caches = qwen.makeLayerCaches()
            let cmd = Device.shared.makeCommandBuffer()
            _ = qwen.forwardMany(
                tokenIds: encoded, startPosition: 0,
                caches: caches, on: cmd, device: Device.shared)
            cmd.commit()
            await cmd.completed()

            let t0 = Date()
            for j in 0 ..< nSteps {
                _ = qwen.forward(
                    tokenId: 0, position: 1024 + j, caches: caches)
            }
            runs.append(Date().timeIntervalSince(t0))
        }
        runs.sort()
        let median = runs[runs.count / 2]
        let tps = Double(nSteps) / median
        print(
            "Qwen3.5-35B-A3B decode T=1 after T=1024 prefill: "
                + "runs=\(runs.map { String(format: "%.3f", $0) })s "
                + "median=\(String(format: "%.3f", median))s → "
                + "\(String(format: "%.2f", tps)) tps")
    }

    private func loadModel() async throws -> Model {
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        return try await ModelLoadLock.shared.loadSerially {
            try await Model.load(qwen35MoELocalPath, options: opts)
        }
    }

    private func runForwardManyBench(targetT: Int) async throws {
        guard FileManager.default.fileExists(atPath: qwen35MoELocalPath) else {
            print("forwardManyBench skipped: \(qwen35MoELocalPath) not found")
            return
        }
        let m = try await loadModel()
        let qwen = try #require(m.qwen35, "expected Qwen35Model engine")

        let seed =
            "The history of the printing press began when European craftsmen of the 15th century combined movable metal type with oil based ink screw presses paper to mass produce printed books pamphlets and broadsheets revolutionising communication"
        let seedEncoded = m.tokenizer.encode(text: seed)
        var encoded = seedEncoded
        while encoded.count < targetT {
            encoded.append(contentsOf: seedEncoded)
        }
        encoded = Array(encoded.prefix(targetT))
        let T = encoded.count
        print("Qwen3.5-35B-A3B forwardManyBench T=\(T)")

        // Warm.
        for _ in 0 ..< 2 {
            let warmCachesP = qwen.makeLayerCaches()
            for (i, tok) in encoded.prefix(2).enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: warmCachesP)
            }
            let warmCachesB = qwen.makeLayerCaches()
            let warmCmd = Device.shared.makeCommandBuffer()
            _ = qwen.forwardMany(
                tokenIds: encoded, startPosition: 0,
                caches: warmCachesB, on: warmCmd, device: Device.shared)
            warmCmd.commit()
            await warmCmd.completed()
        }

        // Per-token loop baseline (5 runs, median).
        var perTokenSecs: [Double] = []
        for _ in 0 ..< 5 {
            let caches = qwen.makeLayerCaches()
            let t0 = Date()
            for (i, tok) in encoded.enumerated() {
                _ = qwen.forward(tokenId: tok, position: i, caches: caches)
            }
            perTokenSecs.append(Date().timeIntervalSince(t0))
        }
        perTokenSecs.sort()
        let perTokenMedian = perTokenSecs[perTokenSecs.count / 2]

        // Batched forwardMany (5 runs, median).
        var batchedSecs: [Double] = []
        for _ in 0 ..< 5 {
            let caches = qwen.makeLayerCaches()
            let bCmd = Device.shared.makeCommandBuffer()
            let t0 = Date()
            _ = qwen.forwardMany(
                tokenIds: encoded, startPosition: 0,
                caches: caches, on: bCmd, device: Device.shared)
            bCmd.commit()
            await bCmd.completed()
            batchedSecs.append(Date().timeIntervalSince(t0))
        }
        batchedSecs.sort()
        let batchedMedian = batchedSecs[batchedSecs.count / 2]

        let speedup = perTokenMedian / batchedMedian
        let batchedTps = Double(T) / batchedMedian
        print(
            "Qwen3.5-35B-A3B RESULT T=\(T): "
                + "per_token=\(String(format: "%.0f", perTokenMedian * 1000))ms "
                + "batched=\(String(format: "%.0f", batchedMedian * 1000))ms "
                + "speedup=\(String(format: "%.2fx", speedup)) "
                + "batched_tps=\(String(format: "%.1f", batchedTps))")
    }
}
