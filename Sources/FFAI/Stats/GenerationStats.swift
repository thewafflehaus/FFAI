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
// GenerationStats — every numeric field one `generate(...)` call
// produces. The CLI's --stats prints this; the bench harness writes
// the same fields into the markdown / JSON sidecar so analysis tooling
// stays cross-compatible with mlx-swift-lm reports.
//
// Field naming and grouping mirrors mlx-swift-lm's `ResultRow` so
// porting analysis scripts is mechanical. Fields not yet meaningful
// (batch decode, speculative decode) are scaffolded as commented
// blocks below — uncomment when the underlying capability lands.

import Foundation

public struct GenerationStats: Sendable {
    // ─── Counts ──────────────────────────────────────────────────────

    /// Number of prompt tokens fed to prefill.
    public let promptTokens: Int
    /// Number of tokens emitted by the decode loop.
    public let generatedTokens: Int
    /// Maximum context window the model supports.
    public let contextSize: Int

    // ─── Latency / throughput ────────────────────────────────────────

    public let prefillTimeS: Double
    public let decodeTimeS: Double
    public let timeToFirstTokenMs: Double

    public var prefillTokensPerSecond: Double {
        prefillTimeS > 0 ? Double(promptTokens) / prefillTimeS : 0
    }
    public var decodeTokensPerSecond: Double {
        decodeTimeS > 0 ? Double(generatedTokens) / decodeTimeS : 0
    }

    /// Per-token decode rate computed from token 11 onward — drops the
    /// first ~10 tokens so PSO compilation, autorelease pool warm-up,
    /// and Metal pipeline-cache misses don't depress the steady-state
    /// number. `nil` when fewer than 11 generated tokens.
    public let steadyTokensPerSecond: Double?

    // ─── Memory (Apple Silicon GPU + wired ticket) ───────────────────

    /// `MTLDevice.currentAllocatedSize` immediately before prefill.
    public let baselineGPUBytes: Int
    /// `MTLDevice.currentAllocatedSize` after the last prefill token.
    public let postPrefillGPUBytes: Int
    /// `MTLDevice.currentAllocatedSize` after the last decode token.
    public let postDecodeGPUBytes: Int
    /// Max GPU bytes seen during the prefill phase.
    public let prefillPeakGPUBytes: Int
    /// Max GPU bytes seen during the decode phase.
    public let decodePeakGPUBytes: Int
    /// `MTLDevice.recommendedMaxWorkingSetSize` — the wired-memory
    /// ticket the OS will try to hold resident.
    public let wiredTicketBytes: Int
    /// Resident weight-tensor bytes (sum of every model parameter
    /// buffer). Computed once at construction.
    public let weightsBytes: Int
    /// Sum of `bytesAllocated` across every per-layer KV cache —
    /// the *capacity*, allocated up-front at the model's max context.
    public let kvCacheAllocatedBytes: Int
    /// Sum of `bytesInUse` across every per-layer KV cache — the
    /// *live* slice (length out of maxSeq).
    public let kvCacheUsedBytes: Int

    public var peakGPUBytes: Int { max(prefillPeakGPUBytes, decodePeakGPUBytes) }
    public var prefillGrowthBytes: Int { postPrefillGPUBytes - baselineGPUBytes }
    public var decodeGrowthBytes: Int { postDecodeGPUBytes - postPrefillGPUBytes }

    // ─── Quality (opt-in; bench harness fills these via Perplexity) ──

    /// Perplexity over the *thinking* segment (tokens between
    /// `<think>` … `</think>`). `nil` when the model didn't emit a
    /// thinking segment, or when the caller didn't request perplexity.
    public let thinkPerplexity: Double?
    /// Perplexity over the post-thinking generation segment.
    public let genPerplexity: Double?
    /// KL divergence vs a reference distribution (e.g. bf16 vs the
    /// quantized variant). Requires a paired reference run.
    public let thinkKLDivergence: Double?
    public let genKLDivergence: Double?

    /// Token counts inside the think / gen segments (sum to
    /// `generatedTokens` when both are non-nil).
    public let thinkTokenCount: Int?
    public let genTokenCount: Int?

    // ─── Scaffold for future modes ───────────────────────────────────
    //
    // Uncomment + populate when the corresponding capability ships.
    // Kept here so the formatted() printer + bench writer schema only
    // grow additively when those land.
    //
    // Batch decoding:
    //   public let batchSize: Int
    //   public let perSequenceDecodeTokensPerSecond: Double?
    //
    // Speculative decoding:
    //   public let acceptanceRate: Double?
    //   public let draftTokensPerSecond: Double?
    //   public let draftAcceptedTokens: Int?

    // ─── Pretty printer for `--stats` ────────────────────────────────

    public func formatted() -> String {
        var out = "[STATS]\n"
        out += "  prompt:           \(promptTokens) tokens\n"
        out += "  generated:        \(generatedTokens) tokens\n"
        out += "  context:          \(contextSize) tokens\n"
        out += String(format: "  ttft:             %.2f ms\n", timeToFirstTokenMs)
        out += String(
            format: "  prefill:          %.2fs (%.2f tok/s)\n",
            prefillTimeS, prefillTokensPerSecond)
        out += String(
            format: "  decode:           %.2fs (%.2f tok/s)\n",
            decodeTimeS, decodeTokensPerSecond)
        if let s = steadyTokensPerSecond {
            out += String(format: "  decode (steady):  %.2f tok/s   (tokens 11+)\n", s)
        }
        out += String(format: "  baseline GPU:     %@\n", Self.fmt(baselineGPUBytes))
        out += String(
            format: "  post-prefill GPU: %@   (+ %@)\n",
            Self.fmt(postPrefillGPUBytes), Self.fmt(prefillGrowthBytes))
        out += String(
            format: "  post-decode  GPU: %@   (+ %@)\n",
            Self.fmt(postDecodeGPUBytes), Self.fmt(decodeGrowthBytes))
        out += String(format: "  prefill peak:     %@\n", Self.fmt(prefillPeakGPUBytes))
        out += String(format: "  decode  peak:     %@\n", Self.fmt(decodePeakGPUBytes))
        out += String(format: "  weights:          %@\n", Self.fmt(weightsBytes))
        out += String(format: "  KV cache (alloc): %@\n", Self.fmt(kvCacheAllocatedBytes))
        out += String(format: "  KV cache (used):  %@\n", Self.fmt(kvCacheUsedBytes))
        out += String(format: "  wired ticket:     %@\n", Self.fmt(wiredTicketBytes))
        if let pp = genPerplexity {
            out += String(format: "  gen perplexity:   %.3f\n", pp)
        }
        if let pp = thinkPerplexity {
            out += String(format: "  think perplexity: %.3f\n", pp)
        }
        if let kl = genKLDivergence {
            out += String(format: "  gen KLD:          %.4f\n", kl)
        }
        if let kl = thinkKLDivergence {
            out += String(format: "  think KLD:        %.4f\n", kl)
        }
        if let t = thinkTokenCount, let g = genTokenCount {
            out += "  think / gen split: \(t) / \(g) tokens\n"
        }
        return out
    }

    private static func fmt(_ bytes: Int) -> String {
        let mb = Double(bytes) / 1024.0 / 1024.0
        if mb >= 1024 { return String(format: "%.2f GB", mb / 1024) }
        return String(format: "%.1f MB", mb)
    }
}
