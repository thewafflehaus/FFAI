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
// BenchMethod — every benchmark method type the FFAI bench harness
// knows about. Mirrors mlx-swift-lm's `MLX_BENCH_METHOD` set so the
// reports are cross-comparable.
//
// Implemented today:                 simple, summarization, wikitext2
// Plumbed but not implemented yet:   niah, multiTurn, toolCalling,
//                                    ngramSpot, ngramSweep,
//                                    ngramSweepSummary, vision
//
// The unimplemented methods are present so the CLI accepts the same
// `--method` strings as mlx-swift-lm; running one prints a clear
// "needs <X>" message + exits non-zero. Each stub names its
// underlying dependency so plumbing them later is straightforward.

import Foundation

public enum BenchMethod: String, Sendable, CaseIterable {
    case simple
    case summarization
    case wikitext2
    case niah
    case multiTurn = "multi-turn"
    case toolCalling = "tool-calling"
    case ngramSpot = "ngram-spot"
    case ngramSweep = "ngram-sweep"
    case ngramSweepSummary = "ngram-sweep-summary"
    case vision

    /// `true` when the bench harness can actually run this method
    /// today. `false` methods print a "needs <X>" message + exit
    /// non-zero rather than silently producing garbage.
    public var isImplemented: Bool {
        switch self {
        case .simple, .summarization, .wikitext2:
            return true
        case .niah, .multiTurn, .toolCalling,
            .ngramSpot, .ngramSweep, .ngramSweepSummary, .vision:
            return false
        }
    }

    /// Short blurb for the CLI's `--help` wall.
    public var description: String {
        switch self {
        case .simple: return "single-prompt generation, throughput + memory"
        case .summarization: return "fixed-size prompts across configurable contexts"
        case .wikitext2: return "perplexity over the WikiText-2 corpus (forced decode)"
        case .niah: return "needle-in-a-haystack retrieval at multiple depths"
        case .multiTurn: return "multi-turn conversation, replies fed back iteratively"
        case .toolCalling: return "tool call generation + validation"
        case .ngramSpot: return "single prompt across N candidate ngram-config cells"
        case .ngramSweep: return "18 prompts × 32 cells, full raw rows"
        case .ngramSweepSummary: return "ngram-sweep matrix plus per-category roll-up"
        case .vision: return "VLM smoke test: image + prompt → text"
        }
    }

    /// What plumbing each unimplemented method is waiting on. Surfaces
    /// in the CLI error message so the user knows whether it's a
    /// today problem or a future-phase problem.
    public var dependency: String? {
        switch self {
        case .simple, .summarization, .wikitext2:
            return nil
        case .niah:
            return "sliding-window attention mask + needle-position bookkeeping"
        case .multiTurn:
            return "ChatSession-style multi-turn cache reuse helper"
        case .toolCalling:
            return "tool-spec rendering in ChatTemplate"
        case .ngramSpot, .ngramSweep, .ngramSweepSummary:
            return "n-gram speculative-decoding lookup"
        case .vision:
            return "vision encoder + multi-modal generate path"
        }
    }
}
