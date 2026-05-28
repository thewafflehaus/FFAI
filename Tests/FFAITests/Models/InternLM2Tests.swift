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
// InternLM2Tests — root-file unit tests for `Sources/FFAI/Models/InternLM2.swift`
// and the fused-wqkv split math in `InternLM2Text.swift`.
//
// Offline. InternLM 2 is Llama-shaped in the forward path but ships
// InternLM2-native tensor names and a fused `wqkv` projection, so it
// has a dedicated loader. These tests guard the dispatch metadata and
// the q/k/v row-range math the loader uses to split `wqkv`.

import Foundation
import Testing

@testable import FFAI

@Suite("InternLM2 Family Root")
struct InternLM2RootTests {

    @Test("modelTypes advertises the internlm2 label")
    func modelTypes() {
        #expect(InternLM2.modelTypes.contains("internlm2"))
        #expect(!InternLM2.modelTypes.isEmpty)
    }

    @Test("architectures advertises InternLM2ForCausalLM")
    func architectures() {
        #expect(InternLM2.architectures.contains("InternLM2ForCausalLM"))
        #expect(!InternLM2.architectures.isEmpty)
    }

    @Test("wqkvRowRanges interleaves q/k/v per KV group (small example)")
    func wqkvRangesSmall() {
        // 4 query heads, 2 KV heads, head_dim 8 → q_per_kv = 2,
        // groupRows = (2 + 2) * 8 = 32. Two groups, total 64 rows.
        let (q, k, v) = InternLM2Dense.wqkvRowRanges(
            nHeads: 4, nKVHeads: 2, headDim: 8)
        // group 0: q [0,16) k [16,24) v [24,32)
        // group 1: q [32,48) k [48,56) v [56,64)
        #expect(q.map { [$0.start, $0.count] } == [[0, 16], [32, 16]])
        #expect(k.map { [$0.start, $0.count] } == [[16, 8], [48, 8]])
        #expect(v.map { [$0.start, $0.count] } == [[24, 8], [56, 8]])
    }

    @Test("wqkvRowRanges partitions the full fused output with no gaps/overlap")
    func wqkvRangesCover() {
        // Production InternLM2 1.8B geometry.
        let nHeads = 16, nKVHeads = 8, headDim = 128
        let (q, k, v) = InternLM2Dense.wqkvRowRanges(
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim)
        let qPerKV = nHeads / nKVHeads
        let total = nKVHeads * (qPerKV + 2) * headDim

        // Output-channel counts match the projections.
        #expect(q.reduce(0) { $0 + $1.count } == nHeads * headDim)
        #expect(k.reduce(0) { $0 + $1.count } == nKVHeads * headDim)
        #expect(v.reduce(0) { $0 + $1.count } == nKVHeads * headDim)

        // Every row in [0, total) is covered exactly once.
        var covered = [Bool](repeating: false, count: total)
        for (start, count) in q + k + v {
            for r in start ..< start + count {
                #expect(!covered[r], "row \(r) covered twice")
                covered[r] = true
            }
        }
        #expect(!covered.contains(false), "every wqkv row must be claimed")
    }
}
