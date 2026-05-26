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
// Integration tests covering the `ffai inspect` path end-to-end
// against a small locally-cached model — asserts the diagnostic
// surface doesn't crash on a working architecture. Doesn't pin the
// exact output (top-K logits / token-id sequence aren't stable
// across tokenizer revisions); just confirms the pipeline runs.
//
// The integration suites already cover coherence for each model
// family; this one specifically protects the InspectCommand glue
// (Model.tokenizer + makeLayerCaches + Sampling.topN integration)
// from silent regressions.

import Foundation
import TestHelpers
import Testing

@testable import FFAI

@Suite("Model Inspection Integration", .serialized)
struct ModelInspectionIntegrationTests {

    @Test("inspect path runs end-to-end on Llama 3.2 1B")
    func inspectLlama() async throws {
        let modelId = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        let prompt = "Once upon a time, in a quiet"

        let m = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelId)
        }

        // Architecture sanity. This is the same surface
        // InspectCommand prints — any field NaN'd / nil-d here
        // would also corrupt the inspect display.
        #expect(m.engine.hidden > 0)
        #expect(m.engine.nLayers > 0)
        #expect(m.engine.nHeads >= m.engine.nKVHeads)
        #expect(m.engine.headDim > 0)
        #expect(m.engine.vocab > 0)

        // Tokenizer round-trips the prompt — same path inspect uses.
        let tokens = m.tokenizer.encode(text: prompt)
        #expect(!tokens.isEmpty)
        let roundtrip = m.tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
        #expect(
            roundtrip.contains("quiet"),
            "tokenizer round-trip should preserve a recognizable substring of the prompt; got \"\(roundtrip)\""
        )

        // Per-layer KV cache layout — inspect prints `bytesAllocated`
        // and the layer 0 stride. Asserting just on the totals
        // (per-test specifics live in the family tests above).
        let caches = m.engine.makeLayerCaches()
        #expect(caches.count == m.engine.nLayers)
        let totalBytes = caches.reduce(0) { $0 + $1.bytesAllocated }
        #expect(totalBytes > 0)

        // Single-step forward (the prefill loop inspect runs) — top-K
        // logits + decoded strings. Asserts no NaN, no inf.
        var lastLogits: Tensor?
        for (i, t) in tokens.enumerated() {
            lastLogits = m.engine.forward(tokenId: t, position: i, caches: caches)
        }
        guard let l = lastLogits else {
            Issue.record("inspect should have run prefill on a non-empty prompt")
            return
        }
        let top = Sampling.topN(l, n: 5)
        #expect(top.count == 5)
        for (_, value) in top {
            #expect(
                value.isFinite, "inspect top-5 logits must be finite (no NaN, no inf); got \(value)"
            )
        }
    }
}
