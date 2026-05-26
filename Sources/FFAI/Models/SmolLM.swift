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
// SmolLM family — HuggingFace's small (135M / 360M / 1.7B / 3B) Llama-3
// shaped text models. Three generations ship: SmolLM 1 (smollm), SmolLM 2
// (smollm2), SmolLM 3 (smollm3). All three use byte-identical weights to
// Llama 3 dense + optional QKV biases that `loadLinear` auto-detects, so
// the family root just declares the dispatch metadata and routes the
// loader through `LlamaDense` — no per-family forward code.

import Foundation

public enum SmolLM {
    /// HuggingFace `model_type` labels SmolLM ships across its three
    /// generations.
    public static let modelTypes: Set<String> = ["smollm", "smollm2", "smollm3"]

    /// HuggingFace `architectures[0]` labels we recognise.
    public static let architectures: Set<String> = [
        "SmolLMForCausalLM",
        "SmolLM2ForCausalLM",
        "SmolLM3ForCausalLM",
    ]
}
