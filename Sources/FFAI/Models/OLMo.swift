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
// OLMo family — Allen AI's open-source Llama-shaped dense decoder.
// OLMo 1 (olmo) and OLMo 2 (olmo2) both ship with byte-identical
// weights to Llama 3 dense + optional QKV biases that `loadLinear`
// auto-detects, so the family root just declares dispatch metadata
// and routes the loader through `LlamaDense`.

import Foundation

public enum OLMo {
    public static let modelTypes: Set<String> = ["olmo", "olmo2"]
    public static let architectures: Set<String> = [
        "OlmoForCausalLM",
        "Olmo2ForCausalLM",
    ]
}
