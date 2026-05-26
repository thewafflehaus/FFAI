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
// InternLM 2 family — Shanghai AI Lab's InternLM v2 dense text models.
// Llama-3-shaped weights — some checkpoints use a fused `wqkv`
// projection that `loadLinear` handles transparently via the
// bias-aware Linear; the family root just declares dispatch metadata
// and routes through `LlamaDense`.

import Foundation

public enum InternLM2 {
    public static let modelTypes: Set<String> = ["internlm2"]
    public static let architectures: Set<String> = ["InternLM2ForCausalLM"]
}
