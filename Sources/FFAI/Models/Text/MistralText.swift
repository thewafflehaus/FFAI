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
// Mistral text — Mistral 7B and Nemo / Small dense decoder support.
//
// The architecture is byte-for-byte identical to Llama 3 except for
// the default `rope_theta` (1_000_000) and a slightly different
// `max_position_embeddings` cap. Both differences flow naturally
// through `LlamaDense` from `config.json`, so the loader aliases
// rather than duplicating: see `Mistral.variant(for:)` in
// `Models/Mistral.swift`, which returns `LlamaDense.self`.
//
// The family enum (`enum Mistral`) lives in `Models/Mistral.swift`
// (the family root / main interface). No Mistral-specific variant
// protocol, error type, or model class is needed today — all the work
// happens in the Llama family.

import Foundation
import Metal
