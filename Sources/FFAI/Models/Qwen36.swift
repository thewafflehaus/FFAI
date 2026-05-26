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
// Qwen 3.6 family root — same Qwen3x hybrid architecture as Qwen 3.5.
//
// Alibaba's Qwen 3.6 line (Qwen3.6-A3B and the larger MoE variants)
// ships under the *same* `qwen3_5*` HuggingFace `model_type` strings as
// Qwen 3.5 — the architecture (stack-interleaved Gated Delta Net ↔
// attention, dense or block-sparse MoE FFN with an always-on shared
// expert) is unchanged. The only practical 3.5 → 3.6 deltas are larger
// checkpoints, more MoE experts, and quantized embeddings — all of
// which the existing loader handles natively.
//
// As a result this file is a *doc-only anchor*. The actual types live
// in `Models/Text/Qwen3xText.swift` (covers Qwen 3.5 AND Qwen 3.6) and
// re-use the `Qwen35*` type prefix because the architecture was named
// at the Qwen 3.5 release. The Models/<F>.swift discoverability rule
// applies here so a developer scanning the `Sources/FFAI/Models/`
// directory finds an obvious entry point for the Qwen 3.6 lineage.
//
// Related files:
//   - Models/Qwen35.swift            — Qwen 3.5 root + Qwen3-VL-MoE
//                                       orchestrator.
//   - Models/Text/Qwen3xText.swift   — the shared hybrid backbone
//                                       (`enum Qwen35`, `Qwen35Hybrid`,
//                                       `Qwen35Model`).
//   - Models/Vision/Qwen3Vision.swift — shared ViT tower (Qwen3-VL
//                                       dense + Qwen3-VL-MoE).

import Foundation
