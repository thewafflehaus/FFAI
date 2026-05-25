// Mistral family root. The family enum (`enum Mistral`), variant
// protocol, and dense-impl class live in `Models/Text/MistralText.swift`.
// Mistral 7B is architecturally identical to Llama 3 dense, so the
// loader just routes through `LlamaDense` with Mistral-specific
// dispatch metadata. This file is the universal family-root anchor —
// every Sources/FFAI model family has a `Models/<F>.swift`
// discoverability entry point, even when (as here) the enum lives one
// folder down.
//
// Variants:
//   - Models/Text/MistralText.swift — Mistral 7B / Mistral Nemo dense
//
// Related (separate families):
//   - Models/Mistral3.swift         — Mistral Small 3.1 vision-language
//                                     (`mistral3` model_type, ViT + MLP
//                                     projector + LlamaDense backbone)

import Foundation
