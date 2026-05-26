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
