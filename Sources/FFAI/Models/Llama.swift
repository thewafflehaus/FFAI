// Llama family root. The family enum (`enum Llama`), variant protocol,
// and dense-impl class live in `Models/Text/LlamaText.swift`. This
// file is the universal family-root anchor — every Sources/FFAI model
// family has a `Models/<F>.swift` discoverability entry point, even
// when (as here) the family is text-only and the enum lives one
// folder down.
//
// Variants:
//   - Models/Text/LlamaText.swift  — LlamaDense (Llama 3.x dense
//                                     text models)

import Foundation
