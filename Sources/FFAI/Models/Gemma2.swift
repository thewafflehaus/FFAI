// Gemma2 family root. The family enum (`enum Gemma2`), variant
// protocol, and dense-impl class live in `Models/Text/Gemma2Text.swift`.
// This file is the universal family-root anchor — every Sources/FFAI
// model family has a `Models/<F>.swift` discoverability entry point,
// even when (as here) the enum lives one folder down.
//
// Variants:
//   - Models/Text/Gemma2Text.swift — Gemma 2 dense (`gemma2` /
//                                     `gemma2_text` model_type). Used
//                                     as the text backbone by
//                                     Paligemma 2 (see Models/Paligemma.swift).

import Foundation
