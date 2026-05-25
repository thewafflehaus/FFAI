// Jamba family root. The family enum (`enum Jamba`), variant protocol,
// and hybrid-impl class live in `Models/Text/JambaText.swift`. This
// file is the universal family-root anchor — every Sources/FFAI model
// family has a `Models/<F>.swift` discoverability entry point, even
// when (as here) the enum lives one folder down.
//
// Variants:
//   - Models/Text/JambaText.swift — AI21's Jamba hybrid (Mamba 2 +
//                                    attention + MoE/MLP alternation,
//                                    `jamba` model_type)

import Foundation
