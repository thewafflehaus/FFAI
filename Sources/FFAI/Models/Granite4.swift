// Granite4 family root — IBM's Granite 4 line (the GraniteMoeHybrid
// architecture). The family enum (`enum Granite4`), variant protocol,
// and hybrid-impl class live in `Models/Text/Granite4Text.swift`. This
// file is the universal family-root anchor — every Sources/FFAI model
// family has a `Models/<F>.swift` discoverability entry point, even
// when (as here) the enum lives one folder down.
//
// Variants:
//   - Models/Text/Granite4Text.swift — IBM Granite 4 hybrid (Mamba 2 +
//                                       attention + MoE,
//                                       `granitemoehybrid` model_type)
//
// Related (separate family):
//   - Models/Granite3.swift — Granite v3 (Llama-shaped dense,
//                              `granite` model_type)

import Foundation
