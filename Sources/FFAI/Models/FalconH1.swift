// FalconH1 family root. The family enum (`enum FalconH1`), variant
// protocol, and hybrid-impl class live in `Models/Text/FalconH1Text.swift`.
// This file is the universal family-root anchor — every Sources/FFAI
// model family has a `Models/<F>.swift` discoverability entry point,
// even when (as here) the enum lives one folder down.
//
// Variants:
//   - Models/Text/FalconH1Text.swift — TII's Falcon H1 hybrid (Mamba 2
//                                       + attention + MLP with per-layer
//                                       multipliers, `falcon_h1`
//                                       model_type)

import Foundation
