// Mamba2 family root. The family enum (`enum Mamba2`), variant
// protocol, and pure-SSM impl class live in `Models/Text/Mamba2Text.swift`.
// This file is the universal family-root anchor — every Sources/FFAI
// model family has a `Models/<F>.swift` discoverability entry point,
// even when (as here) the enum lives one folder down.
//
// Variants:
//   - Models/Text/Mamba2Text.swift — Mamba 2 pure state-space decoder
//                                     (`mamba2` model_type)

import Foundation
