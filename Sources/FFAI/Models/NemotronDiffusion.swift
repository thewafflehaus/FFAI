// NemotronDiffusion family root. The family enum (`enum
// NemotronDiffusion`), variant protocol, and tri-mode-impl class live
// in `Models/Text/NemotronDiffusionText.swift`. This file is the
// universal family-root anchor — every Sources/FFAI model family has a
// `Models/<F>.swift` discoverability entry point, even when (as here)
// the enum lives one folder down.
//
// Variants:
//   - Models/Text/NemotronDiffusionText.swift — Nemotron-Labs-Diffusion
//                                                tri-mode decoder (AR /
//                                                block-diffusion /
//                                                self-speculation)
//
// Note: this is exported through the unified `enum Nemotron` family
// root in `Models/Nemotron.swift` (which unions modelTypes /
// architectures across NemotronH, NemotronVL, NemotronDiffusion) for
// single-membership-check dispatch in the registry. The
// `Models/NemotronDiffusion.swift` anchor here exists so the
// "every family has its own root file" rule holds uniformly.

import Foundation
