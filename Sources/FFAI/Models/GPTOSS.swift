// GPTOSS family root — OpenAI's GPT-OSS open-weights MoE line.
// The family enum (`enum GPTOSS`), variant protocol, and impl classes
// live in `Models/Text/GPTOSSText.swift` + `Models/Text/GPTOSSMoEText.swift`.
// This file is the universal family-root anchor — every Sources/FFAI
// model family has a `Models/<F>.swift` discoverability entry point,
// even when (as here) the enum lives one folder down.
//
// Variants:
//   - Models/Text/GPTOSSText.swift    — shared scaffolding (config,
//                                        sliding-window-vs-full
//                                        attention alternation, sinks
//                                        fold)
//   - Models/Text/GPTOSSMoEText.swift — GPT-OSS-20B MoE decoder
//                                        (`gpt_oss` model_type)

import Foundation
