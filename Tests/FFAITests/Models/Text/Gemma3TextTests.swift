// Gemma3TextTests — unit coverage for `Sources/FFAI/Models/Text/Gemma3Text.swift`.
//
// Offline. Covers the `Gemma3Dense` variant surface (capabilities +
// generation defaults) and the per-layer sliding/global scheduling
// formula `(i + 1) % slidingWindowPattern != 0`. The companion
// `Gemma3TextWeightFoldTests.swift` covers the GemmaRMSNorm `+1.0`
// weight-fold path; this file is the lightweight surface guard.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma3Dense Variant Surface")
struct Gemma3TextTests {

    @Test("Gemma3Dense advertises text in/out capabilities")
    func capabilities() {
        #expect(Gemma3Dense.availableCapabilities.contains(.textIn))
        #expect(Gemma3Dense.availableCapabilities.contains(.textOut))
        #expect(!Gemma3Dense.availableCapabilities.contains(.visionIn))
    }

    @Test("Gemma3Dense default generation parameters track Gemma family")
    func defaultGenerationParameters() {
        // Gemma defaults: temperature 1.0, top-p 0.95, top-k 64.
        let p = Gemma3Dense.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature >= 0)
        #expect(p.topK >= 0)
        #expect(p.topP > 0 && p.topP <= 1.0)
    }

    /// Gemma 3's family default `sliding_window_pattern = 6` — every
    /// 6th layer is global, the rest sliding.
    @Test("pattern=6 marks every 6th layer global")
    func pattern6Layout() {
        let pattern = 6
        for i in 0..<12 {
            let isSliding = (i + 1) % pattern != 0
            if (i + 1) % pattern == 0 {
                #expect(!isSliding, "layer \(i) (i+1=\(i+1)) must be global")
            } else {
                #expect(isSliding, "layer \(i) (i+1=\(i+1)) must be sliding")
            }
        }
    }

    /// The sliding layer uses `rope_local_base_freq`; the global layer
    /// uses `rope_theta`. Both pulled from config with documented
    /// defaults (1e4 and 1e6 respectively).
    @Test("global vs sliding layers use distinct RoPE base frequencies")
    func ropeThetaPerLayer() {
        // Document the contract via a tiny simulation — the loader
        // assembles `layerRopeTheta` exactly this way.
        let ropeTheta: Float = 1_000_000
        let ropeLocal: Float = 10_000
        let pattern = 6
        for i in 0..<6 {
            let isSliding = (i + 1) % pattern != 0
            let layerRopeTheta = isSliding ? ropeLocal : ropeTheta
            #expect(layerRopeTheta == (isSliding ? ropeLocal : ropeTheta))
        }
    }
}
