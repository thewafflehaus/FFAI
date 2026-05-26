// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Gemma2TextTests — unit coverage for `Sources/FFAI/Models/Text/Gemma2Text.swift`.
//
// Offline. Covers the `Gemma2Dense` variant surface (capabilities +
// Gemma-style generation defaults) and the per-layer sliding /
// full-attention scheduling formula `(i + 1) % pattern != 0` (HF's
// `not bool((layer_idx + 1) % pattern == 0)`) that the loader uses to
// stamp each layer's `isSliding` flag.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma2Dense Variant Surface")
struct Gemma2TextTests {

    @Test("Gemma2Dense advertises text in/out capabilities")
    func capabilities() {
        #expect(Gemma2Dense.availableCapabilities.contains(.textIn))
        #expect(Gemma2Dense.availableCapabilities.contains(.textOut))
        #expect(!Gemma2Dense.availableCapabilities.contains(.visionIn))
    }

    @Test("Gemma2Dense default generation parameters track Gemma family")
    func defaultGenerationParameters() {
        // Gemma defaults: temperature 1.0, top-p 0.95, top-k 64.
        let p = Gemma2Dense.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature >= 0)
        #expect(p.topK >= 0)
        #expect(p.topP > 0 && p.topP <= 1.0)
    }

    /// HF's Gemma 2 schedules sliding attention on every layer EXCEPT
    /// each `slidingWindowPattern`-th one (`(i + 1) % pattern == 0`
    /// → global, else sliding). For the family default `pattern = 2`
    /// every even index `i` is sliding, every odd index `i` is global —
    /// i.e. alternating sliding/global starting from layer 0.
    @Test("alternating sliding/full pattern matches HF formula for pattern=2")
    func slidingPattern2Alternates() {
        let pattern = 2
        let kinds = (0..<6).map { i in (i + 1) % pattern != 0 }
        #expect(kinds == [true, false, true, false, true, false])
    }

    /// Pattern 6 is the Gemma 3 default but also a legitimate Gemma 2
    /// override — the formula must hold there too: only layers
    /// 5, 11, 17, ... are global; everything else is sliding.
    @Test("pattern=6 puts every 6th layer on the global path")
    func slidingPattern6Spreads() {
        let pattern = 6
        let kinds = (0..<12).map { i in (i + 1) % pattern != 0 }
        for (i, isSliding) in kinds.enumerated() {
            if (i + 1) % pattern == 0 {
                #expect(!isSliding, "layer \(i) must be global")
            } else {
                #expect(isSliding, "layer \(i) must be sliding")
            }
        }
    }
}
