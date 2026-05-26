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
// Audio-side test helpers — waveform fixtures and the STT phrase
// assertion shared by every audio integration suite.
//
// Two responsibilities:
//
//  1. Waveform fixtures — `clean001Waveform()` ("Sure, I can help you
//     with that.") and `conversationalAWaveform()` (~13 s
//     multi-sentence dialogue) loaded from `Tests/Resources/`.
//
//  2. `expectMentionsSureICanHelp(...)` — STT phrase assertion for the
//     clean_001 fixture, with flexible punctuation + capitalisation
//     matching ("Sure I can help you with that", "sure, i can help
//     you with that.", "Sure! I can help you with that..." all pass).
//
// Checkpoint resolution has moved back into the integration suites
// (each picks a single canonical HF repo id and lets `ModelLocator`
// download / cache-hit) — see the per-suite `loadXYZ()` helpers.

import Foundation
import Testing
import FFAI

public enum AudioTestHelpers {

    // MARK: - Waveform fixtures

    /// Load the bundled 16 kHz speech fixture as a mono float waveform.
    /// "Sure, I can help you with that." — clean synthetic speech, 1.85 s.
    /// Kept for VAD / STS / STT suites whose assertions are tuned to
    /// this clip (see `expectMentionsSureICanHelp(...)`).
    public static func clean001Waveform() throws -> [Float] {
        let url = resourceURL("clean_001.wav")
        return try AudioPreprocessing.loadWaveform(url: url, targetRate: 16_000)
    }

    /// Load the bundled conversational speech fixture as a mono 16 kHz
    /// float waveform. ~13 s of multi-sentence dialogue (24 kHz source,
    /// resampled at load), giving STT suites more text to assert
    /// transcription quality against than the 1.85 s "Sure, I…" clip.
    /// Sourced from ekryski/mlx-audio-swift @ ek/audio-benchmarks.
    public static func conversationalAWaveform() throws -> [Float] {
        let url = resourceURL("conversational_a.wav")
        return try AudioPreprocessing.loadWaveform(url: url, targetRate: 16_000)
    }

    // MARK: - STT phrase assertion

    /// Assert that an STT transcription of the `clean_001.wav` fixture
    /// contains "Sure I can help you with that" — case- and
    /// punctuation-insensitive. The clip's ground truth is exactly
    /// "Sure, I can help you with that." but model outputs vary in
    /// capitalisation, trailing punctuation, and quoting:
    ///
    ///   "Sure, I can help you with that."  ✓
    ///   "sure i can help you with that"    ✓
    ///   "Sure! I can help you with that..." ✓
    ///   "I can help you" (missing "sure")  ✗
    ///
    /// Uses `normalizeForMatch` from TextTestHelpers — lowercases,
    /// strips punctuation, collapses whitespace.
    public static func expectMentionsSureICanHelp(
        _ text: String, label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let normalized = normalizeForMatch(text)
        let expected = "sure i can help you with that"
        let comment = Comment(
            rawValue: "\(label): STT output should contain \"Sure, I can help you with that.\" "
                + "(case + punctuation insensitive). Got: \(text)"
        )
        #expect(
            normalized.contains(expected),
            comment,
            sourceLocation: sourceLocation
        )
    }
}
