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
// Sanity tests for OpsCoverageNotes — the data structure that
// catalogues metaltile kernel families intentionally left unwrapped at
// the Ops surface. Catches truncation regressions on edits.

import Testing
@testable import FFAI

@Suite("OpsCoverageNotes — intentionally-unwrapped kernel inventory")
struct OpsCoverageNotesTests {

    @Test("inventory is non-empty and every entry has a rationale")
    func inventoryNonEmpty() {
        let items = OpsCoverageNotes.intentionallyUnwrapped
        #expect(!items.isEmpty)
        for item in items {
            #expect(!item.familyName.isEmpty)
            #expect(!item.rationale.isEmpty,
                    "kernel \(item.familyName) needs a rationale")
        }
    }

    @Test("count matches the array length")
    func countMatchesLength() {
        #expect(OpsCoverageNotes.count == OpsCoverageNotes.intentionallyUnwrapped.count)
    }

    @Test("at least one entry per major intentionally-skipped category")
    func categoriesCovered() {
        let items = OpsCoverageNotes.intentionallyUnwrapped
        // Cheap sniff-test that the manifest covers the categories the
        // surface-parity README enumerates.
        let categories = [
            "aura_flash",     // internal flash building blocks
            "_record",        // dispatch_chain replay infra
            "smoke",          // probe/test kernels
            "fp4",            // dtype-blocked
            "winograd",       // unused vision specializations
            "scan",           // unused reduction primitives
        ]
        for needle in categories {
            let hit = items.contains { item in
                item.familyName.lowercased().contains(needle) ||
                item.rationale.lowercased().contains(needle)
            }
            #expect(hit, "no manifest entry mentions '\(needle)'")
        }
    }
}
