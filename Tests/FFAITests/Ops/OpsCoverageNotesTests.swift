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
