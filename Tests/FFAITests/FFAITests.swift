import Testing
@testable import FFAI

@Suite("FFAI")
struct FFAITests {
    // Don't pin to a literal — release.sh auto-rewrites FFAI.version
    // to match the tag it's cutting (see publishing.md). Pinning here
    // would force every release to also touch this test.
    @Test("version is a non-empty semver-shaped string")
    func version() {
        let v = FFAI.version
        #expect(!v.isEmpty, "FFAI.version is empty")
        // Expect e.g. "0.1.0", "0.1.0-alpha", "1.2.3-rc.1", "0.2.0-dev".
        // The regex anchors on the leading MAJOR.MINOR.PATCH triple;
        // any non-empty suffix that starts with `-` is fine.
        let pattern = #/^\d+\.\d+\.\d+(-[A-Za-z0-9.-]+)?$/#
        #expect(v.wholeMatch(of: pattern) != nil,
                "FFAI.version (\(v)) doesn't look like semver")
    }
}
