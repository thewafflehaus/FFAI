import Testing
@testable import FFAI

@Suite("FFAI")
struct FFAITests {
    @Test("version is set")
    func version() {
        #expect(FFAI.version == "0.0.1-dev")
    }
}
