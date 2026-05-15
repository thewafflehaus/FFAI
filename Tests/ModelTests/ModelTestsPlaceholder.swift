import Testing
@testable import FFAI

// Placeholder so the ModelTests target has at least one source file.
// Real per-model test files (Llama/, Qwen3/, etc.) land in Phase 2+.
@Suite("ModelTests placeholder")
struct ModelTestsPlaceholder {
    @Test("placeholder")
    func placeholder() {
        #expect(FFAI.version == "0.0.1-dev")
    }
}
