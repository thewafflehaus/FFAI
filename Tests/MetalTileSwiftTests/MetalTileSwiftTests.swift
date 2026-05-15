import Testing
@testable import MetalTileSwift

@Suite("MetalTileSwift")
struct MetalTileSwiftTests {
    @Test("version is set")
    func version() {
        #expect(MetalTileSwift.version == "0.0.1-dev")
    }
}
