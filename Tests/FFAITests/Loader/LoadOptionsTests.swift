import Testing
@testable import FFAI

@Suite("LoadOptions")
struct LoadOptionsTests {
    @Test("defaults — text capabilities + raw KV + eager dispatch + prewarm")
    func defaults() {
        let o = LoadOptions()
        #expect(o.capabilities == Capability.textOnly)
        #expect(o.prewarm == true)
        #expect(o.lazyCapabilities == true)
        #expect(o.revision == "main")
        if case .raw = o.kvCache { /* ok */ } else {
            Issue.record("expected .raw KVCacheKind")
        }
        if case .eager = o.dispatchMode { /* ok */ } else {
            Issue.record("expected .eager DispatchMode")
        }
    }

    @Test("requested capabilities are unioned with text mandatory pair")
    func capabilityUnion() {
        let o = LoadOptions(capabilities: [.visionIn])
        #expect(o.capabilities.contains(.visionIn))
        #expect(o.capabilities.contains(.textIn))
        #expect(o.capabilities.contains(.textOut))
    }

    @Test("prewarm and lazyCapabilities can be overridden")
    func overrides() {
        let o = LoadOptions(prewarm: false, lazyCapabilities: false, revision: "dev")
        #expect(o.prewarm == false)
        #expect(o.lazyCapabilities == false)
        #expect(o.revision == "dev")
    }
}
