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
import Foundation
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
        let o = LoadOptions(capabilities: [.imageIn])
        #expect(o.capabilities.contains(.imageIn))
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

    @Test("cacheDirectory + ModelDownloader convenience init")
    func cacheDirectoryAndDownloader() {
        // Default
        let opts1 = LoadOptions()
        #expect(opts1.cacheDirectory == nil)

        // Custom
        let custom = URL(fileURLWithPath: "/Volumes/Big/hf-cache")
        let opts2 = LoadOptions(cacheDirectory: custom)
        #expect(opts2.cacheDirectory == custom)

        // Convenience init builds without throwing for both nil + non-nil
        let dlNil = ModelDownloader(cacheDirectory: nil)
        let dlSet = ModelDownloader(cacheDirectory: custom)
        // Both should produce a usable client (we don't make network
        // calls — just construct).
        _ = (dlNil, dlSet)
    }
}
