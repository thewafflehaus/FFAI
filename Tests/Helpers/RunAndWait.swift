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
// Shared GPU-test helper: encode + commit + wait inside an
// autoreleasepool. Lives in the shared `TestHelpers` target so every
// test target depending on it (FFAITests, ModelIntegrationTests) can
// invoke `runAndWait { ... }` without duplicating the source file.
//
// Why the pool: `commandQueue.makeCommandBuffer()` returns an
// Objective-C bridged class. Under ARC the local `cb` reference is
// released when the helper returns, but the underlying objc return
// path autoreleases intermediates (the cmdbuf itself, transient
// encoder refs, internal Metal driver objects). Without an explicit
// `autoreleasepool { … }`, those autoreleased objects can sit around
// until the surrounding scope ends — which under Swift Testing's
// scheduler can be much wider than the test method. Across hundreds
// of tests that accumulates into a real cmdbuf-retention problem:
// observed locally as the GPU staying pinned at 100% with the OS
// alternating between responsive and laggy.
//
// `autoreleasepool { … }` forces a deterministic drain at the end of
// each call. Per Apple's Memory Management docs this is the standard
// fix for objc-bridged code in tight loops.

import Foundation
import Metal
import FFAI

/// Run `block` on a fresh `MTLCommandBuffer`, commit it, and wait for
/// completion. Wrapped in an autoreleasepool so cmdbuf retention drains
/// per call rather than accumulating across tests.
///
/// Standard pattern in every GPU-touching test:
///
///     @Test func someTest() {
///         let a = Tensor.empty(shape: [4], dtype: .f32)
///         var out: Tensor!
///         runAndWait { cb in out = Ops.silu(a, on: cb) }
///         #expect(...)
///     }
public func runAndWait(_ block: (MTLCommandBuffer) -> Void) {
    autoreleasepool {
        let cb = Device.shared.makeCommandBuffer()
        block(cb)
        cb.commit()
        cb.waitUntilCompleted()
    }
}
