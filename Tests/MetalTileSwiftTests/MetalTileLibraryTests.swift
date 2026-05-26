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
// MetalTileLibrary surface coverage — verifies the metallib loader
// produces a usable MTLDevice + MTLCommandQueue + MTLLibrary, that
// repeated init succeeds (no global state poisoning), and that
// `library.makeFunction(name:)` resolves known kernels and rejects
// garbage names. KernelManifestSmokeTests already exercises the
// full kernel surface; this file guards the library-loading layer.

import Foundation
import Metal
import Testing

@testable import MetalTileSwift

@Suite("MetalTileLibrary")
struct MetalTileLibraryTests {

    @Test("default init loads the bundled kernels.metallib")
    func defaultInitSucceeds() throws {
        let lib = try MetalTileLibrary()
        #expect(!lib.metallibURL.path.isEmpty)
        #expect(lib.metallibURL.lastPathComponent == "kernels.metallib")
    }

    @Test("a second init returns a fully-formed, independent library")
    func secondInitIsClean() throws {
        // Two back-to-back inits must both succeed — this catches a
        // class of bug where the loader caches a half-built MTLLibrary
        // in a static var and the second caller gets a corrupted one.
        let a = try MetalTileLibrary()
        let b = try MetalTileLibrary()
        #expect(a.metallibURL == b.metallibURL)
        // Both libraries should be able to resolve the same kernel.
        #expect(a.library.makeFunction(name: "vector_add_f32") != nil)
        #expect(b.library.makeFunction(name: "vector_add_f32") != nil)
    }

    @Test("device + command queue are non-nil and usable")
    func deviceAndQueueAreUsable() throws {
        let lib = try MetalTileLibrary()
        // Both surfaces are non-Optional, but exercising them confirms
        // the underlying MTL resources are alive (the queue can make
        // a cmdbuf, the device matches the queue's device).
        #expect(lib.commandQueue.device === lib.device)
        let cb = lib.commandQueue.makeCommandBuffer()
        #expect(cb != nil, "MTLCommandQueue.makeCommandBuffer returned nil")
    }

    @Test("library.makeFunction resolves a known kernel name")
    func makeFunctionKnownKernel() throws {
        let lib = try MetalTileLibrary()
        // vector_add_f32 is a kernel the manifest smoke test depends on,
        // so it's a stable canary across the kernel surface.
        let fn = lib.library.makeFunction(name: "vector_add_f32")
        #expect(fn != nil, "library.makeFunction returned nil for vector_add_f32")
    }

    @Test("library.makeFunction returns nil for an unknown kernel name")
    func makeFunctionUnknownKernel() throws {
        let lib = try MetalTileLibrary()
        // A name that can't possibly exist — MTLLibrary.makeFunction(name:)
        // returns nil rather than throwing.
        let fn = lib.library.makeFunction(name: "absolutely_not_a_real_kernel_xyz")
        #expect(
            fn == nil,
            "library.makeFunction(name:) should return nil for a garbage kernel name")
    }

    @Test("library.functionNames enumerates the bundled kernel surface")
    func functionNamesIsNonEmpty() throws {
        let lib = try MetalTileLibrary()
        let names = lib.library.functionNames
        #expect(
            !names.isEmpty,
            "library.functionNames is empty — the metallib has no kernels?")
        // vector_add_f32 should appear in the enumeration.
        #expect(
            names.contains("vector_add_f32"),
            "expected vector_add_f32 in library.functionNames; got first few: \(names.prefix(5))")
    }

    @Test("shared singleton loads the same metallib as a fresh init")
    func sharedMatchesFreshInit() throws {
        let fresh = try MetalTileLibrary()
        // The shared singleton crashes on init failure rather than
        // throwing — touching `.metallibURL` is enough to verify it
        // initialised successfully.
        let shared = MetalTileLibrary.shared
        #expect(shared.metallibURL == fresh.metallibURL)
    }
}
