// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// PSOCache surface coverage — verifies the PSO cache compiles known
// kernels into pipeline states, dedupes repeat lookups, throws a
// meaningful error for garbage kernel names, and remains correct
// under concurrent access.
//
// KernelManifestSmokeTests already proves "every kernel in the
// manifest produces a usable PSO" and runs the basic dedup check —
// this file targets the error path and thread-safety guarantees the
// cache promises (the inner `compileLock` + double-checked cache
// lookup in `PSOCache.lookup`).

import Foundation
@preconcurrency import Metal
import Testing

@testable import MetalTileSwift

@Suite("PSOCache")
struct PSOCacheTests {

    private func makeCache() throws -> PSOCache {
        let lib = try MetalTileLibrary()
        return PSOCache(library: lib)
    }

    @Test("pipelineState resolves a known kernel into a usable PSO")
    func resolveKnownKernel() throws {
        let cache = try makeCache()
        let pso = try cache.pipelineStateThrowing(for: "vector_add_f32")
        // Metal refuses to instantiate a function it couldn't compile;
        // a positive max-threads value confirms the PSO is alive.
        #expect(
            pso.maxTotalThreadsPerThreadgroup > 0,
            "vector_add_f32 PSO reports zero maxTotalThreadsPerThreadgroup")
    }

    @Test("pipelineStateThrowing throws kernelNotFound for an unknown name")
    func unknownKernelThrows() throws {
        let cache = try makeCache()
        do {
            _ = try cache.pipelineStateThrowing(for: "absolutely_not_a_real_kernel_xyz")
            Issue.record("pipelineStateThrowing should have thrown for a garbage kernel name")
        } catch let err as PSOCacheError {
            // The error description should at least mention the kernel
            // name so failures are diagnosable from logs.
            #expect(
                err.description.contains("absolutely_not_a_real_kernel_xyz"),
                "PSOCacheError description should name the missing kernel; got: \(err)")
        } catch {
            Issue.record("expected PSOCacheError; got \(type(of: error)): \(error)")
        }
    }

    /// Sendable adapter for `MTLComputePipelineState`. Older Xcode
    /// toolchains (16.x SDK) don't mark the Metal protocol Sendable,
    /// so passing one through a `TaskGroup` trips Swift 6 strict
    /// concurrency. The underlying PSO IS thread-safe — Apple's docs
    /// describe `MTLComputePipelineState` as safe to share across
    /// threads for dispatch — so the `@unchecked` is sound here.
    private struct SendablePSO: @unchecked Sendable {
        let pso: any MTLComputePipelineState
    }

    @Test("concurrent lookups of the same kernel return the same PSO instance")
    func concurrentLookupsDedupeUnderRace() async throws {
        let cache = try makeCache()

        // Hammer the cache from many tasks simultaneously. Pre-fix
        // (no compileLock), this could race two compiles and crash;
        // post-fix, every caller receives the same instance because
        // `compileLock` + the double-checked cache read collapse
        // concurrent compiles into one.
        let kernel = "vector_add_f32"
        let psos = await withTaskGroup(of: SendablePSO.self) { group in
            for _ in 0 ..< 8 {
                group.addTask {
                    SendablePSO(pso: cache.pipelineState(for: kernel))
                }
            }
            var collected: [any MTLComputePipelineState] = []
            for await wrapped in group { collected.append(wrapped.pso) }
            return collected
        }
        #expect(psos.count == 8)
        let first = psos[0]
        for (i, pso) in psos.enumerated() {
            #expect(
                pso === first,
                "concurrent PSO lookup \(i) returned a different instance than the first — cache failed to dedupe under race"
            )
        }
    }

    @Test("shared PSOCache resolves the same kernels as a fresh cache")
    func sharedCacheIsUsable() throws {
        // PSOCache.shared is a process-wide singleton; verify it
        // resolves the same canary kernel a fresh PSOCache does.
        let fresh = try makeCache()
        let freshPso = try fresh.pipelineStateThrowing(for: "vector_add_f32")
        let sharedPso = try PSOCache.shared.pipelineStateThrowing(for: "vector_add_f32")
        #expect(freshPso.maxTotalThreadsPerThreadgroup > 0)
        #expect(sharedPso.maxTotalThreadsPerThreadgroup > 0)
        // The two caches are backed by different MTLLibrary instances
        // (each `MetalTileLibrary()` builds its own MTLLibrary), so the
        // PSO instances will differ — but both must be live.
    }

    @Test("PSOCacheError description names the failing kernel")
    func errorDescriptionsAreInformative() {
        let nameMissing = PSOCacheError.kernelNotFound("my_kernel_name")
        #expect(nameMissing.description.contains("my_kernel_name"))

        let underlying = NSError(domain: "test", code: 42)
        let compileFail = PSOCacheError.psoCompileFailed("k_name", underlying)
        #expect(compileFail.description.contains("k_name"))
        #expect(compileFail.description.contains("PSO compile failed"))

        let sourceMissing = PSOCacheError.metalSourceNotFound("mpp_kernel")
        #expect(sourceMissing.description.contains("mpp_kernel"))
    }
}
