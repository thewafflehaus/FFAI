// PSOCache
//
// Lazily compiles MTLComputePipelineState objects from the metallib's
// MTLFunctions and caches them by kernel name. First-call cost is the
// PSO compilation (~100-500µs in Metal driver — no MSL parsing since the
// metallib is already pre-compiled). Subsequent calls are a hash lookup.
//
// Phase 0: keyed by kernel name only. Function-constant specialization
// keys land when we add `#[autotune]` and quantized kernels.

import Foundation
import Metal

public enum PSOCacheError: Error, CustomStringConvertible {
    case kernelNotFound(String)
    case psoCompileFailed(String, Error)
    case metalSourceNotFound(String)
    case metalSourceCompileFailed(String, Error)

    public var description: String {
        switch self {
        case .kernelNotFound(let name):
            return "kernel '\(name)' not found in kernels.metallib"
        case .psoCompileFailed(let name, let underlying):
            return "PSO compile failed for '\(name)': \(underlying)"
        case .metalSourceNotFound(let name):
            return "MPP live-compile: .metal source not found for '\(name)' in Resources/kernels/"
        case .metalSourceCompileFailed(let name, let underlying):
            return "MPP live-compile: makeLibrary(source:) failed for '\(name)': \(underlying)"
        }
    }
}

public final class PSOCache: @unchecked Sendable {
    public static let shared = PSOCache(library: MetalTileLibrary.shared)

    private let library: MetalTileLibrary
    private let lock = NSLock()              // protects `cache` reads + writes
    private let compileLock = NSLock()       // single-flight PSO compilation
    private var cache: [String: MTLComputePipelineState] = [:]

    public init(library: MetalTileLibrary) {
        self.library = library
    }

    /// Get (or build + cache) the PSO for a kernel by name.
    /// `fatalError`s on lookup or compile failure. Use `pipelineState(for:)
    /// throws -> ...` if you want to handle errors at the call site.
    public func pipelineState(for kernelName: String) -> MTLComputePipelineState {
        do {
            return try lookup(kernelName)
        } catch {
            fatalError("PSOCache.pipelineState(for: \"\(kernelName)\") failed: \(error)")
        }
    }

    public func pipelineStateThrowing(for kernelName: String) throws -> MTLComputePipelineState {
        try lookup(kernelName)
    }

    private func lookup(_ name: String) throws -> MTLComputePipelineState {
        // Fast path: cache hit. Hold the lock only long enough to read.
        lock.lock()
        if let cached = cache[name] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        // Slow path: serialize PSO compilation through a dedicated lock.
        // Compiling INSIDE the same `lock` would block all concurrent
        // cache-hit readers; compiling OUTSIDE any lock (the pattern
        // before this fix) lets two threads compile the same PSO
        // concurrently, and the loser's PSO gets dropped on the
        // second `cache[name] = ...` write. Concurrent
        // `makeComputePipelineState` for the same function has also
        // been observed (in unit tests with many parallel suites)
        // to produce a PSO that runs the kernel on garbage instruction
        // memory → NaN output. Single-flight via `compileLock` makes
        // PSO compilation strictly serial, and the inner cache re-check
        // means a waiter that arrived during compile gets the
        // already-built PSO instead of starting a duplicate compile.
        compileLock.lock()
        defer { compileLock.unlock() }

        lock.lock()
        if let cached = cache[name] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        // MPP cooperative-tensor kernels (mt_*_mpp_*) must be live-compiled
        // from their .metal source, NOT loaded from the pre-built metallib.
        // The MetalPerformancePrimitives header inlines cooperative-tensor
        // type IDs + descriptor layouts that drift between SDK versions —
        // when SDK macOS != runtime macOS, an offline-compiled metallib's
        // baked-in MPP types disagree with the device runtime's, producing
        // bit-deterministic wrong output (e.g. cos 0.816 vs 0.999 oracle).
        // Live-compile via `makeLibrary(source:)` resolves MPP against the
        // running OS's header, dodging the skew. See ollama #15594, #14432,
        // llama.cpp PR #16634 for the same class of bug.
        let function: MTLFunction
        if Self.isMppKernel(name) {
            function = try liveCompileMppFunction(name)
        } else {
            guard let fn = library.library.makeFunction(name: name) else {
                throw PSOCacheError.kernelNotFound(name)
            }
            function = fn
        }
        let pso: MTLComputePipelineState
        do {
            pso = try library.device.makeComputePipelineState(function: function)
        } catch {
            throw PSOCacheError.psoCompileFailed(name, error)
        }

        lock.lock()
        cache[name] = pso
        lock.unlock()
        return pso
    }

    /// Returns true for kernels that use `mpp::tensor_ops::matmul2d` and
    /// must be live-compiled from .metal source on the running OS. Detected
    /// purely by name — every metaltile MPP kernel name contains `_mpp_`
    /// (e.g. `mt_qmm_mma_mpp_*`, `mt_moe_gather_qmm_mma_int4_bm{8,16,64}_mpp_*`).
    /// Non-MPP kernels (the bulk of the metallib) still load from
    /// kernels.metallib — much faster PSO build.
    private static func isMppKernel(_ name: String) -> Bool {
        // Opt-out via env var, just in case a future kernel name accidentally
        // matches `_mpp_` (e.g. an `_mppow_` variant). Default on.
        if let raw = ProcessInfo.processInfo.environment["FFAI_PSO_LIVE_COMPILE_MPP"],
           raw == "0" || raw.lowercased() == "false"
        {
            return false
        }
        return name.contains("_mpp_")
    }

    private func liveCompileMppFunction(_ name: String) throws -> MTLFunction {
        let metalURL = try Self.locateMetalSource(name)
        let source: String
        do {
            source = try String(contentsOf: metalURL, encoding: .utf8)
        } catch {
            throw PSOCacheError.metalSourceCompileFailed(name, error)
        }
        let opts = MTLCompileOptions()
        let lib: MTLLibrary
        do {
            lib = try library.device.makeLibrary(source: source, options: opts)
        } catch {
            throw PSOCacheError.metalSourceCompileFailed(name, error)
        }
        guard let fn = lib.makeFunction(name: name) else {
            throw PSOCacheError.kernelNotFound(name)
        }
        return fn
    }

    /// Locate `<name>.metal` next to the loaded metallib. `tile emit` writes
    /// per-kernel sources to `Resources/kernels/<name>.metal` alongside
    /// `Resources/kernels.metallib`, so we anchor the lookup off
    /// `library.metallibURL`.
    private static func locateMetalSource(_ name: String) throws -> URL {
        let metallibURL = MetalTileLibrary.shared.metallibURL
        let kernelsDir = metallibURL.deletingLastPathComponent()
            .appendingPathComponent("kernels", isDirectory: true)
        let url = kernelsDir.appendingPathComponent("\(name).metal")
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        // Fallback: SPM may have flattened Resources/ — try Bundle.module.
        if let bundleURL = Bundle.module.url(
            forResource: name, withExtension: "metal", subdirectory: "Resources/kernels"
        ) {
            return bundleURL
        }
        if let bundleURL = Bundle.module.url(
            forResource: name, withExtension: "metal", subdirectory: "kernels"
        ) {
            return bundleURL
        }
        if let bundleURL = Bundle.module.url(forResource: name, withExtension: "metal") {
            return bundleURL
        }
        throw PSOCacheError.metalSourceNotFound(name)
    }
}
