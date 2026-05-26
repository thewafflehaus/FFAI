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
// MetalTileLibrary
//
// Loads kernels.metallib once at process startup and exposes the underlying
// MTLLibrary + a default MTLDevice and MTLCommandQueue. Designed to be a
// process-wide singleton (`MetalTileLibrary.shared`).
//
// kernels.metallib + manifest.json are produced at build time by
// metaltile-emit. See planning/architecture.md §1.

import Foundation
import Metal

public enum MetalTileLibraryError: Error, CustomStringConvertible {
    case noDefaultDevice
    case noCommandQueue
    case metallibNotFound(URL)
    case metallibLoadFailed(URL, Error)

    public var description: String {
        switch self {
        case .noDefaultDevice:
            return "MTLCreateSystemDefaultDevice() returned nil"
        case .noCommandQueue:
            return "MTLDevice.makeCommandQueue() returned nil"
        case .metallibNotFound(let url):
            return "kernels.metallib not found at \(url.path)"
        case .metallibLoadFailed(let url, let underlying):
            return "Failed to load \(url.path): \(underlying)"
        }
    }
}

public final class MetalTileLibrary: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    public let metallibURL: URL

    /// Maximum number of in-flight (uncompleted) `MTLCommandBuffer`s the
    /// shared command queue keeps before `makeCommandBuffer()` blocks on
    /// the next caller. Default 16.
    ///
    /// Why a cap at all: Metal's default queue depth is effectively
    /// unbounded. With many concurrent callers (e.g. Swift Testing's
    /// default cross-suite parallelism), hundreds of cmdbufs can pile up
    /// in flight, starving the WindowServer compositor of GPU time and
    /// (observed locally) crashing WindowServer → system freeze. A small
    /// cap forces backpressure at the Metal layer so the compositor can
    /// always make progress.
    ///
    /// Why **16** specifically:
    ///
    ///   - Production decode is 1 cmdbuf / token (serial), so 16 is
    ///     16× oversize for steady-state.
    ///   - Hypothetical Phase-8 batched decode with 8 parallel streams
    ///     would use ~8 cmdbufs simultaneously — 16 gives 2× headroom.
    ///   - Apple's MTKView triple-buffering convention is 3 (for display
    ///     smoothness); 16 is "compute-class headroom" but small enough
    ///     that a runaway parallel caller hits backpressure before
    ///     starving the compositor.
    ///   - MLX itself never caps queue depth; they cap per-cmdbuf size
    ///     instead (max_ops / max_mb_per_buffer auto-commit). We pick
    ///     the orthogonal knob because our pattern is many small
    ///     cmdbufs from many callers, not one large cmdbuf from one
    ///     caller.
    ///
    /// Override at runtime via the `FFAI_MAX_COMMAND_BUFFERS` env var
    /// (positive integer). Useful when triaging perf-vs-stability
    /// tradeoffs without rebuilding.
    public static let defaultMaxCommandBufferCount: Int = {
        if let raw = ProcessInfo.processInfo.environment["FFAI_MAX_COMMAND_BUFFERS"],
           let parsed = Int(raw), parsed > 0
        {
            return parsed
        }
        return 16
    }()

    /// Process-wide singleton. Lazily initialized; throws on first access if
    /// the system has no default Metal device or the metallib can't be loaded.
    public static let shared: MetalTileLibrary = {
        do {
            return try MetalTileLibrary()
        } catch {
            fatalError("MetalTileLibrary.shared init failed: \(error)")
        }
    }()

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalTileLibraryError.noDefaultDevice
        }
        // Cap in-flight cmdbuf count to apply backpressure at the Metal
        // layer. See `defaultMaxCommandBufferCount` for the rationale.
        let cap = Self.defaultMaxCommandBufferCount
        // Emit at process startup so we can verify FFAI_MAX_COMMAND_BUFFERS
        // is actually being honored. Cheap (one line on stderr per process).
        // Remove once the freeze diagnostic settles.
        FileHandle.standardError.write(Data(
            "[MetalTileLibrary] maxCommandBufferCount=\(cap) (FFAI_MAX_COMMAND_BUFFERS=\(ProcessInfo.processInfo.environment["FFAI_MAX_COMMAND_BUFFERS"] ?? "<unset>"))\n".utf8
        ))
        guard let queue = device.makeCommandQueue(
            maxCommandBufferCount: cap
        ) else {
            throw MetalTileLibraryError.noCommandQueue
        }
        let url = try Self.locateMetallib()
        do {
            let library = try device.makeLibrary(URL: url)
            self.device = device
            self.commandQueue = queue
            self.library = library
            self.metallibURL = url
        } catch {
            throw MetalTileLibraryError.metallibLoadFailed(url, error)
        }
    }

    /// Find kernels.metallib in the SPM resource bundle.
    private static func locateMetallib() throws -> URL {
        if let url = Bundle.module.url(
            forResource: "kernels",
            withExtension: "metallib",
            subdirectory: "Resources"
        ) {
            return url
        }
        // Fallback: SPM may flatten the Resources/ folder.
        if let url = Bundle.module.url(forResource: "kernels", withExtension: "metallib") {
            return url
        }
        let fallback = Bundle.module.bundleURL.appendingPathComponent(
            "Resources/kernels.metallib"
        )
        throw MetalTileLibraryError.metallibNotFound(fallback)
    }
}
