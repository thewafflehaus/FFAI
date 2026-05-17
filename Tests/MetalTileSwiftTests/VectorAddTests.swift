// Round-trip test for the vector_add kernel.
//
// Validates the full build pipeline:
//   `tile build --emit all` → kernels.metallib + Generated/MetalTileKernels.swift
//   → MetalTileLibrary loads metallib
//   → PSOCache compiles the PSO
//   → MetalTileKernels.vector_add_f32 dispatches on a real MTLCommandBuffer
//   → output matches expected (a[i] + b[i])

import Metal
import Testing
@testable import MetalTileSwift

@Suite("vector_add_f32 round-trip")
struct VectorAddTests {
    @Test("a + b produces expected output")
    func vectorAdd() throws {
        let n = 256
        let lib = try MetalTileLibrary()

        let a: [Float] = (0..<n).map { Float($0) }
        let b: [Float] = Array(repeating: 1.0, count: n)
        let expected: [Float] = zip(a, b).map { $0 + $1 }

        let bytes = n * MemoryLayout<Float>.stride
        guard let aBuf = lib.device.makeBuffer(bytes: a, length: bytes, options: .storageModeShared),
              let bBuf = lib.device.makeBuffer(bytes: b, length: bytes, options: .storageModeShared),
              let cBuf = lib.device.makeBuffer(length: bytes, options: .storageModeShared)
        else {
            Issue.record("failed to allocate MTLBuffers")
            return
        }

        guard let cmd = lib.commandQueue.makeCommandBuffer() else {
            Issue.record("failed to make MTLCommandBuffer")
            return
        }

        let cache = PSOCache(library: lib)
        let pso = try cache.pipelineStateThrowing(for: "vector_add_f32")

        // Dispatch one thread per element. vector_add is Elementwise mode:
        // `program_id::<0>()` returns thread_position_in_grid.
        guard let enc = cmd.makeComputeCommandEncoder() else {
            Issue.record("failed to make encoder")
            return
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(aBuf, offset: 0, index: 0)
        enc.setBuffer(bBuf, offset: 0, index: 1)
        enc.setBuffer(cBuf, offset: 0, index: 2)

        let tpg = min(pso.maxTotalThreadsPerThreadgroup, 256)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        let tgsz = MTLSize(width: tpg, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgsz)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let out = cBuf.contents().bindMemory(to: Float.self, capacity: n)
        for i in 0..<n {
            #expect(out[i] == expected[i], "mismatch at index \(i): got \(out[i]), want \(expected[i])")
        }
    }
}
