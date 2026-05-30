// Copyright 2026 Tom Turney (@TheTom)
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
// GGUF on-disk → GPU-resident dequant pipeline. For each supported
// quant format, splits the packed on-disk blocks into the GPU-resident
// tensor layout the metaltile dequant kernel expects, then dispatches
// the kernel and returns a host-readable `Tensor`.
//
// The CPU split is a one-pass scan over the raw GGUF bytes — fp16
// scales are converted to f32 by host code so the kernel doesn't have
// to bit-cast inside the DSL.

import Foundation
import Metal

enum GGUFDequant {
    // ─── Q8_0 ──────────────────────────────────────────────────────────

    /// Block: `[fp16 d (2 B); int8 qs[32] (32 B)]` = 34 B per 32 values.
    static let q8_0BlockBytes = 34
    static let q8_0BlockValues = 32

    /// Split each on-disk Q8_0 block into the GPU-resident tensors the
    /// kernel expects: a contiguous `[n_blocks * 32]` byte buffer of
    /// int8 quants (kernel sign-reconstructs via `select`) and a
    /// `[n_blocks]` f32 buffer of fp16-converted block super-scales.
    static func dequantQ8_0(
        rawBlocks: Data, nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(nValues % q8_0BlockValues == 0)
        let nBlocks = nValues / q8_0BlockValues
        precondition(
            rawBlocks.count >= nBlocks * q8_0BlockBytes,
            "GGUFDequant.Q8_0: rawBlocks too short for \(nBlocks) blocks")

        // Build the two GPU-resident tensors via CPU buffers; copy
        // once at the end into shared-storage MTLBuffers.
        var qsBytes = [UInt8](repeating: 0, count: nBlocks * q8_0BlockValues)
        var scales = [Float](repeating: 0, count: nBlocks)
        rawBlocks.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            for b in 0..<nBlocks {
                let blockBase = base.advanced(by: b * q8_0BlockBytes)
                let dBits = UInt16(blockBase[0]) | (UInt16(blockBase[1]) << 8)
                scales[b] = Float(Float16(bitPattern: dBits))
                // memcpy the 32 quant bytes
                let dst = qsBytes.withUnsafeMutableBufferPointer { $0.baseAddress! }
                dst.advanced(by: b * q8_0BlockValues).update(
                    from: blockBase.advanced(by: 2), count: q8_0BlockValues)
            }
        }

        let qsTensor = makeU8Tensor(bytes: qsBytes, device: device)
        let scalesTensor = makeF32Tensor(values: scales, device: device)
        return Ops.ggufDequantQ8_0(
            qsSigned: qsTensor, scales: scalesTensor,
            nValues: nValues, outDtype: outDtype,
            on: cmd)
    }

    // ─── Q2_K ──────────────────────────────────────────────────────────

    /// Block: `[u8 scales[16]; u8 qs[64]; fp16 d; fp16 dmin]` = 84 B
    /// per 256 values.
    static let q2_KBlockBytes = 84
    static let q2_KBlockValues = 256

    static func dequantQ2_K(
        rawBlocks: Data, nValues: Int, outDtype: DType,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(nValues % q2_KBlockValues == 0)
        let nBlocks = nValues / q2_KBlockValues
        precondition(
            rawBlocks.count >= nBlocks * q2_KBlockBytes,
            "GGUFDequant.Q2_K: rawBlocks too short for \(nBlocks) blocks")

        // 4 GPU-resident split buffers per block:
        //   qsPacked: [n_blocks * 16] u32  — 64 packed-quant bytes as 16 LE u32 words
        //   scales:   [n_blocks * 16] u8   — raw 4-bit scale + 4-bit min nibble pairs
        //   dF32:     [n_blocks]      f32
        //   dminF32:  [n_blocks]      f32
        var qsPacked = [UInt32](repeating: 0, count: nBlocks * 16)
        var scales = [UInt8](repeating: 0, count: nBlocks * 16)
        var dF32 = [Float](repeating: 0, count: nBlocks)
        var dminF32 = [Float](repeating: 0, count: nBlocks)

        rawBlocks.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            for b in 0..<nBlocks {
                let blockBase = base.advanced(by: b * q2_KBlockBytes)
                // scales[16] at bytes 0..16
                for s in 0..<16 { scales[b * 16 + s] = blockBase[s] }
                // qs[64] at bytes 16..80, re-laid as 16 LE u32 words
                for w in 0..<16 {
                    let off = 16 + w * 4
                    qsPacked[b * 16 + w] =
                        UInt32(blockBase[off]) | (UInt32(blockBase[off + 1]) << 8)
                        | (UInt32(blockBase[off + 2]) << 16) | (UInt32(blockBase[off + 3]) << 24)
                }
                // d at bytes 80..82, dmin at 82..84
                let dBits = UInt16(blockBase[80]) | (UInt16(blockBase[81]) << 8)
                let dminBits = UInt16(blockBase[82]) | (UInt16(blockBase[83]) << 8)
                dF32[b] = Float(Float16(bitPattern: dBits))
                dminF32[b] = Float(Float16(bitPattern: dminBits))
            }
        }

        let qsTensor = makeU32Tensor(values: qsPacked, device: device)
        let scalesTensor = makeU8Tensor(bytes: scales, device: device)
        let dTensor = makeF32Tensor(values: dF32, device: device)
        let dminTensor = makeF32Tensor(values: dminF32, device: device)
        return Ops.ggufDequantQ2_K(
            qsPacked: qsTensor, scales: scalesTensor,
            dF32: dTensor, dminF32: dminTensor,
            nValues: nValues, outDtype: outDtype,
            on: cmd)
    }

    // ─── IQ2_XXS ───────────────────────────────────────────────────────

    /// Block: `[fp16 d (2 B); u16 qs[32] (64 B)]` = 66 B per 256 values.
    static let iq2_xxsBlockBytes = 66
    static let iq2_xxsBlockValues = 256

    static func dequantIQ2_XXS(
        rawBlocks: Data, nValues: Int, outDtype: DType,
        gridTensor: Tensor, signsTensor: Tensor,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(nValues % iq2_xxsBlockValues == 0)
        let nBlocks = nValues / iq2_xxsBlockValues
        precondition(
            rawBlocks.count >= nBlocks * iq2_xxsBlockBytes,
            "GGUFDequant.IQ2_XXS: rawBlocks too short for \(nBlocks) blocks")

        var qsU32 = [UInt32](repeating: 0, count: nBlocks * 16)
        var dF32 = [Float](repeating: 0, count: nBlocks)

        rawBlocks.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            for b in 0..<nBlocks {
                let blockBase = base.advanced(by: b * iq2_xxsBlockBytes)
                let dBits = UInt16(blockBase[0]) | (UInt16(blockBase[1]) << 8)
                dF32[b] = Float(Float16(bitPattern: dBits))
                // qs[32] u16 (64 bytes) starting at byte 2, re-laid as 16 LE u32 words
                for w in 0..<16 {
                    let off = 2 + w * 4
                    qsU32[b * 16 + w] =
                        UInt32(blockBase[off]) | (UInt32(blockBase[off + 1]) << 8)
                        | (UInt32(blockBase[off + 2]) << 16) | (UInt32(blockBase[off + 3]) << 24)
                }
            }
        }

        let qsTensor = makeU32Tensor(values: qsU32, device: device)
        let dTensor = makeF32Tensor(values: dF32, device: device)
        return Ops.ggufDequantIQ2_XXS(
            qsU32: qsTensor, dF32: dTensor,
            grid: gridTensor, signs: signsTensor,
            nValues: nValues, outDtype: outDtype,
            on: cmd)
    }

    // ─── Shared LUT cache ──────────────────────────────────────────────

    /// One-shot upload of the IQ2_XXS lookup tables. Cached so the
    /// 2048+128-byte upload happens at most once per process.
    /// `NSLock`-guarded for the rare cross-thread first-touch case;
    /// after first init the read is a fast pointer compare.
    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var iq2xxsTablesCache:
        (grid: Tensor, signs: Tensor, device: Device)? = nil

    static func iq2xxsTables(device: Device) -> (grid: Tensor, signs: Tensor) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let cached = iq2xxsTablesCache, cached.device === device {
            return (cached.grid, cached.signs)
        }
        let grid = makeU8Tensor(bytes: GGUFIQ2XXSTables.grid, device: device)
        let signs = makeU8Tensor(bytes: GGUFIQ2XXSTables.ksigns, device: device)
        iq2xxsTablesCache = (grid, signs, device)
        return (grid, signs)
    }

    // ─── Buffer construction helpers ───────────────────────────────────

    private static func makeU8Tensor(bytes: [UInt8], device: Device) -> Tensor {
        let buf = device.makeBuffer(length: max(bytes.count, 1))
        bytes.withUnsafeBufferPointer { src in
            buf.contents().copyMemory(
                from: UnsafeRawPointer(src.baseAddress!), byteCount: bytes.count)
        }
        return Tensor(buffer: buf, offset: 0, shape: [bytes.count], dtype: .u8)
    }

    private static func makeU32Tensor(values: [UInt32], device: Device) -> Tensor {
        let buf = device.makeBuffer(length: max(values.count * 4, 4))
        values.withUnsafeBufferPointer { src in
            buf.contents().copyMemory(
                from: UnsafeRawPointer(src.baseAddress!), byteCount: values.count * 4)
        }
        return Tensor(buffer: buf, offset: 0, shape: [values.count], dtype: .u32)
    }

    private static func makeF32Tensor(values: [Float], device: Device) -> Tensor {
        let buf = device.makeBuffer(length: max(values.count * 4, 4))
        values.withUnsafeBufferPointer { src in
            buf.contents().copyMemory(
                from: UnsafeRawPointer(src.baseAddress!), byteCount: values.count * 4)
        }
        return Tensor(buffer: buf, offset: 0, shape: [values.count], dtype: .f32)
    }
}
