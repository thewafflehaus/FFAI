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
// Tests for SafeTensorsWriter — round-trip write + re-load via
// SafeTensorsBundle.
//
// Each test writes one or more tensors with known values, then loads
// the resulting file with SafeTensorsFile / SafeTensorsBundle and
// asserts shape, dtype, and byte content are preserved exactly.

import Foundation
import Testing

@testable import FFAI

@Suite("SafeTensorsWriter")
struct SafeTensorsWriterTests {

    // ─── Single-tensor round-trip ─────────────────────────────────────

    @Test("f16 single tensor round-trips shape, dtype, and byte content")
    func singleTensorF16() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        // [2, 4] f16 tensor with known bit-pattern values.
        let shape = [2, 4]
        let values: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8]
        let bytes = rawData(values)

        let url = dir.appendingPathComponent("model.safetensors")
        let writer = SafeTensorsWriter(url: url)
        try writer.append(name: "linear.weight", dtype: .f16, shape: shape, bytes: bytes)
        try writer.finalize()

        // Re-load via SafeTensorsFile.
        let file = try SafeTensorsFile(url: url)
        #expect(file.entries.count == 1)
        let entry = try file.tensor(named: "linear.weight")

        #expect(entry.shape == shape)
        #expect(entry.dtype == .f16)

        // Byte-exact comparison: read back as Float16 and compare.
        let got = entry.toArray(as: Float16.self)
        #expect(got == values)
    }

    // ─── Multi-tensor round-trip ──────────────────────────────────────

    @Test("three tensors with mixed dtypes preserve ordered offset bookkeeping")
    func multiTensorMixedDtypes() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        // Three tensors: f32, bf16, u32 — different shapes and dtypes.
        let f32Vals: [Float] = [1.0, 2.0, 3.0, 4.0]
        let bf16Vals: [UInt16] = bf16Bits([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])
        let u32Vals: [UInt32] = [0xDEAD_BEEF, 0xCAFE_BABE]

        let url = dir.appendingPathComponent("model.safetensors")
        let writer = SafeTensorsWriter(url: url)
        try writer.append(
            name: "layer.a",
            dtype: .f32, shape: [2, 2], bytes: rawData(f32Vals))
        try writer.append(
            name: "layer.b",
            dtype: .bf16, shape: [2, 3], bytes: rawData(bf16Vals))
        try writer.append(
            name: "layer.c",
            dtype: .u32, shape: [1, 2], bytes: rawData(u32Vals))
        try writer.finalize()

        let file = try SafeTensorsFile(url: url)
        #expect(file.entries.count == 3)

        let a = try file.tensor(named: "layer.a")
        #expect(a.shape == [2, 2])
        #expect(a.dtype == .f32)
        #expect(a.toArray(as: Float.self) == f32Vals)

        let b = try file.tensor(named: "layer.b")
        #expect(b.shape == [2, 3])
        #expect(b.dtype == .bf16)
        #expect(b.toArray(as: UInt16.self) == bf16Vals)

        let c = try file.tensor(named: "layer.c")
        #expect(c.shape == [1, 2])
        #expect(c.dtype == .u32)
        #expect(c.toArray(as: UInt32.self) == u32Vals)
    }

    // ─── SafeTensorsBundle round-trip ─────────────────────────────────

    @Test("SafeTensorsBundle loads a writer-produced file via directory init")
    func bundleRoundTrip() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        let vals: [Float] = [10, 20, 30, 40, 50, 60]
        let url = dir.appendingPathComponent("model.safetensors")
        let writer = SafeTensorsWriter(url: url)
        try writer.append(
            name: "embed.weight", dtype: .f32,
            shape: [2, 3], bytes: rawData(vals))
        try writer.finalize()

        // Bundle discovery picks up "model.safetensors" via the standard
        // single-file path.
        let bundle = try SafeTensorsBundle(directory: dir)
        #expect(bundle.has("embed.weight"))
        let t = try bundle.tensor(named: "embed.weight")
        #expect(t.shape == [2, 3])
        #expect(t.toArray(as: Float.self) == vals)
    }

    // ─── Error cases ──────────────────────────────────────────────────

    @Test("duplicate name throws SafeTensorsWriterError.duplicateName")
    func duplicateNameThrows() throws {
        let dir = tempDir()
        defer { cleanUp(dir) }

        let bytes = rawData([Float(1.0)])
        let url = dir.appendingPathComponent("model.safetensors")
        let writer = SafeTensorsWriter(url: url)
        try writer.append(name: "x", dtype: .f32, shape: [1], bytes: bytes)
        do {
            try writer.append(name: "x", dtype: .f32, shape: [1], bytes: bytes)
            Issue.record("expected SafeTensorsWriterError.duplicateName")
        } catch let e as SafeTensorsWriterError {
            if case .duplicateName = e { /* expected */
            } else {
                Issue.record("wrong error: \(e)")
            }
        }
    }

    @Test("empty bytes throws SafeTensorsWriterError.emptyTensor")
    func emptyBytesThrows() {
        let dir = tempDir()
        defer { cleanUp(dir) }

        let url = dir.appendingPathComponent("model.safetensors")
        let writer = SafeTensorsWriter(url: url)
        do {
            try writer.append(name: "empty", dtype: .f32, shape: [0], bytes: Data())
            Issue.record("expected SafeTensorsWriterError.emptyTensor")
        } catch let e as SafeTensorsWriterError {
            if case .emptyTensor = e { /* expected */
            } else {
                Issue.record("wrong error: \(e)")
            }
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-writer-\(UUID().uuidString)")
        // Create immediately so write(to:options:.atomic) can create its
        // temp file in the same directory.
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func cleanUp(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Reinterpret a typed array as raw Data (little-endian on ARM64).
    private func rawData<T>(_ values: [T]) -> Data {
        values.withUnsafeBufferPointer { buf in
            Data(buffer: buf)
        }
    }

    /// Produce bf16 bit patterns for a sequence of Float values.
    /// bf16 = top 16 bits of f32 (round-to-nearest).
    private func bf16Bits(_ floats: [Float]) -> [UInt16] {
        floats.map { f -> UInt16 in
            let bits = f.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            return UInt16(rounded >> 16)
        }
    }
}
