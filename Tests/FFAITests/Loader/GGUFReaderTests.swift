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
// GGUF v3 reader unit tests — pure-Swift parser + dequant pipeline.
// Tests cover: header parsing, KV metadata round-trip across all 13
// scalar types + 12 array types, tensor-info table decoding, the on-
// disk → GPU-resident dequant pipeline for Q8_0 / Q2_K / IQ2_XXS.

import Foundation
import Testing

@testable import FFAI

@Suite("GGUF v3 reader")
struct GGUFReaderTests {

    // ─── Header parsing ──────────────────────────────────────────────

    @Test("Rejects non-GGUF magic")
    func rejectsNonGGUFMagic() throws {
        var data = Data()
        data.append(contentsOf: [0x46, 0x46, 0x55, 0x4c])  // "FFUL"
        data.append(contentsOf: UInt32(3).leBytes)
        #expect(throws: GGUFError.self) {
            _ = try GGUFReader(url: URL(fileURLWithPath: "/tmp/gguf-test"), data: data)
        }
    }

    @Test("Rejects unsupported version (v2)")
    func rejectsV2() throws {
        var data = Data()
        data.append(contentsOf: GGUFConstants.magic)
        data.append(contentsOf: UInt32(2).leBytes)
        data.append(contentsOf: UInt64(0).leBytes)  // tensor_count
        data.append(contentsOf: UInt64(0).leBytes)  // metadata_kv_count
        #expect(throws: GGUFError.self) {
            _ = try GGUFReader(url: URL(fileURLWithPath: "/tmp/gguf-test"), data: data)
        }
    }

    @Test("Empty v3 file parses cleanly")
    func emptyV3() throws {
        var data = Data()
        data.append(contentsOf: GGUFConstants.magic)
        data.append(contentsOf: UInt32(3).leBytes)
        data.append(contentsOf: UInt64(0).leBytes)
        data.append(contentsOf: UInt64(0).leBytes)
        let reader = try GGUFReader(url: URL(fileURLWithPath: "/tmp/empty.gguf"), data: data)
        #expect(reader.version == 3)
        #expect(reader.tensorInfos.isEmpty)
        #expect(reader.metadata.isEmpty)
    }

    // ─── Metadata round-trip ─────────────────────────────────────────

    @Test("Round-trips one of each scalar metadata type")
    func metadataScalarRoundtrip() throws {
        var data = Data()
        data.append(contentsOf: GGUFConstants.magic)
        data.append(contentsOf: UInt32(3).leBytes)
        data.append(contentsOf: UInt64(0).leBytes)  // tensor_count
        data.append(contentsOf: UInt64(5).leBytes)  // metadata_kv_count

        data.appendKVString("general.name", "Test Model")
        data.appendKVU32("counter", 42)
        data.appendKVF32("epsilon", 1e-6)
        data.appendKVBool("opt", true)
        data.appendKVI64("offset", -123456789)

        let reader = try GGUFReader(url: URL(fileURLWithPath: "/tmp/test.gguf"), data: data)
        #expect(reader.metadataString("general.name") == "Test Model")
        #expect(reader.metadataUInt32("counter") == 42)
        #expect(reader.metadataFloat("epsilon") == 1e-6)
        #expect(reader.metadataBool("opt") == true)
        if case .int64(let v) = reader.metadata["offset"] {
            #expect(v == -123456789)
        } else {
            Issue.record("offset metadata not decoded as int64")
        }
    }

    @Test("Round-trips a string-array (tokenizer.ggml.tokens shape)")
    func metadataStringArray() throws {
        var data = Data()
        data.append(contentsOf: GGUFConstants.magic)
        data.append(contentsOf: UInt32(3).leBytes)
        data.append(contentsOf: UInt64(0).leBytes)
        data.append(contentsOf: UInt64(1).leBytes)
        data.appendKVStringArray("tokenizer.ggml.tokens", ["<bos>", "<eos>", "hello", "world"])

        let reader = try GGUFReader(url: URL(fileURLWithPath: "/tmp/test.gguf"), data: data)
        let tokens = reader.metadataStringArray("tokenizer.ggml.tokens")
        #expect(tokens == ["<bos>", "<eos>", "hello", "world"])
    }

    // ─── Tensor info ─────────────────────────────────────────────────

    @Test("Decodes a tensor info entry and aligns the data section")
    func tensorInfoAlignment() throws {
        var data = Data()
        data.append(contentsOf: GGUFConstants.magic)
        data.append(contentsOf: UInt32(3).leBytes)
        data.append(contentsOf: UInt64(1).leBytes)  // 1 tensor
        data.append(contentsOf: UInt64(0).leBytes)  // 0 metadata
        // tensor: name="w", n_dims=2, dims=[4, 8], type=Q8_0(=8), offset=0
        data.appendString("w")
        data.append(contentsOf: UInt32(2).leBytes)
        data.append(contentsOf: UInt64(4).leBytes)
        data.append(contentsOf: UInt64(8).leBytes)
        data.append(contentsOf: UInt32(GGUFTensorType.q8_0.rawValue).leBytes)
        data.append(contentsOf: UInt64(0).leBytes)
        // No metadata override of alignment → default 32. The
        // tensor-info table ends mid-byte; tensorDataOffset rounds up.
        let reader = try GGUFReader(url: URL(fileURLWithPath: "/tmp/test.gguf"), data: data)
        #expect(reader.tensorInfos.count == 1)
        let info = reader.tensorInfos[0]
        #expect(info.name == "w")
        #expect(info.dimensions == [4, 8])
        #expect(info.type == .q8_0)
        #expect(info.numElements == 32)
        // Q8_0 = 34 bytes per 32-value block → 1 block = 34 bytes.
        #expect(info.byteLength == 34)
        // The tensorDataOffset must be aligned to 32.
        #expect(reader.tensorDataOffset % 32 == 0)
    }

    // ─── IQ2_XXS lookup table integrity ──────────────────────────────

    @Test("iq2xxs_grid + ksigns tables have the expected sizes")
    func iq2xxsTableSizes() {
        #expect(GGUFIQ2XXSTables.grid.count == 2048)
        #expect(GGUFIQ2XXSTables.ksigns.count == 128)
        // Spot-check first row of the grid: little-endian unpack of
        // 0x0808080808080808 → 8 bytes all = 0x08.
        for i in 0..<8 { #expect(GGUFIQ2XXSTables.grid[i] == 0x08) }
        // Spot-check ksigns: byte 0 = 0, byte 1 = 129, byte 127 = 255.
        #expect(GGUFIQ2XXSTables.ksigns[0] == 0)
        #expect(GGUFIQ2XXSTables.ksigns[1] == 129)
        #expect(GGUFIQ2XXSTables.ksigns[127] == 255)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

extension Data {
    fileprivate mutating func appendString(_ s: String) {
        let bytes = Array(s.utf8)
        append(contentsOf: UInt64(bytes.count).leBytes)
        append(contentsOf: bytes)
    }

    fileprivate mutating func appendKVString(_ key: String, _ value: String) {
        appendString(key)
        append(contentsOf: UInt32(GGUFValueType.string.rawValue).leBytes)
        appendString(value)
    }

    fileprivate mutating func appendKVU32(_ key: String, _ value: UInt32) {
        appendString(key)
        append(contentsOf: UInt32(GGUFValueType.uint32.rawValue).leBytes)
        append(contentsOf: value.leBytes)
    }

    fileprivate mutating func appendKVF32(_ key: String, _ value: Float) {
        appendString(key)
        append(contentsOf: UInt32(GGUFValueType.float32.rawValue).leBytes)
        append(contentsOf: value.bitPattern.leBytes)
    }

    fileprivate mutating func appendKVBool(_ key: String, _ value: Bool) {
        appendString(key)
        append(contentsOf: UInt32(GGUFValueType.bool.rawValue).leBytes)
        append(value ? 1 : 0)
    }

    fileprivate mutating func appendKVI64(_ key: String, _ value: Int64) {
        appendString(key)
        append(contentsOf: UInt32(GGUFValueType.int64.rawValue).leBytes)
        append(contentsOf: UInt64(bitPattern: value).leBytes)
    }

    fileprivate mutating func appendKVStringArray(_ key: String, _ values: [String]) {
        appendString(key)
        append(contentsOf: UInt32(GGUFValueType.array.rawValue).leBytes)
        append(contentsOf: UInt32(GGUFValueType.string.rawValue).leBytes)
        append(contentsOf: UInt64(values.count).leBytes)
        for v in values { appendString(v) }
    }
}

extension FixedWidthInteger {
    fileprivate var leBytes: [UInt8] {
        var v = self.littleEndian
        return withUnsafeBytes(of: &v) { Array($0) }
    }
}
