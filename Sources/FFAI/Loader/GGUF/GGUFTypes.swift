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
// GGUF v3 format types — header constants, value-type enum, quant-type
// enum, metadata KV value, error type.
//
// Canonical spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
// Reference impl: https://github.com/ggml-org/llama.cpp (gguf.h /
// ggml-quants.c — MIT). Whole format is little-endian.

import Foundation

// ─── Format constants ────────────────────────────────────────────────

public enum GGUFConstants {
    /// Magic bytes at byte offset 0 of every GGUF file. The four bytes
    /// `G G U F` in ASCII — `0x47 0x47 0x55 0x46`.
    public static let magic: [UInt8] = [0x47, 0x47, 0x55, 0x46]
    /// We support only v3 (the version stable since 2023). v2 (early
    /// 2023) shipped with `u32` array lengths instead of `u64`; v3 fixed
    /// that. Files older than v3 are extremely rare in the wild.
    public static let supportedVersion: UInt32 = 3
    /// Default tensor-data section alignment (overridable via the
    /// `general.alignment` u32 metadata key, but defaults are universal
    /// in practice).
    public static let defaultAlignment: UInt64 = 32
}

// ─── KV metadata value types ─────────────────────────────────────────

/// `GGUF_METADATA_VALUE_TYPE_*` — the u32 tag prefix on every metadata
/// value in the KV block.
public enum GGUFValueType: UInt32 {
    case uint8 = 0
    case int8 = 1
    case uint16 = 2
    case int16 = 3
    case uint32 = 4
    case int32 = 5
    case float32 = 6
    case bool = 7
    case string = 8
    case array = 9
    case uint64 = 10
    case int64 = 11
    case float64 = 12
}

/// One metadata value. Arrays carry their element-type discriminator
/// alongside the elements so a caller can index into `.array(.string([…]))`
/// or `.array(.int32([…]))` without re-querying the file.
public enum GGUFValue: Sendable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case uint64(UInt64)
    case int64(Int64)
    case float32(Float)
    case float64(Double)
    case bool(Bool)
    case string(String)
    case array(GGUFArrayValue)
}

/// Typed array — GGUF's `array` value carries a per-element type tag,
/// so the parser materialises one of these instead of a heterogeneous
/// `[GGUFValue]`. Saves a lot of casting work in the tokenizer adapter
/// + loader.
public enum GGUFArrayValue: Sendable {
    case uint8([UInt8])
    case int8([Int8])
    case uint16([UInt16])
    case int16([Int16])
    case uint32([UInt32])
    case int32([Int32])
    case uint64([UInt64])
    case int64([Int64])
    case float32([Float])
    case float64([Double])
    case bool([Bool])
    case string([String])
}

// ─── Tensor data types ───────────────────────────────────────────────

/// `GGML_TYPE_*` — the on-disk tensor quant format. The enum order
/// mirrors `ggml.h` so the u32 file values cast cleanly. New variants
/// appended at the end of `ggml.h` should be mirrored here in the same
/// order, with new raw values.
public enum GGUFTensorType: UInt32, Sendable {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q4_1 = 3
    // 4 and 5 are removed legacy quants (Q4_2 / Q4_3).
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q8_1 = 9
    case q2_K = 10
    case q3_K = 11
    case q4_K = 12
    case q5_K = 13
    case q6_K = 14
    case q8_K = 15
    case iq2_xxs = 16
    case iq2_xs = 17
    case iq3_xxs = 18
    case iq1_s = 19
    case iq4_nl = 20
    case iq3_s = 21
    case iq2_s = 22
    case iq4_xs = 23
    case i8 = 24
    case i16 = 25
    case i32 = 26
    case i64 = 27
    case f64 = 28
    case iq1_m = 29
    case bf16 = 30
    case tq1_0 = 34
    case tq2_0 = 35
    case mxfp4 = 39

    /// Block-byte size for a tensor of this type. Used to compute the
    /// total byte footprint of a tensor (`n_elements / block_size *
    /// bytes_per_block`).
    public var bytesPerBlock: Int {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .q4_0: return 18
        case .q4_1: return 20
        case .q5_0: return 22
        case .q5_1: return 24
        case .q8_0: return 34
        case .q8_1: return 36
        case .q2_K: return 84
        case .q3_K: return 110
        case .q4_K: return 144
        case .q5_K: return 176
        case .q6_K: return 210
        case .q8_K: return 292
        case .iq1_s: return 50
        case .iq1_m: return 56
        case .iq2_xxs: return 66
        case .iq2_xs: return 74
        case .iq2_s: return 82
        case .iq3_xxs: return 98
        case .iq3_s: return 110
        case .iq4_nl: return 18
        case .iq4_xs: return 136
        case .i8: return 1
        case .i16: return 2
        case .i32: return 4
        case .i64, .f64: return 8
        case .tq1_0: return 34
        case .tq2_0: return 68
        case .mxfp4: return 20
        }
    }

    /// Number of values per block. Legacy `Q*_0/1` use 32; all k-quants
    /// and i-quants use 256; primitive scalars are 1.
    public var blockSize: Int {
        switch self {
        case .f32, .f16, .bf16, .i8, .i16, .i32, .i64, .f64: return 1
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_nl, .mxfp4: return 32
        default: return 256
        }
    }
}

// ─── Tensor descriptor ───────────────────────────────────────────────

/// Static metadata for one tensor — populated from the tensor-info
/// table section. The actual bytes live at
/// `tensor_data_offset + dataOffset`, length
/// `(num_elements / blockSize) * bytesPerBlock`.
public struct GGUFTensorInfo: Sendable {
    public let name: String
    public let dimensions: [UInt64]
    public let type: GGUFTensorType
    /// Offset relative to the start of the tensor-data blob (NOT the
    /// file). Add `fileTensorDataOffset` to get the absolute file
    /// offset.
    public let dataOffset: UInt64

    public var numElements: UInt64 { dimensions.reduce(1, *) }

    /// On-disk byte length for this tensor's data.
    public var byteLength: Int {
        let blocks = Int(numElements) / type.blockSize
        return blocks * type.bytesPerBlock
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum GGUFError: Error, CustomStringConvertible {
    case badMagic
    case unsupportedVersion(UInt32)
    case truncated(at: String)
    case unknownValueType(UInt32, key: String?)
    case unknownTensorType(UInt32, tensor: String?)
    case duplicateKey(String)
    case duplicateTensorName(String)
    case stringNotUTF8(at: String)
    case unsupportedDequant(GGUFTensorType, tensor: String)
    case missingMetadataKey(String)

    public var description: String {
        switch self {
        case .badMagic:
            return "GGUF: first 4 bytes are not 'GGUF' (file is not a GGUF v3 checkpoint)"
        case .unsupportedVersion(let v):
            return "GGUF: file version \(v) is not supported (expected 3)"
        case .truncated(let at):
            return "GGUF: file truncated while reading \(at)"
        case .unknownValueType(let tag, let key):
            let where_ = key.map { " (key=\($0))" } ?? ""
            return "GGUF: unknown metadata value-type tag \(tag)\(where_)"
        case .unknownTensorType(let tag, let tensor):
            let where_ = tensor.map { " (tensor=\($0))" } ?? ""
            return "GGUF: unknown tensor type tag \(tag)\(where_)"
        case .duplicateKey(let k):
            return "GGUF: duplicate metadata key '\(k)'"
        case .duplicateTensorName(let n):
            return "GGUF: duplicate tensor name '\(n)'"
        case .stringNotUTF8(let at):
            return "GGUF: invalid UTF-8 in string at \(at)"
        case .unsupportedDequant(let t, let tensor):
            return
                "GGUF: dequant for \(t) is not yet implemented (tensor '\(tensor)')"
        case .missingMetadataKey(let k):
            return "GGUF: required metadata key '\(k)' is missing"
        }
    }
}
