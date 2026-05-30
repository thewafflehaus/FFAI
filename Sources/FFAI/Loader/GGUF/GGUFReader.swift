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
// GGUF v3 file reader — header + metadata KV + tensor info table.
//
// All scalars are little-endian. Parses lazily where possible: the
// tensor info table is decoded eagerly (small + needed for the load
// dispatch), but tensor data stays mmap'd; the loader copies / dequants
// individual tensors on demand.
//
// Adapts the spec from https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
// and the canonical C reader in ggml.c (MIT). Pure Swift; no FFI.

import Foundation

/// One opened GGUF file — header + metadata KV + tensor info, plus a
/// memory-mapped handle to the raw bytes for on-demand tensor reads.
public final class GGUFReader {
    /// Backing file URL.
    public let url: URL
    /// File version (must be 3 — earlier versions throw at parse).
    public let version: UInt32
    /// Tensor-data section alignment (default 32, may be overridden by
    /// the `general.alignment` metadata key).
    public let alignment: UInt64
    /// Tensor-data section absolute file offset.
    public let tensorDataOffset: UInt64
    /// Metadata KV block.
    public let metadata: [String: GGUFValue]
    /// Tensor info table (ordered as stored on disk).
    public let tensorInfos: [GGUFTensorInfo]
    /// Name → index into `tensorInfos` for O(1) lookup.
    public let tensorIndex: [String: Int]
    /// Memory-mapped backing data. Held to keep the mapping alive
    /// for the lifetime of any tensor read.
    private let mapped: Data

    // ─── Init ──────────────────────────────────────────────────────────

    public convenience init(url: URL) throws {
        // `.mappedIfSafe` lets Foundation fall back to a regular read
        // if the filesystem doesn't support mmap (network shares,
        // some FUSE mounts). Worst case is a slow first read; we
        // tolerate it.
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(url: url, data: data)
    }

    /// In-memory init — useful for tests that synthesise a GGUF
    /// header in `Data` without writing a temp file.
    public init(url: URL, data: Data) throws {
        self.url = url
        self.mapped = data
        var cursor = GGUFCursor(data: data)

        // ── Header ──
        let magic = try cursor.readBytes(GGUFConstants.magic.count, at: "magic")
        guard magic == GGUFConstants.magic else {
            throw GGUFError.badMagic
        }
        let version: UInt32 = try cursor.readLE(at: "version")
        guard version == GGUFConstants.supportedVersion else {
            throw GGUFError.unsupportedVersion(version)
        }
        self.version = version
        let tensorCount: UInt64 = try cursor.readLE(at: "tensor_count")
        let metadataCount: UInt64 = try cursor.readLE(at: "metadata_kv_count")

        // ── Metadata KV block ──
        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(Int(metadataCount))
        for _ in 0..<metadataCount {
            let key = try cursor.readString(at: "metadata key")
            if metadata[key] != nil {
                throw GGUFError.duplicateKey(key)
            }
            let value = try GGUFReader.readValue(cursor: &cursor, key: key)
            metadata[key] = value
        }
        self.metadata = metadata

        // Optional override of the tensor-data section alignment.
        var alignment = GGUFConstants.defaultAlignment
        if case .uint32(let v) = metadata["general.alignment"] {
            alignment = UInt64(v)
        }
        self.alignment = alignment

        // ── Tensor info table ──
        var infos: [GGUFTensorInfo] = []
        infos.reserveCapacity(Int(tensorCount))
        var seenNames = Set<String>()
        seenNames.reserveCapacity(Int(tensorCount))
        var index: [String: Int] = [:]
        index.reserveCapacity(Int(tensorCount))
        for i in 0..<tensorCount {
            let name = try cursor.readString(at: "tensor name")
            if !seenNames.insert(name).inserted {
                throw GGUFError.duplicateTensorName(name)
            }
            let nDims: UInt32 = try cursor.readLE(at: "tensor n_dims")
            var dims: [UInt64] = []
            dims.reserveCapacity(Int(nDims))
            for _ in 0..<nDims {
                dims.append(try cursor.readLE(at: "tensor dim"))
            }
            let typeTag: UInt32 = try cursor.readLE(at: "tensor type")
            guard let type = GGUFTensorType(rawValue: typeTag) else {
                throw GGUFError.unknownTensorType(typeTag, tensor: name)
            }
            let dataOffset: UInt64 = try cursor.readLE(at: "tensor data offset")
            infos.append(
                GGUFTensorInfo(name: name, dimensions: dims, type: type, dataOffset: dataOffset)
            )
            index[name] = Int(i)
        }
        self.tensorInfos = infos
        self.tensorIndex = index

        // ── Padding to alignment boundary ──
        // The tensor-data section starts at the next `alignment`-aligned
        // file offset after the tensor info table. Compute it from the
        // cursor's current position; the padding bytes themselves are
        // not validated (some writers leave garbage, llama.cpp writes
        // zeros — either is spec-conformant).
        let here = UInt64(cursor.offset)
        self.tensorDataOffset = ((here + alignment - 1) / alignment) * alignment
    }

    // ─── Tensor data read ─────────────────────────────────────────────

    /// Return the raw on-disk bytes for tensor `name`. The returned
    /// `Data` shares the underlying mmap'd storage when possible (zero
    /// copy on macOS / iOS for files >16 KB).
    public func rawBytes(named name: String) throws -> Data {
        guard let idx = tensorIndex[name] else {
            throw GGUFError.missingMetadataKey("tensor:\(name)")
        }
        let info = tensorInfos[idx]
        let start = Int(tensorDataOffset + info.dataOffset)
        let end = start + info.byteLength
        return mapped.subdata(in: start..<end)
    }

    /// Convenience: get a metadata value, casted to a specific type.
    /// Returns nil if absent or the type doesn't match.
    public func metadataString(_ key: String) -> String? {
        if case .string(let s) = metadata[key] { return s }
        return nil
    }

    public func metadataUInt32(_ key: String) -> UInt32? {
        switch metadata[key] {
        case .uint32(let v): return v
        case .int32(let v) where v >= 0: return UInt32(v)
        case .uint64(let v) where v <= UInt32.max: return UInt32(v)
        default: return nil
        }
    }

    public func metadataFloat(_ key: String) -> Float? {
        switch metadata[key] {
        case .float32(let v): return v
        case .float64(let v): return Float(v)
        default: return nil
        }
    }

    public func metadataBool(_ key: String) -> Bool? {
        if case .bool(let b) = metadata[key] { return b }
        return nil
    }

    public func metadataStringArray(_ key: String) -> [String]? {
        if case .array(.string(let arr)) = metadata[key] { return arr }
        return nil
    }

    /// Integer array accessor — coerces any of the integer-typed GGUF
    /// array kinds (i32 / u32 / i64 / u64 / i16 / u16 / i8 / u8) to
    /// `[Int]`. Used for per-layer parameter arrays like
    /// `deepseek4.attention.compress_ratios`.
    public func metadataIntArray(_ key: String) -> [Int]? {
        switch metadata[key] {
        case .array(.int32(let a)): return a.map { Int($0) }
        case .array(.uint32(let a)): return a.map { Int($0) }
        case .array(.int64(let a)): return a.map { Int($0) }
        case .array(.uint64(let a)): return a.map { Int($0) }
        case .array(.int16(let a)): return a.map { Int($0) }
        case .array(.uint16(let a)): return a.map { Int($0) }
        case .array(.int8(let a)): return a.map { Int($0) }
        case .array(.uint8(let a)): return a.map { Int($0) }
        default: return nil
        }
    }

    // ─── Value-type decoder (internal) ────────────────────────────────

    private static func readValue(cursor: inout GGUFCursor, key: String) throws -> GGUFValue {
        let tag: UInt32 = try cursor.readLE(at: "value-type tag (key=\(key))")
        guard let kind = GGUFValueType(rawValue: tag) else {
            throw GGUFError.unknownValueType(tag, key: key)
        }
        return try readScalarOrArray(cursor: &cursor, kind: kind, key: key)
    }

    private static func readScalarOrArray(
        cursor: inout GGUFCursor, kind: GGUFValueType, key: String
    ) throws -> GGUFValue {
        switch kind {
        case .uint8: return .uint8(try cursor.readLE(at: "u8 (\(key))"))
        case .int8: return .int8(Int8(bitPattern: try cursor.readLE(at: "i8 (\(key))")))
        case .uint16: return .uint16(try cursor.readLE(at: "u16 (\(key))"))
        case .int16:
            let raw: UInt16 = try cursor.readLE(at: "i16 (\(key))")
            return .int16(Int16(bitPattern: raw))
        case .uint32: return .uint32(try cursor.readLE(at: "u32 (\(key))"))
        case .int32:
            let raw: UInt32 = try cursor.readLE(at: "i32 (\(key))")
            return .int32(Int32(bitPattern: raw))
        case .uint64: return .uint64(try cursor.readLE(at: "u64 (\(key))"))
        case .int64:
            let raw: UInt64 = try cursor.readLE(at: "i64 (\(key))")
            return .int64(Int64(bitPattern: raw))
        case .float32:
            let raw: UInt32 = try cursor.readLE(at: "f32 (\(key))")
            return .float32(Float(bitPattern: raw))
        case .float64:
            let raw: UInt64 = try cursor.readLE(at: "f64 (\(key))")
            return .float64(Double(bitPattern: raw))
        case .bool:
            let b: UInt8 = try cursor.readLE(at: "bool (\(key))")
            return .bool(b != 0)
        case .string:
            return .string(try cursor.readString(at: "string (\(key))"))
        case .array:
            let elemTag: UInt32 = try cursor.readLE(at: "array-elem-type (\(key))")
            guard let elemKind = GGUFValueType(rawValue: elemTag) else {
                throw GGUFError.unknownValueType(elemTag, key: "\(key)[]")
            }
            let n: UInt64 = try cursor.readLE(at: "array-len (\(key))")
            return .array(try readArrayElements(cursor: &cursor, kind: elemKind, count: n, key: key))
        }
    }

    private static func readArrayElements(
        cursor: inout GGUFCursor, kind: GGUFValueType, count: UInt64, key: String
    ) throws -> GGUFArrayValue {
        let n = Int(count)
        switch kind {
        case .uint8:
            var out = [UInt8](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readLE(at: "u8[] (\(key))")) }
            return .uint8(out)
        case .int8:
            var out = [Int8](); out.reserveCapacity(n)
            for _ in 0..<n {
                out.append(Int8(bitPattern: try cursor.readLE(at: "i8[] (\(key))")))
            }
            return .int8(out)
        case .uint16:
            var out = [UInt16](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readLE(at: "u16[] (\(key))")) }
            return .uint16(out)
        case .int16:
            var out = [Int16](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt16 = try cursor.readLE(at: "i16[] (\(key))")
                out.append(Int16(bitPattern: raw))
            }
            return .int16(out)
        case .uint32:
            var out = [UInt32](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readLE(at: "u32[] (\(key))")) }
            return .uint32(out)
        case .int32:
            var out = [Int32](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt32 = try cursor.readLE(at: "i32[] (\(key))")
                out.append(Int32(bitPattern: raw))
            }
            return .int32(out)
        case .uint64:
            var out = [UInt64](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readLE(at: "u64[] (\(key))")) }
            return .uint64(out)
        case .int64:
            var out = [Int64](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt64 = try cursor.readLE(at: "i64[] (\(key))")
                out.append(Int64(bitPattern: raw))
            }
            return .int64(out)
        case .float32:
            var out = [Float](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt32 = try cursor.readLE(at: "f32[] (\(key))")
                out.append(Float(bitPattern: raw))
            }
            return .float32(out)
        case .float64:
            var out = [Double](); out.reserveCapacity(n)
            for _ in 0..<n {
                let raw: UInt64 = try cursor.readLE(at: "f64[] (\(key))")
                out.append(Double(bitPattern: raw))
            }
            return .float64(out)
        case .bool:
            var out = [Bool](); out.reserveCapacity(n)
            for _ in 0..<n {
                let b: UInt8 = try cursor.readLE(at: "bool[] (\(key))")
                out.append(b != 0)
            }
            return .bool(out)
        case .string:
            var out = [String](); out.reserveCapacity(n)
            for _ in 0..<n { out.append(try cursor.readString(at: "string[] (\(key))")) }
            return .string(out)
        case .array:
            // Nested arrays are not supported by GGUF v3. Future-proof
            // by throwing — the spec marks array-of-array as reserved.
            throw GGUFError.unknownValueType(
                GGUFValueType.array.rawValue, key: "\(key)[] (nested array)"
            )
        }
    }
}

// ─── Cursor — bounds-checked LE reader ───────────────────────────────

/// Forward-only byte cursor with bounds checking. Crashes-as-errors:
/// every off-the-end read throws `GGUFError.truncated`. Used only
/// during the parse phase (the read-back path uses direct slicing).
struct GGUFCursor {
    let data: Data
    var offset: Int

    init(data: Data) {
        self.data = data
        self.offset = 0
    }

    mutating func readBytes(_ count: Int, at where_: String) throws -> [UInt8] {
        guard offset + count <= data.count else {
            throw GGUFError.truncated(at: where_)
        }
        let slice = data[offset..<offset + count]
        offset += count
        return Array(slice)
    }

    mutating func readLE<T: FixedWidthInteger & UnsignedInteger>(at where_: String) throws -> T {
        let bytes = MemoryLayout<T>.size
        guard offset + bytes <= data.count else {
            throw GGUFError.truncated(at: where_)
        }
        var value: T = 0
        for i in 0..<bytes {
            value |= T(data[offset + i]) << (8 * i)
        }
        offset += bytes
        return value
    }

    mutating func readString(at where_: String) throws -> String {
        let length: UInt64 = try readLE(at: "\(where_) length")
        let bytes = try readBytes(Int(length), at: where_)
        guard let s = String(bytes: bytes, encoding: .utf8) else {
            throw GGUFError.stringNotUTF8(at: where_)
        }
        return s
    }
}
