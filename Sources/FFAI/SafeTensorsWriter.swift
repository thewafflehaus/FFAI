// SafeTensorsWriter — pure-Swift writer for the safetensors v2 format.
//
// Mirrors exactly the format that SafeTensors.swift (the loader) reads:
//
//   [8 bytes: u64-LE header length N]
//   [N bytes: UTF-8 JSON header]
//   [tensor data, contiguous, in append order]
//
// JSON schema (written with keys sorted alphabetically, matching mlx-lm
// convention so diff tooling produces stable output):
//
//   {
//     "__metadata__": { "format": "pt", "ffai_version": "..." },
//     "tensor_name": {
//       "data_offsets": [start, end],
//       "dtype": "BF16" | "F16" | "F32" | "U32" | ...,
//       "shape": [d0, d1, ...]
//     }
//   }
//
// Usage:
//
//   let w = SafeTensorsWriter(url: outputURL)
//   w.append(name: "a.weight", dtype: .bf16, shape: [16, 64], bytes: rawData)
//   try w.finalize()

import Foundation
import Metal

/// Errors surfaced by SafeTensorsWriter.
public enum SafeTensorsWriterError: Error, CustomStringConvertible {
    case emptyTensor(String)
    case duplicateName(String)
    case jsonSerializationFailed
    case writeFailed(URL, Error)

    public var description: String {
        switch self {
        case .emptyTensor(let n):
            return "SafeTensorsWriter: tensor \"\(n)\" has zero bytes — cannot write empty tensor"
        case .duplicateName(let n):
            return "SafeTensorsWriter: tensor \"\(n)\" appended twice"
        case .jsonSerializationFailed:
            return "SafeTensorsWriter: JSON header serialization failed"
        case .writeFailed(let u, let e):
            return "SafeTensorsWriter: write to \(u.path) failed: \(e)"
        }
    }
}

/// Accumulates tensors in memory and writes them as a single safetensors file.
///
/// Not thread-safe — all `append` calls must come from a single thread
/// before `finalize` is called.
public final class SafeTensorsWriter {
    private let url: URL

    // Each entry records: name, dtype, shape, raw bytes. We buffer all
    // tensor data in-memory so we can compute data_offsets before writing
    // the header. For a 7B model at int4 this is ~3.5 GB RAM — acceptable
    // for a one-shot convert workflow; not suitable for streaming.
    private struct Pending {
        let name: String
        let dtype: DType
        let shape: [Int]
        let bytes: Data
    }

    private var pending: [Pending] = []
    private var usedNames: Set<String> = []

    public init(url: URL) {
        self.url = url
    }

    /// Accumulate one tensor. `bytes` must be raw little-endian tensor data
    /// in `dtype` layout. The tensor is buffered in memory until `finalize`.
    public func append(name: String, dtype: DType, shape: [Int], bytes: Data) throws {
        guard !bytes.isEmpty else { throw SafeTensorsWriterError.emptyTensor(name) }
        guard !usedNames.contains(name) else { throw SafeTensorsWriterError.duplicateName(name) }
        usedNames.insert(name)
        pending.append(Pending(name: name, dtype: dtype, shape: shape, bytes: bytes))
    }

    /// Write the safetensors file. Clears the pending queue after writing.
    /// Throws on JSON serialization failure or I/O error.
    public func finalize() throws {
        // Build header JSON dict. Keys must be sorted alphabetically so
        // the output is deterministic (mlx-lm also sorts alphabetically).
        var headerDict: [String: Any] = [:]

        // __metadata__ block — written first by convention, sorted to top
        // in the final JSON because "_" sorts before letters in ASCII.
        headerDict["__metadata__"] = [
            "format": "pt",
            "ffai_version": FFAI.version,
        ]

        // Compute data_offsets as we walk the pending list in order.
        var offset = 0
        for p in pending {
            let start = offset
            let end = offset + p.bytes.count
            headerDict[p.name] = [
                // Keys sorted: data_offsets < dtype < shape
                "data_offsets": [start, end],
                "dtype": safeTensorsDTypeName(p.dtype),
                "shape": p.shape,
            ] as [String: Any]
            offset = end
        }

        // Serialize JSON with sorted keys (NSJSONSerialization respects
        // `.sortedKeys` on macOS 10.13+).
        let jsonData: Data
        do {
            jsonData = try JSONSerialization.data(
                withJSONObject: headerDict,
                options: [.sortedKeys]
            )
        } catch {
            throw SafeTensorsWriterError.jsonSerializationFailed
        }

        // Assemble the file in-memory:
        //   [8 bytes u64-LE header length]
        //   [N bytes UTF-8 JSON]
        //   [tensor data, contiguous]
        var file = Data()
        var headerLen = UInt64(jsonData.count)
        withUnsafeBytes(of: &headerLen) { file.append(contentsOf: $0) }
        file.append(jsonData)
        for p in pending {
            file.append(p.bytes)
        }

        // Write atomically so partial writes don't leave a corrupt file.
        do {
            try file.write(to: url, options: .atomic)
        } catch {
            throw SafeTensorsWriterError.writeFailed(url, error)
        }

        pending.removeAll()
        usedNames.removeAll()
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    /// Map DType to the uppercase safetensors dtype tag that the loader
    /// (SafeTensors.swift → DType.fromSafeTensors) reads back correctly.
    private func safeTensorsDTypeName(_ dtype: DType) -> String {
        switch dtype {
        case .f32:  return "F32"
        case .f16:  return "F16"
        case .bf16: return "BF16"
        case .i32:  return "I32"
        case .u32:  return "U32"
        case .i8:   return "I8"
        case .u8:   return "U8"
        case .i64:  return "I64"
        case .u64:  return "U64"
        }
    }
}
