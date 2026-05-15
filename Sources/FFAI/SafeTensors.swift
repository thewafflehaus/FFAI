// SafeTensors loader.
//
// SafeTensors format:
//   bytes [0..8):   u64 little-endian header length
//   bytes [8..8+N): UTF-8 JSON header
//   bytes [8+N..]:  raw tensor bytes (concatenated, indexed by header offsets)
//
// JSON header schema:
//   {
//     "tensor_name": {
//       "dtype": "F32" | "F16" | "BF16" | ...,
//       "shape": [d0, d1, ...],
//       "data_offsets": [start, end]   // bytes relative to data section
//     },
//     ...,
//     "__metadata__": { ... optional ... }
//   }
//
// Phase 2: full file load via mmap. Wraps the mmap'd region in an
// MTLBuffer using `device.makeBuffer(bytesNoCopy:)` so Metal sees the
// exact same pages without an extra copy.

import Foundation
import Metal

public enum SafeTensorsError: Error, CustomStringConvertible {
    case fileNotFound(URL)
    case headerTooSmall
    case headerJSONMalformed
    case headerEntryMalformed(String)
    case unsupportedDType(String, key: String)
    case mmapFailed(URL)
    case mtlBufferFailed
    case missingTensor(String)

    public var description: String {
        switch self {
        case .fileNotFound(let u): return "safetensors file not found: \(u.path)"
        case .headerTooSmall: return "safetensors header smaller than 8 bytes"
        case .headerJSONMalformed: return "safetensors JSON header malformed"
        case .headerEntryMalformed(let k): return "malformed entry for tensor \"\(k)\""
        case .unsupportedDType(let dt, let k): return "unsupported dtype \(dt) for tensor \"\(k)\""
        case .mmapFailed(let u): return "mmap failed for \(u.path)"
        case .mtlBufferFailed: return "MTLDevice.makeBuffer(bytesNoCopy:) returned nil"
        case .missingTensor(let k): return "tensor \"\(k)\" not present in safetensors bundle"
        }
    }
}

/// One safetensors file mmap'd into one MTLBuffer (zero-copy on
/// Apple Silicon; Metal sees the same pages as the page cache).
public final class SafeTensorsFile: @unchecked Sendable {
    public struct Entry: Sendable {
        public let name: String
        public let dtype: DType
        public let shape: [Int]
        /// Byte offset within `buffer` (already accounts for the header).
        public let bufferOffset: Int
    }

    public let url: URL
    public let buffer: MTLBuffer
    public let entries: [String: Entry]

    public init(url: URL, device: Device = .shared) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SafeTensorsError.fileNotFound(url)
        }
        // HF cache stores actual blobs symlinked from snapshot dirs.
        // FileManager.attributesOfItem doesn't follow symlinks (returns
        // the symlink's tiny size), so resolve to the underlying blob
        // before stat-ing.
        let resolvedURL = url.resolvingSymlinksInPath()

        let fh = try FileHandle(forReadingFrom: resolvedURL)
        defer { try? fh.close() }

        // Read header length
        let lenData = try fh.read(upToCount: 8) ?? Data()
        guard lenData.count == 8 else { throw SafeTensorsError.headerTooSmall }
        let headerLen: UInt64 = lenData.withUnsafeBytes { $0.load(as: UInt64.self) }

        // Read JSON header
        let headerData = try fh.read(upToCount: Int(headerLen)) ?? Data()
        guard headerData.count == Int(headerLen) else {
            throw SafeTensorsError.headerJSONMalformed
        }
        let parsedJSON: Any
        do {
            parsedJSON = try JSONSerialization.jsonObject(with: headerData)
        } catch {
            throw SafeTensorsError.headerJSONMalformed
        }
        guard let json = parsedJSON as? [String: Any] else {
            throw SafeTensorsError.headerJSONMalformed
        }

        // mmap the whole file (resolved path so we get the blob size).
        let attrs = try FileManager.default.attributesOfItem(atPath: resolvedURL.path)
        guard let fileSize = attrs[.size] as? NSNumber else {
            throw SafeTensorsError.mmapFailed(resolvedURL)
        }
        let totalBytes = Int(truncating: fileSize)

        let fd = open(resolvedURL.path, O_RDONLY)
        guard fd >= 0 else { throw SafeTensorsError.mmapFailed(resolvedURL) }
        defer { close(fd) }

        guard let mapped = mmap(nil, totalBytes, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != MAP_FAILED else {
            throw SafeTensorsError.mmapFailed(resolvedURL)
        }

        // Wrap mmap'd region as a no-copy MTLBuffer. Deallocator unmaps.
        guard let buf = device.mtlDevice.makeBuffer(
            bytesNoCopy: mapped,
            length: totalBytes,
            options: [],
            deallocator: { ptr, len in
                munmap(ptr, len)
            }
        ) else {
            munmap(mapped, totalBytes)
            throw SafeTensorsError.mtlBufferFailed
        }

        // Header + 8 bytes header-length prefix, all skipped from data section
        let dataStart = 8 + Int(headerLen)

        var parsed: [String: Entry] = [:]
        for (name, value) in json {
            if name == "__metadata__" { continue }
            guard let dict = value as? [String: Any],
                  let dtypeStr = dict["dtype"] as? String,
                  let shapeAny = dict["shape"] as? [Any],
                  let offsets = dict["data_offsets"] as? [Any],
                  offsets.count == 2,
                  let startNum = offsets[0] as? NSNumber
            else {
                throw SafeTensorsError.headerEntryMalformed(name)
            }
            guard let dtype = DType.fromSafeTensors(dtypeStr) else {
                throw SafeTensorsError.unsupportedDType(dtypeStr, key: name)
            }
            let shape: [Int] = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeAny.count else {
                throw SafeTensorsError.headerEntryMalformed(name)
            }
            parsed[name] = Entry(
                name: name,
                dtype: dtype,
                shape: shape,
                bufferOffset: dataStart + startNum.intValue
            )
        }

        self.url = url
        self.buffer = buf
        self.entries = parsed
    }

    /// Get a Tensor view over a named entry (no copy).
    public func tensor(named: String) throws -> Tensor {
        guard let e = entries[named] else { throw SafeTensorsError.missingTensor(named) }
        return Tensor(buffer: buffer, offset: e.bufferOffset, shape: e.shape, dtype: e.dtype)
    }
}

/// One or more safetensors files presented as a single tensor namespace.
/// Phase 2 LLM checkpoints are usually a single .safetensors file; sharded
/// checkpoints are also supported via multiple files + the optional
/// `model.safetensors.index.json` map.
public final class SafeTensorsBundle: @unchecked Sendable {
    public let files: [SafeTensorsFile]
    public let directory: URL
    /// tensor name → file index in `files`.
    public let index: [String: Int]

    public init(directory: URL, device: Device = .shared) throws {
        self.directory = directory

        // Look for an index file (sharded model.safetensors.index.json)
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let indexExists = FileManager.default.fileExists(atPath: indexURL.path)

        var fileURLs: [URL] = []
        if indexExists {
            // Sharded: read weight_map → distinct file names
            let data = try Data(contentsOf: indexURL)
            guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = obj["weight_map"] as? [String: String]
            else {
                throw SafeTensorsError.headerJSONMalformed
            }
            let unique = Set(weightMap.values).sorted()
            fileURLs = unique.map { directory.appendingPathComponent($0) }
        } else {
            // Single file at model.safetensors
            let single = directory.appendingPathComponent("model.safetensors")
            if FileManager.default.fileExists(atPath: single.path) {
                fileURLs = [single]
            } else {
                // Fall back: glob for *.safetensors
                let contents = try FileManager.default.contentsOfDirectory(
                    at: directory,
                    includingPropertiesForKeys: nil
                )
                fileURLs = contents
                    .filter { $0.pathExtension == "safetensors" }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
            }
        }

        var loaded: [SafeTensorsFile] = []
        var idx: [String: Int] = [:]
        for (i, url) in fileURLs.enumerated() {
            let f = try SafeTensorsFile(url: url, device: device)
            for name in f.entries.keys { idx[name] = i }
            loaded.append(f)
        }
        self.files = loaded
        self.index = idx
    }

    public func tensor(named: String) throws -> Tensor {
        guard let i = index[named] else { throw SafeTensorsError.missingTensor(named) }
        return try files[i].tensor(named: named)
    }

    public var allKeys: [String] { Array(index.keys).sorted() }
    public var has: (String) -> Bool { { [self] name in index[name] != nil } }
}
