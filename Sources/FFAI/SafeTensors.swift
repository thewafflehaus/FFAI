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

/// One safetensors file mmap'd zero-copy. Each tensor entry gets its
/// own MTLBuffer wrapping the slice of the mmap'd region it occupies.
/// (Apple's Metal driver has aliasing/correctness issues when binding
/// the same MTLBuffer at multiple offsets per dispatch, so we use
/// per-tensor MTLBuffers — still zero-copy because they all wrap the
/// same mmap.)
public final class SafeTensorsFile: @unchecked Sendable {
    public struct Entry: @unchecked Sendable {
        public let name: String
        public let dtype: DType
        public let shape: [Int]
        /// MTLBuffer for this tensor (offset is always 0).
        public let buffer: MTLBuffer
    }

    public let url: URL
    public let entries: [String: Entry]

    /// The mmap is held only while we copy each tensor's bytes into a
    /// freshly-allocated `MTLBuffer` (the copy is unavoidable because
    /// `makeBuffer(bytesNoCopy:)` needs 16 KiB page-aligned pointers
    /// and safetensors offsets aren't aligned). Once every entry is
    /// copied, the mmap has no callers and we `munmap` immediately —
    /// otherwise we'd retain the file's virtual-address footprint for
    /// the whole life of the `SafeTensorsFile`, which can be tens of
    /// GB for a quantized checkpoint.
    private var mappedBase: UnsafeMutableRawPointer?
    private var mappedLength: Int = 0

    /// Test-visible: `true` if init finished without munmap'ing the
    /// region (a bug — see `Phase C #3` post-mortem). Production
    /// expects this to be `false` after a successful `init`.
    internal var isMmapRetained: Bool { mappedBase != nil }

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

        // Header + 8 bytes header-length prefix, all skipped from data section
        let dataStart = 8 + Int(headerLen)

        // Per-tensor MTLBuffers wrapping mmap slices. We retain ownership
        // of the mmap via the `mappedBase` / `mappedLength` fields and
        // unmap in `deinit`. None of the MTLBuffer deallocators get to
        // call munmap directly (we'd unmap mid-execution).
        var parsed: [String: Entry] = [:]
        for (name, value) in json {
            if name == "__metadata__" { continue }
            guard let dict = value as? [String: Any],
                  let dtypeStr = dict["dtype"] as? String,
                  let shapeAny = dict["shape"] as? [Any],
                  let offsets = dict["data_offsets"] as? [Any],
                  offsets.count == 2,
                  let startNum = offsets[0] as? NSNumber,
                  let endNum = offsets[1] as? NSNumber
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
            let absStart = dataStart + startNum.intValue
            let length = endNum.intValue - startNum.intValue
            let ptr = mapped.advanced(by: absStart)
            // Per-tensor MTLBuffer, offset 0. We copy from the mmap into
            // a freshly-allocated page-aligned shared buffer because
            // makeBuffer(bytesNoCopy:) requires page-aligned pointers
            // (Apple Silicon page size = 16 KiB), and arbitrary offsets
            // into a mmap'd safetensors file are not page-aligned.
            // Cost: one extra memcpy of the full file at load time.
            guard let perTensorBuf = device.mtlDevice.makeBuffer(
                bytes: ptr, length: length,
                options: [.storageModeShared]
            ) else {
                munmap(mapped, totalBytes)
                throw SafeTensorsError.mtlBufferFailed
            }
            parsed[name] = Entry(name: name, dtype: dtype,
                                 shape: shape, buffer: perTensorBuf)
        }

        // All tensor bytes have been copied into MTLBuffers above; the
        // mmap is no longer referenced by anything. Drop it now rather
        // than holding it until `deinit` — for a 10 GB quantized
        // checkpoint that's 10 GB of virtual address space freed
        // immediately after load.
        munmap(mapped, totalBytes)

        self.url = url
        self.mappedBase = nil
        self.mappedLength = 0
        self.entries = parsed
    }

    deinit {
        // Defensive — under the current code path the mmap is unmap'd
        // at the end of init, so this is a no-op. Kept in case a
        // future change introduces a path that holds the mapping past
        // init (e.g. for true zero-copy when page-alignment lands).
        if let base = mappedBase, mappedLength > 0 {
            munmap(base, mappedLength)
        }
    }

    /// Get a Tensor view over a named entry (no copy). Each tensor has
    /// its own MTLBuffer at offset 0.
    public func tensor(named: String) throws -> Tensor {
        guard let e = entries[named] else { throw SafeTensorsError.missingTensor(named) }
        return Tensor(buffer: e.buffer, offset: 0, shape: e.shape, dtype: e.dtype)
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
    /// Lookup-key → physical-key translation. Empty for a normally
    /// loaded bundle (the lookup key *is* the physical key); a
    /// `prefixed(_:)` view populates it so a stripped lookup key
    /// resolves back to the full key the underlying `SafeTensorsFile`
    /// stores. `physicalKey(_:)` consults it.
    private let keyTranslation: [String: String]

    public init(directory: URL, device: Device = .shared) throws {
        self.directory = directory
        self.keyTranslation = [:]

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
        // In a `prefixed(_:)` view the lookup key is stripped; the
        // underlying file still stores the full key, so translate back.
        let physical = keyTranslation[named] ?? named
        return try files[i].tensor(named: physical)
    }

    /// Internal init for building a re-indexed view over the same
    /// underlying `SafeTensorsFile`s — used by `prefixed(_:)`.
    private init(files: [SafeTensorsFile], directory: URL,
                 index: [String: Int], keyTranslation: [String: String]) {
        self.files = files
        self.directory = directory
        self.index = index
        self.keyTranslation = keyTranslation
    }

    /// A view onto this bundle with `prefix` prepended to every lookup
    /// key — the inverse of a checkpoint that namespaces its weights
    /// under a sub-module. A VL checkpoint stores its text backbone
    /// under `language_model.`; `bundle.prefixed("language_model.")`
    /// returns a bundle whose `tensor(named: "model.embed_tokens.weight")`
    /// resolves to `language_model.model.embed_tokens.weight`, so the
    /// existing text-family loader runs unchanged on the sub-tree.
    ///
    /// Only keys carrying `prefix` survive into the view; everything
    /// else (vision-tower weights, the multi-modal projector) is hidden.
    public func prefixed(_ prefix: String) -> SafeTensorsBundle {
        var remapped: [String: Int] = [:]
        var translation: [String: String] = [:]
        for (key, fileIndex) in index where key.hasPrefix(prefix) {
            let stripped = String(key.dropFirst(prefix.count))
            remapped[stripped] = fileIndex
            // Resolve through any existing translation so chained
            // `prefixed` calls still reach the real physical key.
            translation[stripped] = keyTranslation[key] ?? key
        }
        return SafeTensorsBundle(files: files, directory: directory,
                                 index: remapped, keyTranslation: translation)
    }

    public var allKeys: [String] { Array(index.keys).sorted() }
    public var has: (String) -> Bool { { [self] name in index[name] != nil } }

    // ─── Quantized weight helpers (mlx int4 layout) ──────────────────

    /// True if `<base>.scales` and `<base>.biases` exist alongside `<base>.weight`,
    /// indicating an MLX-format quantized linear.
    public func isQuantized(_ base: String) -> Bool {
        index["\(base).scales"] != nil && index["\(base).biases"] != nil
    }

    public struct QuantizedTriplet: Sendable {
        public let weight: Tensor
        public let scales: Tensor
        public let biases: Tensor
    }

    /// Load `(weight, scales, biases)` for a quantized linear at `base`.
    /// The weight at `base.weight` is the packed uint32 tensor.
    public func quantizedTriplet(_ base: String) throws -> QuantizedTriplet {
        return QuantizedTriplet(
            weight: try tensor(named: "\(base).weight"),
            scales: try tensor(named: "\(base).scales"),
            biases: try tensor(named: "\(base).biases")
        )
    }
}
