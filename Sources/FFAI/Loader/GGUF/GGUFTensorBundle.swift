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
// `GGUFTensorBundle` — adapter that exposes a single .gguf file (or a
// directory containing one) as a tensor namespace mirroring
// `SafeTensorsBundle`'s shape, so model loaders that already consume
// `SafeTensorsBundle` can dispatch on whichever format is present
// without branching on storage layout.
//
// **Status:** WIP scaffold. The reader (header + KV + tensor-info
// table) is fully implemented in `GGUFReader.swift` and works end-to-
// end on real DeepSeek-V4-Flash IQ2_XXS GGUFs. `tensor(named:)` decodes
// the on-disk bytes into the GPU-resident split that metaltile's GGUF
// dequant kernels expect (Q8_0 / Q2_K / IQ2_XXS) — for formats whose
// dequant kernels haven't landed yet (every i-quant other than
// IQ2_XXS, the FP4 / TQ / MXFP4 variants), it throws
// `GGUFError.unsupportedDequant` so the loader fails fast instead of
// returning garbage.

import Foundation
import Tokenizers

/// A single GGUF file presented as a tensor namespace. Mirrors the
/// `SafeTensorsBundle` shape so the loader's family dispatch can use
/// either format interchangeably.
public final class GGUFTensorBundle: @unchecked Sendable {
    public let directory: URL
    public let reader: GGUFReader

    public init(directory: URL) throws {
        self.directory = directory
        // Locate the .gguf file inside the directory. Conventional
        // single-file layout; sharded GGUF is rare in 2026 (the
        // 4-bit DSv4 file is 86 GB and ships as a single blob).
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        let ggufs = contents.filter { $0.pathExtension == "gguf" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard let url = ggufs.first else {
            throw GGUFError.missingMetadataKey("any .gguf file in \(directory.path)")
        }
        // If the user has both a main weights GGUF and a sibling MTP-only
        // GGUF (DSv4 ships this way), prefer the larger one for the
        // weights bundle; the MTP heads load separately via the same
        // reader once that path is wired.
        let preferred =
            ggufs.max(by: {
                let lhs = (try? FileManager.default.attributesOfItem(atPath: $0.path))?[.size]
                    as? Int ?? 0
                let rhs = (try? FileManager.default.attributesOfItem(atPath: $1.path))?[.size]
                    as? Int ?? 0
                return lhs < rhs
            }) ?? url
        self.reader = try GGUFReader(url: preferred)
    }

    /// Single-file convenience init when the caller already knows the
    /// GGUF URL exactly (tests, the MTP-side load, ...).
    public init(url: URL) throws {
        self.directory = url.deletingLastPathComponent()
        self.reader = try GGUFReader(url: url)
    }

    /// Materialize a tensor from the GGUF as a host-side `Tensor`.
    /// Supported on-disk formats: F32 / F16 / BF16 (direct copy) and
    /// Q8_0 / Q2_K / IQ2_XXS (GPU dequant via the metaltile
    /// `ffai_gguf_dequant_*` kernels). Other quant types raise
    /// `GGUFError.unsupportedDequant` — they land in follow-ups as
    /// the kernel surface grows.
    ///
    /// - Parameters:
    ///   - named: tensor name from the GGUF tensor info table
    ///   - outDtype: target activation dtype for the returned tensor.
    ///     `nil` defaults to f32 for quantized inputs and the on-disk
    ///     dtype for float inputs.
    ///   - device: the device whose command queue handles the dequant
    ///     dispatch. Defaults to `.shared`.
    public func tensor(
        named: String, outDtype: DType? = nil, device: Device = .shared
    ) throws -> Tensor {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        let shape = info.dimensions.map { Int($0) }
        let raw = try reader.rawBytes(named: named)

        switch info.type {
        case .f32, .f16, .bf16:
            let srcDtype: DType = info.type == .f32 ? .f32 : (info.type == .f16 ? .f16 : .bf16)
            let dstDtype = outDtype ?? srcDtype
            if srcDtype == dstDtype {
                // Fast path — direct byte copy.
                let buf = device.makeBuffer(length: max(raw.count, srcDtype.byteSize))
                raw.withUnsafeBytes { src in
                    buf.contents().copyMemory(
                        from: src.baseAddress!, byteCount: raw.count)
                }
                return Tensor(buffer: buf, offset: 0, shape: shape, dtype: srcDtype)
            }
            // Cross-dtype convert path — go through f32, then narrow
            // to the destination dtype. Tensors hit here are small
            // (norms, sinks, biases) so CPU conversion is fine; the
            // bulk weights live on the q8_0 / q2_K / iq2_xxs paths
            // below where the dequant kernel handles the dtype
            // narrowing on-GPU.
            return try Self.convertHalfPrecisionTensor(
                raw: raw, srcDtype: srcDtype, dstDtype: dstDtype,
                shape: shape, device: device)

        case .q8_0:
            let cmd = device.makeCommandBuffer()
            let result = GGUFDequant.dequantQ8_0(
                rawBlocks: raw, nValues: Int(info.numElements),
                outDtype: outDtype ?? .f32,
                on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            return result.reshaped(to: shape)

        case .q2_K:
            let cmd = device.makeCommandBuffer()
            let result = GGUFDequant.dequantQ2_K(
                rawBlocks: raw, nValues: Int(info.numElements),
                outDtype: outDtype ?? .f32,
                on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            return result.reshaped(to: shape)

        case .iq2_xxs:
            let (grid, signs) = GGUFDequant.iq2xxsTables(device: device)
            let cmd = device.makeCommandBuffer()
            let result = GGUFDequant.dequantIQ2_XXS(
                rawBlocks: raw, nValues: Int(info.numElements),
                outDtype: outDtype ?? .f32,
                gridTensor: grid, signsTensor: signs,
                on: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            return result.reshaped(to: shape)

        default:
            throw GGUFError.unsupportedDequant(info.type, tensor: named)
        }
    }

    // ─── Architecture-introspection helpers ──────────────────────────

    /// CPU-side dtype conversion for the small float tensors
    /// (norms, sinks, biases) where the GGUF on-disk dtype differs
    /// from the caller's requested activation dtype. Goes via f32
    /// then narrows; bf16 isn't covered yet (only emerges as a
    /// destination once a model needs bf16 activations and a
    /// dedicated f32-bytes → bf16-bytes helper lands).
    private static func convertHalfPrecisionTensor(
        raw: Data, srcDtype: DType, dstDtype: DType,
        shape: [Int], device: Device
    ) throws -> Tensor {
        // Step 1: decode raw → [Float] in f32.
        var f32s: [Float]
        switch srcDtype {
        case .f32:
            f32s = raw.withUnsafeBytes { rawBuf in
                Array(rawBuf.bindMemory(to: Float.self))
            }
        case .f16:
            f32s = raw.withUnsafeBytes { rawBuf in
                rawBuf.bindMemory(to: Float16.self).map { Float($0) }
            }
        case .bf16:
            f32s = raw.withUnsafeBytes { rawBuf in
                rawBuf.bindMemory(to: UInt16.self).map { bits in
                    Float(bitPattern: UInt32(bits) << 16)
                }
            }
        default:
            throw GGUFError.unsupportedDequant(.f32, tensor: "convert src \(srcDtype)")
        }
        // Step 2: encode f32 → dst bytes.
        let outByteCount = f32s.count * dstDtype.byteSize
        let buf = device.makeBuffer(length: max(outByteCount, dstDtype.byteSize))
        switch dstDtype {
        case .f32:
            buf.contents().assumingMemoryBound(to: Float.self)
                .update(from: &f32s, count: f32s.count)
        case .f16:
            var f16s: [Float16] = f32s.map { Float16($0) }
            buf.contents().assumingMemoryBound(to: Float16.self)
                .update(from: &f16s, count: f16s.count)
        case .bf16:
            var bf16s: [UInt16] = f32s.map { v in
                let bits = v.bitPattern
                // Round-to-nearest-even truncation: add bias before
                // shifting so the round-half-to-even tie-break is
                // approximated (matches PyTorch's bf16 cast).
                let lsb = (bits >> 16) & 1
                let rounded = bits + 0x7FFF + lsb
                return UInt16(rounded >> 16)
            }
            buf.contents().assumingMemoryBound(to: UInt16.self)
                .update(from: &bf16s, count: bf16s.count)
        default:
            throw GGUFError.unsupportedDequant(.f32, tensor: "convert dst \(dstDtype)")
        }
        return Tensor(buffer: buf, offset: 0, shape: shape, dtype: dstDtype)
    }

    // ─── Architecture-introspection helpers ──────────────────────────

    /// `general.architecture` — what the loader's family dispatch
    /// switches on. Returns `nil` if the metadata key is missing
    /// (malformed GGUF).
    public var architecture: String? {
        reader.metadataString("general.architecture")
    }

    /// `general.name` — model display name (e.g.
    /// "DeepSeek V4 Flash"). Optional.
    public var modelName: String? {
        reader.metadataString("general.name")
    }

    /// Build a swift-transformers `Tokenizer` from the embedded
    /// `tokenizer.ggml.*` metadata. Throws when the embedded
    /// tokenizer kind isn't a BPE-family variant the adapter knows
    /// how to translate.
    public func tokenizer() throws -> any Tokenizers.Tokenizer {
        try GGUFTokenizerAdapter.build(reader: reader)
    }
}
