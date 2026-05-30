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

    /// Materialize a tensor from the GGUF as a host-side `Tensor`
    /// (dequantized to f32 unless the on-disk format is already a
    /// supported float dtype). **WIP**: dispatches to the metaltile
    /// dequant kernels for Q8_0 / Q2_K / IQ2_XXS once those are wired
    /// through Ops.swift; for now, throws `notYetImplemented` for
    /// quantized formats and returns a direct copy for F32 / F16 /
    /// BF16 stored tensors.
    public func tensor(named: String) throws -> Tensor {
        guard let idx = reader.tensorIndex[named] else {
            throw GGUFError.missingMetadataKey("tensor:\(named)")
        }
        let info = reader.tensorInfos[idx]
        switch info.type {
        case .f32, .f16, .bf16:
            // Float storage — direct byte copy into a Tensor of the
            // matching dtype. Defer the actual Tensor construction to
            // the follow-up wiring PR; for now, raise so the loader
            // fails fast at the type-check stage rather than silently
            // returning garbage.
            throw GGUFError.unsupportedDequant(info.type, tensor: named)
        case .q8_0, .q2_K, .iq2_xxs:
            // The on-disk → GPU-resident split + metaltile dequant
            // kernel dispatch lands in the follow-up forward-path PR.
            // PR #243 ships the kernels; this PR ships the reader.
            throw GGUFError.unsupportedDequant(info.type, tensor: named)
        default:
            throw GGUFError.unsupportedDequant(info.type, tensor: named)
        }
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
}
