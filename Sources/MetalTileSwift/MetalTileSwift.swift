// MetalTileSwift — Swift runtime for pre-compiled metaltile Metal kernels.
//
// Phase 0 stub. The real implementation lands as part of the Phase 0
// plumbing milestone; see planning/plan.md.
//
// Responsibilities (Phase 0):
// - Load Resources/kernels.metallib once into an MTLLibrary
// - Maintain a PSO cache keyed by (kernel name, function constants)
// - Expose typed Swift wrappers (the wrappers themselves live in
//   Generated/MetalTileKernels.swift, produced by metaltile-emit)

import Foundation

public enum MetalTileSwift {
    public static let version = "0.0.1-dev"
}
