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
