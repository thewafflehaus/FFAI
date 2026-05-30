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
// DeepSeek V4 single-token decode forward path — full-attention
// sub-block. Lands incrementally: this file currently scaffolds the
// attention path against the existing Ops surface; the FFN sub-block
// (MoE + shared expert + mHC), the CSA / HCA paths, and the
// end-to-end `forward(...)` driver land in follow-ups.

import Foundation
import Metal

// MARK: - Per-call decode state

extension DeepSeekV4Model {
    /// Sliding-window MQA KV cache for one layer. Holds up to
    /// `n_swa=128` 512-d entries; appends grow `swCount` until the
    /// cache wraps. Indexing within the window stays in slot order
    /// (the SDPA kernel walks `[0..n_visible)` directly).
    public final class LayerKVState: @unchecked Sendable {
        public var swCache: Tensor   // [n_swa, head_dim]
        public var swCount: Int
        public let nSWA: Int
        public let headDim: Int

        public init(headDim: Int, nSWA: Int, dtype: DType) {
            self.swCache = Tensor.empty(shape: [nSWA, headDim], dtype: dtype)
            self.swCount = 0
            self.nSWA = nSWA
            self.headDim = headDim
        }
    }

    /// One forward-call decode state.
    public final class DecodeState: @unchecked Sendable {
        public var layerStates: [LayerKVState]
        /// 4-channel mHC residual state, `[n_hc=4, hidden]`.
        public var hcState: Tensor
        public var position: Int

        public init(layerStates: [LayerKVState], hcState: Tensor, position: Int = 0) {
            self.layerStates = layerStates
            self.hcState = hcState
            self.position = position
        }
    }

    public func makeDecodeState() -> DecodeState {
        let cfg = textConfig
        let states = (0..<cfg.nLayers).map { _ in
            LayerKVState(
                headDim: cfg.headDim, nSWA: cfg.slidingWindow, dtype: activationDtype)
        }
        let hc = Tensor.empty(shape: [4, cfg.hidden], dtype: activationDtype)
        return DecodeState(layerStates: states, hcState: hc, position: 0)
    }
}

// MARK: - Errors

enum DeepSeekV4ForwardError: Error, CustomStringConvertible {
    case notImplementedForRegime(Int)
    var description: String {
        switch self {
        case .notImplementedForRegime(let r):
            return "DSv4 forward path not yet implemented for compress_ratio=\(r)"
        }
    }
}
