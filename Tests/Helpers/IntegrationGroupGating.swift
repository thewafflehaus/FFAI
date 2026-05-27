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
// IntegrationGroupGating — per-group SKIP/ENABLE flags for integration suites
//
// Why this exists. The 2026-05-27 wrong-dispatch GPU pin (see
// `papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md` for the
// general failure-mode treatment and `planning/known-issues.md` §
// "2026-05-27 GPU pin — mt_sdpa_bidirectional_d64_*" for that specific
// occurrence) cost a full system reboot mid-bisect. Running every
// integration suite as one batch — when many are still unverified at
// production shape — is the bisect-foot-gun the post-mortem warns about.
//
// The defensive posture going forward is:
//
//   1. A suite is `verified` only after a successful per-suite run on
//      current `HEAD`. Until then it is **skipped at the suite level**
//      so a `make test-integration` / `make integration-bisect` run
//      cannot accidentally fire a dispatch that pegs the GPU.
//
//   2. Verification happens in phases ordered by GPU-pin risk:
//      text → quantized → KV-cache-matrix → vision → audio → omni.
//      The flag for the current phase is flipped to `true` while we
//      run the bisect for that group; suites that pass are individually
//      removed from the gate (their `@Suite` line drops the
//      `.enabled(if:)` trait entirely) so they stay green in future
//      full-suite runs without depending on the group flag.
//
//   3. Suites that need metaltile work, an audio CPU→GPU port, or an
//      uncached model **stay skipped** with a `.enabled(if: false,
//      "<deferred reason>")` trait on the suite line, so the SKIP is
//      visible in test output and the deferral reason travels with the
//      test source.
//
// Once every suite is individually verified, this file can be deleted.
// Until then, every group flag below is the single source of truth for
// "is the bisect allowed to fire this suite?".

import Foundation
import Testing

public enum IntegrationGroupGating {

    // ─── Group flags (default: false until the group is verified) ──────
    //
    // Flip a flag to `true` immediately before running the bisect for
    // that group. After the group's suites are individually verified,
    // drop the `.enabled(if:)` trait from each suite's `@Suite` line
    // (so the suite stays enabled regardless of the group flag) and
    // flip the flag back to `false` if any suites remain un-verified.

    /// Dense / hybrid / MoE text-only LLMs (Llama, Qwen2/3/3.5/3.6,
    /// Gemma2/3/4, GPT-OSS, NemotronH, NemotronDiffusion, Phi3, Granite3/4,
    /// FalconH1, InternLM2, Jamba, LFM2, Mamba2, MiniCPM5, Mistral,
    /// OLMo, SmolLM, Starcoder2, DeepSeekR1Distill, Qwen35MoEBench,
    /// SpecDecodeBench, SpecDecodeVerifyCost).
    /// Status: in active verification 2026-05-27.
    public static let enableTextSuites: Bool = true

    /// 2-/3-/4-/5-/6-/8-bit Qwen quantization round-trip suites + the
    /// SlidingWindow KV-cache integration. Pure dequantGemv / GDN
    /// host-loop / sdpaDecode coverage; no vision dispatch.
    /// Status: pending verification (run after text passes).
    public static let enableQuantizedSuites: Bool = false

    /// ModelKVCacheMatrixIntegrationTests — parameterised over many
    /// model × cache-strategy cells. High blast radius (loads many
    /// models). Status: pending verification (run after quantized).
    public static let enableKVCacheMatrixSuite: Bool = false

    /// VLM integration suites (Paligemma, GlmOcr, MiniCPMV, Idefics3,
    /// Mistral3, FastVLM, Gemma3/4 VL, LFM2 VL, NemotronH VL, Qwen2/2.5/3/3.5
    /// VL, SmolVLM2). After commit `f8b87cb` every unverified
    /// sdpaBidirectional shape (d ∈ {32, 64, 80, 96}) routes through
    /// CPU `concurrentPerform` so the pin risk is gone in the obvious
    /// places — but individual suites may still hit other unverified
    /// dispatch paths. Status: pending verification (run after KV-cache).
    public static let enableVisionSuites: Bool = false

    /// Audio STT + TTS + STS + VAD suites. Most STT `transcribe_realSpeech`
    /// sub-tests are CPU-bound (per `known-issues.md` § "Audio model
    /// CPU bottlenecks") and need either a per-test `.disabled` tag,
    /// a maxTokens reduction, or the audio CPU→GPU port. Status:
    /// pending verification (run after vision).
    public static let enableAudioSuites: Bool = false

    /// Omni-modal suites (QwenOmni, LFMAudio). Same CPU-bottleneck
    /// considerations as Audio. Status: pending verification (run last).
    public static let enableOmniSuites: Bool = false

    // ─── Skip-reason strings ──────────────────────────────────────────
    //
    // These appear in test output so the human reader sees WHY a suite
    // didn't run. Keep them descriptive enough that the reader doesn't
    // need to open this file.

    public static let textSkipReason: Comment =
        "Text suite group: not yet verified on current HEAD. Flip IntegrationGroupGating.enableTextSuites to run the text-group bisect."

    public static let quantizedSkipReason: Comment =
        "Quantized suite group: not yet verified on current HEAD. Flip IntegrationGroupGating.enableQuantizedSuites to run."

    public static let kvCacheMatrixSkipReason: Comment =
        "KVCacheMatrix suite: not yet verified on current HEAD. Flip IntegrationGroupGating.enableKVCacheMatrixSuite to run."

    public static let visionSkipReason: Comment =
        "Vision suite group: not yet verified on current HEAD. Vision-tower CPU-attention fallbacks landed in f8b87cb; flip IntegrationGroupGating.enableVisionSuites to run the vision-group bisect once the text + quantized groups are green."

    public static let audioSkipReason: Comment =
        "Audio suite group: not yet verified on current HEAD. Many STT transcribe_realSpeech sub-tests are CPU-bound (see planning/known-issues.md § \"Audio model CPU bottlenecks\"); flip IntegrationGroupGating.enableAudioSuites to run."

    public static let omniSkipReason: Comment =
        "Omni suite group: not yet verified on current HEAD. Flip IntegrationGroupGating.enableOmniSuites to run."
}
