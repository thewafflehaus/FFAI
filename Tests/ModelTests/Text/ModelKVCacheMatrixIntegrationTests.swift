// Model × KV-cache × weight-bitwidth integration matrix.
//
// This is the comprehensive cross-product coverage test: every model
// family FFAI supports, at every published weight quantization, under
// every KV-cache compression scheme that family's engine supports.
// Each cell loads a real checkpoint, greedy-decodes, and asserts the
// output is coherent (`expectCoherentOutput`).
//
// ─── Why almost everything is skipped by default ──────────────────────
//
// The full matrix is ~100+ cells. Running them all would download
// hundreds of GB of checkpoints and take hours — and several large
// variants (35B-A3B MoE, 31B Gemma 4, 120B-class) don't fit on a
// typical dev box at all. So the matrix is *gated*:
//
//   • The smallest checkpoint per family is marked `alwaysRun` — it
//     runs in the normal `make test-integration` gate, exercised
//     against EVERY KV-cache scheme that family supports. This is the
//     "smallest model per family, all KV combos" contract.
//   • Every other cell (larger checkpoints, the weight-bitwidth ladder)
//     is env-gated: it runs only when `FFAI_BUILD_MACHINE` is set in
//     the environment. A dedicated build machine flips the entire
//     matrix on with one env var; nothing source-side changes.
//
// A cell whose checkpoint can't be fetched (offline, gated repo, repo
// renamed) fails the cell — load errors propagate to the test runner.
// The build-machine gate above already keeps the matrix to checkpoints
// the running machine is expected to have.
//
// Set `FFAI_MATRIX_FAMILY=<family>` to run just one row (e.g.
// `FFAI_MATRIX_FAMILY=Gemma4`) — useful for targeted re-runs.
//
// ─── This file subsumes the old KVCacheSchemeIntegrationTests ─────────
//
// The previous `KVCacheSchemeIntegrationTests.swift` ran one model
// (Qwen3-1.7B) under every scheme. That is exactly the Qwen3 row of
// this matrix, so that file was retired and its hard-won commentary
// folded in below.
//
// ─── KV-cache scheme notes (folded from the retired file) ─────────────
//
// affine int4 uses `groupSize: 16`, NOT 64. Affine min-max int4 has
// only 16 quant levels, so the per-group range matters far more than
// it does for int8's 256 levels. Real K/V has sparse "massive
// activation" outliers — one large channel per head. With groupSize=64
// that single outlier inflates the range across 64 dims and the other
// 63 collapse onto 1-2 levels → degenerate decode ("a time, a time").
// Measured mean-abs reconstruction error on outlier-containing K
// (Tests/FFAITests/KVCacheTests.swift `affineInt4GroupSizeErrorCurve`):
//   gs64 → 0.079   gs32 → 0.046   gs16 → 0.027   (int8 gs64 → 0.005)
// groupSize=16 is the smallest power-of-two divisor of headDim=128 and
// the first that restores coherent output. Same outlier-domination
// motivation behind rotation-based KV quant (QuaRot / AURA).
//
// Every AURA recipe runs with a per-layer SRHT rotation Π_l
// (deterministic seed = layer index): Π_l applied to Q after RoPE so
// SDPA scores cancel, Π_l^T applied to the SDPA output so the residual
// stream stays in the original activation space. See the
// `AURAQuantizedKVCache` header for the math.
//
// ─── KV-cache support is per-engine ───────────────────────────────────
//
//   • Full scheme set (raw + affine + AURA): the Llama engine (Llama,
//     Qwen2, Mistral, Phi, the Llama-compatible zoo, DeepSeek-R1
//     distills), the Qwen3 engine, and Nemotron-Labs-Diffusion.
//   • Raw only: Gemma 3 / Gemma 4 (`preconditionFailure` on anything
//     else today), and every hybrid family — FalconH1, NemotronH,
//     GraniteMoeHybrid, Jamba, Qwen3.5, Mamba 2, GPT-OSS — whose
//     `makeLayerCaches` hard-codes a raw `KVCache` for attention
//     layers. When affine/AURA support lands for a family, widen its
//     `kvSchemes` entry below and the matrix picks it up automatically.

import Foundation
import Testing
@testable import FFAI

// ─── Matrix axes ──────────────────────────────────────────────────────

/// One model checkpoint at a specific weight quantization.
struct MatrixModel: Sendable {
    /// Family label — drives the per-engine KV-scheme lookup.
    let family: String
    /// HuggingFace repo id.
    let id: String
    /// Weight bit-width: 16 = unquantized (bf16/fp16); else 3/4/5/6/8.
    let weightBits: Int
    /// `true` → the smallest, always-run checkpoint for its family.
    /// Runs in the normal integration gate against every supported KV
    /// scheme. `false` → env-gated on `FFAI_BUILD_MACHINE`.
    let alwaysRun: Bool
}

/// One concrete (model, KV-scheme) cell of the matrix.
struct MatrixCase: Sendable, CustomTestStringConvertible {
    let model: MatrixModel
    let kv: KVCacheKind

    /// Human-readable cell label, e.g. `Qwen3 / Qwen3-1.7B-bf16 (w16) / aura4v2`.
    var label: String {
        "\(model.family) / \(shortId) (w\(model.weightBits)) / \(MatrixCatalog.kvLabel(kv))"
    }
    private var shortId: String { model.id.split(separator: "/").last.map(String.init) ?? model.id }
    var testDescription: String { label }
}

// ─── The catalog ──────────────────────────────────────────────────────

enum MatrixCatalog {
    /// Fixed decode prompt — neutral, open-ended, no chat template
    /// dependency so it works for base and instruct checkpoints alike.
    static let prompt = "Once upon a time, in a quiet village"
    /// Tokens to greedy-decode per cell. Kept modest so the matrix
    /// stays tractable; `expectCoherentOutput` needs only a handful.
    static let maxTokens = 200

    /// `true` when the env var is set — flips the env-gated cells on.
    static let buildMachineEnabled =
        ProcessInfo.processInfo.environment["FFAI_BUILD_MACHINE"] != nil

    /// Optional family filter — when `FFAI_MATRIX_FAMILY` is set, only
    /// cells whose `family` matches (case-insensitive) run; every other
    /// cell skips. Lets a build machine re-run a single row
    /// (`FFAI_MATRIX_FAMILY=Gemma4`) without the whole sweep, and keeps
    /// targeted verification cheap. `nil` → no filter, normal gating.
    static let familyFilter: String? =
        ProcessInfo.processInfo.environment["FFAI_MATRIX_FAMILY"]
            .map { $0.lowercased() }

    // ── KV-cache scheme sets ──────────────────────────────────────────

    /// Every KV-cache scheme FFAI exposes. `affine4` pinned to
    /// groupSize 16 (see header); `affine8` to 64.
    static let allKVSchemes: [KVCacheKind] = [
        .raw,
        .affineQuantized(bits: 8, groupSize: 64),
        .affineQuantized(bits: 4, groupSize: 16),
        .auraQuantized(scheme: .default),                            // aura4v4
        .auraQuantized(scheme: .aura4v2),                            // aura4v2
        .auraQuantized(scheme: AURAScheme(keyBits: 8, valueBits: 4)), // aura8v4
        .auraQuantized(scheme: AURAScheme(keyBits: 8, valueBits: 8)), // aura8v8
    ]

    /// Families whose engine honors the full scheme set. Everything
    /// else gets `[.raw]`.
    private static let fullSchemeFamilies: Set<String> = [
        "Llama", "Qwen2", "Mistral", "Phi", "LlamaCompatibles",
        "DeepSeekR1Distill", "Qwen3", "NemotronLabsDiffusion",
    ]

    /// KV-cache schemes a given family's engine supports.
    static func kvSchemes(forFamily family: String) -> [KVCacheKind] {
        fullSchemeFamilies.contains(family) ? allKVSchemes : [.raw]
    }

    /// Short scheme label for test/diagnostic names.
    static func kvLabel(_ kv: KVCacheKind) -> String {
        switch kv {
        case .raw:
            return "raw"
        case let .affineQuantized(bits, groupSize):
            return "affine\(bits)g\(groupSize)"
        case let .auraQuantized(scheme):
            return scheme.name   // "aura4" / "aura4v2" / …
        }
    }

    // ── The model list ────────────────────────────────────────────────
    //
    // One `alwaysRun` checkpoint per family (the smallest FFAI can
    // run end-to-end). Larger checkpoints + the weight-bitwidth ladder
    // are present but env-gated. IDs marked `unverified` should be
    // confirmed before a `FFAI_BUILD_MACHINE` run — a wrong id simply
    // logs a skip, never a failure.

    static let models: [MatrixModel] = [
        // ── Llama engine — dense text, full KV-scheme support ─────────
        MatrixModel(family: "Llama", id: "unsloth/Llama-3.2-1B",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "Llama", id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                    weightBits: 4, alwaysRun: false),

        MatrixModel(family: "Qwen2", id: "Qwen/Qwen2.5-0.5B-Instruct",
                    weightBits: 16, alwaysRun: true),

        MatrixModel(family: "Mistral", id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                    weightBits: 4, alwaysRun: true),

        MatrixModel(family: "Phi", id: "mlx-community/Phi-3-mini-4k-instruct-4bit",
                    weightBits: 4, alwaysRun: true),

        // Llama-compatible zoo — all flow through the Llama loader.
        MatrixModel(family: "LlamaCompatibles", id: "mlx-community/SmolLM2-360M-Instruct-bf16",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "LlamaCompatibles", id: "mlx-community/SmolLM-360M-Instruct-bf16",
                    weightBits: 16, alwaysRun: false),
        MatrixModel(family: "LlamaCompatibles", id: "mlx-community/SmolLM3-3B-bf16",
                    weightBits: 16, alwaysRun: false),
        MatrixModel(family: "LlamaCompatibles", id: "mlx-community/OLMo-2-0425-1B-Instruct-bf16",
                    weightBits: 16, alwaysRun: false),
        MatrixModel(family: "LlamaCompatibles", id: "mlx-community/Starcoder2-3B-bf16",
                    weightBits: 16, alwaysRun: false),
        MatrixModel(family: "LlamaCompatibles", id: "mlx-community/granite-3.0-2b-instruct-bf16",
                    weightBits: 16, alwaysRun: false),
        MatrixModel(family: "LlamaCompatibles", id: "mlx-community/internlm2-chat-1_8b-bf16",
                    weightBits: 16, alwaysRun: false),

        // DeepSeek-R1 distills — Qwen2 / Llama architectures.
        MatrixModel(family: "DeepSeekR1Distill",
                    id: "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
                    weightBits: 4, alwaysRun: true),
        MatrixModel(family: "DeepSeekR1Distill",
                    id: "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
                    weightBits: 4, alwaysRun: false),

        // ── Qwen3 engine — full KV-scheme support + weight-bits ladder ─
        MatrixModel(family: "Qwen3", id: "mlx-community/Qwen3-1.7B-bf16",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "Qwen3", id: "mlx-community/Qwen3-1.7B-8bit",
                    weightBits: 8, alwaysRun: false),
        MatrixModel(family: "Qwen3", id: "mlx-community/Qwen3-1.7B-6bit",
                    weightBits: 6, alwaysRun: false),
        MatrixModel(family: "Qwen3", id: "mlx-community/Qwen3-1.7B-5bit",
                    weightBits: 5, alwaysRun: false),
        MatrixModel(family: "Qwen3", id: "mlx-community/Qwen3-1.7B-4bit",
                    weightBits: 4, alwaysRun: false),
        MatrixModel(family: "Qwen3", id: "mlx-community/Qwen3-1.7B-3bit",
                    weightBits: 3, alwaysRun: false),

        // ── Nemotron-Labs-Diffusion — full KV-scheme support ──────────
        MatrixModel(family: "NemotronLabsDiffusion", id: "nvidia/Nemotron-Labs-Diffusion-3B",
                    weightBits: 16, alwaysRun: true),

        // ── Raw-KV-only families ──────────────────────────────────────
        // Gemma 3 / 4: engine `preconditionFailure`s on non-raw today.
        MatrixModel(family: "Gemma3", id: "mlx-community/gemma-3-1b-it-bf16",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "Gemma4", id: "mlx-community/gemma-4-e2b-it-bf16",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "Gemma4", id: "mlx-community/gemma-4-31b-it-4bit",
                    weightBits: 4, alwaysRun: false),

        // Hybrid families — attention layers use a raw `KVCache`.
        MatrixModel(family: "Qwen35Dense", id: "mlx-community/Qwen3.5-0.8B-MLX-bf16",
                    weightBits: 16, alwaysRun: true),
        // Qwen3.5 / 3.6 MoE share the Qwen35 engine (dense vs MoE is a
        // per-checkpoint `num_experts` decision). No small *raw* MoE
        // checkpoint exists — every published MoE conversion is
        // quantized, and the per-expert quantized-slice path is large.
        // env-gated. IDs unverified — confirm before a build-machine run.
        MatrixModel(family: "Qwen35MoE", id: "mlx-community/Qwen3.5-35B-A3B-4bit",
                    weightBits: 4, alwaysRun: false),
        MatrixModel(family: "Qwen36MoE", id: "mlx-community/Qwen3.6-30B-A3B-4bit",
                    weightBits: 4, alwaysRun: false),

        MatrixModel(family: "FalconH1", id: "mlx-community/Falcon-H1-Tiny-90M-Instruct-bf16",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "NemotronH", id: "nvidia/Nemotron-H-4B-Base-8K",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "GraniteMoeHybrid", id: "mlx-community/granite-4.0-h-350m-bf16",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "Jamba", id: "mlx-community/AI21-Jamba-Reasoning-3B-bf16",
                    weightBits: 16, alwaysRun: true),
        MatrixModel(family: "Mamba2", id: "mlx-community/mamba2-130m",
                    weightBits: 16, alwaysRun: true),
        // GPT-OSS — only the 20B exists at small scale; ~11 GB even at
        // MXFP4. env-gated to keep it out of the default gate (the
        // smallest-runnable GPT-OSS coherence check lives in
        // GPTOSSIntegrationTests). A build machine runs it here too.
        MatrixModel(family: "GPTOSS", id: "mlx-community/gpt-oss-20b-MXFP4-Q8",
                    weightBits: 8, alwaysRun: false),
    ]

    /// The flattened (model, KV-scheme) cell list — the parameter set.
    static let cases: [MatrixCase] = models.flatMap { model in
        kvSchemes(forFamily: model.family).map { MatrixCase(model: model, kv: $0) }
    }
}

// ─── The matrix test ──────────────────────────────────────────────────

@Suite("Model × KV-cache Matrix Coherent Output", .serialized)
struct ModelKVCacheMatrixIntegrationTests {

    @Test("matrix cell decodes coherent output", arguments: MatrixCatalog.cases)
    func decodeMatrixCell(_ cell: MatrixCase) async throws {
        // Family filter (FFAI_MATRIX_FAMILY) — when set, only the named
        // family's cells run. Anything outside the filter is gated off
        // for this run; `#require` surfaces that explicitly instead of
        // silently passing.
        if let only = MatrixCatalog.familyFilter {
            try #require(cell.model.family.lowercased() == only,
                         "matrix cell gated by FFAI_MATRIX_FAMILY=\(only): \(cell.label)")
        }

        // Gate: smallest-per-family cells always run; the rest need
        // FFAI_BUILD_MACHINE. Build-machine-gated cells `#require` the
        // env so they fail visibly on non-build machines rather than
        // silently pass.
        try #require(cell.model.alwaysRun || MatrixCatalog.buildMachineEnabled,
                     "matrix cell is build-machine gated; set FFAI_BUILD_MACHINE: \(cell.label)")

        // Load under the cell's KV-cache scheme. Load failures fail the
        // cell — a missing checkpoint is a real failure, not a silent
        // pass. Build-machine gating above keeps the matrix to the
        // checkpoints actually expected on each machine.
        let opts = LoadOptions(kvCache: cell.kv)
        let model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(cell.model.id, options: opts)
        }

        let result = try await model.generate(
            prompt: MatrixCatalog.prompt,
            parameters: GenerationParameters(
                maxTokens: MatrixCatalog.maxTokens, temperature: 0)
        )
        print("[matrix \(cell.label)] \(result.text)")
        expectCoherentOutput(
            result.generatedTokens, minTokens: 8, label: cell.label)
    }
}
