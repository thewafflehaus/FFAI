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
// GenerateCommand — the default `ffai` subcommand. Runs one
// prompt → text generation, optionally streaming, with --stats /
// --debug / --profiling instrumentation.

import ArgumentParser
import FFAI
import Foundation

struct GenerateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate text from a single prompt."
    )

    @Option(name: .shortAndLong, help: "HuggingFace repo id or local model path.")
    var model: String = "unsloth/Llama-3.2-1B"

    @Option(name: .shortAndLong, help: "Prompt to generate from.")
    var prompt: String = "Once upon a time"

    @Option(name: .long, help: "Maximum tokens to generate. Defaults to the model family's value.")
    var maxTokens: Int?

    @Flag(name: .long, help: "Print top-5 next-token distribution instead of generating.")
    var verbose: Bool = false

    @Flag(
        name: .long,
        help: "Print a [STATS] block (per-phase memory, tok/s, TTFT, KV cache, wired ticket).")
    var stats: Bool = false

    /// Stream tokens to stdout as they're generated. Default on.
    /// Disable with `--no-streaming` to print the full text once at the
    /// end (matches the buffered API exactly).
    @Flag(
        name: .long, inversion: .prefixedNo,
        help: "Stream tokens to stdout as they're generated (default). Disable with --no-streaming."
    )
    var streaming: Bool = true

    @Flag(
        name: .long,
        help: "Enable debug logging for every FFAI subsystem (loader, kernels, generate, ...).")
    var debug: Bool = false

    @Option(
        name: .long,
        help:
            "Profiling level: 0 (off), 1 (wallclock breakdown), 2 (level 1 + os_signpost intervals)."
    )
    var profiling: Int = 0

    // ─── Sampling knobs (override the model-family defaults) ─────────

    @Option(
        name: .long, help: "Sampling temperature. 0 = greedy argmax (deterministic, GPU fast path)."
    )
    var temperature: Float?

    @Option(name: .long, help: "Top-K cutoff. 0 = disabled.")
    var topK: Int?

    @Option(name: .long, help: "Top-P / nucleus cutoff. 1.0 = disabled.")
    var topP: Float?

    @Option(name: .long, help: "Min-P cutoff (Qwen-style). 0 = disabled.")
    var minP: Float?

    @Option(name: .long, help: "Repetition penalty. 1.0 = disabled.")
    var repetitionPenalty: Float?

    @Option(name: .long, help: "PRNG seed for reproducible sampling.")
    var seed: UInt64?

    @Option(
        name: .long,
        help:
            "KV cache scheme: \"raw\" (default fp16/bf16), \"affine8\" (~45% smaller), \"affine4\" (~70% smaller), \"aura\" (Phase 5d default aura4v4 ~5x smaller), or any \"auraNvM\" recipe."
    )
    var kvCache: String?

    @Option(
        name: .long,
        help:
            "Maximum positions retained per attention layer. Past this, the cache evicts in FIFO order with the first --kv-keep slots pinned. 0 / unset means unbounded (cap at model max_position_embeddings)."
    )
    var kvWindowSize: Int?

    @Option(
        name: .long,
        help:
            "Number of initial positions kept across FIFO eviction (attention sinks). Default 0. Only meaningful when --kv-window-size is set."
    )
    var kvWindowKeep: Int?

    @Option(
        name: .long,
        help:
            "AURA decode-time attention path: \"compressed\" (default — attend directly on packed K/V codes via aura_flash_p1/pass2; realises AURA's ~4x memory savings) or \"dequant-mirror\" (Stage 1a — bulk-dequant the compressed cache into a full-precision mirror buffer per layer and use sdpaDecode; loses the memory savings but matches the Stage 1a code path for A/B benching). Ignored when --kv-cache isn't an aura scheme."
    )
    var auraDecodePath: String?

    // ─── Nemotron-Labs-Diffusion tri-mode decoding ───────────────────

    @Option(
        name: .long,
        help:
            "Decoding mode: \"ar\" (autoregressive, default), \"diffusion\" (block-wise parallel), or \"self-spec\" (diffusion draft + AR verify). diffusion / self-spec require a Nemotron-Labs-Diffusion model."
    )
    var mode: String = "ar"

    @Option(name: .long, help: "Block length for diffusion / self-spec decoding. Default 32.")
    var blockLength: Int = 32

    @Option(
        name: .long,
        help:
            "Confidence threshold for diffusion-mode denoising (0..1). Unset uses an even per-step transfer budget."
    )
    var confidenceThreshold: Float?

    func run() async throws {
        // Apply --debug + --profiling before any FFAI work so the
        // model-load path is captured.
        if debug { Debug.enableAll() }
        guard let lvl = ProfileLevel(rawValue: profiling) else {
            throw ValidationError("Invalid --profiling level \(profiling). Use 0, 1, or 2.")
        }
        Profile.shared.level = lvl
        Profile.shared.resetPhases()

        print("ffai \(FFAI.version) — loading \(model)…")
        let loadStart = Date()
        // Build LoadOptions with the requested KV cache scheme.
        var loadOpts = LoadOptions()
        let rawKVKind = (kvCache ?? "raw").lowercased()
        switch rawKVKind {
        case "raw":
            loadOpts.kvCache = .raw
        case "affine8":
            loadOpts.kvCache = .affineQuantized(bits: 8, groupSize: 64)
        case "affine4":
            // group_size=16 at 4-bit — much finer groups than affine8's
            // 64. Affine min-max int4 has only 16 quant levels, so a
            // single per-head outlier channel inflates a wide group's
            // range and collapses the other dims onto 1-2 levels. At
            // gs64 decode is fully degenerate ("a time, a time"); gs32
            // is grammatical but loops; gs16 is the first group size
            // that decodes coherently (measured on Qwen3-1.7B — see
            // Tests/ModelTests/KVCacheSchemeIntegrationTests.swift).
            // TurboQuant-style rotation would let group_size=64 work at
            // 4-bit; that's Phase 5d (AURA).
            loadOpts.kvCache = .affineQuantized(bits: 4, groupSize: 16)
        case _ where rawKVKind.hasPrefix("aura"):
            guard let scheme = AURAScheme.parse(rawKVKind) else {
                throw ValidationError(
                    "Unknown AURA recipe \"\(rawKVKind)\". Try \"aura\", \"aura4\", \"aura4v2\", \"aura3\", \"aura8\"."
                )
            }
            loadOpts.kvCache = .auraQuantized(scheme: scheme)
        default:
            throw ValidationError(
                "Unknown --kv-cache \"\(kvCache ?? "")\". Use \"raw\", \"affine8\", \"affine4\", or any \"auraNvM\" recipe."
            )
        }

        // Sliding-window / FIFO eviction. Translates the CLI pair
        // (--kv-window-size, --kv-window-keep) into LoadOptions.kvEviction.
        // Validation deferred to KVEvictionState's preconditions so the
        // error site is colocated with the policy logic.
        if let size = kvWindowSize, size > 0 {
            let keep = kvWindowKeep ?? 0
            loadOpts.kvEviction = .window(maxSize: size, keep: keep)
        } else if (kvWindowKeep ?? 0) != 0 {
            throw ValidationError("--kv-window-keep requires --kv-window-size to be set.")
        }

        // AURA decode-path A/B selection.
        if let raw = auraDecodePath?.lowercased() {
            switch raw {
            case "compressed":
                loadOpts.auraDecodePath = .compressed
            case "dequant-mirror", "dequant_mirror", "mirror":
                loadOpts.auraDecodePath = .dequantMirror
            default:
                throw ValidationError(
                    "Unknown --aura-decode-path \"\(auraDecodePath ?? "")\". Use \"compressed\" or \"dequant-mirror\"."
                )
            }
        }
        let m = try await Model.load(model, options: loadOpts)
        print("loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")

        if verbose {
            // Run prefill once, print top-5 next tokens with their logits.
            // Useful for sanity-checking distributions without committing
            // to a sampling strategy.
            let promptTokens = m.tokenizer.encode(text: prompt)
            print("prompt tokens: \(promptTokens)")

            // Verbose prefill only — size the cache to the probe prompt,
            // not the model's full context window (long-context models
            // would otherwise allocate tens of GB just for this sanity
            // check). The actual streaming generation below goes through
            // Model.generate, which computes its own prompt+maxTokens
            // cache depth.
            let verifyCacheDepth = max(1, Swift.min(m.engine.maxContextWindow, promptTokens.count + 1))
            let caches = m.engine.makeLayerCaches(maxSeq: verifyCacheDepth)
            var lastLogits: Tensor?
            for (i, t) in promptTokens.enumerated() {
                lastLogits = m.engine.forward(tokenId: t, position: i, caches: caches)
            }
            if let l = lastLogits {
                print("top-5 next tokens:")
                for (id, v) in Sampling.topN(l, n: 5) {
                    let s = m.tokenizer.decode(tokens: [id], skipSpecialTokens: false)
                    print("  \(id) (\(String(format: "%.4f", v)))  \"\(s)\"")
                }
            }
            return
        }

        // Non-autoregressive modes — block-wise diffusion or linear
        // self-speculation (Nemotron-Labs-Diffusion only).
        let genMode = mode.lowercased()
        if genMode != "ar" {
            guard genMode == "diffusion" || genMode == "self-spec" || genMode == "selfspec"
            else {
                throw ValidationError(
                    "Unknown --mode \"\(mode)\". Use ar, diffusion, or self-spec.")
            }
            guard m.nemotronLabsDiffusion != nil else {
                throw ValidationError(
                    "--mode \(mode) requires a Nemotron-Labs-Diffusion model "
                        + "(\(model) does not support it).")
            }
            let isDiffusion = genMode == "diffusion"
            // Diffusion requires maxNewTokens to be a multiple of the
            // block length; round down (min one block).
            let requested = maxTokens ?? 128
            let maxNew =
                isDiffusion
                ? max(blockLength, (requested / blockLength) * blockLength)
                : requested
            let diffParams = DiffusionParameters(
                maxNewTokens: maxNew, blockLength: blockLength,
                confidenceThreshold: isDiffusion ? confidenceThreshold : nil)

            print("---")
            print(prompt)
            let start = Date()
            let result =
                isDiffusion
                ? m.generateDiffusion(prompt: prompt, parameters: diffParams)
                : m.generateSelfSpeculative(prompt: prompt, parameters: diffParams)
            let elapsed = Date().timeIntervalSince(start)
            print(result.text)
            print("---")
            let tps = Double(result.generatedTokens.count) / max(elapsed, 1e-9)
            print(
                "generated: \(result.generatedTokens.count) tokens in "
                    + "\(String(format: "%.2f", elapsed))s "
                    + "(\(String(format: "%.2f", tps)) tok/s, "
                    + "\(result.forwardPasses) forward passes, "
                    + "\(String(format: "%.2f", Double(result.generatedTokens.count) / Double(result.forwardPasses))) tokens/forward)"
            )
            return
        }

        // Family defaults + any explicit CLI overrides.
        let params = m.defaultGenerationParameters.with {
            if let n = maxTokens { $0.maxTokens = n }
            if let t = temperature { $0.temperature = t }
            if let k = topK { $0.topK = k }
            if let p = topP { $0.topP = p }
            if let mp = minP { $0.minP = mp }
            if let rp = repetitionPenalty { $0.repetitionPenalty = rp }
            if let s = seed { $0.seed = s }
        }

        print("---")
        print(prompt, terminator: "")

        let result = try await runGenerate(model: m, params: params)

        print("---")
        print(
            "prompt: \(result.promptTokens.count) tokens "
                + "(\(String(format: "%.2f", result.prefillTimeS))s prefill)")
        print(
            "generated: \(result.generatedTokens.count) tokens "
                + "in \(String(format: "%.2f", result.decodeTimeS))s "
                + "(\(String(format: "%.2f", result.tokensPerSecond)) tok/s)")
        if stats {
            print(result.stats.formatted())
        }
        if Profile.shared.level >= .wallclock {
            print(Profile.shared.phases.formatted())
        }
    }

    /// Drive generation either by consuming the stream chunk-by-chunk
    /// (printing as we go) or by collecting once at the end. The
    /// producer loop is identical either way; the only difference is
    /// who prints when. Stream consumption with per-token print adds
    /// ~µs of stdio per token vs ~ms of decode — stats are unaffected.
    private func runGenerate(
        model: Model,
        params: GenerationParameters
    ) async throws -> GenerationResult {
        if streaming {
            var generated: [Int] = []
            var text = ""
            var stats: GenerationStats?
            for try await chunk in model.generateStream(prompt: prompt, parameters: params) {
                if !chunk.text.isEmpty {
                    print(chunk.text, terminator: "")
                    fflush(stdout)
                }
                generated.append(contentsOf: chunk.tokens)
                text += chunk.text
                if let s = chunk.stats { stats = s }
            }
            print("")  // newline after the streamed text
            return GenerationResult(
                promptTokens: model.tokenizer.encode(text: prompt),
                generatedTokens: generated, text: text,
                stats: stats!  // streamEndedWithoutFinalChunk would have thrown above
            )
        } else {
            let result = try await model.generate(prompt: prompt, parameters: params)
            print(result.text)
            return result
        }
    }
}
