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

    @Flag(name: .long, help: "Print a [STATS] block (per-phase memory, tok/s, TTFT, KV cache, wired ticket).")
    var stats: Bool = false

    /// Stream tokens to stdout as they're generated. Default on.
    /// Disable with `--no-streaming` to print the full text once at the
    /// end (matches the buffered API exactly).
    @Flag(name: .long, inversion: .prefixedNo,
          help: "Stream tokens to stdout as they're generated (default). Disable with --no-streaming.")
    var streaming: Bool = true

    @Flag(name: .long, help: "Enable debug logging for every FFAI subsystem (loader, kernels, generate, ...).")
    var debug: Bool = false

    @Option(name: .long,
            help: "Profiling level: 0 (off), 1 (wallclock breakdown), 2 (level 1 + os_signpost intervals).")
    var profiling: Int = 0

    // ─── Sampling knobs (override the model-family defaults) ─────────

    @Option(name: .long, help: "Sampling temperature. 0 = greedy argmax (deterministic, GPU fast path).")
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

    @Option(name: .long, help: "KV cache scheme: \"raw\" (default fp16/bf16), \"int8\" (affine, ~45% smaller), or \"int4\" (affine, ~70% smaller).")
    var kvCache: String?

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
        switch (kvCache ?? "raw").lowercased() {
        case "raw":
            loadOpts.kvCache = .raw
        case "int8", "affine8", "affinequantized":
            loadOpts.kvCache = .affineQuantized(bits: 8, groupSize: 64)
        case "int4", "affine4":
            // group_size=32 at int4 — finer groups than int8's 64 to
            // preserve enough precision; without it K/V loses too much
            // discriminative power and decode degenerates into loops.
            // TurboQuant-style rotation would let group_size=64 work
            // at 4-bit; that's Phase 5d.
            loadOpts.kvCache = .affineQuantized(bits: 4, groupSize: 32)
        default:
            throw ValidationError("Unknown --kv-cache \"\(kvCache ?? "")\". Use \"raw\", \"int8\", or \"int4\".")
        }
        let m = try await Model.load(model, options: loadOpts)
        print("loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")

        if verbose {
            // Run prefill once, print top-5 next tokens with their logits.
            // Useful for sanity-checking distributions without committing
            // to a sampling strategy.
            let promptTokens = m.tokenizer.encode(text: prompt)
            print("prompt tokens: \(promptTokens)")

            let caches = m.engine.makeKVCache()
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

        // Family defaults + any explicit CLI overrides.
        let params = m.defaultGenerationParameters.with {
            if let n = maxTokens          { $0.maxTokens = n }
            if let t = temperature        { $0.temperature = t }
            if let k = topK               { $0.topK = k }
            if let p = topP               { $0.topP = p }
            if let mp = minP              { $0.minP = mp }
            if let rp = repetitionPenalty { $0.repetitionPenalty = rp }
            if let s = seed               { $0.seed = s }
        }

        print("---")
        print(prompt, terminator: "")

        let result = try await runGenerate(model: m, params: params)

        print("---")
        print("prompt: \(result.promptTokens.count) tokens "
              + "(\(String(format: "%.2f", result.prefillTimeS))s prefill)")
        print("generated: \(result.generatedTokens.count) tokens "
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
    private func runGenerate(model: Model,
                             params: GenerationParameters) async throws -> GenerationResult {
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
            print("")   // newline after the streamed text
            return GenerationResult(
                promptTokens: model.tokenizer.encode(text: prompt),
                generatedTokens: generated, text: text,
                stats: stats!   // streamEndedWithoutFinalChunk would have thrown above
            )
        } else {
            let result = try await model.generate(prompt: prompt, parameters: params)
            print(result.text)
            return result
        }
    }
}
