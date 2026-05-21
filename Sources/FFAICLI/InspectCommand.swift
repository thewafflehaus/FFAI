// `ffai inspect` — model bring-up diagnostic.
//
// Load a model and print everything that matters for "is this thing
// wired up correctly":
//
//   1. **Architecture** — family, dtype, hidden/nLayers/nHeads/
//      nKVHeads/headDim/vocab/maxSeq, tied vs untied lm_head,
//      quantization scheme if any.
//   2. **Capabilities** — text-only / vision / audio / etc., which
//      are available vs currently enabled.
//   3. **Tokenizer** — vocab size, special tokens, how the test
//      prompt encodes to ids + how those decode back.
//   4. **KV cache layout** — bytes allocated per layer and total,
//      eviction policy, working buffer sharing.
//   5. **Single-step forward** — top-5 next-token logits with
//      decoded strings. Catches NaN logits + lets the user
//      eyeball whether the distribution is plausible (e.g. for
//      "Once upon a time, in a quiet" the top tokens should be
//      " village", " forest", " place", etc., not " <pad>").
//
// Replaces the ad-hoc `generate --verbose` for model triage. The
// `--debug` / `--profiling` flags from generate carry over so
// callers can drive the whole telemetry surface from one command.

import ArgumentParser
import FFAI
import Foundation

struct InspectCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "inspect",
        abstract: "Load a model and print architecture, tokens, and top-5 logits for a fixed probe prompt."
    )

    @Option(name: .shortAndLong,
            help: "HuggingFace repo id or local model path.")
    var model: String = "unsloth/Llama-3.2-1B"

    @Option(name: .shortAndLong,
            help: "Probe prompt — kept short on purpose to make the top-5 output easy to eyeball.")
    var prompt: String = "Once upon a time, in a quiet"

    @Option(name: .long,
            help: "How many of the top next-token logits to print (default 5).")
    var topK: Int = 5

    @Option(name: .long,
            help: "KV cache scheme: \"raw\", \"affine8\", \"affine4\", or any \"auraNvM\" recipe.")
    var kvCache: String?

    @Option(name: .long,
            help: "Maximum positions retained per attention layer. 0 / unset = unbounded.")
    var kvWindowSize: Int?

    @Option(name: .long,
            help: "Attention-sink positions to pin across FIFO eviction. Default 0.")
    var kvWindowKeep: Int?

    @Flag(name: .long, help: "Enable debug logging for every FFAI subsystem.")
    var debug: Bool = false

    @Option(name: .long,
            help: "Profiling level: 0 (off), 1 (wallclock breakdown), 2 (level 1 + os_signpost).")
    var profiling: Int = 0

    @Flag(name: .long,
          help: "Print per-layer intermediate-value stats (min/max/nan/inf + first 4 values) at every layer boundary during prefill. Slow — first-light debugging only. Sets FFAI_INSPECT=1.")
    var layerTrace: Bool = false

    @Option(name: .long,
            help: "Comma-separated layer indices to trace. Only meaningful with --layer-trace. Example: --trace-layers 0,1,5,15")
    var traceLayers: String?

    func run() async throws {
        if debug { Debug.enableAll() }
        guard let lvl = ProfileLevel(rawValue: profiling) else {
            throw ValidationError("Invalid --profiling level \(profiling). Use 0, 1, or 2.")
        }
        Profile.shared.level = lvl
        Profile.shared.resetPhases()

        // --layer-trace is a passthrough for FFAI_INSPECT — set
        // the env var so every model.forward() the inspect command
        // triggers picks it up from `InspectTap.fromEnvironment`.
        if layerTrace {
            setenv("FFAI_INSPECT", "1", 1)
            if let layers = traceLayers {
                setenv("FFAI_INSPECT_LAYERS", layers, 1)
            }
        }

        print("ffai \(FFAI.version) — inspecting \(model)")

        // ─── Build LoadOptions ─────────────────────────────────────
        var opts = LoadOptions()
        let rawKVKind = (kvCache ?? "raw").lowercased()
        switch rawKVKind {
        case "raw":
            opts.kvCache = .raw
        case "affine8":
            opts.kvCache = .affineQuantized(bits: 8, groupSize: 64)
        case "affine4":
            // group_size=16 — see GenerateCommand for the rationale
            // (affine int4 needs tight groups to survive per-head
            // outlier channels; gs64/gs32 decode degenerately).
            opts.kvCache = .affineQuantized(bits: 4, groupSize: 16)
        case _ where rawKVKind.hasPrefix("aura"):
            guard let scheme = AURAScheme.parse(rawKVKind) else {
                throw ValidationError("Unknown AURA recipe \"\(rawKVKind)\".")
            }
            opts.kvCache = .auraQuantized(scheme: scheme)
        default:
            throw ValidationError("Unknown --kv-cache \"\(kvCache ?? "")\".")
        }
        if let size = kvWindowSize, size > 0 {
            opts.kvEviction = .window(maxSize: size, keep: kvWindowKeep ?? 0)
        }

        // ─── Load ────────────────────────────────────────────────
        let loadStart = Date()
        let m = try await Model.load(model, options: opts)
        let loadSecs = Date().timeIntervalSince(loadStart)
        print("loaded in \(String(format: "%.2f", loadSecs))s")

        // ─── Architecture ────────────────────────────────────────
        print("")
        print("┌─ Architecture ─────────────────────────────────────────")
        let family = familyTag(for: m)
        print("│ family             \(family)")
        print("│ model_type         \(m.config.modelType ?? "—")")
        print("│ architecture       \(m.config.architecture ?? "—")")
        print("│ activation dtype   \(m.engine.dtype)")
        print("│ hidden_size        \(m.engine.hidden)")
        print("│ num_layers         \(m.engine.nLayers)")
        print("│ num_heads          \(m.engine.nHeads)")
        print("│ num_kv_heads       \(m.engine.nKVHeads) (GQA fan-out \(m.engine.nHeads / max(m.engine.nKVHeads, 1)))")
        print("│ head_dim           \(m.engine.headDim)")
        print("│ vocab_size         \(m.engine.vocab)")
        print("│ max_position_emb   \(m.engine.maxSeq)")
        if let q = m.config.quantization {
            print("│ weight quant       int\(q.bits) group_size=\(q.groupSize)")
        } else {
            print("│ weight quant       (none — full precision)")
        }
        print("└────────────────────────────────────────────────────────")

        // ─── Capabilities ────────────────────────────────────────
        print("")
        print("┌─ Capabilities ─────────────────────────────────────────")
        let availableSorted = m.availableCapabilities.map { "\($0)" }.sorted()
        let enabledSorted = m.enabledCapabilities.map { "\($0)" }.sorted()
        print("│ available  \(availableSorted.joined(separator: ", "))")
        print("│ enabled    \(enabledSorted.joined(separator: ", "))")
        print("└────────────────────────────────────────────────────────")

        // ─── Tokenizer ───────────────────────────────────────────
        let promptTokens = m.tokenizer.encode(text: prompt)
        print("")
        print("┌─ Tokenizer ────────────────────────────────────────────")
        print("│ probe prompt       \"\(prompt)\"")
        print("│ prompt tokens      \(promptTokens.count): \(promptTokens)")
        let perTokenDecoded = promptTokens.map { id -> String in
            let s = m.tokenizer.decode(tokens: [id], skipSpecialTokens: false)
            return "\(id)=\"\(s)\""
        }
        print("│ per-token          \(perTokenDecoded.joined(separator: ", "))")
        let roundtrip = m.tokenizer.decode(tokens: promptTokens, skipSpecialTokens: false)
        print("│ roundtrip          \"\(roundtrip)\"")
        print("└────────────────────────────────────────────────────────")

        // ─── Special tokens ────────────────────────────────────────
        // Parsed from tokenizer_config.json + categorized so chat-
        // template / tool-calling / multi-turn debugging starts here
        // instead of with `cat tokenizer_config.json | jq`.
        let tokInfo = inspectTokenizer(
            modelDirectory: m.modelDirectory, config: m.config
        )
        print("")
        print("┌─ Special Tokens ───────────────────────────────────────")
        // BOS — union of config.bos_token_id + heuristically-detected
        // BOS tokens (multiple shouldn't normally exist, but we union
        // for symmetry).
        let bosIds: [Int] = {
            var s = Set<Int>()
            if let b = tokInfo.bosTokenId { s.insert(b) }
            for t in tokInfo.tokens(in: .bos) { s.insert(t.id) }
            return s.sorted()
        }()
        printIdLine("BOS", ids: bosIds, model: m)
        // EOS — union of config.eos_token_id (often [<|end_of_text|>])
        // + every heuristic-matched end-of-turn token (Llama 3 declares
        // <|eot_id|> and <|eom_id|> as added_tokens but only the base
        // <|end_of_text|> in eos_token_id; the chat template uses the
        // others to signal end of message / end of turn).
        let eosIds: [Int] = {
            var s = Set(tokInfo.eosTokenIds)
            for t in tokInfo.tokens(in: .eos) { s.insert(t.id) }
            return s.sorted()
        }()
        printIdLine("EOS / end-of-turn", ids: eosIds, model: m)
        for category in SpecialTokenCategory.allCases where category != .bos && category != .eos {
            let toks = tokInfo.tokens(in: category)
            guard !toks.isEmpty else { continue }
            let label = category.rawValue.padding(toLength: 17, withPad: " ", startingAt: 0)
            let rendered = toks.prefix(6)
                .map { "\($0.id) \"\($0.content)\"" }
                .joined(separator: ", ")
            let tail = toks.count > 6 ? " … (+\(toks.count - 6) more)" : ""
            print("│ \(label) \(rendered)\(tail)")
        }
        if tokInfo.hasChatTemplate {
            let markers = tokInfo.chatTemplateMarkers
            let markersStr = markers.isEmpty
                ? "(no recognized markers)"
                : markers.joined(separator: ", ")
            print("│ chat_template     present — mentions: \(markersStr)")
        } else {
            print("│ chat_template     (not present — caller must compose chat input manually)")
        }
        print("└────────────────────────────────────────────────────────")

        // ─── KV Cache layout ─────────────────────────────────────
        let caches = m.engine.makeLayerCaches()
        let bytesAllocated = caches.reduce(0) { $0 + $1.bytesAllocated }
        print("")
        print("┌─ KV Cache ─────────────────────────────────────────────")
        print("│ scheme             \(opts.kvCache)")
        print("│ eviction policy    \(opts.kvEviction)")
        print("│ per-layer caches   \(caches.count)")
        print("│ total bytes alloc  \(formatBytes(bytesAllocated))")
        if let kv = caches.first as? any KVCacheProtocol {
            print("│ layer 0 stride     [nKVHeads=\(kv.nKVHeads), maxSeq=\(kv.maxSeq), headDim=\(kv.headDim)]")
            print("│ layer 0 dtype      \(kv.dtype)")
            print("│ layer 0 maxSize    \(kv.effectiveMaxSize)")
        }
        print("└────────────────────────────────────────────────────────")

        // ─── Single-step forward + top-K logits ──────────────────
        print("")
        print("┌─ Top-\(topK) next tokens ──────────────────────────────────")
        var lastLogits: Tensor?
        for (i, t) in promptTokens.enumerated() {
            lastLogits = m.engine.forward(tokenId: t, position: i, caches: caches)
        }
        guard let l = lastLogits else {
            print("│ (no prompt tokens — nothing to forward)")
            print("└────────────────────────────────────────────────────────")
            return
        }
        let top = Sampling.topN(l, n: topK)
        var anyNaN = false
        for (id, value) in top {
            let s = m.tokenizer.decode(tokens: [id], skipSpecialTokens: false)
            let valueStr: String
            if value.isNaN { valueStr = "NaN"; anyNaN = true }
            else if !value.isFinite { valueStr = "inf" }
            else { valueStr = String(format: "%+.4f", value) }
            print("│ \(String(format: "%6d", id))  \(valueStr)  \"\(s)\"")
        }
        print("└────────────────────────────────────────────────────────")

        if anyNaN {
            print("")
            print("⚠️  NaN logits detected — model forward pass is broken.")
            print("    Likely causes: kernel-side overflow (often bf16 in")
            print("    activations like gelu/tanh/exp), missing weight tie,")
            print("    or a layer-input/weight shape slip. Re-run with")
            print("    --debug to see per-op kernel dispatch traces.")
        }

        // ─── Profile breakdown ───────────────────────────────────
        if Profile.shared.level >= .wallclock {
            print("")
            print(Profile.shared.phases.formatted())
        }
    }

    /// One row of the Special Tokens table for BOS / EOS — IDs
    /// resolved to content strings via the tokenizer's
    /// decode-with-specials path.
    private func printIdLine(_ label: String, ids: [Int], model m: Model) {
        let padded = label.padding(toLength: 17, withPad: " ", startingAt: 0)
        if ids.isEmpty {
            print("│ \(padded) (not declared)")
            return
        }
        let parts: [String] = ids.map { id in
            let decoded = m.tokenizer.decode(tokens: [id], skipSpecialTokens: false)
            return "\(id) \"\(decoded)\""
        }
        print("│ \(padded) \(parts.joined(separator: ", "))")
    }

    // Pretty-print "1.34 GB" / "12.6 MB" / "2.3 kB" / "512 B".
    private func formatBytes(_ b: Int) -> String {
        let kB = 1024.0
        let mB = kB * 1024
        let gB = mB * 1024
        let v = Double(b)
        if v >= gB { return String(format: "%.2f GB", v / gB) }
        if v >= mB { return String(format: "%.2f MB", v / mB) }
        if v >= kB { return String(format: "%.2f kB", v / kB) }
        return "\(b) B"
    }

    // Best-effort family tag derived from the engine's concrete type.
    // Used purely for the inspect printout — production code should
    // dispatch via `Model.engine` casts, not string matching.
    private func familyTag(for m: Model) -> String {
        let typeName = String(describing: type(of: m.engine))
        // LlamaModel covers everything that routes through the Llama
        // loader (Llama, Mistral, Qwen 2.x, SmolLM, OLMo, Granite,
        // Yi, InternLM 2, Starcoder 2, DeepSeek R1 Distill). The
        // config's architectures[0] / model_type discriminates.
        let arch = m.config.architecture ?? m.config.modelType ?? "?"
        return "\(typeName) (\(arch))"
    }
}
