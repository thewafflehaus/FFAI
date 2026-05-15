// FFAI CLI — `ffai --model <id-or-path> --prompt "..."`
//
// Phase 2: end-to-end Llama 3.2 1B inference.

import ArgumentParser
import FFAI
import Foundation

@main
struct FFAICommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ffai",
        abstract: "Fucking Fast Apple Inference — Apple Silicon LLM CLI."
    )

    @Option(name: .shortAndLong, help: "HuggingFace repo id or local model path.")
    var model: String = "unsloth/Llama-3.2-1B"

    @Option(name: .shortAndLong, help: "Prompt to generate from.")
    var prompt: String = "Once upon a time"

    @Option(name: .long, help: "Maximum tokens to generate.")
    var maxTokens: Int = 64

    @Flag(name: .long, help: "Print top-5 next-token distribution instead of generating.")
    var verbose: Bool = false

    func run() async throws {
        print("ffai \(FFAI.version) — loading \(model)…")
        let loadStart = Date()
        let m = try await Model.load(model)
        print("loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")

        if verbose {
            // Run prefill once, print top-5 next tokens with their logits.
            // Useful for sanity-checking distributions without committing
            // to a sampling strategy.
            let promptTokens = m.tokenizer.encode(text: prompt)
            print("prompt tokens: \(promptTokens)")
            let caches = m.llama.makeKVCache()
            var lastLogits: Tensor?
            for (i, t) in promptTokens.enumerated() {
                lastLogits = m.llama.forward(tokenId: t, position: i, caches: caches)
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

        print("---")
        print(prompt, terminator: "")

        let result = try await m.generate(
            prompt: prompt,
            options: GenerateOptions(maxNewTokens: maxTokens)
        )
        print(result.text)
        print("---")
        print("prompt: \(result.promptTokens.count) tokens "
              + "(\(String(format: "%.2f", result.prefillTimeS))s prefill)")
        print("generated: \(result.generatedTokens.count) tokens "
              + "in \(String(format: "%.2f", result.decodeTimeS))s "
              + "(\(String(format: "%.2f", result.tokensPerSecond)) tok/s)")
    }
}
