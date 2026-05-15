// FFAI CLI — `ffai --model <id-or-path> --prompt "..."`
//
// Phase 0 stub. The real implementation lands in Phase 2.

import ArgumentParser
import FFAI

@main
struct FFAICommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ffai",
        abstract: "Fucking Fast Apple Inference — Apple Silicon LLM CLI."
    )

    @Option(name: .shortAndLong, help: "HuggingFace repo id or local model path.")
    var model: String?

    @Option(name: .shortAndLong, help: "Prompt to generate from.")
    var prompt: String?

    func run() async throws {
        print("ffai \(FFAI.version) — Phase 0 stub")
        print("model:  \(model ?? "<none>")")
        print("prompt: \(prompt ?? "<none>")")
    }
}
