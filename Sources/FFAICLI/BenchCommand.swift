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
// BenchCommand — `ffai bench --method <name> --model <repo> ...`.
//
// Runs a `BenchMethod` against the given model and appends the
// resulting row to a per-day report (`<chip>-YYYY-MM-DD.md` plus a
// JSON sidecar) under `--report-dir`. Same row schema as
// mlx-swift-lm so analysis tooling stays cross-compatible.
//
// Implemented methods bottom out in the same `Model.generate(...)` /
// `Perplexity.compute(...)` paths the library uses; unimplemented
// methods (NIAH, multi-turn, tool-calling, ngram-*, vision) fail
// fast with the dependency name they're waiting on.

import ArgumentParser
import FFAI
import Foundation

struct BenchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Run a single benchmark method against a model and append the result to the day's report.",
        discussion: """
        Methods (mirrors mlx-swift-lm's --method):

          simple              \(BenchMethod.simple.description)
          summarization       \(BenchMethod.summarization.description)
          wikitext2           \(BenchMethod.wikitext2.description)
          niah                \(BenchMethod.niah.description) [TODO: \(BenchMethod.niah.dependency ?? "?")]
          multi-turn          \(BenchMethod.multiTurn.description) [TODO: \(BenchMethod.multiTurn.dependency ?? "?")]
          tool-calling        \(BenchMethod.toolCalling.description) [TODO: \(BenchMethod.toolCalling.dependency ?? "?")]
          ngram-spot          \(BenchMethod.ngramSpot.description) [TODO: \(BenchMethod.ngramSpot.dependency ?? "?")]
          ngram-sweep         \(BenchMethod.ngramSweep.description) [TODO]
          ngram-sweep-summary \(BenchMethod.ngramSweepSummary.description) [TODO]
          vision              \(BenchMethod.vision.description) [TODO: \(BenchMethod.vision.dependency ?? "?")]
        """
    )

    @Option(name: .shortAndLong, help: "HuggingFace repo id or local model path.")
    var model: String

    @Option(name: .long, help: "Benchmark method (run `ffai bench --help` for the list).")
    var method: String = BenchMethod.simple.rawValue

    @Option(name: .shortAndLong, help: "Prompt to use for prompt-based methods (simple, summarization).")
    var prompt: String?

    @Option(name: .long, help: "Maximum tokens to generate.")
    var maxTokens: Int = 64

    @Option(name: .long,
            help: "HuggingFace repo id of the reference model for KLD comparison. Use the bf16 unquantized variant of the same architecture if it fits on the machine.")
    var refModel: String?

    @Option(name: .long, help: "Where to write the per-day report (markdown + JSON sidecar). Default: ./benchmarks/")
    var reportDir: String = "./benchmarks"

    @Option(name: .long, help: "Quantization label for the report column (e.g. \"4bit\", \"bf16\"). Optional.")
    var quantization: String?

    @Option(name: .long, help: "Path to the WikiText-2 corpus file (`wiki.test.raw`). Required for --method wikitext2.")
    var wikitext2Corpus: String?

    @Option(name: .long, help: "Cap WikiText-2 token count (default 2048).")
    var wikitext2MaxTokens: Int = 2048

    @Flag(name: .long, help: "Enable debug logging.")
    var debug: Bool = false

    @Option(name: .long, help: "Profiling level: 0 (off), 1 (wallclock breakdown), 2 (level 1 + os_signpost intervals).")
    var profiling: Int = 0

    func run() async throws {
        if debug { Debug.enableAll() }
        guard let lvl = ProfileLevel(rawValue: profiling) else {
            throw ValidationError("Invalid --profiling level \(profiling). Use 0, 1, or 2.")
        }
        Profile.shared.level = lvl
        Profile.shared.resetPhases()

        guard let benchMethod = BenchMethod(rawValue: method) else {
            throw ValidationError("Unknown --method \(method). Run `ffai bench --help` for the list.")
        }

        // Fail fast on unimplemented methods rather than loading a
        // multi-GB model first.
        if !benchMethod.isImplemented {
            throw BenchRunnerError.notImplemented(
                method: benchMethod,
                dependency: benchMethod.dependency ?? "<unknown>"
            )
        }

        print("ffai bench \(benchMethod.rawValue) — loading \(model)…")
        let candidate = try await Model.load(model)

        let reference: Model? = try await {
            guard let ref = refModel else { return nil }
            print("loading reference \(ref)…")
            return try await Model.load(ref)
        }()

        let opts = BenchOptions(
            prompt: prompt,
            maxTokens: maxTokens,
            quantization: quantization,
            wikitext2Corpus: wikitext2Corpus.map { URL(fileURLWithPath: $0) },
            wikitext2MaxTokens: wikitext2MaxTokens,
            referenceModel: reference
        )
        let runner = BenchRunner(model: candidate, modelLabel: model)
        let row = try await runner.run(method: benchMethod, options: opts)

        let writer = BenchmarkWriter(
            reportDirectory: URL(fileURLWithPath: reportDir)
        )
        let urls = try writer.append(row)

        print("[BENCH] \(benchMethod.rawValue) \(model)")
        print("  prefill tok/s:  \(String(format: "%.2f", row.prefillTokensPerSecond))")
        print("  decode tok/s:   \(String(format: "%.2f", row.decodeTokensPerSecond))")
        if let s = row.steadyTokensPerSecond {
            print("  steady tok/s:   \(String(format: "%.2f", s))")
        }
        print("  TTFT:           \(String(format: "%.2f ms", row.timeToFirstTokenMs))")
        if let p = row.genPerplexity {
            print("  gen perplexity: \(String(format: "%.3f", p))")
        }
        if let k = row.genKLDivergence {
            print("  gen KLD:        \(String(format: "%.4f", k))")
        }
        print("  report:         \(urls.markdown.path)")
        print("  sidecar:        \(urls.sidecar.path)")
        if Profile.shared.level >= .wallclock {
            print(Profile.shared.phases.formatted())
        }
    }
}
