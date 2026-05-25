// FFAI CLI — `ffai <subcommand>`.
//
// Default subcommand is `generate`, so `ffai --model X --prompt Y`
// keeps working without typing the subcommand. `ffai bench --method
// simple --model X --prompt Y` runs a benchmark instead.

import ArgumentParser
import FFAI
import Foundation

@main
struct FFAIRoot: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ffai",
        abstract: "Fucking Fast Apple Inference — Apple Silicon LLM CLI.",
        // `--version` is auto-wired by swift-argument-parser when this is
        // non-empty; it prints the string + exits 0 before any subcommand
        // dispatches. Sourced from `FFAI.version` so there's one source of
        // truth (bumped at release time — see
        // documentation/developing/publishing.md).
        version: FFAI.version,
        subcommands: [GenerateCommand.self, BenchCommand.self,
                      InspectCommand.self, ModelsCommand.self,
                      ConvertCommand.self],
        defaultSubcommand: GenerateCommand.self
    )
}
