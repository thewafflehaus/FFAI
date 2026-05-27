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
        abstract: "F*cking Fast Apple Inference — Apple Silicon LLM CLI.",
        // `--version` is auto-wired by swift-argument-parser when this is
        // non-empty; it prints the string + exits 0 before any subcommand
        // dispatches. Sourced from `FFAI.version` so there's one source of
        // truth (bumped at release time — see
        // documentation/developing/publishing.md).
        version: FFAI.version,
        subcommands: [
            GenerateCommand.self, BenchCommand.self,
            InspectCommand.self, ModelsCommand.self,
            ConvertCommand.self, DownloadCommand.self,
        ],
        defaultSubcommand: GenerateCommand.self
    )
}
