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
// `ffai download <repo-id> [...]` — pre-fetch one or more HuggingFace
// repos into the local cache without loading the model into memory or
// dispatching any GPU work. Wraps `ModelDownloader.download` (which in
// turn uses the same swift-huggingface `HubClient` as the runtime
// `Model.load` path) so the cache layout is byte-identical to what
// the runtime loader expects.
//
// Why a dedicated CLI command. Phase I.1 of `planning/session-plan.md`:
// today the only way to pull a model is via `Model.load(...)`, which
// also dispatches a prewarm forward pass and (for VLMs) the vision
// tower init. That's a lot of GPU time for what's conceptually a
// cache-warm step. `ffai download` is the cache-warm primitive:
// network + disk only, no GPU.
//
// Typical use:
//   ffai download mlx-community/Qwen3-1.7B-4bit
//   ffai download mlx-community/Qwen3-1.7B-4bit mlx-community/Llama-3.2-1B-Instruct-4bit
//   ffai download --revision dev mlx-community/Qwen3.5-0.8B-4bit
//
// Companion to:
//   - `ffai inspect`   — load + 1 forward pass (cheap GPU probe)
//   - `ffai generate`  — load + full decode
//   - `ffai bench`     — load + benchmarked decode
//   - `ffai models`    — list known model identities (registry)

import ArgumentParser
import FFAI
import Foundation

struct DownloadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract:
            "Pre-fetch one or more HuggingFace model repos into the local cache (no GPU work).",
        discussion: """
            Examples:

              # Single repo, default revision (main):
              ffai download mlx-community/Qwen3-1.7B-4bit

              # Multiple repos in one command:
              ffai download \\
                mlx-community/Llama-3.2-1B-Instruct-4bit \\
                mlx-community/Qwen3-1.7B-4bit

              # Override the revision:
              ffai download --revision dev mlx-community/Qwen3.5-0.8B-4bit

              # Point the cache at an external SSD (otherwise uses
              # $HF_HOME or ~/.cache/huggingface/hub):
              ffai download --cache /Volumes/Scratch/hf mlx-community/Qwen3-1.7B-4bit

            Returns non-zero exit on the first failure; remaining repos
            in the same invocation are skipped. Use `--continue-on-error`
            to drain the full list and only fail at the end if any
            individual download failed.
            """
    )

    @Argument(
        help: "One or more HuggingFace repo ids (e.g. `mlx-community/Qwen3-1.7B-4bit`).")
    var repoIDs: [String]

    @Option(
        name: .long,
        help: "Git revision (branch / tag / commit) to pull. Default: `main`.")
    var revision: String = "main"

    @Option(
        name: .long,
        help:
            "Optional cache root override. Default: `$HF_HOME` if set, else `~/.cache/huggingface/hub`.")
    var cache: String?

    @Flag(
        name: .long,
        help:
            "If any individual download fails, keep going through the remaining repos and exit non-zero at the end (instead of stopping at the first failure)."
    )
    var continueOnError: Bool = false

    @Flag(
        name: .long,
        help:
            "Don't hit the network — succeed only if the snapshot is already on disk. Useful for verifying a cache without pulling."
    )
    var localFilesOnly: Bool = false

    func run() async throws {
        guard !repoIDs.isEmpty else {
            throw ValidationError(
                "ffai download: pass at least one repo id "
                    + "(e.g. `ffai download mlx-community/Qwen3-1.7B-4bit`).")
        }

        let cacheURL: URL? = cache.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let downloader = ModelDownloader(cacheDirectory: cacheURL)

        var failures: [(String, Error)] = []
        for (i, id) in repoIDs.enumerated() {
            let header = repoIDs.count > 1 ? "[\(i + 1)/\(repoIDs.count)] " : ""
            print("\(header)Downloading \(id) @ \(revision) …")
            do {
                let path = try await downloader.download(
                    id: id,
                    revision: revision,
                    localFilesOnly: localFilesOnly,
                    progressHandler: nil  // swift-huggingface logs its own progress
                )
                print("\(header)✓ \(id) → \(path.path)")
            } catch {
                let line = "\(header)✗ \(id) — \(error)"
                if continueOnError {
                    print(line)
                    failures.append((id, error))
                } else {
                    FileHandle.standardError.write(Data("\(line)\n".utf8))
                    throw error
                }
            }
        }

        if !failures.isEmpty {
            FileHandle.standardError.write(
                Data("\nffai download: \(failures.count) of \(repoIDs.count) repo(s) failed.\n".utf8)
            )
            throw ExitCode.failure
        }
    }
}
