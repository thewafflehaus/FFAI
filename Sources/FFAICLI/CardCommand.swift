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
// `ffai card` — render (and optionally upload) the HF README.md model
// card for a `ffai convert` output, without re-running the conversion.
//
// Background. `ffai convert --upload-repo` now emits a README.md
// alongside the safetensors + config.json (`ConvertDriver.renderModel-
// Card`, committed 6d56ba8). For checkpoints that were uploaded BEFORE
// that change landed — or any HF repo that doesn't have a card —
// this command produces the same body without touching the model
// weights, prints it to stdout (or writes it to a file), and
// optionally shells out to `hf upload <repo> README.md` to publish.
//
// Typical use:
//
//   # Print to stdout to eyeball before publishing.
//   ffai card --source bigcode/starcoder2-3b --bits 4 \
//       --upload-repo ekryski/starcoder2-3b-4bit
//
//   # Write to a file (e.g. for a multi-repo batch script).
//   ffai card --source bigcode/starcoder2-3b --bits 4 \
//       --upload-repo ekryski/starcoder2-3b-4bit \
//       --output /tmp/starcoder2-card.md
//
//   # Publish straight to HF (requires `hf auth login`).
//   ffai card --source bigcode/starcoder2-3b --bits 4 \
//       --upload-repo ekryski/starcoder2-3b-4bit --publish

import ArgumentParser
import FFAI
import Foundation

struct CardCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "card",
        abstract:
            "Render (and optionally publish) the HF README.md model card for a conversion.",
        discussion: """
            Mirrors what `ffai convert` now emits inline. Use this when
            you need to publish a card to a repo that was uploaded
            before the convert-side card emission landed, or for any
            existing HF repo that's missing a README.

            Examples:

              # Print to stdout.
              ffai card --source HuggingFaceTB/SmolLM2-360M-Instruct --bits 4 \\
                  --upload-repo ekryski/SmolLM2-360M-Instruct-4bit

              # Write to a file.
              ffai card --source nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1 \\
                  --bits 4 --upload-repo ekryski/Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit \\
                  --output /tmp/card.md

              # Render + publish to HF.
              ffai card --source bigcode/starcoder2-3b --bits 4 \\
                  --upload-repo ekryski/starcoder2-3b-4bit --publish
            """
    )

    @Option(
        name: .shortAndLong,
        help: "Source HF repo id (`org/repo`) or local path the conversion was based on.")
    var source: String

    @Option(
        name: .shortAndLong,
        help: "Main quantization spec. Accepts `2..8` (affine bit-widths) or `fp16` / `bf16`.")
    var bits: QuantSpec = .bits(4)

    @Option(
        name: .long,
        help: "Embedding spec. Same accepted values as `--bits`.")
    var embeddingBits: QuantSpec?

    @Option(
        name: .long,
        help: "lm_head spec. Same accepted values as `--bits`.")
    var lmHeadBits: QuantSpec?

    @Option(
        name: .long,
        help: "Vision-tower spec. Same accepted values as `--bits`.")
    var visionBits: QuantSpec?

    @Option(
        name: .long,
        help:
            "HF repo id the card describes (e.g. `ekryski/SmolLM2-360M-Instruct-4bit`). Appears in the title, in the `Model.load(...)` example, and in the `--upload-repo` line of the reproducible command."
    )
    var uploadRepo: String?

    @Option(
        name: .shortAndLong,
        help: "Write the rendered card to this file instead of stdout.")
    var output: String?

    @Flag(
        name: .long,
        help:
            "After rendering, shell out to `hf upload <upload-repo> <file> README.md` to publish the card. Requires `--upload-repo` to be set and `hf auth login` to have been run."
    )
    var publish: Bool = false

    func run() async throws {
        var opts = ConvertOptions()
        opts.bits = bits
        opts.embeddingSpec = embeddingBits
        opts.lmHeadSpec = lmHeadBits
        opts.visionSpec = visionBits
        opts.sourceID = source
        opts.uploadRepo = uploadRepo

        let body = ConvertDriver.renderModelCard(sourceID: source, options: opts)

        // Resolve the destination file path — either the caller-supplied
        // `--output`, or a temp file when --publish needs something to
        // hand to `hf upload`. Stdout is the default when neither is set.
        let outFile: URL?
        if let out = output {
            let expanded = (out as NSString).expandingTildeInPath
            outFile = URL(fileURLWithPath: expanded)
        } else if publish {
            // `hf upload` needs a real file on disk; spill into a temp.
            outFile = FileManager.default.temporaryDirectory
                .appendingPathComponent("ffai-card-\(UUID().uuidString).md")
        } else {
            outFile = nil
        }

        if let url = outFile {
            try body.write(to: url, atomically: true, encoding: String.Encoding.utf8)
            // Only echo the path when --publish wasn't set — the
            // publish path prints its own status lines below.
            if !publish {
                print("wrote \(url.path)")
            }
        } else {
            print(body)
        }

        if publish {
            guard let repo = uploadRepo else {
                throw ValidationError(
                    "ffai card --publish requires --upload-repo (the HF repo to push the card to).")
            }
            guard let cardFile = outFile else {
                // Defensive — the temp-path branch above sets outFile
                // whenever `publish` is true, so this should be unreachable.
                throw ValidationError(
                    "ffai card --publish: internal error — no file to upload.")
            }
            try uploadCard(repoId: repo, cardFile: cardFile)
        }
    }

    /// Shell out to `hf upload <repo> <card-file> README.md` to publish
    /// the rendered card to HF. The `path_in_repo=README.md` argument
    /// forces the upload to land at the repo root regardless of the
    /// local file's name (we use a UUID temp file in the --publish
    /// path, so the local name carries no meaning).
    private func uploadCard(repoId: String, cardFile: URL) throws {
        print("publishing card to \(repoId) …")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["hf", "upload", repoId, cardFile.path, "README.md"]
        try process.run()
        process.waitUntilExit()
        if process.terminationStatus == 0 {
            print("published: https://huggingface.co/\(repoId)")
        } else {
            throw ValidationError(
                "ffai card --publish: hf upload exited \(process.terminationStatus).")
        }
    }
}
