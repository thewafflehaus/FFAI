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
// `ffai convert` — quantize a bf16/fp16 HuggingFace checkpoint to
// MLX affine format using FFAI's own GPU kernels.
//
// Per-tensor-class specs: `--bits` controls the attention + MLP
// linears (the bulk of the model), and `--embedding-bits` /
// `--lm-head-bits` / `--vision-bits` override the spec for those
// specific tensors. Each flag accepts any of:
//
//   2 / 3 / 4 / 5 / 6 / 8   → affine-quantize to that bit-width
//   fp16 / bf16             → downcast to that dtype (no quant)
//
// The `--*-bits` overrides are optional — omit one and that tensor
// keeps its source dtype (mlx-lm convention for embeddings, lm_head,
// vision tower).
//
// Examples:
//   ffai convert HuggingFaceTB/SmolLM2-360M-Instruct --bits 4
//
//   # Quantize text + embeddings (both at 4-bit), keep an untied
//   # lm_head at 8-bit, leave vision full precision:
//   ffai convert <repo> --bits 4 --embedding-bits 4 --lm-head-bits 8
//
//   # Mixed: text + embed at 3-bit, vision tower at 6-bit (requires
//   # a VL tower that consumes QuantizedLinear — none ship today):
//   ffai convert <vlm> --bits 3 --embedding-bits 3 --vision-bits 6
//
//   # Pure-downcast — no quantization, just publish the bf16 model
//   # as fp16 (typical for downstream platforms that prefer fp16):
//   ffai convert <repo> --bits fp16
//
//   ffai convert /local/path/to/model --bits 8 --output /tmp/out
//   ffai convert mlx-community/Llama-3.2-1B-4bit --upload-repo ekryski/my-4bit

import ArgumentParser
import FFAI
import Foundation

/// Wrap `QuantSpec` for ArgumentParser. Lives in the CLI module so the
/// FFAI library doesn't pull in an `ArgumentParser` dependency. The
/// parser delegates to `QuantSpec.init(parsing:)` for the actual logic.
extension QuantSpec: ExpressibleByArgument {
    public init?(argument: String) {
        self.init(parsing: argument)
    }
}

struct ConvertCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "convert",
        abstract: "Quantize a bf16/fp16 HuggingFace checkpoint to MLX 4-bit affine."
    )

    @Argument(
        help: "HF repo id (e.g. HuggingFaceTB/SmolLM2-360M-Instruct) or local directory path.")
    var source: String

    @Option(
        name: .shortAndLong,
        help: ArgumentHelp(
            "Spec for the main linear projections (q/k/v/o, gate/up/down, MoE experts). "
                + "Accepts: 2 / 3 / 4 / 5 / 6 / 8 (affine bits) or fp16 / bf16 (pure downcast)."))
    var bits: QuantSpec = .bits(4)

    @Option(
        name: .long, help: "Output directory. Defaults to ~/.cache/ffai/converts/<repo>-<bits>bit.")
    var output: String?

    @Option(
        name: .long,
        help: "Upload to HF repo (e.g. ekryski/foo-4bit). Requires `hf` CLI authenticated.")
    var uploadRepo: String?

    @Option(
        name: .long,
        help: ArgumentHelp(
            "Spec for embed_tokens (independent of --bits). Same accepted values as --bits. "
                + "Omit to keep the embedding in its source dtype."))
    var embeddingBits: QuantSpec?

    @Option(
        name: .long,
        help: ArgumentHelp(
            "Spec for lm_head when untied (independent of --bits). Same accepted values as "
                + "--bits. Omit to keep the head in its source dtype."))
    var lmHeadBits: QuantSpec?

    @Option(
        name: .long,
        help: ArgumentHelp(
            "Spec for vision-tower weights (independent of --bits). Same accepted values as "
                + "--bits. Omit to keep the tower in its source dtype — FFAI VL towers run plain "
                + "Linear today, so a quantized spec here is only useful when wiring a new "
                + "tower that consumes QuantizedLinear."))
    var visionBits: QuantSpec?

    @Option(name: .long, help: "Revision (branch/tag/commit) to download from HF. Default: main.")
    var revision: String = "main"

    func run() async throws {
        print("ffai \(FFAI.version) — convert")

        // ─── Resolve source ──────────────────────────────────────────
        print("resolving \(source) …")
        let locator = ModelLocator()
        let sourceDir = try await locator.resolve(
            idOrPath: source,
            revision: revision,
            progressHandler: { p in
                Task { @MainActor in
                    let frac = p.fractionCompleted
                    if frac > 0 {
                        let pct = Int(frac * 100)
                        print("  download \(pct)%", terminator: "\r")
                    }
                }
            }
        )
        print("source dir: \(sourceDir.path)")

        // ─── Compute output path ─────────────────────────────────────
        let destDir: URL
        if let out = output {
            let expanded = (out as NSString).expandingTildeInPath
            destDir = URL(fileURLWithPath: expanded, isDirectory: true)
        } else {
            destDir = defaultOutputDir(for: source, spec: bits)
        }
        print("output dir: \(destDir.path)")

        // ─── Build options ───────────────────────────────────────────
        var opts = ConvertOptions()
        opts.bits = bits
        opts.embeddingSpec = embeddingBits
        opts.lmHeadSpec = lmHeadBits
        opts.visionSpec = visionBits
        // The driver emits a README.md (HF model card) only when
        // sourceID is set; pass the user-supplied source verbatim so
        // the card's `base_model:` field and the example `ffai convert`
        // command match what the user typed. Same goes for
        // `uploadRepo` — it appears in the example command when set
        // so the reader can reproduce the upload step.
        opts.sourceID = source
        opts.uploadRepo = uploadRepo

        // ─── Run conversion ──────────────────────────────────────────
        // Swift 6 strict concurrency: the progress closure is @Sendable so
        // it cannot capture a local `var` by mutation. Use a simple print
        // without a counter (the conversion is synchronous on this thread).
        let startTime = Date()
        try ConvertDriver.convert(
            sourceDir: sourceDir,
            destDir: destDir,
            options: opts,
            progress: { msg in
                print(msg)
            }
        )
        let elapsed = Date().timeIntervalSince(startTime)
        print(String(format: "\nconvert done in %.1fs", elapsed))
        print("output: \(destDir.path)")

        // ─── Optional HF upload ──────────────────────────────────────
        if let repo = uploadRepo {
            try uploadToHuggingFace(repoId: repo, directory: destDir)
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────

    /// Default output path: `~/.cache/ffai/converts/<safe-name>-<spec>`.
    /// `safe-name` is the source with "/" replaced by "--" so it stays
    /// one directory level deep and is human-readable. `<spec>` is the
    /// `QuantSpec.label` — `4bit`, `fp16`, etc.
    private func defaultOutputDir(for source: String, spec: QuantSpec) -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let cacheRoot =
            home
            .appendingPathComponent(".cache")
            .appendingPathComponent("ffai")
            .appendingPathComponent("converts")

        // For HF repo ids like "org/model", produce "org--model-4bit".
        // For local paths, use the last path component.
        let baseName: String
        if ModelLocator.isLocalPath(source) {
            baseName = URL(fileURLWithPath: source).lastPathComponent
        } else {
            baseName = source.replacingOccurrences(of: "/", with: "--")
        }
        let dirName = "\(baseName)-\(spec.label)"
        return cacheRoot.appendingPathComponent(dirName)
    }

    /// Shell out to `hf upload <repo> <dir>` for the optional upload step.
    /// The HF Python SDK (huggingface_hub) is a thin Python CLI; no Swift
    /// SDK is available in this codebase, so we use Process.
    private func uploadToHuggingFace(repoId: String, directory: URL) throws {
        print("\nuploading to \(repoId) …")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["hf", "upload", repoId, directory.path]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        let output =
            String(
                data: pipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8) ?? ""
        if !output.isEmpty { print(output) }

        if process.terminationStatus != 0 {
            // Non-fatal: the model was written locally even if upload fails.
            print(
                "warning: hf upload exited \(process.terminationStatus) — "
                    + "model is still at \(directory.path)")
        } else {
            print("uploaded: https://huggingface.co/\(repoId)")
        }
    }
}
