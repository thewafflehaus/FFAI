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
// DownloadCommandTests — argument-parsing + validation coverage for
// `ffai download`. The actual network / disk side of the command is
// covered by `Tests/FFAITests/Loader/ModelDownloaderTests.swift`; the
// tests here pin the CLI surface so flag renames / defaults shifts /
// new validation rules surface as test failures rather than as runtime
// breakage in scripts that call `ffai download` with specific flags.
//
// No tests in this file touch the network or run the command body —
// every test exercises only the `parse(...)` path (no `.run()` call).
// The single exception is `run-with-no-args-throws-validation`, which
// invokes `run()` so it can assert the ValidationError fires; that path
// also has no network call because it errors out before the
// `ModelDownloader.download` call.

import ArgumentParser
import Foundation
import Testing

@testable import FFAICLI

@Suite("DownloadCommand — argument parsing + validation")
struct DownloadCommandTests {

    // ─── Single-repo parse ─────────────────────────────────────────────

    @Test("parse — single repo id, all defaults")
    func parseSingleRepo() async throws {
        let cmd = try await DownloadCommand.parse(["mlx-community/Qwen3-1.7B-4bit"])
        #expect(cmd.repoIDs == ["mlx-community/Qwen3-1.7B-4bit"])
        #expect(cmd.revision == "main")
        #expect(cmd.cache == nil)
        #expect(cmd.continueOnError == false)
        #expect(cmd.localFilesOnly == false)
    }

    // ─── Multi-repo parse ──────────────────────────────────────────────

    @Test("parse — multiple repo ids in one invocation")
    func parseMultipleRepos() async throws {
        let cmd = try await DownloadCommand.parse([
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "mlx-community/Qwen3-1.7B-4bit",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        ])
        #expect(cmd.repoIDs.count == 3)
        #expect(cmd.repoIDs[0] == "mlx-community/Llama-3.2-1B-Instruct-4bit")
        #expect(cmd.repoIDs[1] == "mlx-community/Qwen3-1.7B-4bit")
        #expect(cmd.repoIDs[2] == "mlx-community/Qwen3.5-0.8B-MLX-4bit")
        // Order preserved — the batch is processed sequentially in
        // input order so the user can rely on the per-repo progress
        // lines lining up with their script.
    }

    // ─── --revision ────────────────────────────────────────────────────

    @Test("parse — --revision overrides the default `main`")
    func parseRevision() async throws {
        let cmd = try await DownloadCommand.parse([
            "--revision", "dev",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        ])
        #expect(cmd.revision == "dev")
        #expect(cmd.repoIDs == ["mlx-community/Qwen3.5-0.8B-MLX-4bit"])
    }

    @Test("parse — --revision accepts a long commit sha")
    func parseRevisionCommitSha() async throws {
        let sha = "0123456789abcdef0123456789abcdef01234567"
        let cmd = try await DownloadCommand.parse([
            "--revision", sha,
            "mlx-community/Qwen3-1.7B-4bit",
        ])
        #expect(cmd.revision == sha)
    }

    // ─── --cache ───────────────────────────────────────────────────────

    @Test("parse — --cache populates the override path verbatim")
    func parseCacheOverride() async throws {
        let cmd = try await DownloadCommand.parse([
            "--cache", "/Volumes/Scratch/hf",
            "mlx-community/Qwen3-1.7B-4bit",
        ])
        // The command stores the raw string — URL construction happens
        // inside run(). This pins the parse-side contract.
        #expect(cmd.cache == "/Volumes/Scratch/hf")
    }

    @Test("parse — --cache with a tilde-prefixed path stays unexpanded at parse time")
    func parseCacheTilde() async throws {
        // The command stores the user-supplied string as-is. Tilde
        // expansion (if any) is a downstream concern of ModelDownloader
        // and the HubClient. Test pins the no-expansion contract so the
        // surface doesn't accidentally start mangling user input.
        let cmd = try await DownloadCommand.parse([
            "--cache", "~/my-models",
            "mlx-community/Qwen3-1.7B-4bit",
        ])
        #expect(cmd.cache == "~/my-models")
    }

    // ─── --continue-on-error ──────────────────────────────────────────

    @Test("parse — --continue-on-error flips the flag")
    func parseContinueOnError() async throws {
        let cmd = try await DownloadCommand.parse([
            "--continue-on-error",
            "mlx-community/Qwen3-1.7B-4bit",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        ])
        #expect(cmd.continueOnError == true)
        #expect(cmd.repoIDs.count == 2)
    }

    // ─── --local-files-only ────────────────────────────────────────────

    @Test("parse — --local-files-only flips the flag")
    func parseLocalFilesOnly() async throws {
        let cmd = try await DownloadCommand.parse([
            "--local-files-only",
            "mlx-community/Qwen3-1.7B-4bit",
        ])
        #expect(cmd.localFilesOnly == true)
    }

    // ─── Combined flags ────────────────────────────────────────────────

    @Test("parse — all flags + multiple repos compose cleanly")
    func parseEverythingTogether() async throws {
        let cmd = try await DownloadCommand.parse([
            "--revision", "v1.2",
            "--cache", "/tmp/hf-cache",
            "--continue-on-error",
            "--local-files-only",
            "mlx-community/Qwen3-1.7B-4bit",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        ])
        #expect(cmd.revision == "v1.2")
        #expect(cmd.cache == "/tmp/hf-cache")
        #expect(cmd.continueOnError == true)
        #expect(cmd.localFilesOnly == true)
        #expect(cmd.repoIDs == [
            "mlx-community/Qwen3-1.7B-4bit",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        ])
    }

    @Test("parse — flags can appear after the positional repo ids")
    func parseFlagsAfterPositionals() async throws {
        // swift-argument-parser allows interleaved positionals + options
        // by default. Make sure the user can rearrange them without the
        // command rejecting their input.
        let cmd = try await DownloadCommand.parse([
            "mlx-community/Qwen3-1.7B-4bit",
            "--revision", "dev",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "--continue-on-error",
        ])
        #expect(cmd.repoIDs == [
            "mlx-community/Qwen3-1.7B-4bit",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        ])
        #expect(cmd.revision == "dev")
        #expect(cmd.continueOnError == true)
    }

    // ─── Validation: empty positional list ─────────────────────────────

    @Test("run — empty repo list throws ValidationError")
    func runEmptyReposThrowsValidation() async throws {
        // ArgumentParser permits `repoIDs: [String]` to be empty at
        // parse time (no positional arg required), so the empty-list
        // check has to live in `run()`. Pin the contract: calling
        // `run()` with no positionals throws `ValidationError` BEFORE
        // any HubClient / network call would happen.
        let cmd = try await DownloadCommand.parse([])
        #expect(cmd.repoIDs.isEmpty)
        do {
            try await cmd.run()
            Issue.record("expected ValidationError, got success")
        } catch is ValidationError {
            // Expected.
        } catch {
            Issue.record(
                "expected ValidationError, got \(type(of: error)): \(error)")
        }
    }

    // ─── Help text smoke test ─────────────────────────────────────────

    @Test("help text contains the multi-repo example")
    func helpTextMentionsMultiRepo() {
        let help = DownloadCommand.helpMessage()
        #expect(help.contains("ffai download"))
        // The discussion includes a multi-repo example; this pins it so
        // a future doc rewrite doesn't silently drop the batch syntax
        // demo that scripts learn from.
        #expect(help.contains("Llama-3.2-1B-Instruct-4bit"))
        #expect(help.contains("--continue-on-error"))
    }

    @Test("help text documents --cache override")
    func helpTextMentionsCache() {
        let help = DownloadCommand.helpMessage()
        #expect(help.contains("--cache"))
        // The cache discovery order is documented in quickstart.md —
        // make sure the CLI help points at the same defaults so users
        // don't end up consulting two contradictory sources.
        #expect(
            help.contains("HF_HOME") || help.contains("~/.cache/huggingface"),
            "help should hint at the default cache discovery order")
    }
}
