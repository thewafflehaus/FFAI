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
// ConvertDriverModelCardTests — pure-function coverage for the HF
// README.md emitter that ships alongside every `ffai convert` output
// when `ConvertOptions.sourceID` is set. The card body is reconstructed
// for the live convert flow + every uploaded ekryski/ checkpoint; if
// these tests regress, every future upload would publish a malformed
// card silently.
//
// The render path is filesystem-free, so these tests touch neither HF
// nor the local cache.

import Foundation
import Testing

@testable import FFAI

@Suite("ConvertDriver.renderModelCard — HF README.md emitter")
struct ConvertDriverModelCardTests {

    // ─── Frontmatter ─────────────────────────────────────────────────

    @Test("4-bit conversion of an HF repo emits expected frontmatter")
    func frontmatter4bitHFRepo() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
        opts.uploadRepo = "ekryski/Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)

        #expect(card.hasPrefix("---\n"))
        #expect(card.contains("license: apache-2.0"))
        #expect(card.contains("base_model: nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"))
        #expect(card.contains("  - mlx"))
        #expect(card.contains("  - ffai"))
        #expect(card.contains("  - quantized"))
        #expect(card.contains("  - 4bit"))
        #expect(card.contains("  - affine"))
        // Frontmatter is bounded by two `---\n` markers.
        let frontmatterEnd = card.range(of: "\n---\n", range: card.range(of: "---\n")!.upperBound ..< card.endIndex)
        #expect(frontmatterEnd != nil, "card should have a closing --- delimiter")
    }

    @Test("fp16 downcast emits an fp16 tag and matching CLI value")
    func fp16DowncastTagsAndCommand() {
        var opts = ConvertOptions()
        opts.bits = .fp16
        opts.sourceID = "HuggingFaceTB/SmolLM-360M-Instruct"
        opts.uploadRepo = "ekryski/SmolLM-360M-Instruct-fp16"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)

        #expect(card.contains("  - fp16"))
        // No `quantized` / `Nbit` tags on a pure downcast — those signal
        // affine quantization specifically.
        #expect(!card.contains("  - quantized"))
        #expect(!card.contains("  - affine"))
        // Conversion command uses the `--bits fp16` CLI form (not `4`
        // / not `.fp16` / not the `fp16` label).
        #expect(card.contains("--bits fp16"))
    }

    @Test("bf16 downcast emits a bf16 tag")
    func bf16DowncastTag() {
        var opts = ConvertOptions()
        opts.bits = .bf16
        opts.sourceID = "allenai/OLMo-2-0425-1B-Instruct"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)

        #expect(card.contains("  - bf16"))
        #expect(card.contains("--bits bf16"))
    }

    // ─── Title + source link ─────────────────────────────────────────

    @Test("title prefers uploadRepo's last segment when present")
    func titleFromUploadRepo() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "bigcode/starcoder2-3b"
        opts.uploadRepo = "ekryski/starcoder2-3b-4bit"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)
        #expect(card.contains("# starcoder2-3b-4bit"))
    }

    @Test("title falls back to source's last segment when uploadRepo is nil")
    func titleFromSourceWhenNoUpload() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "ibm-granite/granite-3.0-2b-instruct"
        opts.uploadRepo = nil
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)
        #expect(card.contains("# granite-3.0-2b-instruct"))
    }

    @Test("HF repo source IDs render as Markdown links to huggingface.co")
    func sourceLinksToHuggingFace() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "bigcode/starcoder2-3b"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)
        #expect(card.contains("[bigcode/starcoder2-3b](https://huggingface.co/bigcode/starcoder2-3b)"))
    }

    @Test("absolute local paths render as inline code, not as HF links")
    func localPathRendersInlineCode() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "/Users/eric/models/my-local-checkpoint"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)
        // A local path starts with `/` and should NOT become an HF link.
        #expect(!card.contains("huggingface.co/Users/eric/models"))
        #expect(card.contains("`/Users/eric/models/my-local-checkpoint`"))
    }

    // ─── Conversion command ──────────────────────────────────────────

    @Test("simple 4-bit conversion command")
    func simpleConvertCommand() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "bigcode/starcoder2-3b"
        opts.uploadRepo = "ekryski/starcoder2-3b-4bit"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)

        let expected =
            "ffai convert bigcode/starcoder2-3b --bits 4 \\\n    --upload-repo ekryski/starcoder2-3b-4bit"
        #expect(card.contains(expected))
    }

    @Test("mixed-spec conversion shows every --*-bits flag")
    func mixedSpecConversionCommand() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.embeddingSpec = .bits(8)
        opts.lmHeadSpec = .bits(6)
        opts.visionSpec = .fp16
        opts.sourceID = "org/example-vlm"
        opts.uploadRepo = "ekryski/example-vlm-mixed"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)

        #expect(card.contains("--bits 4"))
        #expect(card.contains("--embedding-bits 8"))
        #expect(card.contains("--lm-head-bits 6"))
        #expect(card.contains("--vision-bits fp16"))
        #expect(card.contains("--upload-repo ekryski/example-vlm-mixed"))
    }

    @Test("mixed-spec overrides surface as separate frontmatter tags")
    func mixedSpecTags() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.embeddingSpec = .bits(8)
        opts.lmHeadSpec = .bits(6)
        opts.visionSpec = .fp16
        opts.sourceID = "org/example-vlm"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)

        // Main bits tag stays as the canonical "4bit"/"quantized"/"affine"
        // triple; per-role overrides get their own `<role>-<label>` tag.
        #expect(card.contains("  - 4bit"))
        #expect(card.contains("  - embed-8bit"))
        #expect(card.contains("  - lmhead-6bit"))
        #expect(card.contains("  - vision-fp16"))
    }

    @Test("uploadRepo omitted from command when nil")
    func commandWithoutUpload() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "bigcode/starcoder2-3b"
        opts.uploadRepo = nil
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)
        #expect(card.contains("ffai convert bigcode/starcoder2-3b --bits 4"))
        #expect(!card.contains("--upload-repo"))
    }

    // ─── Provenance / FFAI version ───────────────────────────────────

    @Test("provenance lead sentence cites the live FFAI version constant")
    func leadSentenceMentionsFFAIVersion() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "bigcode/starcoder2-3b"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)

        // Body line should be of the shape:
        //   "4-bit affine quantization of [bigcode/starcoder2-3b](...),
        //    produced with [FFAI](https://github.com/thewafflehaus/FFAI)
        //    <version>'s `ffai convert` (mlx-affine format, `group_size=64`)."
        #expect(card.contains("4-bit affine quantization of "))
        #expect(card.contains("[FFAI](https://github.com/thewafflehaus/FFAI) \(FFAI.version)"))
        #expect(card.contains("`group_size=64`"))
    }

    @Test("`See also` section links the live ffai.Model.load call")
    func seeAlsoMentionsModelLoad() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "bigcode/starcoder2-3b"
        opts.uploadRepo = "ekryski/starcoder2-3b-4bit"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)
        // When an uploadRepo is set, the Model.load example uses it
        // (that's the published id readers should consume).
        #expect(card.contains(#"Model.load("ekryski/starcoder2-3b-4bit")"#))
    }

    @Test("`See also` falls back to source when uploadRepo is nil")
    func seeAlsoUsesSourceWhenNoUpload() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "bigcode/starcoder2-3b"
        opts.uploadRepo = nil
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)
        #expect(card.contains(#"Model.load("bigcode/starcoder2-3b")"#))
    }

    // ─── Smoke test on a realistic full card ─────────────────────────

    @Test("full card for a Nemotron-VL 4-bit conversion validates end-to-end")
    func realCardSmoke() {
        var opts = ConvertOptions()
        opts.bits = .bits(4)
        opts.sourceID = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
        opts.uploadRepo = "ekryski/Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit"
        let card = ConvertDriver.renderModelCard(
            sourceID: opts.sourceID!, options: opts)

        // Sanity: complete card has all the structural sections.
        #expect(card.contains("---\nlicense: apache-2.0"))
        #expect(card.contains("\n---\n\n# Llama-3.1-Nemotron-Nano-VL-8B-V1-4bit\n"))
        #expect(card.contains("## Conversion"))
        #expect(card.contains("## See also"))
        // No trailing whitespace artefacts that would render as extra
        // line breaks on HF.
        #expect(!card.hasSuffix("\n\n"))
    }
}
