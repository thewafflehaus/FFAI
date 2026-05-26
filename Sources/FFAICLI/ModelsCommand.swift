// `ffai models` — list every supported model family with example
// HuggingFace repo IDs, so a user can copy-paste an ID straight into
// `ffai generate --model <id>` or `ffai bench --model <id>`.

import ArgumentParser
import FFAI
import Foundation

/// One supported model family + a few example checkpoints.
private struct CatalogEntry {
    let family: String
    /// `config.json` `model_type` value(s) the loader dispatches on.
    let modelType: String
    let summary: String
    /// Example HuggingFace repo IDs — typically a bf16 + an 8-bit + a
    /// 4-bit mlx conversion where published.
    let repos: [String]
}

private struct CatalogGroup {
    let title: String
    let entries: [CatalogEntry]
}

/// The curated catalog. Any mlx-format 3/4/5/6/8-bit conversion of a
/// listed architecture also loads — these are just convenient examples.
private let modelCatalog: [CatalogGroup] = [
    CatalogGroup(title: "Dense text", entries: [
        CatalogEntry(
            family: "Llama 3.x", modelType: "llama",
            summary: "Meta Llama 3 / 3.1 / 3.2 dense GQA transformer.",
            repos: ["unsloth/Llama-3.2-1B",
                    "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"]),
        CatalogEntry(
            family: "SmolLM", modelType: "smollm / smollm2 / smollm3",
            summary: "HuggingFace SmolLM 1 / 2 / 3 — small (135M–3B) Llama-shaped models.",
            repos: ["mlx-community/SmolLM2-360M-Instruct-bf16",
                    "mlx-community/SmolLM3-3B-bf16"]),
        CatalogEntry(
            family: "OLMo", modelType: "olmo / olmo2",
            summary: "AI2 OLMo 1 / 2 — open Llama-shaped research models.",
            repos: ["mlx-community/OLMo-2-0425-1B-Instruct-bf16",
                    "mlx-community/OLMo-2-1124-7B-Instruct-bf16"]),
        CatalogEntry(
            family: "Granite 3", modelType: "granite",
            summary: "IBM Granite v3 dense — Llama-shaped GQA backbone.",
            repos: ["mlx-community/granite-3.2-2b-instruct-bf16"]),
        CatalogEntry(
            family: "Starcoder 2", modelType: "starcoder2",
            summary: "BigCode Starcoder 2 — Llama-shaped code model with attention biases.",
            repos: ["mlx-community/starcoder2-3b-4bit"]),
        CatalogEntry(
            family: "InternLM 2", modelType: "internlm2",
            summary: "Shanghai AI Lab InternLM 2 / 2.5 — Llama-shaped GQA backbone.",
            repos: ["mlx-community/internlm2_5-7b-chat-bf16"]),
        CatalogEntry(
            family: "Yi", modelType: "yi",
            summary: "01.AI Yi — Llama-shaped dense backbone.",
            repos: ["mlx-community/Yi-1.5-6B-Chat-bf16"]),
        CatalogEntry(
            family: "Qwen 2", modelType: "qwen2",
            summary: "Qwen 2 / 2.5 dense transformer.",
            repos: ["Qwen/Qwen2.5-0.5B-Instruct",
                    "mlx-community/Qwen2.5-7B-Instruct-4bit"]),
        CatalogEntry(
            family: "Qwen 3", modelType: "qwen3",
            summary: "Qwen 3 dense — per-head q/k RMSNorm before RoPE.",
            repos: ["mlx-community/Qwen3-1.7B-bf16",
                    "mlx-community/Qwen3-1.7B-8bit",
                    "mlx-community/Qwen3-1.7B-4bit"]),
        CatalogEntry(
            family: "Mistral", modelType: "mistral",
            summary: "Mistral 7B — Llama-shaped GQA backbone.",
            repos: ["mlx-community/Mistral-7B-Instruct-v0.3-bf16",
                    "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
                    "mlx-community/Mistral-7B-Instruct-v0.3-4bit"]),
        CatalogEntry(
            family: "Phi 3", modelType: "phi3",
            summary: "Microsoft Phi-3 / 3.5 dense transformer.",
            repos: ["mlx-community/Phi-3-mini-4k-instruct-8bit",
                    "mlx-community/Phi-3-mini-4k-instruct-4bit"]),
        CatalogEntry(
            family: "Gemma 3", modelType: "gemma3",
            summary: "Google Gemma 3 text decoder.",
            repos: ["mlx-community/gemma-3-1b-it-bf16",
                    "mlx-community/gemma-3-4b-it-8bit",
                    "mlx-community/gemma-3-4b-it-4bit"]),
        CatalogEntry(
            family: "Gemma 4", modelType: "gemma4",
            summary: "Gemma 4 — Dense / E-series PLE / MoE variants.",
            repos: ["mlx-community/gemma-4-e2b-it-bf16",
                    "mlx-community/gemma-4-26b-a4b-it-8bit",
                    "mlx-community/gemma-4-31b-it-4bit"]),
    ]),
    CatalogGroup(title: "Mixture-of-experts", entries: [
        CatalogEntry(
            family: "GPT-OSS-20B", modelType: "gpt_oss",
            summary: "OpenAI GPT-OSS — 32-expert MoE, sliding/full attention, learned sinks.",
            repos: ["mlx-community/gpt-oss-20b-MXFP4-Q8"]),
    ]),
    CatalogGroup(title: "SSM / GDN hybrid", entries: [
        CatalogEntry(
            family: "Mamba 2", modelType: "mamba2",
            summary: "Dense Mamba 2 selective-SSM (no attention).",
            repos: ["mlx-community/mamba2-130m",
                    "mlx-community/mamba2-1.3b"]),
        CatalogEntry(
            family: "Qwen 3.5", modelType: "qwen3_5 / qwen3_5_moe",
            summary: "Gated Delta Net ↔ attention hybrid; dense or MoE FFN.",
            repos: ["mlx-community/Qwen3.5-0.8B-MLX-bf16",
                    "mlx-community/Qwen3.5-0.8B-MLX-8bit",
                    "mlx-community/Qwen3.5-0.8B-MLX-4bit"]),
        CatalogEntry(
            family: "FalconH1", modelType: "falcon_h1",
            summary: "Per-layer Mamba 2 + attention (within-layer hybrid). Raw bf16/f16 only.",
            repos: ["mlx-community/Falcon-H1-Tiny-90M-Instruct-bf16",
                    "mlx-community/Falcon-H1-1.5B-Instruct-bf16"]),
        CatalogEntry(
            family: "NemotronH", modelType: "nemotron_h",
            summary: "Stack-interleaved Mamba 2 / attention / MLP or MoE. Raw bf16/f16 only.",
            repos: ["nvidia/Nemotron-H-4B-Base-8K",
                    "nvidia/Nemotron-Cascade-2-30B-A3B",
                    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"]),
        CatalogEntry(
            family: "Granite4", modelType: "granitemoehybrid",
            summary: "Stack-interleaved Mamba 2 / attention + dense or MoE FFN. Raw bf16/f16 only.",
            repos: ["mlx-community/granite-4.0-h-350m-bf16"]),
        CatalogEntry(
            family: "Jamba", modelType: "jamba",
            summary: "Mamba 1 + attention + dense/MoE FFN. Raw bf16/f16 only.",
            repos: ["mlx-community/AI21-Jamba-Reasoning-3B-bf16"]),
    ]),
    CatalogGroup(title: "Diffusion", entries: [
        CatalogEntry(
            family: "Nemotron-Labs-Diffusion", modelType: "nemotron_labs_diffusion",
            summary: "Tri-mode — autoregressive / block diffusion / self-speculation.",
            repos: ["nvidia/Nemotron-Labs-Diffusion-3B"]),
        CatalogEntry(
            family: "Nemotron-Labs-Diffusion VLM",
            modelType: "nemotron_labs_diffusion_vlm",
            summary: "Tri-mode diffusion text backbone + Pixtral ViT vision tower.",
            repos: ["nvidia/Nemotron-Labs-Diffusion-VLM-8B"]),
    ]),
]

struct ModelsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "List the model families FFAI supports, with example repo IDs."
    )

    func run() throws {
        print("FFAI — supported model families\n")
        for group in modelCatalog {
            print("── \(group.title) " + String(repeating: "─", count: max(0, 50 - group.title.count)))
            for e in group.entries {
                print("  \(e.family)  [\(e.modelType)]")
                print("    \(e.summary)")
                for repo in e.repos {
                    print("    • \(repo)")
                }
                print("")
            }
        }
        print("""
        Pass any example repo ID to a command:

          ffai generate --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Hello"
          ffai bench    --model mlx-community/Qwen3-1.7B-8bit --prompt "Hello" --stats

        Any mlx-format 3/4/5/6/8-bit conversion of a supported
        architecture also loads — the IDs above are just examples.
        Repos are resolved + cached through the HuggingFace Hub.
        """)
    }
}
