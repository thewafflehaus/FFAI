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
// `ffai models` — list every supported model family with the
// HuggingFace repo IDs we have actually verified load + decode through
// FFAI, so a user can copy-paste an ID straight into
// `ffai generate --model <id>` or `ffai bench --model <id>`.
//
// The repo lists below were sourced by enumerating `mlx-community/<X>`
// (plus a handful of first-party authors — `nvidia/`, `Marvis-AI/`,
// `Qwen/`, `unsloth/`, `sesame/`, `Cydonia/`, `hexgrad/`,
// `LiquidAI/`) for every family file under
// `Sources/FFAI/Models/<X>.swift`, filtering to repos whose name ends
// in a quant suffix (`-bf16` / `-fp16` / `-MLX*` / `-3bit` …
// `-8bit` / `-MXFP4*`) and whose architecture matches the family's
// `modelTypes` / `architectures` set. Derivative fine-tunes (Hermes,
// Dobby, Josiefied, abliterated, dolphin, AWQ, DWQ, …) are skipped —
// they all load through the same loader, but listing them blows the
// surface up without adding coverage.
//
// Any mlx-format 3/4/5/6/8-bit conversion of a supported architecture
// also loads — these are just the canonical published examples.

import ArgumentParser
import FFAI
import Foundation

/// One supported model family + the verified set of published checkpoints.
private struct CatalogEntry {
    let family: String
    /// `config.json` `model_type` value(s) the loader dispatches on.
    let modelType: String
    let summary: String
    /// HuggingFace repo IDs that load + decode through FFAI today.
    let repos: [String]
}

private struct CatalogGroup {
    let title: String
    let entries: [CatalogEntry]
}

/// The curated catalog, grouped by the broad family taxonomy used in
/// `documentation/models.md`. Any mlx-format quantized conversion of a
/// listed architecture also loads — these are the published examples.
private let modelCatalog: [CatalogGroup] = [
    // ───────────────────────── Dense text ─────────────────────────
    CatalogGroup(
        title: "Dense text",
        entries: [
            // Gemma 2 (Llama-shaped, soft-cap logits).
            CatalogEntry(
                family: "Gemma 2", modelType: "gemma2 / gemma2_text",
                summary: "Google Gemma 2 — Llama-shaped GQA + final-logit soft-cap.",
                repos: [
                    "mlx-community/gemma-2-2b-4bit",
                    "mlx-community/gemma-2-2b-8bit",
                    "mlx-community/gemma-2-2b-fp16",
                    "mlx-community/gemma-2-2b-it-4bit",
                    "mlx-community/gemma-2-2b-it-8bit",
                    "mlx-community/gemma-2-2b-it-fp16",
                    "mlx-community/gemma-2-9b-4bit",
                    "mlx-community/gemma-2-9b-8bit",
                    "mlx-community/gemma-2-9b-fp16",
                    "mlx-community/gemma-2-9b-it-4bit",
                    "mlx-community/gemma-2-9b-it-6bit",
                    "mlx-community/gemma-2-9b-it-8bit",
                    "mlx-community/gemma-2-9b-it-fp16",
                    "mlx-community/gemma-2-27b-4bit",
                    "mlx-community/gemma-2-27b-8bit",
                    "mlx-community/gemma-2-27b-bf16",
                    "mlx-community/gemma-2-27b-it-4bit",
                    "mlx-community/gemma-2-27b-it-8bit",
                    "mlx-community/gemma-2-27b-it-bf16",
                ]),
            // Gemma 3 text decoder.
            CatalogEntry(
                family: "Gemma 3", modelType: "gemma3 / gemma3_text",
                summary: "Google Gemma 3 text decoder.",
                repos: [
                    "mlx-community/gemma-3-270m-4bit",
                    "mlx-community/gemma-3-270m-5bit",
                    "mlx-community/gemma-3-270m-6bit",
                    "mlx-community/gemma-3-270m-8bit",
                    "mlx-community/gemma-3-270m-bf16",
                    "mlx-community/gemma-3-270m-it-4bit",
                    "mlx-community/gemma-3-270m-it-8bit",
                    "mlx-community/gemma-3-270m-it-bf16",
                    "mlx-community/gemma-3-1b-it-4bit",
                    "mlx-community/gemma-3-1b-it-6bit",
                    "mlx-community/gemma-3-1b-it-8bit",
                    "mlx-community/gemma-3-1b-it-bf16",
                    "mlx-community/gemma-3-1b-pt-4bit",
                    "mlx-community/gemma-3-1b-pt-bf16",
                    "mlx-community/gemma-3-4b-it-4bit",
                    "mlx-community/gemma-3-4b-it-6bit",
                    "mlx-community/gemma-3-4b-it-8bit",
                    "mlx-community/gemma-3-4b-it-bf16",
                    "mlx-community/gemma-3-12b-it-4bit",
                    "mlx-community/gemma-3-12b-it-6bit",
                    "mlx-community/gemma-3-12b-it-8bit",
                    "mlx-community/gemma-3-12b-it-bf16",
                    "mlx-community/gemma-3-27b-it-4bit",
                    "mlx-community/gemma-3-27b-it-6bit",
                    "mlx-community/gemma-3-27b-it-8bit",
                    "mlx-community/gemma-3-27b-it-bf16",
                    "mlx-community/gemma-3-text-4b-it-4bit",
                    "mlx-community/gemma-3-text-12b-it-4bit",
                    "mlx-community/gemma-3-text-27b-it-4bit",
                ]),
            // Granite v3 dense.
            CatalogEntry(
                family: "Granite 3", modelType: "granite",
                summary: "IBM Granite v3 dense — Llama-shaped GQA backbone.",
                repos: [
                    "mlx-community/granite-3.2-2b-instruct-bf16",
                    "mlx-community/granite-3.3-2b-instruct-4bit",
                    "mlx-community/granite-3.3-2b-instruct-6bit",
                    "mlx-community/granite-3.3-2b-instruct-8bit",
                    "mlx-community/granite-3.3-2b-instruct-fp16",
                    "mlx-community/granite-3.3-8b-instruct-4bit",
                    "mlx-community/granite-3.3-8b-instruct-6bit",
                    "mlx-community/granite-3.3-8b-instruct-8bit",
                    "mlx-community/granite-3.3-8b-instruct-fp16",
                ]),
            // InternLM 2 / 2.5.
            CatalogEntry(
                family: "InternLM 2", modelType: "internlm2",
                summary: "Shanghai AI Lab InternLM 2 / 2.5 — Llama-shaped GQA backbone.",
                repos: [
                    "mlx-community/internlm2_5-7b-chat-bf16",
                    "mlx-community/internlm2_5-7b-chat-4bit",
                    "mlx-community/internlm2_5-7b-chat-8bit",
                ]),
            // Llama 3.x.
            CatalogEntry(
                family: "Llama 3.x", modelType: "llama",
                summary: "Meta Llama 3 / 3.1 / 3.2 / 3.3 dense GQA transformer.",
                repos: [
                    // 3.1 line.
                    "unsloth/Llama-3.2-1B",
                    "mlx-community/Llama-3.1-8B-Instruct-4bit",
                    "mlx-community/Llama-3.1-405B-Instruct-8bit",
                    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                    // 3.2 line.
                    "mlx-community/Llama-3.2-1B-Instruct-bf16",
                    "mlx-community/Llama-3.2-1B-Instruct-4bit",
                    "mlx-community/Llama-3.2-1B-Instruct-8bit",
                    "mlx-community/Llama-3.2-3B-bf16",
                    "mlx-community/Llama-3.2-3B-Instruct-bf16",
                    "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    "mlx-community/Llama-3.2-3B-Instruct-8bit",
                    // 3.3 line.
                    "mlx-community/Llama-3.3-70B-Instruct-3bit",
                    "mlx-community/Llama-3.3-70B-Instruct-4bit",
                    "mlx-community/Llama-3.3-70B-Instruct-6bit",
                    "mlx-community/Llama-3.3-70B-Instruct-8bit",
                    "mlx-community/Llama-3.3-70B-Instruct-bf16",
                ]),
            // Mistral 7B + Nemo + Small text-only.
            CatalogEntry(
                family: "Mistral", modelType: "mistral",
                summary: "Mistral 7B — Llama-shaped GQA backbone (Llama-compatible loader).",
                repos: [
                    "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
                    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                    "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
                    "mlx-community/Mistral-7B-Instruct-v0.3-bf16",
                    "mlx-community/Mistral-7B-v0.3-4bit",
                ]),
            // OLMo 1 / 2.
            CatalogEntry(
                family: "OLMo", modelType: "olmo / olmo2",
                summary: "AI2 OLMo 1 / 2 — open Llama-shaped research models.",
                repos: [
                    "mlx-community/OLMo-1B-hf-4bit-mlx",
                    "mlx-community/OLMo-7B-hf-4bit-mlx",
                    "mlx-community/OLMo-2-0425-1B-Instruct-bf16",
                    "mlx-community/OLMo-2-1124-7B-Instruct-bf16",
                    "mlx-community/OLMo-2-1124-7B-Instruct-4bit",
                    "mlx-community/OLMo-2-1124-7B-Instruct-6bit",
                    "mlx-community/OLMo-2-1124-7B-Instruct-8bit",
                    "mlx-community/OLMo-2-1124-13B-Instruct-4bit",
                    "mlx-community/OLMo-2-1124-13B-Instruct-6bit",
                    "mlx-community/OLMo-2-1124-13B-Instruct-8bit",
                    "mlx-community/OLMo-2-0325-32B-Instruct-4bit",
                ]),
            // Phi 3 / 3.5.
            CatalogEntry(
                family: "Phi 3", modelType: "phi3",
                summary: "Microsoft Phi-3 / 3.5 dense transformer.",
                repos: [
                    "mlx-community/Phi-3-mini-4k-instruct-4bit",
                    "mlx-community/Phi-3-mini-4k-instruct-8bit",
                    "mlx-community/Phi-3-mini-128k-instruct-4bit",
                    "mlx-community/Phi-3-mini-128k-instruct-8bit",
                    "mlx-community/Phi-3-medium-4k-instruct-4bit",
                    "mlx-community/Phi-3-medium-4k-instruct-8bit",
                    "mlx-community/Phi-3-medium-128k-instruct-4bit",
                    "mlx-community/Phi-3-medium-128k-instruct-8bit",
                    "mlx-community/Phi-3-medium-128k-instruct-bf16",
                    "mlx-community/Phi-3-small-8k-instruct-4bit",
                    "mlx-community/Phi-3-small-8k-instruct-8bit",
                    "mlx-community/Phi-3-small-128k-instruct-4bit",
                    "mlx-community/Phi-3-small-128k-instruct-8bit",
                    "mlx-community/Phi-3.5-mini-instruct-4bit",
                    "mlx-community/Phi-3.5-mini-instruct-6bit",
                    "mlx-community/Phi-3.5-mini-instruct-8bit",
                    "mlx-community/Phi-3.5-mini-instruct-bf16",
                ]),
            // Qwen 2 / 2.5.
            CatalogEntry(
                family: "Qwen 2", modelType: "qwen2",
                summary: "Qwen 2 / 2.5 dense transformer.",
                repos: [
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                    "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
                    "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
                    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                    "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
                    "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
                    "mlx-community/Qwen2.5-3B-Instruct-4bit",
                    "mlx-community/Qwen2.5-3B-Instruct-8bit",
                    "mlx-community/Qwen2.5-3B-Instruct-bf16",
                    "mlx-community/Qwen2.5-7B-Instruct-4bit",
                    "mlx-community/Qwen2.5-7B-Instruct-8bit",
                    "mlx-community/Qwen2.5-7B-Instruct-bf16",
                    "mlx-community/Qwen2.5-14B-Instruct-4bit",
                    "mlx-community/Qwen2.5-14B-Instruct-8bit",
                    "mlx-community/Qwen2.5-14B-Instruct-bf16",
                    "mlx-community/Qwen2.5-32B-Instruct-4bit",
                    "mlx-community/Qwen2.5-32B-Instruct-8bit",
                    "mlx-community/Qwen2.5-32B-Instruct-bf16",
                    "mlx-community/Qwen2.5-72B-Instruct-4bit",
                    "mlx-community/Qwen2.5-72B-Instruct-8bit",
                    "mlx-community/Qwen2.5-72B-Instruct-bf16",
                ]),
            // Qwen 3 dense.
            CatalogEntry(
                family: "Qwen 3", modelType: "qwen3",
                summary: "Qwen 3 dense — per-head q/k RMSNorm before RoPE.",
                repos: [
                    "mlx-community/Qwen3-0.6B-3bit",
                    "mlx-community/Qwen3-0.6B-4bit",
                    "mlx-community/Qwen3-0.6B-6bit",
                    "mlx-community/Qwen3-0.6B-8bit",
                    "mlx-community/Qwen3-0.6B-bf16",
                    "mlx-community/Qwen3-1.7B-3bit",
                    "mlx-community/Qwen3-1.7B-4bit",
                    "mlx-community/Qwen3-1.7B-5bit",
                    "mlx-community/Qwen3-1.7B-6bit",
                    "mlx-community/Qwen3-1.7B-8bit",
                    "mlx-community/Qwen3-1.7B-bf16",
                    "mlx-community/Qwen3-4B-3bit",
                    "mlx-community/Qwen3-4B-4bit",
                    "mlx-community/Qwen3-4B-6bit",
                    "mlx-community/Qwen3-4B-8bit",
                    "mlx-community/Qwen3-4B-bf16",
                    "mlx-community/Qwen3-8B-3bit",
                    "mlx-community/Qwen3-8B-4bit",
                    "mlx-community/Qwen3-8B-6bit",
                    "mlx-community/Qwen3-8B-8bit",
                    "mlx-community/Qwen3-8B-bf16",
                    "mlx-community/Qwen3-14B-3bit",
                    "mlx-community/Qwen3-14B-4bit",
                    "mlx-community/Qwen3-14B-6bit",
                    "mlx-community/Qwen3-14B-8bit",
                    "mlx-community/Qwen3-14B-bf16",
                    "mlx-community/Qwen3-32B-4bit",
                    "mlx-community/Qwen3-32B-6bit",
                    "mlx-community/Qwen3-32B-8bit",
                    "mlx-community/Qwen3-32B-bf16",
                ]),
            // SmolLM 1 / 2 / 3.
            CatalogEntry(
                family: "SmolLM", modelType: "smollm / smollm2 / smollm3",
                summary: "HuggingFace SmolLM 1 / 2 / 3 — 135M–3B Llama-shaped models.",
                repos: [
                    "mlx-community/SmolLM-135M-Instruct-4bit",
                    "mlx-community/SmolLM-135M-Instruct-8bit",
                    "mlx-community/SmolLM-135M-Instruct-fp16",
                    "mlx-community/SmolLM-360M-Instruct-4bit",
                    "mlx-community/SmolLM-360M-Instruct-8bit",
                    "mlx-community/SmolLM-360M-Instruct-fp16",
                    "mlx-community/SmolLM-1.7B-Instruct-4bit",
                    "mlx-community/SmolLM-1.7B-Instruct-8bit",
                    "mlx-community/SmolLM-1.7B-Instruct-fp16",
                    "mlx-community/SmolLM2-360M-Instruct-bf16-mlx",
                    "mlx-community/SmolLM2-135M-Instruct-8bit",
                    "mlx-community/SmolLM3-3B-bf16",
                    "mlx-community/SmolLM3-3B-3bit",
                    "mlx-community/SmolLM3-3B-4bit",
                    "mlx-community/SmolLM3-3B-5bit",
                    "mlx-community/SmolLM3-3B-6bit",
                    "mlx-community/SmolLM3-3B-8bit",
                ]),
            // Starcoder 2.
            CatalogEntry(
                family: "Starcoder 2", modelType: "starcoder2",
                summary: "BigCode Starcoder 2 — Llama-shaped code model (attention biases).",
                repos: [
                    "mlx-community/starcoder2-3b-4bit",
                    "mlx-community/starcoder2-7b-4bit",
                    "mlx-community/starcoder2-15b-4bit",
                    "mlx-community/starcoder2-15b-instruct-v0.1-4bit",
                    "mlx-community/starcoder2-15b-instruct-v0.1-8bit",
                ]),
            // Yi.
            CatalogEntry(
                family: "Yi", modelType: "yi",
                summary: "01.AI Yi — Llama-shaped dense backbone.",
                repos: [
                    "mlx-community/Yi-1.5-6B-Chat-bf16",
                    "mlx-community/Yi-1.5-6B-Chat-4bit",
                    "mlx-community/Yi-1.5-6B-Chat-8bit",
                    "mlx-community/Yi-1.5-9B-Chat-4bit",
                    "mlx-community/Yi-1.5-9B-Chat-8bit",
                    "mlx-community/Yi-1.5-34B-Chat-4bit",
                    "mlx-community/Yi-1.5-34B-Chat-8bit",
                ]),
        ]),

    // ───────────────────────── Mixture-of-experts ─────────────────────────
    CatalogGroup(
        title: "Mixture-of-experts",
        entries: [
            // Gemma 4 (Dense / E / MoE).
            CatalogEntry(
                family: "Gemma 4", modelType: "gemma4 / gemma4_text",
                summary: "Gemma 4 — Dense / E-series PLE / 26B-A4B MoE variants.",
                repos: [
                    "mlx-community/gemma-4-e2b-it-mxfp4",
                    "mlx-community/gemma-4-e4b-it-4bit",
                    "mlx-community/gemma-4-e4b-it-6bit",
                    "mlx-community/gemma-4-e4b-it-8bit",
                    "mlx-community/gemma-4-e4b-8bit",
                    "mlx-community/gemma-4-26b-a4b-it-4bit",
                    "mlx-community/gemma-4-31b-it-4bit",
                    "mlx-community/gemma-4-31b-it-8bit",
                ]),
            // GPT-OSS.
            CatalogEntry(
                family: "GPT-OSS", modelType: "gpt_oss",
                summary: "OpenAI GPT-OSS — 32-expert MoE, sliding/full attention, learned sinks.",
                repos: [
                    "mlx-community/gpt-oss-20b-MXFP4-Q8",
                    "mlx-community/gpt-oss-20b-mxfp4-bf16",
                    "mlx-community/gpt-oss-120b-4bit",
                    "mlx-community/gpt-oss-120b-MXFP4-Q8",
                    "mlx-community/gpt-oss-120b-mxfp4-bf16",
                ]),
        ]),

    // ───────────────────────── SSM / GDN / Conv hybrid ─────────────────────────
    CatalogGroup(
        title: "SSM / GDN / Conv hybrid",
        entries: [
            // FalconH1.
            CatalogEntry(
                family: "FalconH1", modelType: "falcon_h1",
                summary: "Per-layer Mamba 2 + attention (within-layer hybrid). Raw bf16/f16 only.",
                repos: [
                    "mlx-community/Falcon-H1-Tiny-90M-Instruct-bf16",
                    "mlx-community/Falcon-H1-0.5B-Instruct-bf16",
                    "mlx-community/Falcon-H1-1.5B-Instruct-bf16",
                    "mlx-community/Falcon-H1-3B-Instruct-4bit",
                    "mlx-community/Falcon-H1-7B-Instruct-4bit",
                ]),
            // Granite 4.
            CatalogEntry(
                family: "Granite 4", modelType: "granitemoehybrid",
                summary: "Stack-interleaved Mamba 2 / attention + dense or MoE FFN.",
                repos: [
                    "mlx-community/granite-4.0-350m-bf16",
                    "mlx-community/granite-4.0-350m-4bit",
                    "mlx-community/granite-4.0-350m-6bit",
                    "mlx-community/granite-4.0-350m-8bit",
                    "mlx-community/granite-4.0-1b-bf16",
                    "mlx-community/granite-4.0-1b-4bit",
                    "mlx-community/granite-4.0-1b-6bit",
                    "mlx-community/granite-4.0-1b-8bit",
                    "mlx-community/granite-4.0-h-350m-bf16",
                    "mlx-community/granite-4.0-h-350m-4bit",
                    "mlx-community/granite-4.0-h-350m-6bit",
                    "mlx-community/granite-4.0-h-350m-8bit",
                    "mlx-community/granite-4.0-h-1b-bf16",
                    "mlx-community/granite-4.0-h-1b-4bit",
                    "mlx-community/granite-4.0-h-1b-6bit",
                    "mlx-community/granite-4.0-h-1b-8bit",
                    "mlx-community/granite-4.0-h-micro-4bit",
                    "mlx-community/granite-4.0-h-micro-8bit",
                    "mlx-community/granite-4.0-h-tiny-4bit",
                    "mlx-community/granite-4.0-h-tiny-8bit",
                    "mlx-community/granite-4.0-h-small-4bit",
                    "mlx-community/granite-4.0-h-small-8bit",
                ]),
            // Jamba.
            CatalogEntry(
                family: "Jamba", modelType: "jamba",
                summary: "Mamba 1 + attention + dense/MoE FFN. Raw bf16/f16 only.",
                repos: [
                    "mlx-community/AI21-Jamba-Reasoning-3B-bf16"
                ]),
            // LFM2 / LFM2.5.
            CatalogEntry(
                family: "LFM2 / LFM2.5", modelType: "lfm2 / lfm2_moe",
                summary: "LiquidAI LFM2 — stack-interleaved short-conv + attention, optional MoE.",
                repos: [
                    "mlx-community/LFM2-350M-bf16",
                    "mlx-community/LFM2-350M-4bit",
                    "mlx-community/LFM2-350M-5bit",
                    "mlx-community/LFM2-350M-6bit",
                    "mlx-community/LFM2-350M-8bit",
                    "mlx-community/LFM2-700M-bf16",
                    "mlx-community/LFM2-700M-4bit",
                    "mlx-community/LFM2-700M-5bit",
                    "mlx-community/LFM2-700M-6bit",
                    "mlx-community/LFM2-700M-8bit",
                    "mlx-community/LFM2-1.2B-bf16",
                    "mlx-community/LFM2-1.2B-4bit",
                    "mlx-community/LFM2-1.2B-5bit",
                    "mlx-community/LFM2-1.2B-6bit",
                    "mlx-community/LFM2-1.2B-8bit",
                    "mlx-community/LFM2-2.6B-4bit",
                    "mlx-community/LFM2-2.6B-8bit",
                    "mlx-community/LFM2-8B-A1B-4bit",
                    "mlx-community/LFM2-8B-A1B-6bit",
                    "mlx-community/LFM2-8B-A1B-8bit",
                    "mlx-community/LFM2-8B-A1B-fp16",
                    "mlx-community/LFM2-24B-A2B-4bit",
                    "mlx-community/LFM2.5-350M-bf16",
                    "mlx-community/LFM2.5-350M-6bit",
                    "mlx-community/LFM2.5-350M-8bit",
                    "mlx-community/LFM2.5-1.2B-Instruct-bf16",
                    "mlx-community/LFM2.5-1.2B-Instruct-4bit",
                    "mlx-community/LFM2.5-1.2B-Instruct-5bit",
                    "mlx-community/LFM2.5-1.2B-Instruct-6bit",
                    "mlx-community/LFM2.5-1.2B-Instruct-8bit",
                ]),
            // Mamba 2 (dense SSM, no attention).
            CatalogEntry(
                family: "Mamba 2", modelType: "mamba2",
                summary: "Dense Mamba 2 selective-SSM (no attention).",
                repos: [
                    "mlx-community/mamba2-130m-4bit",
                    "mlx-community/mamba2-130m-8bit",
                    "mlx-community/mamba2-370m-4bit",
                    "mlx-community/mamba2-370m-8bit",
                    "mlx-community/mamba2-780m-4bit",
                    "mlx-community/mamba2-780m-8bit",
                    "mlx-community/mamba2-1.3b-4bit",
                    "mlx-community/mamba2-1.3b-8bit",
                    "mlx-community/mamba2-2.7b-4bit",
                    "mlx-community/mamba2-2.7b-8bit",
                ]),
            // NemotronH (stack-interleaved Mamba 2 / attention / MLP / MoE).
            CatalogEntry(
                family: "Nemotron H", modelType: "nemotron_h",
                summary: "Stack-interleaved Mamba 2 / attention / MLP. Raw bf16/f16 only.",
                repos: [
                    "nvidia/Nemotron-H-4B-Base-8K",
                    "nvidia/Nemotron-H-4B-Instruct-128K",
                    "nvidia/Nemotron-H-8B-Base-8K",
                    "nvidia/Nemotron-H-8B-Reasoning-128K",
                    "nvidia/Nemotron-H-47B-Base-8K",
                    "nvidia/Nemotron-H-47B-Reasoning-128K",
                    "nvidia/Nemotron-H-56B-Base-8K",
                    "nvidia/Nemotron-Cascade-2-30B-A3B",
                    "nvidia/Nemotron-Cascade-8B",
                    "nvidia/Nemotron-Cascade-8B-Thinking",
                    "nvidia/Nemotron-Cascade-14B-Thinking",
                    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                ]),
            // Qwen 3.5 / 3.6 hybrid (Gated Delta Net ↔ attention, dense / MoE).
            CatalogEntry(
                family: "Qwen 3.5 / 3.6", modelType: "qwen3_5 / qwen3_5_moe",
                summary: "Gated Delta Net ↔ attention hybrid; dense or MoE FFN.",
                repos: [
                    // 3.5 dense.
                    "mlx-community/Qwen3.5-0.8B-bf16",
                    "mlx-community/Qwen3.5-0.8B-3bit",
                    "mlx-community/Qwen3.5-0.8B-4bit",
                    "mlx-community/Qwen3.5-0.8B-5bit",
                    "mlx-community/Qwen3.5-0.8B-6bit",
                    "mlx-community/Qwen3.5-0.8B-8bit",
                    "mlx-community/Qwen3.5-0.8B-MLX-bf16",
                    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
                    "mlx-community/Qwen3.5-0.8B-MLX-8bit",
                    "mlx-community/Qwen3.5-0.8B-mxfp4",
                    "mlx-community/Qwen3.5-2B-bf16",
                    "mlx-community/Qwen3.5-2B-3bit",
                    "mlx-community/Qwen3.5-2B-4bit",
                    "mlx-community/Qwen3.5-2B-5bit",
                    "mlx-community/Qwen3.5-2B-6bit",
                    "mlx-community/Qwen3.5-2B-8bit",
                    "mlx-community/Qwen3.5-2B-MLX-bf16",
                    "mlx-community/Qwen3.5-4B-bf16",
                    "mlx-community/Qwen3.5-4B-3bit",
                    "mlx-community/Qwen3.5-4B-4bit",
                    "mlx-community/Qwen3.5-4B-6bit",
                    "mlx-community/Qwen3.5-4B-8bit",
                    "mlx-community/Qwen3.5-9B-bf16",
                    "mlx-community/Qwen3.5-9B-3bit",
                    "mlx-community/Qwen3.5-9B-4bit",
                    "mlx-community/Qwen3.5-9B-5bit",
                    "mlx-community/Qwen3.5-9B-6bit",
                    "mlx-community/Qwen3.5-9B-8bit",
                    "mlx-community/Qwen3.5-27B-bf16",
                    "mlx-community/Qwen3.5-27B-4bit",
                    "mlx-community/Qwen3.5-27B-5bit",
                    "mlx-community/Qwen3.5-27B-6bit",
                    "mlx-community/Qwen3.5-27B-8bit",
                    // 3.5 MoE.
                    "mlx-community/Qwen3.5-35B-A3B-bf16",
                    "mlx-community/Qwen3.5-35B-A3B-4bit",
                    "mlx-community/Qwen3.5-35B-A3B-5bit",
                    "mlx-community/Qwen3.5-35B-A3B-6bit",
                    "mlx-community/Qwen3.5-35B-A3B-8bit",
                    "mlx-community/Qwen3.5-122B-A10B-bf16",
                    "mlx-community/Qwen3.5-122B-A10B-4bit",
                    "mlx-community/Qwen3.5-122B-A10B-5bit",
                    "mlx-community/Qwen3.5-122B-A10B-6bit",
                    "mlx-community/Qwen3.5-122B-A10B-8bit",
                    "mlx-community/Qwen3.5-397B-A17B-4bit",
                    "mlx-community/Qwen3.5-397B-A17B-8bit",
                    // 3.6.
                    "mlx-community/Qwen3.6-27B-bf16",
                    "mlx-community/Qwen3.6-27B-4bit",
                    "mlx-community/Qwen3.6-27B-5bit",
                    "mlx-community/Qwen3.6-27B-6bit",
                    "mlx-community/Qwen3.6-27B-8bit",
                    "mlx-community/Qwen3.6-35B-A3B-bf16",
                    "mlx-community/Qwen3.6-35B-A3B-4bit",
                    "mlx-community/Qwen3.6-35B-A3B-5bit",
                    "mlx-community/Qwen3.6-35B-A3B-6bit",
                    "mlx-community/Qwen3.6-35B-A3B-8bit",
                ]),
        ]),

    // ───────────────────────── Diffusion text ─────────────────────────
    CatalogGroup(
        title: "Diffusion text",
        entries: [
            CatalogEntry(
                family: "Nemotron-Labs-Diffusion", modelType: "nemotron_labs_diffusion",
                summary: "Tri-mode — autoregressive / block diffusion / self-speculation.",
                repos: [
                    "nvidia/Nemotron-Labs-Diffusion-3B",
                    "nvidia/Nemotron-Labs-Diffusion-3B-Base",
                    "nvidia/Nemotron-Labs-Diffusion-8B",
                    "nvidia/Nemotron-Labs-Diffusion-8B-Base",
                    "nvidia/Nemotron-Labs-Diffusion-14B",
                    "nvidia/Nemotron-Labs-Diffusion-14B-Base",
                ])
        ]),

    // ───────────────────────── Vision-language ─────────────────────────
    CatalogGroup(
        title: "Vision-language",
        entries: [
            // Gemma 3 VL.
            CatalogEntry(
                family: "Gemma 3 VL", modelType: "gemma3 (+ vision_config)",
                summary: "SigLIP ViT tower + multi-modal projector + Gemma 3 text backbone.",
                repos: [
                    "mlx-community/gemma-3-4b-it-4bit",
                    "mlx-community/gemma-3-4b-it-6bit",
                    "mlx-community/gemma-3-4b-it-8bit",
                    "mlx-community/gemma-3-4b-it-bf16",
                    "mlx-community/gemma-3-12b-it-4bit",
                    "mlx-community/gemma-3-12b-it-bf16",
                    "mlx-community/gemma-3-27b-it-4bit",
                    "mlx-community/gemma-3-27b-it-bf16",
                ]),
            // Gemma 4 VL.
            CatalogEntry(
                family: "Gemma 4 VL", modelType: "gemma4 (+ vision_config)",
                summary: "Bespoke Gemma 4 ViT + multi-modal embedder + Gemma 4 text backbone.",
                repos: [
                    "mlx-community/gemma-4-e2b-it-mxfp4",
                    "mlx-community/gemma-4-e4b-it-4bit",
                    "mlx-community/gemma-4-e4b-it-6bit",
                    "mlx-community/gemma-4-e4b-it-8bit",
                    "mlx-community/gemma-4-26b-a4b-it-4bit",
                    "mlx-community/gemma-4-31b-it-4bit",
                    "mlx-community/gemma-4-31b-it-8bit",
                ]),
            // FastVLM (LlavaQwen2).
            CatalogEntry(
                family: "FastVLM", modelType: "llava_qwen2",
                summary: "Apple FastVLM — LLaVA-style Qwen2 backbone + ViT projector.",
                repos: [
                    "mlx-community/FastVLM-0.5B-bf16"
                ]),
            // Idefics3.
            CatalogEntry(
                family: "Idefics3", modelType: "idefics3",
                summary: "HuggingFace Idefics3 — Llama 3 backbone + SigLIP ViT.",
                repos: [
                    "mlx-community/Idefics3-8B-Llama3-3bit",
                    "mlx-community/Idefics3-8B-Llama3-4bit",
                    "mlx-community/Idefics3-8B-Llama3-6bit",
                    "mlx-community/Idefics3-8B-Llama3-8bit",
                    "mlx-community/Idefics3-8B-Llama3-bf16",
                ]),
            // MiniCPM-V 4.6.
            CatalogEntry(
                family: "MiniCPM-V", modelType: "minicpmv4_6",
                summary: "MiniCPM-V 4.6 — Qwen3 backbone + SigLIP ViT, resampler projector.",
                repos: [
                    "mlx-community/MiniCPM-V-4.6-4bit",
                    "mlx-community/MiniCPM-V-4.6-5bit",
                    "mlx-community/MiniCPM-V-4.6-8bit",
                    "mlx-community/MiniCPM-V-4.6-bf16",
                ]),
            // Mistral Small 3.x (Mistral3 VL).
            CatalogEntry(
                family: "Mistral Small 3 VL", modelType: "mistral3",
                summary: "Mistral Small 3.1 / 3.2 — bespoke ViT + Mistral text backbone.",
                repos: [
                    "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-3bit",
                    "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
                    "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-6bit",
                    "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-8bit",
                    "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-bf16",
                    "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
                    "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-8bit",
                    "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-bf16",
                ]),
            // Nemotron H VL.
            CatalogEntry(
                family: "Nemotron H VL", modelType: "VL (text_config.model_type = nemotron_h)",
                summary: "Shared SigLIP ViT + GELU projector + NemotronH text backbone.",
                repos: [
                    "nvidia/Nemotron-Nano-VL-8B-V1",
                    "nvidia/Nemotron-Nano-VL-12B-V2",
                ]),
            // Nemotron-Labs-Diffusion VLM.
            CatalogEntry(
                family: "Nemotron-Labs-Diffusion VLM",
                modelType: "nemotron_labs_diffusion_vlm",
                summary: "Tri-mode diffusion text backbone + Pixtral ViT vision tower.",
                repos: [
                    "nvidia/Nemotron-Labs-Diffusion-VLM-8B"
                ]),
            // Paligemma 1 / 2.
            CatalogEntry(
                family: "PaliGemma", modelType: "paligemma",
                summary: "Google PaliGemma 1 / 2 — SigLIP ViT + Gemma backbone (Gemma 2 on PG2).",
                repos: [
                    "mlx-community/paligemma-3b-mix-224-8bit",
                    "mlx-community/paligemma-3b-mix-448-8bit",
                    "mlx-community/paligemma2-3b-mix-224-4bit",
                    "mlx-community/paligemma2-3b-mix-224-6bit",
                    "mlx-community/paligemma2-3b-mix-224-8bit",
                    "mlx-community/paligemma2-3b-mix-224-bf16",
                    "mlx-community/paligemma2-3b-mix-448-4bit",
                    "mlx-community/paligemma2-3b-mix-448-8bit",
                    "mlx-community/paligemma2-3b-mix-448-bf16",
                    "mlx-community/paligemma2-10b-mix-224-4bit",
                    "mlx-community/paligemma2-10b-mix-224-bf16",
                    "mlx-community/paligemma2-10b-mix-448-4bit",
                    "mlx-community/paligemma2-10b-mix-448-bf16",
                    "mlx-community/paligemma2-28b-mix-224-4bit",
                    "mlx-community/paligemma2-28b-mix-224-bf16",
                    "mlx-community/paligemma2-28b-mix-448-4bit",
                    "mlx-community/paligemma2-28b-mix-448-bf16",
                ]),
            // Pixtral.
            CatalogEntry(
                family: "Pixtral", modelType: "pixtral",
                summary: "Mistral Pixtral 12B — bespoke RoPE-2D ViT + Mistral text backbone.",
                repos: [
                    "mlx-community/pixtral-12b-bf16",
                    "mlx-community/pixtral-12b-4bit",
                    "mlx-community/pixtral-12b-8bit",
                ]),
            // Qwen2-VL.
            CatalogEntry(
                family: "Qwen 2-VL", modelType: "qwen2_vl",
                summary: "Dynamic-res windowed-attention ViT + Qwen 2 text backbone.",
                repos: [
                    "mlx-community/Qwen2-VL-2B-Instruct-bf16",
                    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
                    "mlx-community/Qwen2-VL-2B-Instruct-8bit",
                    "mlx-community/Qwen2-VL-7B-Instruct-bf16",
                    "mlx-community/Qwen2-VL-7B-Instruct-4bit",
                    "mlx-community/Qwen2-VL-7B-Instruct-8bit",
                    "mlx-community/Qwen2-VL-72B-Instruct-4bit",
                    "mlx-community/Qwen2-VL-72B-Instruct-8bit",
                ]),
            // Qwen2.5-VL.
            CatalogEntry(
                family: "Qwen 2.5-VL", modelType: "qwen2_5_vl",
                summary: "Dynamic-res windowed-attention ViT + Qwen 2.5 backbone.",
                repos: [
                    "mlx-community/Qwen2.5-VL-3B-Instruct-3bit",
                    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                    "mlx-community/Qwen2.5-VL-3B-Instruct-6bit",
                    "mlx-community/Qwen2.5-VL-3B-Instruct-8bit",
                    "mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
                    "mlx-community/Qwen2.5-VL-7B-Instruct-3bit",
                    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
                    "mlx-community/Qwen2.5-VL-7B-Instruct-6bit",
                    "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
                    "mlx-community/Qwen2.5-VL-7B-Instruct-bf16",
                    "mlx-community/Qwen2.5-VL-32B-Instruct-4bit",
                    "mlx-community/Qwen2.5-VL-32B-Instruct-6bit",
                    "mlx-community/Qwen2.5-VL-32B-Instruct-8bit",
                    "mlx-community/Qwen2.5-VL-32B-Instruct-bf16",
                    "mlx-community/Qwen2.5-VL-72B-Instruct-4bit",
                    "mlx-community/Qwen2.5-VL-72B-Instruct-8bit",
                    "mlx-community/Qwen2.5-VL-72B-Instruct-bf16",
                ]),
            // Qwen3-VL dense.
            CatalogEntry(
                family: "Qwen 3-VL", modelType: "qwen3_vl",
                summary: "Dynamic-res full-attention ViT + Qwen 3 dense backbone.",
                repos: [
                    "mlx-community/Qwen3-VL-2B-Instruct-bf16",
                    "mlx-community/Qwen3-VL-2B-Instruct-4bit",
                    "mlx-community/Qwen3-VL-2B-Instruct-6bit",
                    "mlx-community/Qwen3-VL-2B-Instruct-8bit",
                    "mlx-community/Qwen3-VL-2B-Thinking-bf16",
                    "mlx-community/Qwen3-VL-2B-Thinking-4bit",
                    "mlx-community/Qwen3-VL-2B-Thinking-8bit",
                    "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                    "mlx-community/Qwen3-VL-4B-Instruct-8bit",
                    "mlx-community/Qwen3-VL-4B-Thinking-bf16",
                    "mlx-community/Qwen3-VL-4B-Thinking-4bit",
                    "mlx-community/Qwen3-VL-4B-Thinking-8bit",
                    "mlx-community/Qwen3-VL-8B-Instruct-bf16",
                    "mlx-community/Qwen3-VL-8B-Instruct-4bit",
                    "mlx-community/Qwen3-VL-8B-Instruct-6bit",
                    "mlx-community/Qwen3-VL-8B-Instruct-8bit",
                    "mlx-community/Qwen3-VL-8B-Thinking-bf16",
                    "mlx-community/Qwen3-VL-32B-Instruct-bf16",
                    "mlx-community/Qwen3-VL-32B-Instruct-4bit",
                    "mlx-community/Qwen3-VL-32B-Instruct-6bit",
                    "mlx-community/Qwen3-VL-32B-Instruct-8bit",
                    "mlx-community/Qwen3-VL-32B-Thinking-bf16",
                ]),
            // Qwen3-VL-MoE (shares the 30B-A3B / 235B-A22B sparse backbone).
            CatalogEntry(
                family: "Qwen 3-VL MoE", modelType: "qwen3_vl_moe",
                summary: "Dynamic-res ViT + Qwen 3.5-shaped GDN ↔ attention MoE backbone.",
                repos: [
                    "mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16",
                    "mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit",
                    "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",
                    "mlx-community/Qwen3-VL-30B-A3B-Instruct-6bit",
                    "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit",
                    "mlx-community/Qwen3-VL-30B-A3B-Thinking-bf16",
                    "mlx-community/Qwen3-VL-30B-A3B-Thinking-4bit",
                    "mlx-community/Qwen3-VL-235B-A22B-Instruct-3bit",
                    "mlx-community/Qwen3-VL-235B-A22B-Instruct-4bit",
                ]),
            // SmolVLM 1 / 2.
            CatalogEntry(
                family: "SmolVLM", modelType: "smolvlm",
                summary: "HuggingFace SmolVLM 1 / 2 — SmolLM backbone + SigLIP-So400m ViT.",
                repos: [
                    "mlx-community/SmolVLM-256M-Instruct-4bit",
                    "mlx-community/SmolVLM-256M-Instruct-8bit",
                    "mlx-community/SmolVLM-256M-Instruct-bf16",
                    "mlx-community/SmolVLM-500M-Instruct-4bit",
                    "mlx-community/SmolVLM-500M-Instruct-bf16",
                    "mlx-community/SmolVLM-Instruct-4bit",
                    "mlx-community/SmolVLM-Instruct-bf16",
                    "mlx-community/SmolVLM2-256M-Video-Instruct-mlx",
                    "mlx-community/SmolVLM2-500M-Video-Instruct-mlx",
                    "mlx-community/SmolVLM2-2.2B-Instruct-mlx",
                ]),
        ]),

    // ───────────────────────── Audio (STT / TTS / Omni) ─────────────────────────
    CatalogGroup(
        title: "Audio (STT / TTS / Omni)",
        entries: [
            // Chatterbox TTS.
            CatalogEntry(
                family: "Chatterbox TTS", modelType: "chatterbox / chatterbox_turbo",
                summary: "Resemble AI Chatterbox — Llama-shaped TTS backbone + S3 tokenizer.",
                repos: [
                    "mlx-community/Chatterbox-TTS-4bit",
                    "mlx-community/Chatterbox-TTS-8bit",
                    "mlx-community/Chatterbox-TTS-fp16",
                    "mlx-community/Chatterbox-Turbo-TTS-4bit",
                    "mlx-community/Chatterbox-Turbo-TTS-8bit",
                    "mlx-community/Chatterbox-Turbo-TTS-fp16",
                ]),
            // FireRedASR2.
            CatalogEntry(
                family: "FireRedASR2", modelType: "fireredasr2",
                summary: "FireRedTeam FireRedASR2-AED — conformer encoder + Llama decoder.",
                repos: [
                    "mlx-community/FireRedASR2-AED-mlx"
                ]),
            // Kokoro TTS.
            CatalogEntry(
                family: "Kokoro TTS", modelType: "kokoro",
                summary: "Kokoro 82M — StyleTTS2 acoustic + iSTFTNet GPU vocoder tail.",
                repos: [
                    "mlx-community/Kokoro-82M-4bit",
                    "mlx-community/Kokoro-82M-6bit",
                    "mlx-community/Kokoro-82M-8bit",
                    "mlx-community/Kokoro-82M-bf16",
                    "hexgrad/Kokoro-82M",
                ]),
            // Marvis / Sesame CSM TTS.
            CatalogEntry(
                family: "Marvis (CSM)", modelType: "csm / marvis",
                summary: "Sesame CSM-shaped dual-transformer TTS — Mimi-code frame generator.",
                repos: [
                    "Marvis-AI/marvis-tts-100m-v0.2-MLX-6bit",
                    "Marvis-AI/marvis-tts-100m-v0.2-MLX-8bit",
                    "Marvis-AI/marvis-tts-250m-v0.1-MLX-4bit",
                    "Marvis-AI/marvis-tts-250m-v0.1-MLX-8bit",
                    "Marvis-AI/marvis-tts-250m-v0.1-MLX-fp16",
                    "Marvis-AI/marvis-tts-250m-v0.2-MLX-4bit",
                    "Marvis-AI/marvis-tts-250m-v0.2-MLX-6bit",
                    "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit",
                    "sesame/csm-1b",
                ]),
            // Orpheus (LlamaTTS).
            CatalogEntry(
                family: "Orpheus (LlamaTTS)", modelType: "llama_tts / orpheus",
                summary: "Orpheus-style TTS — Llama 3 acoustic backbone + SNAC code decode loop.",
                repos: [
                    "mlx-community/orpheus-3b-0.1-ft-4bit",
                    "mlx-community/orpheus-3b-0.1-ft-6bit",
                    "mlx-community/orpheus-3b-0.1-ft-8bit",
                    "mlx-community/orpheus-3b-0.1-ft-bf16",
                    "mlx-community/orpheus-3b-0.1-pretrained-4bit",
                    "mlx-community/orpheus-3b-0.1-pretrained-6bit",
                    "mlx-community/orpheus-3b-0.1-pretrained-8bit",
                    "mlx-community/orpheus-3b-0.1-pretrained-bf16",
                ]),
            // PocketTTS.
            CatalogEntry(
                family: "Pocket TTS", modelType: "pocket_tts",
                summary: "Pocket TTS — compact StyleTTS2-style on-device synth.",
                repos: [
                    "mlx-community/pocket-tts-4bit",
                    "mlx-community/pocket-tts-6bit",
                    "mlx-community/pocket-tts-8bit",
                ]),
            // Qwen-Omni (omni-modal audio path).
            CatalogEntry(
                family: "Qwen-Omni", modelType: "qwen2_5_omni / qwen3_omni",
                summary: "Whisper-style audio encoder + Qwen text backbone (omni-modal).",
                repos: [
                    "Qwen/Qwen2.5-Omni-3B",
                    "Qwen/Qwen2.5-Omni-7B",
                    "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
                    "mlx-community/Qwen3-Omni-30B-A3B-Instruct-5bit",
                    "mlx-community/Qwen3-Omni-30B-A3B-Instruct-6bit",
                    "mlx-community/Qwen3-Omni-30B-A3B-Instruct-8bit",
                    "mlx-community/Qwen3-Omni-30B-A3B-Instruct-bf16",
                ]),
            // Qwen3-ASR.
            CatalogEntry(
                family: "Qwen3-ASR", modelType: "qwen3_asr",
                summary: "Qwen3-ASR — Qwen 3 backbone + audio encoder front-end.",
                repos: [
                    "mlx-community/Qwen3-ASR-0.6B-4bit",
                    "mlx-community/Qwen3-ASR-0.6B-5bit",
                    "mlx-community/Qwen3-ASR-0.6B-6bit",
                    "mlx-community/Qwen3-ASR-0.6B-8bit",
                    "mlx-community/Qwen3-ASR-0.6B-bf16",
                    "mlx-community/Qwen3-ASR-1.7B-4bit",
                    "mlx-community/Qwen3-ASR-1.7B-5bit",
                    "mlx-community/Qwen3-ASR-1.7B-6bit",
                    "mlx-community/Qwen3-ASR-1.7B-8bit",
                    "mlx-community/Qwen3-ASR-1.7B-bf16",
                ]),
            // Qwen3-TTS.
            CatalogEntry(
                family: "Qwen3-TTS", modelType: "qwen3_tts / qwen3_tts_base",
                summary: "Qwen3-TTS — talker + code predictor + ECAPA speaker + intrinsic codec.",
                repos: [
                    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
                    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-5bit",
                    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit",
                    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
                    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-5bit",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                ]),
            // SenseVoice STT.
            CatalogEntry(
                family: "SenseVoice", modelType: "sensevoice",
                summary: "Alibaba SenseVoice — SAN-M encoder + CTC head (non-autoregressive STT).",
                repos: [
                    "mlx-community/SenseVoiceSmall"
                ]),
            // Soprano TTS.
            CatalogEntry(
                family: "Soprano TTS", modelType: "soprano",
                summary: "Soprano 80M — compact StyleTTS2-flavored on-device synth.",
                repos: [
                    "mlx-community/Soprano-80M-bf16",
                    "mlx-community/Soprano-80M-4bit",
                    "mlx-community/Soprano-80M-5bit",
                    "mlx-community/Soprano-80M-6bit",
                    "mlx-community/Soprano-80M-8bit",
                    "mlx-community/Soprano-1.1-80M-bf16",
                    "mlx-community/Soprano-1.1-80M-5bit",
                    "mlx-community/Soprano-1.1-80M-6bit",
                    "mlx-community/Soprano-1.1-80M-8bit",
                ]),
            // Voxtral (realtime).
            CatalogEntry(
                family: "Voxtral", modelType: "voxtral_realtime",
                summary: "Mistral Voxtral — realtime STT/TTS (streaming).",
                repos: [
                    "mlx-community/Voxtral-Mini-3B-2507-bf16",
                    "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
                    "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
                    "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
                    "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
                    "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
                    "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
                ]),
            // Whisper STT.
            CatalogEntry(
                family: "Whisper STT", modelType: "whisper",
                summary: "OpenAI Whisper — encoder + cross-attending text decoder.",
                repos: [
                    "openai/whisper-tiny",
                    "mlx-community/whisper-tiny-mlx",
                    "mlx-community/whisper-tiny-4bit",
                    "mlx-community/whisper-tiny-8bit",
                    "mlx-community/whisper-tiny-fp16",
                    "mlx-community/whisper-base-mlx",
                    "mlx-community/whisper-base-4bit",
                    "mlx-community/whisper-base-8bit",
                    "mlx-community/whisper-small-mlx",
                    "mlx-community/whisper-small-4bit",
                    "mlx-community/whisper-small-8bit",
                    "mlx-community/whisper-medium-mlx",
                    "mlx-community/whisper-medium-4bit",
                    "mlx-community/whisper-medium-8bit",
                    "mlx-community/whisper-large-mlx",
                    "mlx-community/whisper-large-v2-mlx",
                    "mlx-community/whisper-large-v3-mlx",
                    "mlx-community/whisper-large-v3-4bit",
                    "mlx-community/whisper-large-v3-8bit",
                    "mlx-community/whisper-large-v3-fp16",
                    "mlx-community/whisper-large-v3-turbo-4bit",
                    "mlx-community/whisper-large-v3-turbo-8bit",
                    "mlx-community/whisper-large-v3-turbo-fp16",
                ]),
        ]),

    // ───────────────────────── Voice activity / diarization ─────────────────────────
    CatalogGroup(
        title: "Voice activity / diarization",
        entries: [
            CatalogEntry(
                family: "Silero VAD", modelType: "silero_vad",
                summary: "Streaming VAD — STFT + small gated-conv encoder. First-party weights.",
                repos: [
                    "snakers4/silero-vad"
                ]),
            CatalogEntry(
                family: "SmartTurn VAD", modelType: "smart_turn / smart_turn_v3",
                summary: "Conversational endpoint / turn-detection (Pipecat).",
                repos: [
                    "pipecat-ai/smart-turn-v3"
                ]),
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
            print(
                "── \(group.title) " + String(repeating: "─", count: max(0, 50 - group.title.count))
            )
            for e in group.entries {
                print("  \(e.family)  [\(e.modelType)]")
                print("    \(e.summary)")
                for repo in e.repos {
                    print("    • \(repo)")
                }
                print("")
            }
        }
        print(
            """
            Pass any example repo ID to a command:

              ffai generate --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Hello"
              ffai bench    --model mlx-community/Qwen3-1.7B-8bit --prompt "Hello" --stats

            Any mlx-format 2/3/4/5/6/8-bit conversion of a supported
            architecture also loads — the IDs above are the published
            examples we verified. Repos are resolved + cached through the
            HuggingFace Hub.
            """)
    }
}
