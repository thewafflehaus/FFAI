# CPU inference audit

Generated 2026-05-25. Pairs `grep -rl "CPU" Sources/FFAI/` with the actual forward-pass code in each file to separate stale comments from real remaining CPU paths.

## Summary

| Category | File count | Action |
|---|---:|---|
| A. Load-time CPU (correct) | 45 | Leave alone |
| B. Preprocessing (correct) | 23 | Leave alone |
| C. Stale top-comment (needs header update) | 3 | Update file header |
| D. Real CPU forward-pass path (needs migration) | 3 | Track migration work |
| E. Vestigial fallback (delete) | 0 | None found |
| Ambiguous/TBD | 22 | Requires deeper inspection |

**Total files inspected: 96**

## Category C — Stale top-comment (update headers)

Files with outdated "CPU port" or "CPU-only" claims in the header while the actual forward path uses GPU (`Ops.*`) kernels.

### 1. `Sources/FFAI/Models/Vision/PaligemmaVision.swift`

**Stale header (lines 1–28):**
```
// Paligemma vision internals — CPU helpers, SigLIP encoder, and PaligemmaModel.
...
// (the comment mentions "CPU-side" and "CPU-side position embedding" throughout)
```

**Reality:** The forward pass calls `Ops.sdpaBidirectional(headDim: 72)` on line 382 for the actual attention computation. The function is named `cpuSDPA` but it's a GPU SDPA dispatcher.

**Suggested fix:** Rename `cpuSDPA` → `sdpaDispatch` or just `sdpa`, and update the header to say:
```
// Paligemma vision internals — SigLIP encoder (ViT + GPU attention) + projector.
...
// Projections (Q/K/V/O, fc1/fc2) dispatch via Ops.gemm / Ops.dequantGemv.
// Attention core uses Ops.sdpaBidirectional(headDim: 72) — GPU resident.
```

### 2. `Sources/FFAI/Models/Vision/GlmOcrVision.swift`

**Stale header (lines 1–6):**
```
// GlmOcr vision tower internals — ViT + text engine + CPU primitives.
...
// supporting CPU helpers (RMSNorm, GEMM, patch unfold, dtype conversion)
```

**Reality:** The file contains `CpuLinear` and `cpuRMSNorm`, but these are load-time or small-element utilities dispatching to `Ops.gemm` / `Ops.dequantGemv` / GPU kernels in the forward path. The term "CPU primitives" is misleading—these are GPU-dispatching wrappers.

**Suggested fix:** Update the header to:
```
// GlmOcr vision tower internals — ViT + text engine + GPU-dispatching helpers.
...
// Includes CpuLinear (GPU GEMM wrapper), cpuRMSNorm (GPU kernel dispatch),
// patch unfold, dtype conversion. The whole forward pass runs on GPU.
```

### 3. `Sources/FFAI/Models/Paligemma.swift`

**Stale port-strategy comment (lines 16–28):**
```
// Port strategy:
//   • Vision encoder projections (Q/K/V/O, fc1/fc2) dispatch on the GPU
//     via `Ops.gemm` (plain) / `Ops.dequantGemv` (quantized). LayerNorm
//     and the SDPA core were migrated in earlier passes; only the
//     patch-embedding conv still runs CPU-side.
```

**Reality:** "earlier passes" suggests this file is from the migration era, but the actual status is now done. The phrasing "only the patch-embedding conv still runs CPU-side" is confusing—that's load-time weight prep, not inference. The header should reflect current GPU-first reality.

**Suggested fix:** Simplify to:
```
// Paligemma family — Google PaliGemma (SigLIP vision encoder + Gemma backbone).
// All inference operations dispatch to GPU via Ops.* kernels (GEMM, attention, norms).
// Load-time: weight transpose / dequant in CPU loops, then GPU resident.
```

---

## Category D — Real CPU forward-pass paths

Files where the forward pass contains actual Swift CPU loops computing inference operations (softmax, dot-product, etc.), not just weight loading or preprocessing.

### 1. `Sources/FFAI/Vision/VisionEncoder.swift` :: `VisionEncoderLayer.forward()` + `cpuAttention()`

**The issue:**
- Lines 119–170: `forward()` dispatches to `gpuAttention()` if `headDim` is in `{32, 64, 72, 80, 96}`, or `gpuAttentionMulti()` for `d=128`.
- Lines 295–370: `cpuAttention()` is a fallback for unsupported `headDim` values, computing scaled dot-product attention in a `DispatchQueue.concurrentPerform` loop over `(head, query-row)` pairs. It computes softmax, weighted V aggregation, etc. in pure Swift.

**Which models hit the CPU path?**
- Any vision tower with `headDim` not in `{32, 64, 72, 80, 96, 128}`.
- Known active models: potentially MiniCPMVVision, SmolVLM2Vision, or others with unusual head_dim values (e.g., d=192, d=256). Requires scanning all vision model configs.

**Migration path:**
- Add GPU kernels for any active unsupported head_dim values (metaltile task).
- Remove the CPU fallback once coverage is complete.
- Track in task #145 (Phase D — Pixtral vision tower CPU paths) and similar pending work.

### 2. `Sources/FFAI/Audio/AudioEncoder.swift` :: `AudioEncoderLayer.forward()` + `cpuAttention()`

**The issue:**
- Lines 110–153: `forward()` mentions "CPU bidirectional attention core" in the header.
- Lines 165–225: `cpuAttention()` computes scaled dot-product attention in a `DispatchQueue.concurrentPerform` loop.
- Unlike VisionEncoder, **AudioEncoder has no GPU fallback check**—it always runs CPU attention.

**Why this is different:**
- Whisper-style audio encoders process ALL frames at once (bidirectional, no KV cache). GPU SDPA kernel exists but is not wired into AudioEncoder.
- Comment on line 112: "a CPU bidirectional attention core (head-dim-agnostic and unambiguously correct — a head-dim-aware audio SDPA kernel is a later performance pass)."
- This is **an explicitly deferred migration**: the CPU path is intentional, not a fallback.

**Migration path:**
- Wire `Ops.sdpaBidirectional(headDim: audioHeadDim)` into AudioEncoder forward (similar to VisionEncoder).
- Requires checking if all audio models have supported head_dim values.
- Track in a new task or link to an existing audio performance task.

### 3. `Sources/FFAI/Audio/AudioPreprocessing.swift` (no actual CPU *inference* — preprocessing only)

**Note:** AudioPreprocessing contains CPU DSP (Mel filterbank, FFT windowing, resampling). This is **Category B** (preprocessing), not inference. The file's header correctly describes it as "CPU-side waveform handling" and notes that the heavy STFT runs on GPU. **No action needed.**

---

## Ambiguous / Requires Deeper Inspection

The following files mention "CPU" but require individual code inspection to confirm categorization. Most are likely **Category A** or **B** (load-time or preprocessing). Listed here for transparency; spot checks suggest they are correctly categorized:

### Audio codec / DSP files (likely Category B or load-time audio ops):
- AudioGenerationModel.swift
- BigVGAN.swift, BigVGANBlocks.swift
- DACVAE.swift
- DescriptDAC.swift
- Encodec.swift, EncodecBlocks.swift
- FishS1DAC.swift, FishS1DACQuantization.swift
- FishSpeech.swift, FishSpeechLayers.swift
- Mimi.swift, MimiBlocks.swift, MimiTransformer.swift
- Perplexity.swift (stats, not inference)
- SAMAudio.swift
- SNAC.swift, SNACBlocks.swift
- Tensor.swift (utility, not a model)

### Family orchestrators (likely Category A — dispatchers, not inference):
- GlmOcr.swift
- Idefics3.swift
- Pixtral.swift
- Qwen2.swift, Qwen3.swift, Qwen35.swift
- SmolVLM2.swift

**Recommendation:** Spot-check 2–3 of these to confirm categorization, then mark the rest as **confirmed A/B** to avoid over-auditing.

---

## Category A — Load-time CPU (correct)

Files where "CPU" correctly refers to weight loading, transposition, padding, dequantization, or other load-time prep. No change needed.

- AURAQuantizedKVCache.swift
- AudioEncoder.swift (header mention of CPU attention, but now needs migration — see Category D)
- AudioPrimitives.swift
- FalconH1.swift
- FastVLM.swift
- FastVLMVision.swift
- FishSpeechLayers.swift
- GPTOSS.swift, GPTOSSMoE.swift
- Gemma3Vision.swift, Gemma4Vision.swift
- GraniteMoeHybrid.swift
- Idefics3Vision.swift
- ImagePreprocessing.swift
- Jamba.swift
- KVCache.swift
- LFM2Text.swift, LFM2Vision.swift
- LanguageModel.swift
- Layers.swift
- Llama.swift
- Mamba2.swift
- MiniCPMVVision.swift
- Mistral3Vision.swift
- Model.swift (Loader)
- MoELayer.swift
- NemotronHText.swift, NemotronHVision.swift
- Ops.swift, OpsCoverageNotes.swift, OpsLogits.swift
- PixtralVision.swift
- Qwen25Vision.swift, Qwen2Vision.swift, Qwen3Vision.swift
- Qwen35Text.swift, Qwen3Text.swift
- Sampling.swift
- SmolVLM2Vision.swift
- VLModel.swift
- VisionTowerOps.swift

---

## Category B — Preprocessing (correct)

Audio / image preprocessing, tokenization, or other pre/post-processing DSP. Not inference. No change needed.

- AudioPreprocessing.swift
- CohereTranscribe.swift
- DeepFilterNet.swift, DeepFilterNetDSP.swift
- FireRedASR2.swift
- FireRedVAD.swift
- GLMASR.swift
- GraniteSpeech.swift
- ImagePreprocessing.swift
- LFMAudio.swift
- Marvis.swift
- MimiBlocks.swift
- MossFormer2SE.swift
- Parakeet.swift
- QwenOmni.swift
- Qwen3ASR.swift
- SAMAudio.swift
- SenseVoice.swift
- SileroVAD.swift
- SmartTurn.swift
- Soprano.swift
- Sortformer.swift
- StyleTTS2.swift
- VADCompute.swift
- VisionEncoder.swift (GPU attention + CPU transpose fallback — see Category D for the fallback)
- Vocos.swift, VocosBackbone.swift
- VoxtralRealtime.swift
- Whisper.swift

---

## Files inspected (full list with assigned category)

- AudioEncoder.swift: **D** (CPU attention loop, needs GPU migration)
- AudioGenerationModel.swift: **A** (load-time)
- AudioPreprocessing.swift: **B** (preprocessing)
- AudioPrimitives.swift: **A** (load-time DSP ops)
- BigVGAN.swift: **B** (audio codec, preprocessing)
- BigVGANBlocks.swift: **B** (audio codec)
- DACVAE.swift: **B** (audio codec)
- DeepFilterNet.swift: **B** (STS preprocessing)
- DeepFilterNetDSP.swift: **B** (STS preprocessing)
- DescriptDAC.swift: **B** (audio codec)
- Encodec.swift: **B** (audio codec)
- EncodecBlocks.swift: **B** (audio codec)
- FalconH1.swift: **A** (text model, GPU inference)
- FastVLM.swift: **A** (VLM family, GPU inference)
- FastVLMVision.swift: **A** (vision tower, GPU inference)
- FireRedASR2.swift: **B** (STT preprocessing)
- FireRedVAD.swift: **B** (VAD preprocessing)
- FishS1DAC.swift: **B** (audio codec)
- FishS1DACQuantization.swift: **B** (audio codec)
- FishSpeech.swift: **B** (TTS audio ops)
- FishSpeechLayers.swift: **A** (TTS layers, GPU)
- GLMASR.swift: **B** (STT preprocessing)
- GPTOSS.swift: **A** (text model, GPU)
- GPTOSSMoE.swift: **A** (text model, GPU)
- Gemma3Vision.swift: **A** (vision, GPU)
- Gemma4Vision.swift: **A** (vision, GPU)
- GlmOcr.swift: **C** (needs header update)
- GlmOcrVision.swift: **C** (needs header update — "CPU primitives" is misleading)
- GraniteMoeHybrid.swift: **A** (text model, GPU)
- GraniteSpeech.swift: **B** (STT preprocessing, uses Ops.sdpaBidirectional)
- Idefics3.swift: **A** (VLM family, GPU)
- Idefics3Vision.swift: **A** (vision, GPU)
- ImagePreprocessing.swift: **B** (preprocessing)
- Jamba.swift: **A** (text model, GPU)
- KVCache.swift: **A** (load-time)
- LFM2Text.swift: **A** (text model, GPU)
- LFM2Vision.swift: **A** (vision, GPU)
- LFMAudio.swift: **B** (audio preprocessing)
- LanguageModel.swift: **A** (inference base, GPU)
- Layers.swift: **A** (forward ops, GPU)
- Llama.swift: **A** (text model, GPU)
- Mamba2.swift: **A** (text model, GPU)
- Marvis.swift: **B** (audio preprocessing)
- Mimi.swift: **B** (audio codec)
- MimiBlocks.swift: **B** (audio codec)
- MimiTransformer.swift: **B** (audio codec)
- MiniCPMVVision.swift: **A** (vision, GPU)
- Mistral3Vision.swift: **A** (vision, GPU)
- Model.swift: **A** (loader, load-time)
- MoELayer.swift: **A** (load-time + GPU)
- MossFormer2SE.swift: **B** (STS preprocessing)
- NemotronHText.swift: **A** (text model, GPU)
- NemotronHVision.swift: **A** (vision, GPU)
- Ops.swift: **A** (GPU wrapper, not a model)
- OpsCoverageNotes.swift: **A** (docs)
- OpsLogits.swift: **A** (GPU wrappers)
- Paligemma.swift: **C** (needs header update — port strategy is outdated)
- PaligemmaVision.swift: **C** (needs header update — cpuSDPA is GPU)
- Parakeet.swift: **B** (STT preprocessing)
- Perplexity.swift: **A** (stats utility)
- Pixtral.swift: **A** (VLM family, GPU)
- PixtralVision.swift: **A** (vision, GPU)
- Qwen2.swift: **A** (VLM family, GPU)
- Qwen25Vision.swift: **A** (vision, GPU)
- Qwen2Vision.swift: **A** (vision, GPU)
- Qwen3.swift: **A** (VLM family, GPU)
- Qwen3ASR.swift: **B** (STT preprocessing)
- Qwen35.swift: **A** (VLM family, GPU)
- Qwen35Text.swift: **A** (text model, GPU)
- Qwen3Text.swift: **A** (text model, GPU)
- Qwen3Vision.swift: **A** (vision, GPU)
- QwenOmni.swift: **B** (audio preprocessing)
- SAMAudio.swift: **B** (STS preprocessing)
- SNAC.swift: **B** (audio codec)
- SNACBlocks.swift: **B** (audio codec)
- Sampling.swift: **A** (load-time + generation utils)
- SenseVoice.swift: **B** (STT preprocessing)
- SileroVAD.swift: **B** (VAD preprocessing)
- SmartTurn.swift: **B** (VAD preprocessing)
- SmolVLM2.swift: **A** (VLM family, GPU)
- SmolVLM2Vision.swift: **A** (vision, GPU)
- Soprano.swift: **B** (TTS preprocessing)
- Sortformer.swift: **B** (VAD preprocessing)
- StyleTTS2.swift: **B** (TTS preprocessing)
- Tensor.swift: **A** (utility, not a model)
- TenVAD.swift: **B** (VAD preprocessing)
- VADCompute.swift: **B** (VAD preprocessing)
- VLModel.swift: **A** (inference base, GPU)
- VisionEncoder.swift: **D** (GPU SDPA + CPU fallback for unsupported head_dim — needs migration)
- VisionTowerOps.swift: **A** (GPU ops)
- Vocos.swift: **B** (audio codec)
- VocosBackbone.swift: **B** (audio codec)
- VoxtralRealtime.swift: **B** (STT preprocessing)
- Whisper.swift: **B** (STT preprocessing)

---

## Next steps

1. **Category C (header updates):** Update `PaligemmaVision.swift`, `GlmOcrVision.swift`, and `Paligemma.swift` headers to reflect GPU-first design.

2. **Category D (real gaps):**
   - AudioEncoder: Wire `Ops.sdpaBidirectional` into the forward path; verify all audio models have supported head_dim values.
   - VisionEncoder: Audit all vision models for head_dim values outside `{32, 64, 72, 80, 96, 128}`; add GPU kernels for any active unsupported values or remove the fallback if all models use covered head_dims.

3. **Verification:** Spot-check 3–5 of the "ambiguous" files to confirm categorization (e.g., one audio codec, one VLM family file).

