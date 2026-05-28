# Known Issues — to investigate after the model-implementation wave lands

Open bugs and rough notes captured during the model-family port. Each entry has enough context that a future session can pick it up cold.

## GPU pin at 100% — long-running

**Symptom.** During (some) model loads / runs the GPU pegs at 100 % utilisation and stays there. The machine does NOT freeze — `FFAI_MAX_COMMAND_BUFFERS` caps the in-flight queue so we don't OOM the GPU — but the pin is real and the application becomes effectively unresponsive on that model.

**Status.** Cause unknown. Could be:
- A metaltile kernel with a dispatch shape that never converges (a reduction kernel encoded as `elementwiseGrid`, or an indirect dispatch reading a still-uninitialised threadgroup-count buffer).
- An FFAI-side path that re-encodes the same buffer in a busy loop without checking the prior commit's `status`.
- A specific kernel emitted by metaltile that has a per-thread loop parameterised on a stride that goes to zero / negative at certain shapes.

**Next steps when this lands as a priority.**
1. Reproduce deterministically — capture the exact model + prompt that pins the GPU.
2. Attach Metal System Trace (Instruments) while the pin is live. The trace shows which command buffer is in flight and which kernel is hung — that tells us metaltile-side vs FFAI-side immediately.
3. If metaltile-side: pull the kernel name from the trace, find the dispatch in `Ops.*`, audit the grid / threadgroup shape against the kernel's `## DISPATCH INVARIANTS` block.
4. If FFAI-side: check for a busy-loop that re-encodes the same compute pass without a `cmd.status` guard, or a missing `waitUntilCompleted` that lets the runtime queue grow unbounded.

**Mitigation today.** `FFAI_MAX_COMMAND_BUFFERS=16` (the production default since the wrong-dispatch GPU-pin post-mortem in `papers/optimizing-kernels-for-apple-m-series-architecture.md`) prevents the runaway queue from OOMing the device — that's why the pin is annoying but not fatal.

### 2026-05-23 status — post-Phase 6.5 / 7 wave + recent metaltile churn

**Hypothesis stack (ranked).** The Phase 6.5 (Vision) + Phase 7 (Audio) wave landed 18 commits on `ek/aura-port` without re-running any kernel or integration tests — every sub-agent prompt was "swift build only, don't run swift test on ModelTests". On the metaltile side, the `Sources/MetalTileSwift/Resources/kernels.metallib` was regenerated 2026-05-23 01:24 against `metaltile@840d281`. That regen pulled in:

- **#150** `mt_gated_delta_prep_chunk` — new chunked GDN prep+recurrence kernel.
- **#149** "remove all hand-written InlineMsl — port the MPP kernels to the coop_tile DSL" — the int4/int8/fp16 spread the user has been working on in another session.
- **#147** "All the remaining kernels" — large surface, broad.
- **#144** bm8 MoE BGEMM + dynamic-M qmm + naming standardisation.

Most likely cause: a kernel in #147 or #149 with a dispatch shape that goes pathological at production sizes but isn't caught by the small-N unit tests. The MPP coop_tile DSL port is the highest-suspicion bucket because cooperative-tensor type IDs drift across SDK versions and the DSL refactor reshuffled the lowering path.

**Test plan to localise (when the machine has been rebooted and the pin cleared).**

1. **Spot-check unit suite first.** `make test-unit` runs FFAITests + MetalTileSwiftTests at small problem sizes against the real GPU. If it fails, the failing test name pins down the broken kernel immediately. ~3 minutes.
2. **If unit suite passes, drop in a small integration.** `make test-integration --filter LlamaIntegrationTests` — Llama 3.2 1B exercises the core decode path (RMSNorm, RoPE, SDPA d128, gemv, argmax) without the multi-modal or hybrid kernels. If this pins the GPU, the bug is on a core kernel.
3. **Bisect by family if step 2 passes.** `make test-integration --filter <Qwen3 | Gemma3 | NemotronH | Jamba | GraniteMoeHybrid | GPTOSS | Pixtral | Mistral3 | Whisper | Marvis>` — pick the families that cover the most distinct kernels (MoE, GDN, SSM, AURA, VLM, audio).
4. **Once a family pins reliably:**
   - **Path A — capture in Instruments (Metal System Trace).** Launch the failing test under Instruments → Metal System Trace. The trace shows the in-flight command buffer + its current dispatch. Read off the kernel name. If it's specialised (e.g. `mt_qmm_int4_f16_bm8_*`), that's our suspect.
   - **Path B — `MTLCaptureManager` Metal frame capture from FFAI.** Wrap the suspect dispatch with `MTLCaptureManager.shared()` `.startCapture` / `.stopCapture`. Saves a `.gputrace` you can open in Xcode and step through.
5. **Localised? Audit the kernel's `## DISPATCH INVARIANTS` block in metaltile vs the FFAI `Ops.*` wrapper that calls it.** Reduction-mode kernel encoded as `elementwiseGrid` is the canonical freeze pattern (see `papers/optimizing-kernels-for-apple-m-series-architecture.md`).

**Capturing the pin live (if it can be done without reboot).** It's possible to attach Instruments to an already-pinned process and see which command buffer is in flight — `xcrun xctrace record --template "Metal System Trace" --attach <PID>`. Stop after ~5 seconds; the trace shows the queue depth and the hung dispatch. This is preferable to a reboot+rerun because we get the actual culprit rather than a candidate from rerunning.

**Live signal at the moment.** The user reports the pin reproduces right now (2026-05-23) after a fresh reboot — i.e. it's not a sticky-from-previous-session situation. Some path the OS goes through on this branch's `kernels.metallib` is enough to trigger it. That makes the kernel-side hypothesis (one of the freshly-regenerated kernels) substantially more likely than an FFAI-side busy-loop.

## CPU pin — vision / audio models burn all CPU cores

**Symptom.** During some integration tests (observed via `mactop`), all CPU cores spike to 100 % while the GPU stays well below saturation. GPU never sitting at 100 % during these runs is the giveaway — work that should be on the GPU is silently running on the CPU instead.

**Status.** Cause unknown. The vision (Phase 6.5) and audio (Phase 7) model families landed without a CPU-fallback audit, so the suspect surface is anything those pipelines touch where:

- A model op silently falls through to the parallel-CPU `concurrentPerform` core because no GPU wrapper was wired up (e.g. unknown SDPA `head_dim`, missing `Ops.conv2dDepthwise`, a custom audio op).
- A test or harness allocates / keeps tensors on CPU instead of dispatching to GPU.
- A test fixture / oracle is running its CPU reference path (parallelised across cores) and we're misreading that as the model itself pinning the CPU.

**Live signal.** `mactop` shows full CPU saturation across all cores during some currently-running tests; GPU has not been seen pegged during the same window. Indicates CPU is doing real work the GPU should be doing — or that a test-side CPU oracle dwarfs the GPU dispatch.

**Next steps when this lands as a priority.** Deferred until the current integration-test sweep finishes — that pass will surface more failures and give a fuller picture of which model families need attention. Then:

1. Re-run vision / audio model tests under `mactop` (or `powermetrics --samplers cpu_power,gpu_power`) to confirm the CPU pin reproduces and pin down which families.
2. For each pinning family, instrument the forward pass to log every op dispatch — find the calls that hit `concurrentPerform` / `cpuFallback` paths instead of `Ops.*` GPU dispatch.
3. Cross-reference against the FastVLM CPU depthwise-conv outlier in [§ Vision tower SDPA](#vision-tower-sdpa--head_dim-coverage) — likely the same class of gap, just not yet enumerated for the other vision / audio families.
4. File missing `Ops.*` wrappers as separate metaltile / FFAI follow-ups (kernel exists in metaltile but no Swift wrapper, or kernel itself missing).

## Vision tower SDPA — head_dim coverage

VLM vision towers run bidirectional multi-head attention; FFAI now ships GPU kernels for every head_dim in the cached zoo:

| head_dim | Kernel | Models |
|---|---|---|
| 32  | `Ops.sdpaBidirectional` (d32) | FastVLM (FastViT-HD) |
| 64  | `Ops.sdpaBidirectional` (d64) | SigLIP-base, CLIP-L, Mistral3, Gemma 4 E2B/E4B, Qwen3-VL 2B/4B |
| 72  | `Ops.sdpaBidirectional` (d72, ragged) | SigLIP-So400m → Paligemma, Gemma 3 VL, Gemma 4 26B/31B, Idefics3, Qwen3-VL-30B-A3B |
| 80  | `Ops.sdpaBidirectional` (d80, ragged) | Qwen2.5-VL |
| 96  | `Ops.sdpaBidirectional` (d96) | Qwen2-VL |
| 128 | `Ops.sdpaMulti(causal: false)` | Pixtral, Mistral3 Pixtral-base, GlmOcr |

`Sources/FFAI/VisionEncoder.swift::forward` dispatches based on the tower's `headDim`. Unknown head_dims (e.g. d80 at SigLIP-So-400m variants not yet seen) fall back to the parallel CPU `concurrentPerform` core. No tracked head_dim gaps as of 2026-05-24.

The remaining known cold-inference outlier is FastVLM at 1024px: its FastViT-HD stem has 256×256×96 + 128×128×192 depthwise convs that still run CPU because no `Ops.conv2dDepthwise(...)` wrapper exists. Tracked as a separate metaltile follow-up.

## Integration bisect — 2026-05-23/24 first-run findings

Per-suite serialised bisect run via `make integration-bisect` (commit `6047340`+) surfacing failures across all 74 integration tests. Fixed during the run:

- **ChatterboxIntegrationTests** — was failing on `unsupported dtype I64 for tensor "s3gen.speaker_encoder.xvector.block1.tdnnd6. nonlinear2.batchnorm.num_batches_tracked"`. Fixed in commit `af74ce7` by adding `.i64` / `.u64` to `DType` + making `SafeTensorsBundle` skip tensors with unsupported dtypes (Debug log only). Now PASSes in 21s.

- **CohereTranscribeIntegrationTests** — HF 404. Original `mlx-community/c4ai-aya-expanse-transcribe-mlx` archived; Cohere re-released as `cohere-transcribe-03-2026`. Fixed in commit `6ad9e73` pointing test at `mlx-community/cohere-transcribe-03-2026-mlx-8bit`.

Triage queue from same run (root-cause TBD — most look pre-existing):

- **DeepSeekR1DistillIntegrationTests** — R1-Distill-Llama-8B PASSes; R1-Distill-Qwen-1.5B produces degenerate output (token 15 repeated). Manually verified `mt_add_rms_norm` at hidden=1536 with `maxResidErr=0`, `maxNormedErr=1.4e-6` against CPU reference — fusion is NOT the regressor. Likely a pre- existing Qwen2 (R1-Distill base architecture) issue.

- **FastVLMIntegrationTests** — `Swift/ContiguousArrayBuffer.swift: 692: Fatal error: Index out of range` during the `load` test. Crash happens before the GPU-SDPA path runs, so the VLM agent's migration is not the cause. Suspect the SafeTensors-skip change or a pre-existing array index in the FastViT-HD loader.

- **FireRedASR2IntegrationTests** — `FFAI/Layers.swift:19: Precondition failed: Linear: weight must be 2D`. Some weight tensor is non-2D where a `Linear` is expected — checkpoint shape vs loader-side reshape mismatch.

- **FishSpeechIntegrationTests** — `safetensors file not found: models--mlx-community--fish-audio-s2-pro-8bit/snapshots/.../ model.safetensors`. Snapshot directory exists, file is missing — incomplete cache. Needs re-download or upstream check.

- **GLMASRIntegrationTests** — `Ops.swift:1072 Precondition failed: dequantGemv: input 65280 ≠ in_dim 1280`. Caller is feeding the wrong-rank tensor (looks like 51 rows of in_dim=1280 flat). Pre- existing shape bug in the GLMASR encoder/decoder.

Bisect continues; this list will be appended-to as more suites complete.

### 2026-05-27 integration-bisect findings (post-PR-#14)

Fresh bisect over a priority subset (Llama, GlmOcr, Paligemma, DeepSeekR1Distill, Qwen35Text, Qwen36Text, Gemma3Vision, MiniCPMV). Fixes landed in this pass:

- **GlmOcr** — Two loader bugs against the actual mlx-community `GLM-OCR-4bit` layout:
  1. mlx-community fuses the text MLP as `mlp.gate_up_proj` (uint32 weight + scales + biases triplet shaped `[2 × intermediate, ...]`) — the loader was requesting split `mlp.gate_proj` / `mlp.up_proj` and throwing on the first tensor lookup. Added `loadFusedGateUp` helper in `Sources/FFAI/Models/GlmOcr.swift` that slices the fused tensors along dim 0 and emits two independent `QuantizedLinear` halves.
  2. Vision tower prefix is `vision_tower.` in the actual checkpoint, not `model.visual.` (`Sources/FFAI/Models/GlmOcr.swift` line ~235). Corrected.
  Pre-existing sandwich-norm ordering bug also surfaced once load succeeded: `post_self_attn_layernorm` (shape `[1536]`) was being applied to `attnOut` (head-space `[2048]`) **before** `o_proj`, instead of after. Re-ordered so `o_proj` brings back to the residual stream first, then `post_self_attn_layernorm` (`Sources/FFAI/Models/Vision/GlmOcrVision.swift` line ~140). Also switched vision-tower attention from `Ops.sdpaMulti` (only d=128/256 supported) to `Ops.sdpaBidirectional` for the d=64 SigLIP-style heads the real `GLM-OCR-4bit` ships (hidden=1024 / heads=16).

- **MiniCPMV** — Three loader / prompt bugs against `mlx-community/MiniCPM-V-4.6-4bit`:
  1. Text-prefix rewrite was wrong: bundle ships `language_model.model.X` (outer `language_model.` wrapping inner `model.`) but loader did `prefixed("model.language_model.").withAddedPrefix("model.")` which returned an empty view (no key carries `model.language_model.` in this checkpoint). Replaced with passing `weights` straight through to `Qwen35Hybrid.loadModel` — Qwen3.5 auto-detects the `language_model.model.` prefix candidate.
  2. `vit_merger.*` keys live at top level of the bundle (sibling of `vision_tower.*`), not nested under `vision_tower.vit_merger.*` as the previous loader assumed. Added a `vitMergerOverride:` parameter to `MiniCPMVComposedEncoder.load` so the orchestrator can pass `weights.prefixed("vit_merger.")` directly. (`Sources/FFAI/Models/MiniCPMV.swift` + `Sources/FFAI/Models/Vision/MiniCPMVVision.swift`).
  3. SigLIP-2 `embeddings.patch_embedding.weight` ships in MLX's OHWI layout `[hidden, patch, patch, 3]` — `Ops.conv2d` expects PyTorch OIHW `[hidden, 3, patch, patch]`. Added the `transposeOHWItoOIHW` detection (mirrors Gemma3 / NemotronH / FastVLM / Pixtral).
  4. Chat-template fix: hand-rolled assistant prefix omitted the `<think>\n\n</think>\n\n` opener the Qwen3.5 backbone is fine-tuned on (`enable_thinking=false` default). Without it the model degenerated into a newline-loop. Added the opener to both image and video tests.

- **Qwen35TextIntegrationTests** (dense Qwen3.5-0.8B-4bit) — `mt_gated_delta_prep_step_{bf16,f16}` (PR #14's fused GDN-prep kernel) silently produces degenerate output at the dense `numValueHeads = 16` shape (kernel tuned for Qwen3.6-A3B's `numValueHeads = 32`). The integration test caught it as a 4 %-diversity Chinese math-loop ("根号内是负数…"). Diagnostic: setting `FFAI_GDN_NO_FUSED_PREP=1` restored coherent English. Added a `numValueHeads % 32 == 0` gate on `Qwen35GDNMixer.fused` (`Sources/FFAI/Models/Text/Qwen3xText.swift:1302-1305`); also tightened the existing fused-4-projection gate from `% 8 == 0` to `% 32 == 0` for the same reason. Also added an `FFAI_NO_FUSED_LM_HEAD` diagnostic env-var on `qwen35FinalNormLmHead`. **TODO (metaltile):** a paired GPU correctness test for `mt_gated_delta_prep_step_*` at `hv=16, dk=128, dv=128` against a CPU oracle, and a fix to the kernel's inner-loop / bounds-select arithmetic so the gate can be loosened back to `% 8 == 0` or removed.
- **Gemma3VisionIntegrationTests** — cache corruption: `~/.cache/huggingface/hub/models--mlx-community--gemma-3-4b-it-4bit/snapshots/.../model.safetensors.index.json` lists two shards but only a single `model.safetensors` is on disk (huggingface_hub partial-download / re-resume mismatch). Loader hardening landed in `Sources/FFAI/Loader/SafeTensors.swift:266-292`: if any index-listed shard is missing AND a single `model.safetensors` exists, fall back to the single file. Resolves the load failure on this exact corruption pattern. (The pre-existing 7-token-loop after preamble per F.9 in `session-plan.md` is a separate decode-time bug that needs the model to actually load to diagnose; loader fallback unblocks the investigation.)

- **PaligemmaIntegrationTests** — switched to use `Tests/Resources/testocr.png` + `ocr en` task prefix instead of `caption en` on `dog.jpeg`, and now asserts `VisionTestHelpers.expectRecognizesOCRText`. Generated: "This is a lot of 12 point text to test the" — matches the printed passage. PASSES in 234s.

Still pending (from earlier triage queue, not touched this pass): DeepSeekR1Distill (greedy-loop at temp=0 on R1-Distill-Qwen-1.5B — likely a model property, not a kernel bug; needs `mlx-lm` baseline comparison), FastVLM (likely 4-bit vision-tower not supported in `loadPW2D`; needs the checkpoint downloaded to confirm), FireRedASR2 (depthwise weight axis pick at `loadDepthwiseWeight`), GLMASR (audio adapter forward feeds quantized weights into `Ops.gemm` without dequant — verify the adapter load path). See agent diagnoses in the conversation transcript for line-level recommendations.

### 2026-05-27 Qwen3.6-27B-4bit degenerate output — likely a kernel-shape issue at hidden=5120

**Symptom.** `Qwen36TextIntegrationTests` (the new dense integration suite added 2026-05-27) loads `mlx-community/Qwen3.6-27B-4bit` successfully, the engine selection + shape assertions + cache-kind alignment all pass, the single-token forward smoke test produces top-5 ordered logits (`top[0] > top[4]`) — but greedy decode emits token 0 (`"!"`) every step. 32-token sample output: `"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"` — pure degenerate.

**Same engine, smaller model, works.** `mlx-community/Qwen3.5-0.8B-4bit` runs on the identical Qwen3.5 dense GDN-hybrid engine and produces coherent English. The regression is 27B-scale-specific.

**Hypothesis stack.** The 27B-4bit checkpoint differs from the 0.8B-4bit only at config-level dimensions; the engine code path is the same except for fast-path-eligibility predicates:

| Field                       | 0.8B (works)            | 27B (broken)              |
| --------------------------- | ----------------------- | ------------------------- |
| `hidden_size`               | 1024                    | 5120                      |
| `num_hidden_layers`         | 24                      | 64                        |
| `num_attention_heads`       | 8                       | 24                        |
| `num_key_value_heads`       | 2 (4:1 GQA)             | 4 (6:1 GQA)               |
| `head_dim`                  | 256                     | 256                       |
| `intermediate_size`         | 3072                    | 17408                     |
| `linear_num_value_heads`    | 16                      | 48                        |
| `linear_num_key_heads`      | 16                      | 16                        |
| `tie_word_embeddings`       | true (tied lm_head)     | false (untied quantized lm_head) |
| `vocab_size`                | 248_320                 | 248_320                   |

The `numValueHeads % 32 == 0` gate (added during the 2026-05-27 Qwen3.5 GDN-prep regression) excludes both 16 and 48 from the fused-prep fast path, so the suspect is NOT `mt_gated_delta_prep_step_*`. The remaining fast paths that DO fire at the 27B shape:

1. **`Ops.batchedQkvQmmFast`** — eligible when all 3 of q/k/v are int4 group_size=64 and `inDim % 512 == 0 && outDim % 8 == 0`. At 27B: inDim=5120 ✓, qDim=12288 ✓, kDim=vDim=1024 ✓. Was tuned against Qwen3.6-A3B's (hidden=2048, qDim=2048, kDim=vDim=512). The qDim difference (2048 vs 12288) is the largest delta; if the kernel has a fixed-tile assumption tied to `outDim ≤ N`, this is where it'd break.

2. **`Ops.rmsNormQgemvInt4Fast`** (the fused finalNorm + lm_head path in `qwen35FinalNormLmHead`) — eligible when lmHead is int4 group_size=64 and `inDim % 512 == 0 && outDim % 8 == 0`. At 27B: inDim=5120 ✓, outDim=248_320 ✓. Was tuned against Qwen3.6-A3B's (hidden=2048, vocab=248_320). Same in_dim delta (2048 → 5120) as #1.

3. **`Qwen35GDNMixer.fused4Eligible`** — already gated at `numValueHeads % 32 == 0`. 48 % 32 = 16 ≠ 0 → gate excludes 27B → fused4 path NOT fired. Safe.

4. **Per-token GDN host-loop** (the fallback when `fused == false`) — same code as 0.8B, but iterating across `numValueHeads = 48` instead of 16. Bug here would need a fixed-buffer-size assumption or a numerical accumulation issue.

**Recommended next-session investigation.** The 27B test cycle is 5–15 min per run (14 GB load + multi-token decode), so a one-at-a-time env-var A/B is the right shape:

1. `FFAI_NO_FUSED_LM_HEAD=1 swift test --filter Qwen36TextIntegrationTests` — if this restores coherent output, the bug is in `Ops.rmsNormQgemvInt4Fast` at the (in=5120, out=248_320) shape. Fix: gate the fused path on a verified `inDim ∈ {1024, 2048}` allowlist (or whatever the post-investigation safe list ends up being).
2. `FFAI_NO_FUSED_QKV=1 swift test --filter Qwen36TextIntegrationTests` — if this restores coherent output, the bug is in `Ops.batchedQkvQmmFast` at the (qDim=12288, kDim=vDim=1024, inDim=5120) shape. Fix: same gate pattern.
3. If neither flag alone fixes it, try them combined.
4. If still broken, the bug is in the legacy (non-fast) path — most likely `numValueHeads = 48` in the GDN host loop. Add a test-side print of `top[0]` token-id from the single-token smoke test to confirm whether even position 0 is degenerate. If yes, the per-token path itself is broken at this shape (not just the prefill).

Test stays **enabled** so the failure is visible in every bisect run rather than disappearing behind a skip-reason. Tracked in the task list as a follow-up.

### 2026-05-27 LFM2-MoE router shape mismatch — `gate.weight [E, 512]` vs hidden=2048

**Symptom.** `LFM2TextIntegrationTests` dense-350M case PASSES; the 8B-A1B MoE case crashes at load-time prewarm with `Ops.swift:406: Precondition failed: gemv: in_dim mismatch 512 vs 2048`.

**Root cause.** LFM2-8B-A1B-4bit ships the MoE router as `feed_forward.gate.weight` shape `[numExperts=32, 512]` — NOT the `[numExperts, hidden=2048]` shape `MoELayer.decode` expects. `MoELayer.decode` feeds the post-norm hidden state (`[hidden=2048]`) into the router, which fails the gemv precondition. LFM2-MoE likely uses a low-rank router projection (hidden → 512 → numExperts) or some other reduced-input routing scheme that the current `MoELayer` doesn't model.

**Status.** Pre-existing bug surfaced when the quantized stacked-switch_mlp expert slicing landed (`683f002`) and unblocked the MoE load path past the prior `unsupportedConfig` throw. Tracked as a separate task.

**Next steps.** Read the LFM2-MoE reference impl (`mlx-lm/models/lfm2_moe.py` or equivalent) to identify the actual routing pipeline. Likely needs either a router-input projection layer or a router-input-dim override on `MoELayer.decode`. Once the routing fires correctly, the rest of the MoE forward path (which uses standard expert dispatch on top of the quantized stacked triplets) should follow.

### 2026-05-27 GPU pin — `mt_sdpa_bidirectional_d64_*` at vision-tower production shape

**Symptom.** During the 2026-05-27 post-fix-wave bisect, `GlmOcrIntegrationTests` produced zero test-side output for 600 s, then `MiniCPMVIntegrationTests` did the same, then the OS pinned (full WindowServer freeze, hard reboot required). Diagnostic via `ffai inspect` confirmed both checkpoints load + text-decode cleanly — pin is isolated to the **vision-tower forward**.

**Root cause hypothesis.** `Ops.sdpaBidirectional` dispatched to the `mt_sdpa_bidirectional_d64_f32` kernel at the production shape `(nQHeads = 16, nQuery = 576, kvStride = 576)` never returns — the kernel's reduction / bounds arithmetic is presumed to have an inner-loop pathology at this geometry that no FFAI-side or metaltile-side test has ever exercised. The pre-PR-#14 dispatch was `Ops.sdpaMulti(d=64)` which had a precondition crash (sdpaMulti only supports d=128 / d=256), so the GLM-OCR vision tower had been dead-on-arrival for that checkpoint and the kernel was never actually fired against it. This pin is the canonical "wrong-dispatch reduction-mode kernel" failure pattern described in `papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md`.

**Blast radius.** The same `sdpaBidirectional` d=64 path is dispatched by every VLM with a SigLIP-base / CLIP-L vision tower: Gemma 4 E2B/E4B, SmolVLM2, LFM2-VL, plus any future consumer of `Vision/VisionEncoder.swift` at head_dim=64. None of these production shapes had a paired GPU correctness test in metaltile-std. Verified-safe shapes today: only **d=72** (SigLIP-So400m → Paligemma's 2026-05-27 PASS).

**Mitigation landed (commit `e2ee05d` + `f8b87cb`).** Every consumer of `Ops.sdpaBidirectional` at d ∈ {32, 64, 80, 96} now routes through a pure-CPU `DispatchQueue.concurrentPerform` per-(head, query) softmax loop. The shared `Vision/VisionEncoder.swift::forward` only takes the GPU path at d=72 (the confirmed-safe shape); every other dim falls through to the existing `cpuAttention` core. The per-family GlmOcr / Gemma4 / SmolVLM2 attention sites that bypass the shared encoder got the same CPU-loop treatment in-file. Qwen2.5-VL's d=80 full-attention block (`Qwen25Vision.swift`) is gated behind `FFAI_QWEN25VL_GPU_SDPA=1` so the bench surface stays available but the integration tests default to CPU. Slow (4096 patches × 27 layers ≈ minutes on M-series CPU) but safe — **no GPU pin under any production shape**.

**Open follow-up (metaltile).** Per dim ∈ {32, 64, 80, 96} that the audit flags as untested at production shape, ship a paired GPU correctness test in `crates/metaltile-std/tests/sdpa_bidirectional_d{N}_gpu_correctness.rs` that dispatches the kernel at the actual VLM-tower (nQHeads, nQuery, kvStride) shape against a naive CPU oracle. Fix whatever in the kernel's inner-loop / bounds-select / reduction-collapse arithmetic pegs the GPU at that shape. Then drop the per-family CPU gate in the corresponding `*Vision.swift` (and revert `useGPUSdpaBidirectional = headDim == 72` to the full `sdpaBidirectionalSupportedHeadDims.contains(headDim)` form in `Vision/VisionEncoder.swift`).

### 2026-05-28 Qwen3-1.7B-3bit degenerate output — model quality floor, NOT a kernel bug

**Symptom.** `Quantized3bitIntegrationTests` (mlx-community/Qwen3-1.7B-3bit) loads, passes every shape/config assertion, and the single-token forward smoke test returns finite top-5 logits — but greedy temp=0 decode of "Once upon a time, in a quiet village" collapses to a low-diversity loop: 19 unique tokens of 200 (10 %, `expectCoherentOutput` floor is 20 %). First 16 token ids `[11, 1052, 525, 2326, 4780, 11, 6941, 1283, 279, 11931, 304, 279, 829, 315, 279, 14126]` — real words ("…in the name of the…"), then repeats. **Deterministic**: identical output when the suite runs alone (NOT a memory-pressure artifact).

**Investigated 2026-05-28 — the int3 kernels are correct and unchanged.** Initial hypothesis was a metaltile codegen regression from `1522fbd`. Ruled out with hard evidence:
1. The entire 3-bit inference path (embedding gather + every projection/MLP + lm_head, both prefill and decode) uses exactly two int3 kernels: `dequant_gather_int3` and `dequant_gemv_int3`. (For `bits=3`, `QuantizedLinear.callMany` falls through to the per-row `dequantGemv` loop — there is no `mt_qmm_int3` path — see `Sources/FFAI/Layers.swift:240`.)
2. Added production-shape GPU correctness cells for `dequant_gemv_int3` at FFAI's exact decode geometry (in_dim=2048, group_size=64, f32/f16/bf16) in metaltile `crates/metaltile-std/tests/dequant_gemv_gpu_correctness.rs` (commit `5cca5ac` on `feat/sdpa-decode-d96`). All three PASS against the CPU oracle — the decode kernel is numerically correct at the production shape+dtype.
3. Emitted the int3 MSL at `0c0aef9` (pre-`1522fbd`) vs HEAD and diffed: `dequant_gemv_int3` differs only by instruction-scheduling order (declarations moved, zero numeric effect); `dequant_gather_int3` differs only by a missed integer-index CSE (`v55 = v50 + v_g` recomputed instead of reusing `v51` — same value, same address). Both kernels are **behaviorally identical** before and after `1522fbd`.

**Conclusion.** No FFAI bug, no metaltile bug. Pure 3-bit Qwen3-1.7B sits at its greedy-decode quality floor: temp=0 loops on real-but-repetitive text. Same class as the documented **Qwen3.5-0.8B** case (loops "even when content is correct — separate from the kernel bug"; that suite relaxed its diversity floor to 0.05) and the **Quantized2bitIntegrationTests** decision (no coherence assertion — pure 2-bit at 0.8B is below threshold). The 2026-05-24 "pass" was a marginal greedy trajectory that a harmless rounding shift (e.g. the int3 GEMV's reordered accumulation, or an FFAI-side quant-dispatch change from the Phase E / PR #14 wave) tipped below 20 %.

**Resolution (deferred, 2026-05-28).** `Quantized3bitIntegrationTests` is left **failing on purpose** as a documented known-issue (red signal kept visible) rather than relaxed — decision to regroup before deciding. The int3 kernels need no change; the new production-shape correctness cells (metaltile `5cca5ac`) are the durable signal that the path is correct. Two not-yet-ruled-out contributors to the marginal pass→fail since 2026-05-24, to revisit when this is picked back up: (a) this session's KV-cache changes (capacity/contextLength rename + incremental growth + residency-release) subtly shifting the decode trajectory, or (b) plain int3 fragility — 3-bit on a 1.7B model is close enough to collapse that small forward-pass rounding differences flip it. When ready, either bisect those or relax the test to the Qwen3.5-0.8B precedent (0.05 floor) / 2-bit-test style (drop coherence assert).

## Audio model CPU bottlenecks — Whisper + QwenOmni timeout

**Symptom.** `WhisperIntegrationTests` and `QwenOmniIntegrationTests` both hit the 900 s `gtimeout` cap during the integration bisect; the log captures only the build banner — the test process was SIGKILL'd before any test boundary was reported.

**Root cause — shared audio-encoder CPU attention.** Both families route through `Sources/FFAI/AudioEncoder.swift::cpuAttention` (line ~165). The Whisper-style encoder runs N encoder blocks (24 for Whisper-large-v3 / Qwen2.5-Omni) of multi-head bidirectional attention over `nAudioCtx = 1500` frames; each block performs `O(nFrames² · nHeads · headDim)` work on the CPU via `DispatchQueue.concurrentPerform`. With 1500² · 20 · 64 ≈ 2.9 G FLOPs per layer × 24 layers ≈ 70 G FLOPs of pure CPU work per `encodeAudio` call. That dominates the wall clock and pegs all cores (matches the broader CPU-pin symptom in [§ CPU pin](#cpu-pin--vision--audio-models-burn-all-cpu-cores)). The Q/K/V projections + LayerNorm + GELU + final GEMM are already on GPU; only the SDPA-equivalent attention core falls back to CPU.

**Whisper additional site — decoder.** `Sources/FFAI/Models/Whisper.swift::cpuAttention` (line ~424) runs the SAME CPU multi-head attention for the **decoder's self- AND cross-attention**, once per generated token. `generateTranscript` then loops `decoderLogits` for up to 224 tokens. The cross-attention K/V is recomputed against the full 1500-frame audio every step (no KV cache on the cross side), so each generated token re-pays a `nQuery · nKV · headDim` matmul that should be a single GPU SDPA dispatch. Whisper's transcribe test pays the encoder cost once + the decoder cost ≈ 224 times.

**QwenOmni site.** `QwenOmniModel.encodeAudio` (Sources/FFAI/Models/QwenOmni.swift:163) calls into the same shared `AudioEncoder`, so it inherits the encoder-side bottleneck. QwenOmni has no decoder cpuAttention of its own (it splices feature tokens into the Qwen3 text backbone, which is already on GPU). So the expected wall-clock improvement after migration is proportional to the time spent in the audio tower alone — but for a multi-second input that's still tens of seconds of pure CPU at the encoder.

**Migration needed.** Replace both `cpuAttention` cores (`AudioEncoder.swift::cpuAttention` and `Whisper.swift::cpuAttention`) with `Ops.sdpaBidirectional` (encoder) and `Ops.sdpaMulti(causal: true)` (decoder self-attn, already cached) + a cross-attention GPU dispatch. Whisper's head_dim is 64 (small) / 80 (medium) / 80 (large-v3) — all already covered by [§ Vision tower SDPA](#vision-tower-sdpa--head_dim-coverage). The cross-attention path also needs a precomputed-K/V variant: encoder K/V should be projected once per utterance, then reused for every decoder step (eliminates the per-token K/V recompute). Mirrors the VLM splice pattern.

**Status.** Diagnosed only; not in scope for the current bisect pass. Tracked here so the audio-tower migration can pick this up as the unified "audio CPU-attention → GPU SDPA" port.

## Quantization — missing 2-bit support + mixed-precision schemes

**Gap.** FFAI's quantized weight surface (`QuantizedLinear` / `QuantizedEmbedding` + the `int4` / `int8` MetalTile kernels) covers the symmetric per-group `affineQuantized` cases we ship today: `bits ∈ {3, 4, 5, 6, 8}`. The integration matrix (`Tests/ModelTests/Quantized{3,4,5,6,8}bitIntegrationTests.swift`) exercises each; all five passed in the 2026-05-24 bisect; as of 2026-05-28 the 3-bit case fails coherence at its greedy-decode quality floor (the int3 kernels are verified correct + unchanged — see § 2026-05-28 Qwen3-1.7B-3bit degenerate output). 4/5/6/8-bit still PASS.

What's NOT covered yet:

1. **2-bit quantization.** No kernel variant, no `QuantizedLinear` path. `mlx-community` ships 2-bit conversions of several models (Qwen 3 32B, Llama 3 70B, etc.) that we can't load. Need:
   - `dequant_gemv_int2_{f16,bf16}` and `mt_qmm_int2_*` kernels in metaltile (mirror the int4 codegen — same per-group `(scale, bias)` layout, just 2 bits per index).
   - `Ops.dequantGemv` + `Ops.qmm` dispatch table extended with `bits == 2` cases.
   - `QuantizedLinear` accepts `bits == 2` (currently the validator rejects it).
   - `Tests/ModelTests/Quantized2bitIntegrationTests.swift` mirroring the existing pattern.

2. **Mixed-precision (per-tensor-class) quantization.** Some `mlx-community` checkpoints quantize each weight class at a different bit-width — e.g. attention projections at int4, MLP gate/up/down at int3, embeddings at int8, lm_head at int6. These are advertised as `mixed_3_6`, `mixed_4_8` etc. in the model card. The current loader (`AnyLinear.load(...)`) assumes a single uniform `bits` value from the config's `quantization` block; it can't decode a heterogeneous scheme.

Sketch: extend the config decoder to recognise per-key bit-widths (probably a `quantization.weight_specs` dict in the config), thread the per-tensor `bits` through `loadLinear` / `loadEmbedding`, and let each call pick the matching kernel variant. No new kernels required — every selected bit-width already has one (modulo gap (1) for int2).

**Why it matters.** 2-bit is the only path to fit the 70B-class dense models (Llama 3 70B, Qwen 3 70B) in a single Apple Silicon device's wired memory; mixed-precision is what the better `mlx-community` conversions are starting to ship (preserves quality on the attention projections + lm_head while keeping the MLP slim). Without these we can't load 30-40 % of the recent zoo.

**Test bar.** When implemented:
- `make test-unit` must add a 2-bit dequant_gemv GPU correctness test mirroring the existing int4/int8 ones, against a CPU reference.
- `Quantized2bitIntegrationTests` + `MixedPrecisionIntegrationTests` added to the bisect runner.

## Tokenizers — InternLM2Tokenizer not registered

**Symptom.** Loading any InternLM2 checkpoint (e.g. `mlx-community/internlm2_5-7b-chat-4bit`, `internlm/internlm2-chat-1_8b`) logs: `Warning: Tokenizer model class InternLM2Tokenizer is not registered, falling back to a standard BPE implementation.`

**Status.** Low priority — InternLM2 is a niche family and the BPE fallback decodes coherently (verified: greedy generation produces correct output). The InternLM2-native tokenizer is SentencePiece-based with its own special-token handling, so the fallback could mishandle some special tokens / chat-template edges, but nothing wrong has been observed in practice.

**Next steps when this lands as a priority.** Find FFAI's tokenizer model-class registration (where `LlamaTokenizer` / `Qwen2Tokenizer` etc. are mapped) and register `InternLM2Tokenizer` against the SentencePiece/BPE handler it needs. Verify the warning is gone and special tokens resolve on `internlm2_5-7b-chat-4bit`.

**Context.** Surfaced 2026-05-28 while adding the dedicated InternLM2 loader (name remap + fused-`wqkv` split). Independent of the loader fix.


