# Supported Models

FFAI ships the full text-LLM family set today; all run real
HuggingFace checkpoints end-to-end through `Model.load("org/repo")`.
That spans the dense transformer families (Llama + the
Llama-compatible zoo, Qwen 2 / 3, Mistral, Phi, Gemma 3, Gemma 4),
the GPT-OSS-20B MoE, the dense SSM family Mamba 2, the SSM/GDN hybrid
families (FalconH1, NemotronH, Granite4, Jamba, Qwen 3.5), the
LFM2 / LFM2.5 conv+attention hybrid family, and
Nemotron-Labs-Diffusion. Vision-language (VLM) and audio families ship
too — see the [vision-language](#vision-language-families-phase-65),
[audio](#audio-families-phase-7) and [neural codec](#neural-audio-codecs-phase-7)
sections below.

This page is the canonical landing for:

- which architectures are in tree
- which sizes / quantizations have been exercised
- known gaps per family

For porting a new architecture, see
[developing/adding-a-model.md](developing/adding-a-model.md).

## In-tree families

| Family | File | `model_type` | `architectures` | Variants |
|---|---|---|---|---|
| **Llama 3.x** | [`Models/Text/Llama.swift`](../Sources/FFAI/Models/Text/Llama.swift) | `llama` | `LlamaForCausalLM` | `LlamaDense` |
| **SmolLM 1/2/3** | [`Models/SmolLM.swift`](../Sources/FFAI/Models/SmolLM.swift) | `smollm`, `smollm2`, `smollm3` | `SmolLM{,2,3}ForCausalLM` | reuses the Llama loader |
| **OLMo 1/2** | [`Models/OLMo.swift`](../Sources/FFAI/Models/OLMo.swift) | `olmo`, `olmo2` | `Olmo{,2}ForCausalLM` | reuses the Llama loader |
| **Granite 3** | [`Models/Granite3.swift`](../Sources/FFAI/Models/Granite3.swift) | `granite` | `GraniteForCausalLM` | IBM Granite v3 dense; reuses the Llama loader |
| **Yi** | [`Models/Yi.swift`](../Sources/FFAI/Models/Yi.swift) | `yi` | `YiForCausalLM` | reuses the Llama loader |
| **InternLM 2** | [`Models/InternLM2.swift`](../Sources/FFAI/Models/InternLM2.swift) | `internlm2` | `InternLM2ForCausalLM` | reuses the Llama loader |
| **Starcoder 2** | [`Models/Starcoder2.swift`](../Sources/FFAI/Models/Starcoder2.swift) | `starcoder2` | `Starcoder2ForCausalLM` | reuses the Llama loader |
| **Qwen 2** | [`Models/Text/Qwen2Text.swift`](../Sources/FFAI/Models/Text/Qwen2Text.swift) | `qwen2` | `Qwen2ForCausalLM` | reuses the Llama loader |
| **Qwen 3** | [`Models/Text/Qwen3Text.swift`](../Sources/FFAI/Models/Text/Qwen3Text.swift) | `qwen3` | `Qwen3ForCausalLM` | `Qwen3Dense` |
| **Mistral** | [`Models/Text/Mistral.swift`](../Sources/FFAI/Models/Text/Mistral.swift) | `mistral` | `MistralForCausalLM` | reuses the Llama loader |
| **Phi 3** | [`Models/Text/Phi.swift`](../Sources/FFAI/Models/Text/Phi.swift) | `phi3` | `Phi3ForCausalLM` | `Phi3Dense` |
| **Gemma 2** | [`Models/Text/Gemma2.swift`](../Sources/FFAI/Models/Text/Gemma2.swift) | `gemma2` | `Gemma2ForCausalLM` | `Gemma2Dense` |
| **Gemma 3** | [`Models/Text/Gemma3Text.swift`](../Sources/FFAI/Models/Text/Gemma3Text.swift) | `gemma3`, `gemma3_text` | `Gemma3ForCausalLM` | `Gemma3Dense` |
| **Mamba 2** | [`Models/Text/Mamba2.swift`](../Sources/FFAI/Models/Text/Mamba2.swift) | `mamba2` | `Mamba2ForCausalLM` | `Mamba2Dense` |
| **FalconH1** | [`Models/Text/FalconH1.swift`](../Sources/FFAI/Models/Text/FalconH1.swift) | `falcon_h1` | `FalconH1ForCausalLM` | `FalconH1Hybrid` |
| **NemotronH** | [`Models/Text/NemotronHText.swift`](../Sources/FFAI/Models/Text/NemotronHText.swift) | `nemotron_h` | `NemotronHForCausalLM` | `NemotronHHybrid` |
| **Nemotron-Labs-Diffusion** | [`Models/Text/NemotronDiffusionText.swift`](../Sources/FFAI/Models/Text/NemotronDiffusionText.swift) | `nemotron_labs_diffusion` | `NemotronDiffusionModel` | `NemotronDiffusionDense` |
| **Granite4** | [`Models/Text/Granite4Text.swift`](../Sources/FFAI/Models/Text/Granite4Text.swift) | `granitemoehybrid` | `Granite4ForCausalLM` | `Granite4Hybrid` |
| **Jamba** | [`Models/Text/Jamba.swift`](../Sources/FFAI/Models/Text/Jamba.swift) | `jamba` | `JambaForCausalLM` | `JambaHybrid` |
| **LFM2 / LFM2.5** | [`Models/Text/LFM2Text.swift`](../Sources/FFAI/Models/Text/LFM2Text.swift) | `lfm2`, `lfm2_moe` | `Lfm2ForCausalLM`, `Lfm2MoeForCausalLM` | `LFM2Dense`, `LFM2MoE` |
| **Qwen 3.5 / 3.6** | [`Models/Text/Qwen3xText.swift`](../Sources/FFAI/Models/Text/Qwen3xText.swift) | `qwen3_5`, `qwen3_5_moe` | `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration` | `Qwen35Hybrid` |
| **Gemma 4** | [`Models/Text/Gemma4Text.swift`](../Sources/FFAI/Models/Text/Gemma4Text.swift) | `gemma4`, `gemma4_text` | `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration` | `Gemma4Dense`, `Gemma4E`, `Gemma4MoE` |
| **GPT-OSS** | [`Models/Text/GPTOSS.swift`](../Sources/FFAI/Models/Text/GPTOSS.swift) | `gpt_oss` | `GptOssForCausalLM` | `GPTOSSMoEVariant` |

### Audio families (Phase 7)

The audio families do **not** route through `ModelRegistry` /
`LanguageModel` — those describe a pure text-in / text-out causal
decoder. An STT model is audio-in / text-out, a TTS model is text-in /
audio-out. They load through [`AudioModelRegistry`](../Sources/FFAI/AudioModelRegistry.swift)
instead, which inspects `config.json`, picks the family, and reports
the audio `Capability` set.

| Family | File | `model_type` | Capability | Notes |
|---|---|---|---|---|
| **Whisper** | [`Models/Whisper.swift`](../Sources/FFAI/Models/Whisper.swift) | `whisper` | `speechToText` | STT, tiny → large-v3 (one variant). `AudioEncoder` + a causal text decoder cross-attending to the audio features. |
| **SenseVoice** | [`Models/SenseVoice.swift`](../Sources/FFAI/Models/SenseVoice.swift) | `sensevoice` | `speechToText` | Non-autoregressive STT — a SAN-M encoder (fused QKV + FSMN depthwise-conv memory block) and a CTC head. Greedy CTC collapse instead of a decoder loop. Kaldi-style FBANK + LFR front-end. |
| **Kokoro** | [`Models/Kokoro.swift`](../Sources/FFAI/Models/Kokoro.swift) | `kokoro` | `textToSpeech` | TTS. Ships the GPU iSTFTNet vocoder tail (`Ops.vocoderISTFT`); the StyleTTS2 acoustic front-end is a later port. |
| **Qwen-Omni** | [`Models/QwenOmni.swift`](../Sources/FFAI/Models/QwenOmni.swift) | `qwen2_5_omni`, `qwen3_omni` | `omniAudio` | Audio-in path: a Whisper-style encoder projecting into the text backbone hidden dim. Vision path is the Qwen-VL port. |
| **LlamaTTS** | [`Models/LlamaTTS.swift`](../Sources/FFAI/Models/LlamaTTS.swift) | `llama_tts`, `orpheus` | `textToSpeech` | Orpheus-style TTS on a Llama 3.x backbone (reuses the `LlamaModel` engine). Adds the Orpheus token protocol + autoregressive SNAC-code decode loop; `generateCodes` emits de-interleaved SNAC code planes. The SNAC neural codec (waveform tail) is a separate codec port. |
| **Marvis** | [`Models/Marvis.swift`](../Sources/FFAI/Models/Marvis.swift) | `csm`, `marvis` | `textToSpeech` | Sesame CSM dual-transformer TTS — a backbone + depth-decoder (both built on FFAI's `LlamaLayer` blocks), embedding tables + per-codebook audio heads. `generateFrames` emits the `[K, nFrames]` Mimi code matrix. The Mimi neural codec (waveform tail) is a separate codec port. |
| **Qwen3TTS** | [`Models/Qwen3TTS.swift`](../Sources/FFAI/Models/Qwen3TTS.swift) | `qwen3_tts` | `textToSpeech` | Qwen's four-part TTS (talker + code predictor + ECAPA speaker encoder + intrinsic speech-tokenizer codec). **Staged port** — stage 1 ships config decoding + family detection; the talker (Qwen3 stack + 3D mRoPE), code predictor and codec are follow-on stages. `synthesize` throws `synthesisNotWired` until then. |

Whisper, Kokoro and Qwen-Omni share the
[`AudioEncoder`](../Sources/FFAI/AudioEncoder.swift)
module (a Whisper-style conv stem + bidirectional transformer) and the
[`AudioPreprocessing`](../Sources/FFAI/AudioPreprocessing.swift)
front-end (log-Mel STFT framing). The three FFAI audio kernels —
`mel_spectrogram`, `audio_conv1d`, `vocoder_istft` — are wrapped by
`Ops.melSpectrogram` / `Ops.audioConv1d` / `Ops.vocoderISTFT`.

SenseVoice is a standalone SAN-M family: its Kaldi-style FBANK front-end
(per-frame mean removal, pre-emphasis, power-of-two FFT, HTK Mel, plain
log) and low-frame-rate stacking differ from the shared log-Mel
front-end, so it carries its own `SenseVoiceFrontEnd` CPU path. The
FSMN memory block is a depthwise (per-channel) 1-D convolution — also a
CPU path, since `Ops.audioConv1d` is a dense (non-grouped) conv.

### Voice-activity-detection families (Phase 7)

VAD models have a third contract — audio waveform in, per-frame
speech-probability stream out — so they load through their own
[`VADModelRegistry`](../Sources/FFAI/VADModelRegistry.swift) rather than
`ModelRegistry` or `AudioModelRegistry`. Each family exposes
`loadFromDirectory` / `fromPretrained` and a `detect(audio:sampleRate:)`
returning a [`VADOutput`](../Sources/FFAI/VADOutput.swift).

| Family | File | `model_type` | Notes |
|---|---|---|---|
| **SileroVAD** | [`Models/SileroVAD.swift`](../Sources/FFAI/Models/SileroVAD.swift) | `silero_vad` | Streaming VAD — STFT front-end + a small gated-conv encoder, 16 kHz and 8 kHz branch configs. |
| **SmartTurn** | [`Models/SmartTurn.swift`](../Sources/FFAI/Models/SmartTurn.swift) | `smart_turn`, `smart_turn_v3` | Conversational endpoint / turn detection. |

Sortformer (multi-speaker diarization) is recognized by the registry;
its loader is a follow-on port.

### Vision-language families (Phase 6.5)

VL checkpoints store a text backbone under `language_model.` plus a
vision tower; `SafeTensorsBundle.prefixed(_:)` lets the existing text
loader run unchanged on the sub-tree, and [`VisionModel`](../Sources/FFAI/VisionModel.swift)
splices the projected image tokens into the text stream.

| Family | File | `architectures` | Notes |
|---|---|---|---|
| **Gemma 3 VL** | [`Models/Gemma3.swift`](../Sources/FFAI/Models/Gemma3.swift) | `Gemma3ForConditionalGeneration` | SigLIP ViT tower + multi-modal projector (patch-grid pool) + Gemma 3 text backbone. |
| **Qwen 2.5-VL** | [`Models/Qwen25.swift`](../Sources/FFAI/Models/Qwen25.swift) | `Qwen2_5_VLForConditionalGeneration` | Dynamic-resolution windowed-attention ViT tower + the Qwen 2.x text backbone routed through the Llama dense engine (embedding-input forward). |
| **Qwen 3-VL** | [`Models/Qwen3.swift`](../Sources/FFAI/Models/Qwen3.swift) | `Qwen3VLForConditionalGeneration` | Dynamic-resolution full-attention ViT tower (LayerNorm pre-norms, GELU MLP, learned position table) + the Qwen 3 dense text backbone (embedding-input forward). |
| **Qwen 3-VL-MoE** | [`Models/Qwen35.swift`](../Sources/FFAI/Models/Qwen35.swift) | `Qwen3VLMoeForConditionalGeneration` | The Qwen3-VL ViT tower + the Qwen 3.5 mixture-of-experts hybrid text backbone (Gated Delta Net ↔ attention, block-sparse MoE FFN), embedding-input forward. |
| **Gemma 4 VL** | [`Models/Gemma4.swift`](../Sources/FFAI/Models/Gemma4.swift) | `Gemma4ForConditionalGeneration` (+ `vision_config`) | Bespoke Gemma 4 ViT tower (RoPE attention with multi-dimensional positions, q/k/v RMSNorms, four per-block GemmaRMSNorms, attention-pooling head) + multi-modal embedder + Gemma 4 text backbone (embedding-input forward). |
| **Nemotron-VLM** | [`Models/NemotronH.swift`](../Sources/FFAI/Models/NemotronH.swift) | VL checkpoint whose `text_config.model_type` is `nemotron_h` | ViT tower (shared SigLIP-shaped `VisionEncoder`) + two-layer GELU-MLP projector + the NemotronH stack-interleaved hybrid text backbone (Mamba 2 / attention / dense-MLP, embedding-input forward). |

All VL families recognized by `ModelRegistry` are now wired to a
checkpoint loader.

### Neural audio codecs (Phase 7)

Codecs (encoder + quantizer + decoder) turn a waveform into discrete
codes and back; the autoregressive TTS families (LlamaTTS, Marvis, …)
emit codes that a codec renders to audio. They live under
[`Sources/FFAI/Audio/`](../Sources/FFAI/Audio).

| Codec | File | Notes |
|---|---|---|
| **SNAC** | [`Audio/SNAC.swift`](../Sources/FFAI/Audio/SNAC.swift) | Multi-scale residual-VQ codec — the waveform tail for Orpheus-style (LlamaTTS) synthesis. `encode(waveform:)` / `decode(codes:)`. |
| **Encodec** | [`Audio/Encodec.swift`](../Sources/FFAI/Audio/Encodec.swift) | Meta's EnCodec — SEANet encoder/decoder with a 2-layer LSTM bottleneck and a residual-VQ quantizer. Bandwidth-selectable codebook count; optional per-utterance normalization. Single-frame (no-chunking) path. `encode(waveform:)` / `decode(codes:)`. |
| **Mimi** | [`Audio/Mimi.swift`](../Sources/FFAI/Audio/Mimi.swift) | Kyutai's Mimi/Moshi codec — SEANet encoder/decoder, an 8-layer latent Transformer (traditional RoPE, LayerScale, windowed-causal attention), and a *split* residual-VQ (semantic codebook + acoustic residual stack). The waveform tail for Mimi-code TTS families (e.g. Marvis). Whole-utterance (non-streaming) port. `encode(waveform:)` / `decode(codes:)`. |
| **Descript DAC** | [`Audio/DescriptDAC.swift`](../Sources/FFAI/Audio/DescriptDAC.swift) | The Descript Audio Codec — a high-fidelity single-scale residual-VQ codec (Snake-conv encoder/decoder, L2-normalized codebook lookup). Shares the proven `WeightNorm` conv path with SNAC. `encode(waveform:)` / `decode(codes:)`. |
| **Vocos** | [`Audio/Vocos.swift`](../Sources/FFAI/Audio/Vocos.swift) | A *decode-only* vocoder — a ConvNeXt backbone plus an ISTFT head that turns a mel-spectrogram (or summed EnCodec embeddings) into a waveform. The backbone is CPU-native; the ISTFT head reuses the fused GPU `Ops.vocoderISTFT` kernel. `decode(features:)`. |
| **BigVGAN** | [`Audio/BigVGAN.swift`](../Sources/FFAI/Audio/BigVGAN.swift) | NVIDIA's BigVGAN GAN vocoder — a *decode-only* mel→waveform model: `conv_pre`, transposed-conv upsample stages interleaved with multi-receptive-field "AMP" residual blocks, anti-aliased Snake/SnakeBeta activations (Kaiser-sinc up/downsampling). Shares the `WeightNorm` conv path with SNAC. `decode(mel:)`. |
| **DACVAE** | [`Audio/DACVAE.swift`](../Sources/FFAI/Audio/DACVAE.swift) | The VAE-style Descript Audio Codec used by SAM-Audio — a *continuous* (no residual-VQ) codec: a Snake-conv encoder produces a VAE `mean` latent, a mirrored decoder reconstructs the waveform. Conv weights are transposed from MLX `[Cout,K,Cin]` to PyTorch layout at load; shares the `WeightNorm` conv path with SNAC. Standard (non-watermark) path only. `encode(waveform:)` → latent, `decode(latents:)` → waveform. |

**GPT-OSS** is OpenAI's GPT-OSS-20B — a 24-layer mixture-of-experts
transformer (~20B total / ~3.6B active params). Three structural
features distinguish it from the dense families: (1) an **alternating
attention schedule** — `layer_types` assigns each layer
`sliding_attention` or `full_attention`; sliding layers get a `.window`
eviction KV cache (128-token cap), full layers stay unbounded; (2)
**learned per-head attention sinks** — a `self_attn.sinks` logit vector
folded into the softmax denominator. Because the `head_dim=64`
`Ops.sdpaDecode` kernel has no native learned-sink support, GPT-OSS
folds the sink as a per-head *post-hoc rescale* of the plain SDPA
output (`O' = O · Z/(Z + exp(sink − M))`, with `M` / `Z` recovered by a
CPU dot-product over the KV cache); (3) **bias-corrected** q/k/v/o
projections (`attention_bias`). Every layer's feed-forward half is a
32-expert block-sparse MoE with top-4 `topK-then-softmax` routing and a
*clipped* SwiGLU expert activation. The published checkpoints ship the
MoE experts **MXFP4**-quantized while the attention / router /
embedding / lm_head tensors are mlx *affine*-quantized — a mixed-
precision checkpoint; the loader transcodes the MXFP4 experts to FFAI's
affine-int4 format ([`GPTOSSMoE.swift`](../Sources/FFAI/Models/GPTOSSMoE.swift)).
The sink correction and the MoE router both need a CPU sync, so every
GPT-OSS layer commits the command buffer mid-`decode` and
`GPTOSSModel.forward` runs the layers on internal buffers, queuing only
the final norm + lm_head onto the caller's `cmd` — the Jamba contract.

**FalconH1** is FFAI's first *hybrid* family: every decoder layer runs
BOTH a Mamba 2 selective-SSM mixer AND a grouped-query attention path
on the same normalized input, sums their outputs into the residual,
then applies a SwiGLU MLP. There is no layer-schedule interleave — all
layers are identically shaped (the hybrid-ness is *within* the layer).
It is the proving ground for the `DecoderLayer` protocol: the engine
holds `[any DecoderLayer]` and walks it in lockstep with a per-layer
`FalconH1LayerCache` bundling a Mamba SSM/conv state and an attention
KV cache. FalconH1 reuses the shipped Mamba 2 SSM kernels (`ssm_step`,
`conv1d_causal_step`) and the `sdpaDecode` attention path — no new
kernels were needed.

**NemotronH** is FFAI's first *stack-interleaved* hybrid: a
`hybrid_override_pattern` string assigns each decoder layer exactly one
mixer kind — `M` = Mamba 2 selective-SSM mixer, `*` = multi-head
attention, `-` = dense squared-ReLU MLP — and the kinds genuinely vary
down the stack (NemotronH-4B is 24 Mamba / 4 attention / 24 dense MLP).
Every layer shares one pre-mixer RMSNorm + one residual add; there is
no separate pre-FF norm. The layer array is therefore *heterogeneous*:
the engine holds `[any DecoderLayer]` and walks it in lockstep with a
per-index cache array (`Mamba2LayerCache` for `M`, `KVCache` for `*`,
`StatelessLayerCache` for `-`). Two structural deltas vs other
families: NemotronH attention uses **no positional embedding** (no
RoPE — the Mamba layers carry sequence order), and its Mamba layers run
**`n_groups = 8` grouped B/C** plus a **gated mixer RMSNorm**. The
grouped SSM reuses the shipped scalar `ssm_step` kernel by dispatching
it once per group over a contiguous head sub-slab — no new kernel. The
`E` (mixture-of-experts) layer kind is recognised but rejected at load:
NemotronH's MoE diverges from the shipped SwiGLU `MoELayer`, and no
small published checkpoint exercises it.

**Granite4** is a *stack-interleaved* hybrid like NemotronH — a
`layer_types` array assigns each decoder layer one mixer kind (`mamba`
or `attention`) — but the feed-forward half of every layer is uniform:
either a block-sparse **MoE** (top-K SwiGLU experts plus an always-on
shared SwiGLU expert) when `num_local_experts > 0`, or a dense SwiGLU
MLP when it is `0`. Each layer is two pre-norm + residual blocks
(`input_layernorm` → mixer, `post_attention_layernorm` → FFN), unlike
NemotronH's single norm/residual. Attention uses **no positional
embedding** (`position_embedding_type: "nope"`) like NemotronH; the
Mamba mixer runs a single full-width gated RMSNorm over `d_inner`.
Granite's four scalar multipliers are handled without double-folding:
`embedding_multiplier` folds into a dedicated scaled embedding copy,
`residual_multiplier` folds into every mixer/FFN output projection,
`attention_multiplier` is the SDPA scale, and `logits_scaling` divides
the final logits. The MoE feed-forward reuses the shared `MoELayer`
(`.topKThenSoftmax` gating). It is the first family to exercise the
MoE command-buffer contract end-to-end: `MoELayer.decode` commits the
command buffer, so `Granite4Model.forward` allocates a fresh
buffer after every MoE-bearing layer.

**Jamba** is a *stack-interleaved* hybrid like NemotronH /
Granite4 — a `layers_block_type` schedule (derived from
`attn_layer_period` / `attn_layer_offset` when not given explicitly)
assigns each decoder layer one mixer kind (`mamba` or `attention`), and
every layer carries a feed-forward half: a dense SwiGLU MLP
(`num_experts == 1`) or a block-sparse **MoE** block. Each layer is two
pre-norm + residual blocks (`input_layernorm` → mixer,
`pre_ff_layernorm` → FFN). Attention uses **no positional embedding**
(no RoPE) like NemotronH. The structural delta vs every other FFAI
hybrid: Jamba's mixer is the *original* **Mamba 1** selective SSM, not
the Mamba 2 SSD form. Mamba 1's `A` is per-`(channel, state)` (a 2-D
`A_log` of shape `[d_inner, d_state]`) and `dt` is per-channel, so the
recurrence decay `exp(A·dt)` varies with the state index — which the
shipped Mamba 2 `ssm_step` Metal kernel (one scalar `A`/`dt` per head)
cannot express. Jamba therefore runs the selective-scan core on the
**CPU** (the per-token cost is `d_inner · d_state` ≈ 82K MACs,
negligible); the GPU still owns every projection, attention, and the
MLP. Because the scan is host-side, every Jamba *mamba* layer commits
the command buffer mid-`decode`, so `JambaModel.forward` refreshes the
command buffer after each such layer — the same contract
Granite4's MoE layers use. The MoE feed-forward reuses the
shared `MoELayer` (`.topKThenSoftmax` gating).

**Qwen 3.5** is a *stack-interleaved* hybrid like Jamba — an explicit
`layer_types` schedule (with a `(i + 1) % full_attention_interval`
fallback) assigns each decoder layer one mixer kind (`linear_attention`
or `full_attention`), and every layer carries a feed-forward half: a
dense SwiGLU MLP (`num_experts == 0`) or a block-sparse **MoE** block
with a *sigmoid-gated always-on shared expert*. Each layer is two
pre-norm + residual blocks (`input_layernorm` → mixer,
`post_attention_layernorm` → FFN). The structural deltas vs every other
FFAI hybrid: the recurrent mixer is a **Gated Delta Net** (GDN, not
Mamba) — the first FFAI consumer of the `gated_delta_step` kernel and
`GDNStateCache`; attention is **gated** (`attn_output_gate` makes
`q_proj` emit `2 × heads`, the second half a `sigmoid` gate on the SDPA
output) with **partial RoPE** (`partial_rotary_factor` rotates only the
first quarter of each head — `Ops.ropePartial`). The GDN block runs the
*standard* (non-fused) kernel, so it pre-computes the per-head q/k
RMSNorm + scale and the per-value-head gates `g = exp(-exp(A_log) ·
softplus(a + dt_bias))` / `beta = sigmoid(b)` host-side; the gated
mixer RMSNorm also runs host-side (the kernel emits fp32 and there is
no GPU cast). Because the GDN host prep and any MoE FFN both commit the
command buffer mid-`decode`, `Qwen35Model.forward` runs every layer on
internal work buffers and queues only the final `norm` + `lm_head`
onto the caller's pristine command buffer — the Jamba contract. Each
GDN layer's cache is a composite `Qwen35GDNLayerCache` bundling a
`ConvStateCache` and a double-buffered `GDNStateCache`. The MoE
feed-forward reuses the shared `MoELayer` (`.softmaxThenTopK` gating)
for the routed experts and applies the sigmoid-gated shared expert
separately.

**LFM2 / LFM2.5** is LiquidAI's Liquid Foundation Models 2 — a
*stack-interleaved* hybrid like NemotronH / Granite4 / Jamba,
but the two mixer kinds are **conv** and **attention** (no SSM). A
`layer_types` array (or the older `full_attn_idxs` index list) assigns
each decoder layer one mixer: `conv` is LFM2's double-gated short
convolution — `in_proj` splits into `(B, C, x)`, the input gate `B·x`
feeds a depthwise causal conv1d, the output gate `C·conv` feeds
`out_proj` (no activation between the conv and the gate, unlike
Mamba's post-conv SiLU); `full_attention` is grouped-query attention
with RoPE and a per-head RMSNorm on Q/K (like Qwen 3). The conv mixer
reuses the shipped `conv1d_causal_step` kernel + a `ConvStateCache` —
no new kernel. LFM2.5 is architecturally identical (`model_type:
lfm2`); the "Thinking" / "Instruct" / "Base" variants differ only by
training run + chat template, so they load through the same family.
LFM2's head_dim (64) is not 128-aligned, so the per-head Q/K RMSNorm
runs **host-side** (a 64-wide norm over one decode token is trivial on
the CPU) — the GPT-OSS host-norm precedent. The **`lfm2_moe`** variant
(`Lfm2MoeForCausalLM` — LFM2-8B-A1B / LFM2-24B-A2B) swaps the dense
SwiGLU FFN for a block-sparse `MoELayer` on every layer at index ≥
`num_dense_layers`; the first `num_dense_layers` stay dense. Its
router is softmax → add a per-expert load-balancing `expert_bias` →
top-K of the biased values → optional `norm_topk_prob` re-normalise —
the biased value is *both* the selector and the combine weight, wired
through `MoERouter`'s `expertBias` parameter on `.softmaxThenTopK`.
Both an attention layer (host-side Q/K norm) and a MoE FFN (router CPU
readback) commit the command buffer mid-layer, so `LFM2Model.forward`
runs the stack on internal work buffers and refreshes after each
committing layer — the Granite4 contract. Only raw bf16 / f16
LFM2 checkpoints are supported; quantized variants are rejected with a
clear error.

**Gemma 4** is Google's Gemma 4 text decoder — three checkpoint shapes
under the single `gemma4` model_type, picked from config by the family
file: `Gemma4Dense` (31B, plain Gemma backbone), `Gemma4E` (E2B / E4B,
adds *Per-Layer Embeddings*) and `Gemma4MoE` (26B-A4B, mixture-of-experts
FFN). It keeps the Gemma 3 backbone — four per-block norms, the Gemma
`(1 + weight)` RMSNorm fold, per-head q/k norms, `sqrt(hidden)` embed
scale, GELU MLP, tied embeddings — and adds: **two attention
geometries** (`layer_types` labels each layer `sliding_attention`,
head_dim 256, or `full_attention`, `global_head_dim` 512), a scale-free
**value RMSNorm**, SDPA scale `1.0`, a learned **per-layer scalar**,
**ProportionalRoPE** on the global layers (only the first
`partial_rotary_factor · 512` dims rotate, paired across the full head),
**Per-Layer Embeddings** (Gemma4E mixes a second small embedding into
every block) and **final logit soft-capping**. Sliding layers run the
GPU `Ops.sdpaDecode` 256-wide kernel; the 512-wide global layers have no
specialised kernel yet, so global-layer attention runs through a
host-side single-token SDPA (`globalAttention`) — a bounded, deterministic
readback in the one-token decode loop. A 512-wide `sdpaDecode`
specialization is the perf follow-up. Each `Gemma4Layer` is
self-contained (runs on its own command buffers, commits before
returning) so the global-layer CPU SDPA and the MoE FFN's mid-layer
commit compose cleanly. The MoE variant reuses the shared `MoELayer`
(`.topKThenSoftmax` gating).

Both Llama / Qwen 3 variants share the same Llama-shaped core: GQA
attention with RoPE, RMSNorm, SwiGLU MLP. Qwen 3 adds per-head q_norm / k_norm
RMSNorms applied to queries/keys *before* RoPE — the only structural
difference vs Llama. No new kernels were needed for Qwen 3; just an
extra RMSNorm site.

Nemotron-Labs-Diffusion is NVIDIA's "tri-mode" model: the same dense
Ministral/Llama-shaped weights decode as **autoregressive**, **block-wise
diffusion**, or **linear self-speculation** (diffusion draft + AR
verify). AR runs through the standard decode loop; the two
non-autoregressive modes run through
[`generateDiffusion` / `generateSelfSpeculative`](../Sources/FFAI/GenerateDiffusion.swift)
or `ffai generate --mode diffusion|self-spec`. The diffusion /
self-speculation modes require a raw KV cache (`LoadOptions.kvCache =
.raw`, the default). RoPE uses the checkpoint's YaRN scaling via the
`ffai_rope_yarn` kernel; the block forward uses the multi-query
`ffai_sdpa_multi` attention kernel and the tiled `ffai_gemm` kernel for
its projections (weight reused across the block's rows). The
`linear_spec_lora` adapter auto-attaches
for the self-speculation drafter and can be hot-swapped at runtime
(`Model.loadLoRA` / `unloadLoRA`).

The KV cache defaults to an 8192-token window — the checkpoint's YaRN
context is 262144; set `LoadOptions.maxContextLength` to size the cache
for a longer (or shorter) context.

Not implemented: the trained diffusion sampler (its weights are not in
the public checkpoint), quadratic self-speculation (paper-only; not in
the released inference code), and the VLM variant (needs a vision
encoder stack). Distinct from the `NemotronH` stack-interleaved hybrid
family above.

## Sizes exercised

These are the checkpoints regression-swept in the test suite and used
for the [performance](performance.md) numbers. Any other Llama 3.x or
Qwen 3 size from HuggingFace should work — the family files don't hard-code
size-specific paths.

### Llama 3.x

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `unsloth/Llama-3.2-1B` | 1B | bf16 | Phase 2 reference. |
| `unsloth/Llama-3.2-3B` | 3B | bf16 | |
| `mlx-community/Llama-3.2-1B-4bit` | 1B | mlx 4-bit | |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 3B | mlx 4-bit | |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | 8B | mlx 4-bit | |

Llama 3 RoPE scaling (`factor`, `low_freq_factor`, `high_freq_factor`,
`original_max_position`) is honored; checkpoints with `rope_type:
"llama3"` in `config.json` route through the scaled variant
automatically.

### Qwen 3

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `mlx-community/Qwen3-0.6B-bf16` | 0.6B | bf16 | |
| `mlx-community/Qwen3-1.7B-bf16` | 1.7B | bf16 | Integration-test baseline. |
| `mlx-community/Qwen3-1.7B-3bit` | 1.7B | mlx 3-bit | Integration-tested. |
| `mlx-community/Qwen3-1.7B-4bit` | 1.7B | mlx 4-bit | Integration-tested. |
| `mlx-community/Qwen3-1.7B-5bit` | 1.7B | mlx 5-bit | Integration-tested. Published by us — mlx-community didn't ship a plain text-only 5-bit Qwen3-1.7B (only TTS / ASR variants existed). |
| `mlx-community/Qwen3-1.7B-6bit` | 1.7B | mlx 6-bit | Integration-tested. |
| `mlx-community/Qwen3-1.7B-8bit` | 1.7B | mlx 8-bit | Integration-tested. |
| `mlx-community/Qwen3-4B-bf16` | 4B | bf16 | |
| `mlx-community/Qwen3-4B-4bit` | 4B | mlx 4-bit | |
| `mlx-community/Qwen3-4B-8bit` | 4B | mlx 8-bit | |
| `mlx-community/Qwen3-8B-4bit` | 8B | mlx 4-bit | |
| `mlx-community/Qwen3-14B-4bit` | 14B | mlx 4-bit | |

The integration-test suite uses Qwen 3 1.7B across every supported
bit width to keep CI fast — the per-bit-width quantization paths
don't depend on model size, so 1.7B (~3.5 GB bf16) covers the same
codepaths as 4B / 8B / 14B at a fraction of the download cost.

mlx-community didn't have a plain text-only 5-bit Qwen3-1.7B
(only TTS / ASR variants), so we converted + uploaded one ourselves:

```bash
mlx_lm.convert --hf-path Qwen/Qwen3-1.7B \
               --mlx-path ./Qwen3-1.7B-5bit \
               -q --q-bits 5 --q-group-size 64 \
               --upload-repo mlx-community/Qwen3-1.7B-5bit
```

### FalconH1

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `mlx-community/Falcon-H1-Tiny-90M-Instruct-bf16` | 90M | bf16 | Integration-test baseline (~173 MB). |
| `mlx-community/Falcon-H1-0.5B-Instruct-bf16` | 0.5B | bf16 | |
| `mlx-community/Falcon-H1-1.5B-Instruct-bf16` | 1.5B | bf16 | |
| `mlx-community/Falcon-H1-3B-Instruct-bf16` | 3B | bf16 | |
| `mlx-community/Falcon-H1-7B-Instruct-bf16` | 7B | bf16 | |

The integration suite uses Falcon-H1-Tiny-90M-Instruct — the smallest
published FalconH1 checkpoint — so the hybrid decode path (dual
Mamba+attention mixers, `FalconH1LayerCache`, the `DecoderLayer`
protocol stack) is exercised at minimal download cost. The architecture
is identical in shape across 0.5B / 1.5B / 3B / 7B.

**Known gaps.** Only raw bf16 / f16 FalconH1 checkpoints are supported
today — quantized (`-4bit` / `-8bit`) variants are rejected with a
clear error (the µP scaling interacts with packed-weight dequant in a
way that needs dedicated handling). `mamba_rms_norm=true` checkpoints
(gated mixer RMSNorm) and `mamba_n_groups > 1` are likewise rejected;
the shipped 90M / 0.5B / 1.5B all use `mamba_rms_norm=false` +
`n_groups=1`. mlx-community checkpoints ship *pre-sanitized* (the
scalar multipliers are already folded into the saved weights); the
loader detects this via a conv1d-weight-shape probe and skips
re-folding to avoid double-applying the multipliers.

### NemotronH

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `nvidia/Nemotron-H-4B-Base-8K` | 4B | bf16 | Integration-test baseline. |
| `nvidia/Nemotron-H-4B-Instruct-128K` | 4B | bf16 | Same shape, instruction-tuned. |
| `nvidia/Nemotron-H-8B-Base-8K` | 8B | bf16 | |

The integration suite uses Nemotron-H-4B-Base-8K — the smallest
published NemotronH checkpoint whose config the shipped FFAI path can
run end-to-end. It exercises the heterogeneous `[any DecoderLayer]`
decode loop, the per-index cache array, the grouped-B/C Mamba path,
the gated mixer RMSNorm, and no-RoPE attention.

**Known gaps.** Only raw bf16 / f16 NemotronH checkpoints are supported
— quantized variants are rejected with a clear error. The `E`
(mixture-of-experts) layer kind — used by the Nemotron-Cascade-2 /
Nemotron-3 MoE checkpoints — is rejected at load: NemotronH MoE uses
squared-ReLU experts with sigmoid group-expert-select routing, both of
which diverge from the shipped SwiGLU `MoELayer`. A NemotronH Mamba
layer whose per-group RMSNorm row size (`d_inner / n_groups`) is not a
multiple of 128 (e.g. Nemotron-3-Nano-4B's 960) is also rejected — the
`rmsNormRows` kernel requires a 128-aligned row.

### Granite4

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `mlx-community/granite-4.0-h-350m-bf16` | 350M | bf16 | Integration-test baseline (dense FFN). |
| `mlx-community/granite-4.0-h-1b-bf16` | 1B | bf16 | Dense FFN, same hybrid shape. |
| `ibm-granite/granite-4.0-h-tiny` | 7B | bf16 | 64-expert MoE FFN + shared expert. |

The integration suite uses granite-4.0-h-350m — the smallest published
Granite4 checkpoint (32 layers, 28 Mamba + 4 attention,
`num_local_experts = 0` → dense SwiGLU FFN). It exercises the
heterogeneous `[any DecoderLayer]` decode loop, the per-index cache
array, no-RoPE attention, the gated mixer RMSNorm, and the four Granite
scalar multipliers.

**Known gaps.** Only raw bf16 / f16 Granite4 checkpoints are
supported — quantized variants are rejected with a clear error. The MoE
feed-forward path (block-sparse experts + shared expert) is implemented
and unit-covered via `MoELayerTests`, but the published MoE checkpoints
(H-Tiny / H-Small, 7B+) ship only quantized on mlx-community, so the
integration suite cannot exercise the MoE path on a small raw
checkpoint. A Mamba layer whose `d_inner` is not a multiple of 128 or
exceeds 4096 is rejected — the gated mixer RMSNorm uses the single-row
`rmsNorm` reduction kernel.

### Jamba

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `mlx-community/AI21-Jamba-Reasoning-3B-bf16` | 3B | bf16 | Integration-test baseline (dense FFN). |

The integration suite uses AI21-Jamba-Reasoning-3B — the smallest
published Jamba checkpoint that runs end-to-end without quantized-MoE
expert slicing (28 layers, 26 Mamba 1 + 2 attention, `num_experts = 1`
→ dense SwiGLU FFN on every layer). It exercises the heterogeneous
`[any DecoderLayer]` decode loop, the per-index cache array
(`JambaMambaLayerCache` / `KVCache`), the host-side Mamba 1 selective
scan, the 2-D `A_log` handling, and no-RoPE attention.

**Known gaps.** Only raw bf16 / f16 Jamba checkpoints are supported —
quantized variants (the `-4bit` / `-6bit` conversions) are rejected
with a clear error. The MoE feed-forward path (`num_experts > 1`,
e.g. the original Jamba-v0.1's 16-expert blocks) is implemented and
reuses the shared `MoELayer`, but every small raw Jamba checkpoint on
mlx-community is dense, so the integration suite exercises only the
dense FFN. The Mamba 1 selective scan runs on the CPU — the shipped
Mamba 2 `ssm_step` kernel cannot express Mamba 1's per-`(channel,
state)` decay — which makes every Jamba mamba layer commit the command
buffer mid-decode.

### Nemotron-Labs-Diffusion

| Repo | Size | Quant | Notes |
|---|---|---|---|
| `nvidia/Nemotron-Labs-Diffusion-3B` | 3B | bf16 | Integration-test baseline — all three modes. |
| `nvidia/Nemotron-Labs-Diffusion-8B` | 8B | bf16 | |
| `nvidia/Nemotron-Labs-Diffusion-14B` | 14B | bf16 | |

The integration test exercises autoregressive, block-wise diffusion,
and linear self-speculation on the 3B checkpoint. The `-Base` repos and
the `-VLM-8B` vision variant are untested. Self-speculation reports a
forward-pass count (NFE) on `DiffusionResult` — the diffusion
efficiency metric.

### Audio (Phase 7)

| Repo | Family | Notes |
|---|---|---|
| `openai/whisper-tiny` | Whisper STT | Integration-test baseline — encoder + cross-attending decoder. |
| `mlx-community/SenseVoiceSmall` | SenseVoice STT | Integration-test baseline — SAN-M encoder + CTC head, greedy CTC decode. |
| `hexgrad/Kokoro-82M` | Kokoro TTS | Integration-test baseline — the iSTFTNet vocoder tail. |
| `Qwen/Qwen2.5-Omni-3B` | Qwen-Omni | Integration-test baseline — the audio-in encoder path. |
| `mlx-community/orpheus-3b-0.1-ft-bf16` | LlamaTTS | Integration-test baseline — the Llama acoustic backbone + Orpheus SNAC-code decode loop. |
| `Marvis-AI/marvis-tts-250m-v0.2-MLX-fp16` | Marvis (CSM) | Integration-test baseline — the CSM dual-transformer + Mimi frame-generation loop. |
| `mlx-community/Qwen3-TTS-Flash-bf16` | Qwen3TTS | Integration-test baseline — stage-1 config decode + family detection. |

The Whisper integration test verifies the audio encoder produces
finite features, the decoder emits a non-degenerate logit distribution
cross-attending to the audio, and greedy decode runs. The Kokoro test
verifies the vocoder synthesizes a non-degenerate (finite, non-silent)
waveform from a predicted spectrogram. The Qwen-Omni test verifies the
audio tower encodes a waveform into feature tokens in the text-backbone
hidden dim. Each suite prints a skip and passes when its checkpoint
cannot be fetched.

## Quantization

All in-tree families support every bit width FFAI implements:

| Bit width | Format | Status |
|---|---|---|
| **bf16 / fp16** | Plain safetensors | ✅ |
| **8-bit** | mlx-format affine | ✅ |
| **6-bit** | mlx-format affine (byte-packed) | ✅ |
| **5-bit** | mlx-format affine (byte-packed) | ✅ |
| **4-bit** | mlx-format affine | ✅ |
| **3-bit** | mlx-format affine (byte-packed) | ✅ |

See [quantization.md](quantization.md) for the packing layout, the
sub-group split dispatch trick that closed the perf gap on 4-bit
Qwen3 4B, and how the loader auto-detects mlx-format weights.

## Loading any other repo

Pass any HuggingFace repo ID with one of the in-tree
`model_type` / `architectures` strings to `Model.load`:

```swift
let model = try await Model.load("mlx-community/Qwen3-14B-4bit")
```

The loader resolves the snapshot, parses `config.json`, picks the
right family via `ModelRegistry.dispatchAndLoad`, and builds the
variant. If the architecture isn't in the registry yet, you get a
`ModelError.unsupportedArchitecture(...)`.

## Known gaps

| Item | Status |
|---|---|
| Multi-modal (vision, audio) | Capability infrastructure in place; the VLM wave is Phase 6.5, audio is Phase 7. No VL / audio family variants yet. |
| Chat templates | The tokenizer's chat template is not auto-applied by `generate(...)` — pass the templated prompt yourself. Auto-apply lands with the first instruct-tuned VL model. |
| GPU sampling filters | Greedy (argmax) and categorical (`temperature`) run fully on-GPU. `top-K` / `top-P` / `min-P` filters fall back to the CPU-sample path; GPU filter kernels are a follow-up. |
| AURA performance | AURA KV schemes are correct and decode coherently, but still run the dequant-then-`sdpaDecode` path with a working-buffer mirror. Compressed-domain attention is the Phase 6.3 perf pass. |
| Chunked prefill | Prefill walks the prompt one token per dispatch. Batched (chunked) prefill is Phase 6.6 — a large TTFT win on long prompts. |
| Cross-request prompt caching | The KV cache lives for one `generate(...)` call. Prefix-cache reuse across requests is Phase 8.2. |
| Quantized hybrid checkpoints | NemotronH / Granite4 / Jamba / FalconH1 load raw bf16/f16 only; quantized variants are rejected with a clear error. |

## Coming next

Per [`planning/plan.md`](../planning/plan.md):

- **Phase 6.1–6.4** — perf + infra: sliding-window SDPA fast path,
  AURA MSL snapshot tests, AURA performance (compressed-domain
  attention), injectable `Profile`.
- **Phase 6.6** — chunked (batched) prefill: process the prompt N
  tokens per dispatch instead of one — a large TTFT win.
- **Phase 6.5** — Vision (VLM): Qwen 2.5/3.5-VL, Gemma 3/4-VL, plus
  the `VisionEncoder` + `conv2d` / `patch_embed` / `rope_2d` kernels.
- **Phase 7** — Audio: Whisper STT, Kokoro TTS, Qwen-Omni.
- **Phase 8** — speculative decoding, prefix KV cache, batched /
  continuous decode, and the serving wave (specs 013–043).
