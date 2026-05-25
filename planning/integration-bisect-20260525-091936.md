# Integration bisect — 2026-05-25T09:19:36Z

Per-suite serial integration run with GPU pinning probe.

- Per-test timeout: 600s
- GPU settle: 3s
- GPU pin threshold: 50%
- GPU sampling: DISABLED (sudo unavailable)

| Suite | Status | Duration | GPU post-test | Notes |
|---|---|---|---|---|
| CohereTranscribeIntegrationTests | FAIL | 11s | ? | 􀢄  Test "load — config and weight shapes bind correctly" recorded an issue at CohereTranscribeI |
| DeepFilterNetIntegrationTests | FAIL | 2s | ? | 􀢄  Test "enhance returns non-empty waveform with same length as input" recorded an issue at DeepF |
| DeepSeekR1DistillIntegrationTests | FAIL | 28s | ? | 􀢄  Test "R1-Distill-Qwen-1.5B (Qwen 2 architecture) generates coherent output" recorded an issue  |
| FalconH1IntegrationTests | PASS | 8s | ? | — |
| FastVLMIntegrationTests | FAIL | 9s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| FireRedASR2IntegrationTests | FAIL | 16s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| FishSpeechIntegrationTests | FAIL | 1s | ? | 􀢄  Test "load config + weights from cached checkpoint" recorded an issue at FishSpeechIntegration |
| GLMASRIntegrationTests | FAIL | 36s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| GPTOSSIntegrationTests | PASS | 7s | ? | — |
| Gemma2IntegrationTests | PASS | 18s | ? | — |
| Gemma3IntegrationTests | PASS | 13s | ? | — |
| Gemma3VLIntegrationTests | FAIL | 414s | ? | 􀢄  Test "image + text prompt — describes the dog photo" recorded an issue at Gemma3VLIntegratio |
| Gemma4IntegrationTests | PASS | 51s | ? | — |
| Gemma4VLIntegrationTests | FAIL | 70s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| GlmOcrIntegrationTests | FAIL | 13s | ? | 􀢄  Test "load — GLM-OCR checkpoint loads and is recognised as GlmOcrModel" recorded an issue at |
| GraniteMoeHybridIntegrationTests | FAIL | 170s | ? | 􀢄  Test "load + greedy generate produces coherent hybrid output" recorded an issue at GraniteMoeH |
| GraniteSpeechIntegrationTests | TIMEOUT | 600s | ? | exceeded 600s |
| Idefics3IntegrationTests | FAIL | 217s | ? | 􀢄  Test "load Idefics3-8B + shape check + image encode + generate" recorded an issue at Idefics3I |
| JambaIntegrationTests | FAIL | 120s | ? | 􀢄  Test "load + greedy generate produces coherent hybrid output" recorded an issue at JambaIntegr |
| KokoroIntegrationTests | PASS | 9s | ? | — |
| LFM2IntegrationTests | PASS | 53s | ? | — |
| LFM2VLIntegrationTests | FAIL | 13s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| LFMAudioIntegrationTests | FAIL | 192s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| LlamaCompatiblesIntegrationTests | FAIL | 21s | ? | 􀢄  Test "SmolLM2-360M-Instruct (LlamaForCausalLM, no biases) decodes coherently" recorded an issu |
| LlamaIntegrationTests | PASS | 11s | ? | — |
| LlamaTTSIntegrationTests | PASS | 31s | ? | — |
| Mamba2IntegrationTests | FAIL | 8s | ? | 􀢄  Test "load + greedy generate produces non-degenerate text" recorded an issue at Mamba2Integrat |
| MarvisIntegrationTests | PASS | 39s | ? | — |
| MiniCPMVIntegrationTests | FAIL | 71s | ? | 􀢄  Test "image + text prompt — coherent multi-modal generation" recorded an issue at MiniCPMVIn |
| MiniCPMVVideoIntegrationTests | FAIL | 197s | ? | 􀢄  Test "video + text prompt — describes the cat clip" recorded an issue at MiniCPMVVideoIntegr |
| Mistral3IntegrationTests | FAIL | 1s | ? | 􀢄  Test "load — Mistral3 checkpoint loads with vision capability" recorded an issue at Mistral3 |
| MistralIntegrationTests | PASS | 15s | ? | — |
| ModelKVCacheMatrixIntegrationTests | TIMEOUT | 600s | ? | exceeded 600s |
| MossFormer2SEIntegrationTests | FAIL | 2s | ? | 􀢄  Test "enhance — produces non-empty output matching input length" recorded an issue at MossFo |
| NemotronHIntegrationTests | PASS | 17s | ? | — |
| NemotronLabsDiffusionIntegrationTests | PASS | 206s | ? | — |
| NemotronVLIntegrationTests | FAIL | 1s | ? | 􀢄  Test "load — Nemotron-VLM checkpoint loads with vision capability" recorded an issue at Nemo |
