# Integration bisect — 2026-05-24T19:23:33Z

Per-suite serial integration run with GPU pinning probe.

- Per-test timeout: 900s
- GPU settle: 3s
- GPU pin threshold: 50%
- GPU sampling: powermetrics (sudo OK)

| Suite | Status | Duration | GPU post-test | Notes |
|---|---|---|---|---|
| ChatterboxIntegrationTests | PASS | 23s | ? | — |
| CohereTranscribeIntegrationTests | FAIL | 146s | ? | 􀢄  Test "load — config and weight shapes bind correctly" recorded an issue at CohereTranscribeI |
| DeepFilterNetIntegrationTests | PASS | 1s | ? | — |
| DeepSeekR1DistillIntegrationTests | FAIL | 22s | ? | 􀢄  Test "R1-Distill-Qwen-1.5B (Qwen 2 architecture) generates coherent output" recorded an issue  |
| EchoTTSIntegrationTests | PASS | 14s | ? | — |
| FalconH1IntegrationTests | PASS | 12s | ? | — |
| FastVLMIntegrationTests | FAIL | 9s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| FireRedASR2IntegrationTests | FAIL | 10s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| FireRedVADIntegrationTests | PASS | 1s | ? | — |
| FishSpeechIntegrationTests | FAIL | 1s | ? | 􀢄  Test "load config + weights from cached checkpoint" recorded an issue at FishSpeechIntegration |
| GLMASRIntegrationTests | FAIL | 26s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| GPTOSSIntegrationTests | PASS | 1s | ? | — |
| Gemma3IntegrationTests | PASS | 12s | ? | — |
| Gemma3VLIntegrationTests | FAIL | 393s | ? | 􀢄  Test "image + text prompt — describes the dog photo" recorded an issue at Gemma3VLIntegratio |
| Gemma4IntegrationTests | PASS | 207s | ? | — |
| Gemma4VLIntegrationTests | TIMEOUT | 900s | ? | exceeded 900s |
| GlmOcrIntegrationTests | PASS | 5s | ? | — |
| GraniteMoeHybridIntegrationTests | PASS | 62s | ? | — |
| GraniteSpeechIntegrationTests | TIMEOUT | 901s | ? | exceeded 900s |
| Idefics3IntegrationTests | PASS | 1s | ? | — |
| JambaIntegrationTests | PASS | 120s | ? | — |
| KokoroIntegrationTests | PASS | 2s | ? | — |
| LFM2IntegrationTests | PASS | 52s | ? | — |
| LFM2VLIntegrationTests | FAIL | 4s | ? | 􀢄  Test "load — LFM2-VL checkpoint loads with vision capability" recorded an issue at LFM2VLInt |
| LFMAudioIntegrationTests | FAIL | 377s | ? | 􀢄  Test "load — conformer + adapter + backbone weights bind correctly" recorded an issue at LFM |
| LlamaCompatiblesIntegrationTests | PASS | 24s | ? | — |
| LlamaIntegrationTests | FAIL | 15s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| LlamaTTSIntegrationTests | PASS | 51s | ? | — |
| Mamba2IntegrationTests | PASS | 4s | ? | — |
| MarvisIntegrationTests | PASS | 67s | ? | — |
| MiniCPMVIntegrationTests | FAIL | 130s | ? | 􀢄  Test "image + text prompt — coherent multi-modal generation" recorded an issue at MiniCPMVIn |
| Mistral3IntegrationTests | FAIL | 1s | ? | 􀢄  Test "load — Mistral3 checkpoint loads with vision capability" recorded an issue at Mistral3 |
| MistralIntegrationTests | PASS | 12s | ? | — |
| ModelKVCacheMatrixIntegrationTests | FAIL | 11s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| MossFormer2SEIntegrationTests | PASS | 1s | ? | — |
| MossTTSIntegrationTests | PASS | 151s | ? | — |
| MossTTSNanoIntegrationTests | PASS | 7s | ? | — |
| NemotronHIntegrationTests | PASS | 27s | ? | — |
| NemotronLabsDiffusionIntegrationTests | PASS | 275s | ? | — |
| NemotronVLIntegrationTests | PASS | 1s | ? | — |
| PaligemmaIntegrationTests | TIMEOUT | 900s | ? | exceeded 900s |
| ParakeetIntegrationTests | FAIL | 7s | ? | 􀢄  Test "load — Parakeet config + weights bind correctly" recorded an issue at ParakeetIntegrat |
| Phi3IntegrationTests | PASS | 5s | ? | — |
| PixtralIntegrationTests | PASS | 2s | ? | — |
| PocketTTSIntegrationTests | PASS | 7s | ? | — |
| Quantized3bitIntegrationTests | PASS | 11s | ? | — |
| Quantized4bitIntegrationTests | PASS | 8s | ? | — |
| Quantized5bitIntegrationTests | PASS | 11s | ? | — |
| Quantized6bitIntegrationTests | PASS | 12s | ? | — |
| Quantized8bitIntegrationTests | PASS | 11s | ? | — |
| Qwen25VLIntegrationTests | TIMEOUT | 900s | ? | exceeded 900s |
| Qwen2IntegrationTests | FAIL | 6s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| Qwen2VLIntegrationTests | FAIL | 6s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| Qwen35IntegrationTests | PASS | 24s | ? | — |
| Qwen3ASRIntegrationTests | PASS | 99s | ? | — |
| Qwen3IntegrationTests | PASS | 16s | ? | — |
| Qwen3TTSBaseIntegrationTests | FAIL | 71s | ? | 􀢄  Test "registry — VyvoTTS routes through AudioModelRegistry" recorded an issue at Qwen3TTSBas |
| Qwen3TTSIntegrationTests | PASS | 5s | ? | — |
| Qwen3VLIntegrationTests | FAIL | 6s | ? | 􀢄  Test "load — Qwen 3-VL checkpoint loads with vision capability" recorded an issue at Qwen3VL |
| Qwen3VLMoeIntegrationTests | FAIL | 1s | ? | 􀢄  Test "load — Qwen 3-VL-MoE checkpoint loads with vision capability" recorded an issue at Qwe |
| QwenOmniIntegrationTests | TIMEOUT | 900s | ? | exceeded 900s |
| SAMAudioIntegrationTests | FAIL | 12s | ? | 􀢄  Test "load + segment produces correctly-shaped output" recorded an issue at SAMAudioIntegratio |
| SenseVoiceIntegrationTests | PASS | 334s | ? | — |
| SileroVADIntegrationTests | PASS | 23s | ? | — |
| SlidingWindowIntegrationTests | FAIL | 14s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| SmartTurnIntegrationTests | PASS | 828s | ? | — |
| SmolVLM2IntegrationTests | FAIL | 26s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| SopranoIntegrationTests | PASS | 34s | ? | — |
| SortformerIntegrationTests | PASS | 602s | ? | — |
| StyleTTS2IntegrationTests | FAIL | 1s | ? | 􀢄  Test "load kitten-tts-nano checkpoint + synthesize placeholder produces sane waveform" recorde |
| TenVADIntegrationTests | PASS | 1s | ? | — |
| VoxtralRealtimeIntegrationTests | FAIL | 55s | ? | error: Process '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/l |
| WhisperIntegrationTests | TIMEOUT | 900s | ? | exceeded 900s |

## Summary

- Total: 73
- PASS: 43
- FAIL: 24
- TIMEOUT: 6
- GPU-pinned after exit: 0

Per-suite swift-test logs: `planning/integration-bisect-logs/`

Next step: for each TIMEOUT or PINNED suite, attach Instruments
(`xcrun xctrace record --template 'Metal System Trace' --attach <pid>`)
during a re-run to identify the in-flight kernel + dispatch shape.
