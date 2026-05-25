# Unit-test coverage audit

Generated `2026-05-25` against `ek/aura-port` after the test-layout
reorg. Pairing rule: `Sources/FFAI/<rel>/<F>.swift` is "covered" if a
matching `Tests/FFAITests/<rel>/<F>Tests.swift` exists.

| Bucket | Total | With matching test | Missing test file |
|---|---:|---:|---:|
| **All Sources/FFAI/\*.swift** | 171 | 64 | **107** |

The headline number overstates the gap — most "missing" files **are
exercised** by an integration test in `Tests/ModelTests/` or by an
infrastructure unit test that covers several source files together
(`LayersTests`, `OpsTests`, `KVCacheTests`, …). The audit just flags
files that don't have a same-name unit-test sibling.

This document categorises the 107 gaps so we can decide what to fill
first.

## Bucket A — VL family roots (16 gaps)

`Models/<F>.swift` files (the VL orchestrators, Phase B output).
Each is exercised by `Tests/ModelTests/Vision/<F>VisionIntegrationTests.swift`
end-to-end, but has no fast unit test that targets just the load
dispatch / capability set / token-id constants without spinning up
the GPU + downloading weights.

- FastVLM, Gemma3, Gemma4, GlmOcr, Idefics3, LFM2, MiniCPMV,
  Mistral3, NemotronH, Paligemma, Pixtral, Qwen2, Qwen25, Qwen3,
  Qwen35, SmolVLM2

**Recommended fix:** small `<F>RegistrationTests` files asserting
`Family.modelTypes`, `Family.architectures`, `Family.availableCapabilities`
match the source — no GPU needed.

## Bucket B — Text family files (14 gaps)

`Models/Text/<F>.swift` or `<F>Text.swift`. Each is exercised by
`Tests/ModelTests/Text/<F>IntegrationTests.swift` end-to-end.

- FalconH1, GPTOSS, GPTOSSMoE, Gemma2, Gemma3Text, Gemma4Text,
  GraniteMoeHybrid, Jamba, Llama, LlamaCompatibles, Mamba2, Mistral,
  Phi, Qwen2Text, Qwen35Text, Qwen3Text

**Recommended fix:** same `<F>RegistrationTests` pattern as Bucket A.
The existing `Text/LFM2TextTests.swift`, `Text/NemotronHTextTests.swift`,
`Text/Gemma3TextWeightFoldTests.swift`, `Text/GraniteMoeHybridForwardTests.swift`,
`Text/NemotronLabsDiffusionTests.swift` show the shape — extend
that pattern.

## Bucket C — Vision tower internals (15 gaps)

`Models/Vision/<F>Vision.swift`. Several have `<F>VisionConfigTests.swift`
(the config-decode + registry tests landed in the layout reorg) but
many don't. The vision tower itself is tested implicitly when the
matching integration test runs the encoder end-to-end.

Missing config tests:
- FastVLMVision, GlmOcrVision, Idefics3Vision, MiniCPMVVision,
  Mistral3Vision, PaligemmaVision (those VL-only families' tests
  live as `Vision/<F>Tests.swift` already — partial coverage)

Have config tests already:
- Gemma3Vision, Gemma4Vision, LFM2Vision, NemotronHVision,
  PixtralVision, Qwen2Vision, Qwen25Vision, Qwen3Vision, Qwen35Vision,
  SmolVLM2Vision

Plus `VisionTowerOps.swift` — pure helper functions, would benefit
from a dedicated `VisionTowerOpsTests.swift` (padding correctness,
patch-embed reshape vs reference).

## Bucket D — Audio codecs (16 gaps, mostly indirectly covered)

`Sources/FFAI/Audio/*.swift`. Some have dedicated codec round-trip
tests already (BigVGAN, DACVAE, DescriptDAC, Encodec, FishS1DAC, Mimi,
SNAC, Vocos — `Tests/FFAITests/Audio/<Codec>CodecTests.swift`).

Missing same-name unit-test siblings:
- AudioPrimitives, BigVGANBlocks, EncodecBlocks, FishS1DACConfig,
  FishS1DACQuantization, MimiBlocks, MimiTransformer, SNACBlocks,
  VocosBackbone

These are internal block / config types tested via the parent codec's
round-trip test. The gap is largely cosmetic — splitting the existing
codec tests into block-level files would add ceremony without new
coverage.

## Bucket E — Audio families (13 gaps)

`Models/Audio/{STT,TTS,VAD,Omni}/<F>.swift`. Each has an integration
test (`Tests/ModelTests/Audio/<sub>/<F>IntegrationTests.swift`). Some
have unit tests in `Tests/FFAITests/Models/Audio/<sub>/<F>Tests.swift`
already (Chatterbox, FishSpeech, GraniteSpeech, Parakeet, PocketTTS,
Qwen3TTSBase, Soprano, StyleTTS2, MossTTS, MossTTSNano, EchoTTS,
DeepFilterNet, MossFormer2SE, SAMAudio, SileroVAD … wait, no — those
were moved in the Audio sub-folder migration).

Truly missing unit-test files:
- Audio/Omni/QwenOmni — has integration only
- Audio/STT/Whisper, SenseVoice — has integration only
- Audio/TTS/Kokoro, LlamaTTS, Marvis, Qwen3TTS, FishSpeechConfig,
  FishSpeechLayers — integration only
- Audio/VAD/SileroVAD, SmartTurn, VADCompute — integration only

## Bucket F — Loader / Generation / Ops cross-cutting (8 gaps)

These touch multiple source files; existing tests cover behaviour
rather than per-file boundaries.

- `Loader/Model.swift` — covered by `ModelLifecycleTests`,
  `ModelRegistryTests`, every integration test. No `ModelTests.swift`.
- `Loader/ConvertDriver.swift` — covered by `SafeTensorsWriterTests` +
  one-shot conversion smoke. No `ConvertDriverTests.swift`.
- `Loader/TokenizerLoader.swift` — covered by `Tokenizer/*Tests.swift`.
  No `TokenizerLoaderTests.swift`.
- `Generation/Generate.swift` — covered by every integration test.
  No `GenerateTests.swift`.
- `Generation/GenerateDiffusion.swift` — covered by
  `NemotronLabsDiffusionIntegrationTests.swift`. No
  `GenerateDiffusionTests.swift`.

## Bucket G — Other infrastructure (9 gaps)

- `AudioEncoder`, `AudioPreprocessing`, `AudioGenerationModel`,
  `DeepFilterNetDSP` — covered by `Audio/AudioEncoderTests.swift`
  and codec tests.
- `ImagePreprocessing` — covered indirectly by VLM tests. No
  `ImagePreprocessingTests.swift`.
- `BufferPool` — covered by `Benchmark/BufferPoolTests.swift`.
- `LanguageModel` — protocol; tested by every family test.
- `Profiling` — covered by `Benchmark/ProfilingTests.swift`.
- `VADOutput` — covered by `Audio/VADModelTests.swift`.
- `Benchmark/{BenchMethod,BenchRunner,BenchmarkWriter}` — covered
  by `Benchmark/BenchTests.swift`.
- `CLI/{AudioModelRegistry,VADModelRegistry}` — covered by audio /
  VAD integration paths.
- `Inspect/{InspectTap,TokenizerInspection}` — covered by
  `Text/InspectSmokeTests.swift`.
- `KVCache/{AURAQuantizedKVCache,AURAScheme,KVCacheEviction,Mamba2LayerCache}`
  — covered by `KVCache/*Tests.swift` (AURA codec round-trip, KV
  eviction, state-replay tests).
- `Stats/{GenerationStats,MemoryStats,Perplexity,ThinkingSplit}` —
  covered by `Generation/*Tests.swift` and
  `Benchmark/MemoryStatsTests.swift`.

## Priority recommendation

| Priority | Bucket | Action | Effort |
|---|---|---|---|
| **High** | A + B | `<F>RegistrationTests.swift` for every family root — load-free assertions on modelTypes / architectures / capabilities / token ids. Fast, parallel-safe, catches registry typos that integration tests only catch after downloading weights. | 30 family-root files × ~30 LOC each = ~1 day |
| **Medium** | C (VL-only) | Add `<F>VisionConfigTests.swift` for FastVLM / GlmOcr / Idefics3 / MiniCPMV / Mistral3 / Paligemma so every VL family has a config test. | 6 files × ~50 LOC = half day |
| **Medium** | F + G subset | `GenerateTests.swift`, `ImagePreprocessingTests.swift`, `VisionTowerOpsTests.swift` for the unit-testable cross-cutting bits that currently rely on integration tests for coverage. | 3 files × ~150 LOC = 1 day |
| **Low** | D + E + G rest | Block-level codec + per-audio-family unit tests duplicate what integration tests already exercise. Defer unless a regression motivates it. | Per-request |

## What's already in great shape

- `Ops/*Tests.swift` — every public Ops wrapper has a unit test
  (audited per #122 last cycle).
- `KVCache/*Tests.swift` — every cache subclass + AURA codec
  round-trip + state-replay coverage.
- `Layers/*Tests.swift` — RMSNorm / RoPE / Linear / Embedding
  forward correctness.
- `Loader/SafeTensors*Tests.swift` — header parse, mmap drop, error
  paths.
- `Generation/{Sampling,GenerationParameters,Perplexity,ChatTemplate,
  SpeculativeAccept,ThinkingSplit}Tests.swift` — each Generation
  surface has a unit test.

## Total estimate

- **High-value gap fill (Bucket A + B)**: ~30 stub registration tests
  in a single follow-up commit.
- **Full parity (every source file gets a unit test)**: ~5 working
  days; not all of it adds real coverage (much is duplicative with
  integration tests).

Recommendation: tackle Bucket A + B as a one-day follow-up, then
treat Bucket C + F as needed (when bugs surface or APIs change).
Skip D + E + most of G as "covered enough" via the integration
suite + the existing thematic tests.
