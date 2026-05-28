We should add support for the following model families:

- Mamba - https://github.com/state-spaces/mamba. We have support for Mamba2 but not regular Mamba
- Mamba 3 - https://arxiv.org/html/2603.15569v1, https://www.together.ai/blog/mamba-3
- MOSS TTS 1.5 - https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5
- MiniCPM 2 - use https://huggingface.co/openbmb/MiniCPM-V-2 as a reference
- MiniCPM 3 - use https://huggingface.co/openbmb/MiniCPM3-4B as a reference
- Llama 2 - use https://huggingface.co/meta-llama/Llama-2-7b as a reference
- Microsoft Fara - use https://huggingface.co/microsoft/Fara-7B as a reference
- Llama 4 - use https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E as a reference (multi-modal MoE, vision + text)
- Falcon OCR - use https://huggingface.co/tiiuae/Falcon-OCR as a reference (image OCR model)
- LightOn OCR - use https://huggingface.co/lightonai/LightOnOCR-2-1B as reference (image OCR model)
- LightOn OCR BBOX - use https://huggingface.co/lightonai/LightOnOCR-2-1B-bbox as reference (image OCR model with bounding box support)
- Surya OCR 2 - use https://huggingface.co/datalab-to/surya-ocr-2 as reference (image OCR model with bounding box support, SOTA)
- SAM 3.1 - use https://huggingface.co/facebook/sam3.1 as a reference (vision/video segmentation model)
- Kimi K2 - use https://huggingface.co/moonshotai/Kimi-K2-Thinking as a reference
- Kimi K2.5 - use https://huggingface.co/moonshotai/Kimi-K2.5 as a reference (multi-modal, vision + text)
- Kimi K2.6 - use https://huggingface.co/moonshotai/Kimi-K2.6 as a reference (multi-modal, vision + text)
- Minimax M2 - use https://huggingface.co/MiniMaxAI/MiniMax-M2 as a reference
- Minimax M2.1 - use https://huggingface.co/MiniMaxAI/MiniMax-M2.1 as a reference
- Minimax M2.5 - use https://huggingface.co/MiniMaxAI/MiniMax-M2.5 as a reference
- Minimax M2.7 - use https://huggingface.co/MiniMaxAI/MiniMax-M2.7 as a reference
- GLM 4 - use https://huggingface.co/zai-org/glm-4-9b-chat-hf as a reference
- GLM 4.5 - use https://huggingface.co/zai-org/GLM-4.5 as a reference
- GLM 4.5 V - use https://huggingface.co/zai-org/GLM-4.5V as a reference (vision model)
- GLM 4.6 - use https://huggingface.co/zai-org/GLM-4.6 as a reference
- GLM 4.6 V - use https://huggingface.co/zai-org/GLM-4.5V as a reference (vision model)
- GLM 5 - use https://huggingface.co/zai-org/GLM-5 as a reference
- GLM 5.1 - use https://huggingface.co/zai-org/GLM-5.1 as a reference
- Deepseek OCR 1 - use https://huggingface.co/deepseek-ai/DeepSeek-OCR (ocr image model)
- Deepseek OCR 2 - use https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 (ocr image model)
- Deepseek V2 - use https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat as reference
- Deepsseek V3 - use https://huggingface.co/deepseek-ai/DeepSeek-V3 as reference
- Deepsseek V3.1 - use https://huggingface.co/deepseek-ai/DeepSeek-V3.1 as reference
- Deepseek V3.2 - use https://huggingface.co/deepseek-ai/DeepSeek-V3.2 as reference
- Deepseek V4 Flash - use https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash as reference
- Cohere A Plus - use https://huggingface.co/CohereLabs/command-a-plus-05-2026-w4a4 as a reference
- Tencent Hy3 - use https://huggingface.co/tencent/Hy3-preview as a reference
- Tencent Hy2 - use https://huggingface.co/tencent/Hy-MT2-1.8B as our reference and test model. They have other model sizes including an MoE https://huggingface.co/tencent/Hy-MT2-30B-A3B. We should support them all.
- LFM2 24B - use https://huggingface.co/LiquidAI/LFM2-24B-A2B as a reference
- Microsoft Bitnet - use https://huggingface.co/microsoft/bitnet-b1.58-2B-4T as a reference
- Bonsai Ternary - https://huggingface.co/collections/prism-ml/ternary-bonsai (ternary bit text llm, same concept as Bitnet, high priority!)
- Bonsai Image - https://huggingface.co/collections/prism-ml/bonsai-image (ternary image gen model)
- Nemotron OCR 2 - use https://huggingface.co/nvidia/nemotron-ocr-v2 as reference (image OCR)
- Nvidia Canary - https://huggingface.co/nvidia/canary-1b-v2 (multi-lingual audio model)
- Nvidia Nemotron Speech Streaming - https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b (streaming audio)
- Nvidia Parakeet Realtime EOU - https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 (realtime streaming audio)
- Nvidia Multi-talker Parakeet - https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 (multiple speaker streaming audio)
- Nvidia Nemotron Nano VL - https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 (vision model)
- Nvidia Llama Nemotron Nano VL - https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1 (vision/ocr model with llama backbone)
- Nvidia Nemotron Labs Diffusion VL - https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B (vision diffusion model)
- NVidia Locate Anything VL - https://huggingface.co/nvidia/LocateAnything-3B (vision language model)
- EPIC SHARC MOHTE - https://github.com/DjDevilCloud/EPIC-SHARC-MOHTE (unclear on architecture, needs analysis first to see if even relevant/novel)

Yes I know that some of the models are too big to run on my machine. Code them up anyway. We can check out the model format and config files and still build out reference implementations and add unit tests. We can create integration tests but we'll just mark them as skipped with a reason of "requires hardware with {{amount_of_ram_needed}} unified memory". Follow the same patterns for implementation as we have with other models.

## Ablated Models

Community-distilled / "uncensored" / personality-merged derivatives of the supported families. These load through the parent family's loader unchanged — the weight surface is the same; only training data + final-stage merges differ — so coverage here is about pinning the loader behaviour on real-world derivative checkpoints, not net-new architectures. Add integration tests only once the parent family's coherence assertions are green; otherwise an ablation-side regression looks like a loader bug.

- Qwen 3.6 35B-A3B Uncensored Heretic (MLX 4-bit) - https://huggingface.co/froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit (Qwen 3.6 MoE, abliterated)
- Qwopus 3.6 27B v2 (MLX 4-bit) - https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-MLX-4bit (Qwen 3.6 27B dense merge)
- Qwopus 3.5 27B v3 - https://huggingface.co/Jackrong/Qwopus3.5-27B-v3 (Qwen 3.5 27B dense merge, raw)