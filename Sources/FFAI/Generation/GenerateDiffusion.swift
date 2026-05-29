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
// GenerateDiffusion — block-wise diffusion + linear self-speculation
// decoding for NemotronDiffusion. AR decoding uses the standard
// `Generate.swift` path; the two non-autoregressive tri-modes live here.
//
// Diffusion (`generateDiffusion`): causal-prefill the prompt, then
// decode block-by-block. Each block starts as `mask` tokens (slot 0
// seeded by the prior AR next-token) and is iteratively denoised — each
// step runs a bidirectional forward over the block and commits the
// highest-confidence masked positions. A causal forward over the
// finished block appends it to the KV cache and seeds the next block.
//
// Linear self-speculation (`generateSelfSpeculative`): per block,
// diffusion-draft the block bidirectionally, then AR-verify it causally;
// accept the longest matching prefix plus one bonus token and roll the
// KV cache back to the accepted length.
//
// Both ports follow `modeling_nemotron_labs_diffusion.py`
// (`generate` / `linear_spec_generate`). Greedy (temperature 0) only in
// this first cut — the integration contract is coherent text.

import Foundation

// ─── Public parameters + result ──────────────────────────────────────

/// Knobs for the diffusion / self-speculation decode paths.
public struct DiffusionParameters: Sendable, Equatable {
    /// Total new tokens to generate. Must be a multiple of `blockLength`.
    public var maxNewTokens: Int
    /// Block size for parallel decoding (the checkpoint's `block_size`).
    public var blockLength: Int
    /// Confidence threshold for committing a masked position during
    /// denoising. `nil` falls back to an even per-step transfer budget.
    /// For self-speculation, `nil`/`0` means a single-pass full draft.
    public var confidenceThreshold: Float?
    /// Stop when the model's EOS token is produced.
    public var stopOnEOS: Bool

    public init(
        maxNewTokens: Int = 64,
        blockLength: Int = 32,
        confidenceThreshold: Float? = 0.9,
        stopOnEOS: Bool = true
    ) {
        self.maxNewTokens = maxNewTokens
        self.blockLength = blockLength
        self.confidenceThreshold = confidenceThreshold
        self.stopOnEOS = stopOnEOS
    }
}

/// Result of a diffusion / self-speculation decode.
public struct DiffusionResult: Sendable {
    public let promptTokens: [Int]
    public let generatedTokens: [Int]
    public let text: String
    /// Number of model forward passes (NFE) — the diffusion efficiency
    /// metric. Lower is better for a given token count.
    public let forwardPasses: Int
}

/// Which decode strategy a NemotronDiffusion model runs. Self-speculation
/// is the default — it drafts a block with the diffusion head then
/// AR-verifies it, matching the reference `linear_spec_generate` default
/// and beating plain AR while staying coherent. Pick `.diffusion` for the
/// pure block-denoising path, or `.autoregressive` for a standard greedy
/// token-by-token decode.
public enum DiffusionMode: String, Sendable, CaseIterable {
    case autoregressive
    case diffusion
    case selfSpeculative
}

// ─── Diffusion + self-speculation entry points ───────────────────────

extension Model {

    /// Unified NemotronDiffusion decode entry. `mode` selects the
    /// strategy; an explicit value wins, and `nil` (the default) falls
    /// back to the mode the model was loaded with
    /// (`LoadOptions.diffusionMode`, itself defaulting to
    /// `.selfSpeculative`). So callers pick AR / diffusion /
    /// self-speculation at load time *or* per call instead of reaching
    /// for three separate methods. Requires a NemotronDiffusion engine;
    /// all three modes return a `DiffusionResult` (NFE-counted).
    public func generate(
        prompt: String,
        mode: DiffusionMode? = nil,
        diffusionParameters: DiffusionParameters = DiffusionParameters()
    )
        -> DiffusionResult
    {
        let resolvedMode = mode ?? loadOptions.diffusionMode
        let promptTokens = tokenizer.encode(text: prompt)
        let generated: (tokens: [Int], nfe: Int)
        switch resolvedMode {
        case .autoregressive:
            generated = driveAutoregressive(
                promptTokens: promptTokens, params: diffusionParameters)
        case .diffusion:
            generated = driveDiffusion(
                promptTokens: promptTokens, params: diffusionParameters)
        case .selfSpeculative:
            generated = driveSelfSpeculative(
                promptTokens: promptTokens, params: diffusionParameters)
        }
        return makeResult(promptTokens: promptTokens, generated: generated)
    }

    /// Block-wise diffusion decoding. Requires a NemotronDiffusion
    /// engine loaded with a raw KV cache (`LoadOptions.kvCache = .raw`).
    public func generateDiffusion(
        prompt: String,
        parameters: DiffusionParameters = DiffusionParameters()
    )
        -> DiffusionResult
    {
        let promptTokens = tokenizer.encode(text: prompt)
        let generated = driveDiffusion(promptTokens: promptTokens, params: parameters)
        return makeResult(promptTokens: promptTokens, generated: generated)
    }

    /// Linear self-speculation: diffusion drafts a block, AR verifies it,
    /// the longest matching prefix (plus one bonus token) is committed.
    public func generateSelfSpeculative(
        prompt: String,
        parameters: DiffusionParameters = DiffusionParameters()
    )
        -> DiffusionResult
    {
        let promptTokens = tokenizer.encode(text: prompt)
        let generated = driveSelfSpeculative(promptTokens: promptTokens, params: parameters)
        return makeResult(promptTokens: promptTokens, generated: generated)
    }

    // ─── Internal drivers ────────────────────────────────────────────

    private func makeResult(
        promptTokens: [Int],
        generated: (tokens: [Int], nfe: Int)
    ) -> DiffusionResult {
        let text = tokenizer.decode(tokens: generated.tokens, skipSpecialTokens: true)
        return DiffusionResult(
            promptTokens: promptTokens,
            generatedTokens: generated.tokens,
            text: text, forwardPasses: generated.nfe)
    }

    private func diffusionEngine() -> NemotronDiffusionModel {
        guard let m = engine as? NemotronDiffusionModel else {
            preconditionFailure(
                "generateDiffusion / generateSelfSpeculative require a "
                    + "NemotronDiffusion model")
        }
        return m
    }

    /// Standard greedy autoregressive decode over the diffusion backbone:
    /// causal-prefill the prompt, then emit one token per causal
    /// `forwardBlock` step. Unlike the diffusion / self-spec paths,
    /// `maxNewTokens` need not be a multiple of `blockLength` and no
    /// block-staging headroom is reserved. Returns the generated tokens +
    /// the NFE count (prefill + one forward per token).
    private func driveAutoregressive(
        promptTokens: [Int],
        params: DiffusionParameters
    )
        -> (tokens: [Int], nfe: Int)
    {
        let m = diffusionEngine()
        precondition(!promptTokens.isEmpty, "generate(.autoregressive): prompt is empty")
        let eos = params.stopOnEOS ? m.eosTokenId : nil
        let cacheDepth = min(
            m.maxContextWindow, promptTokens.count + params.maxNewTokens)
        let caches = m.makeLayerCaches(maxSeq: cacheDepth)
        var nfe = 0

        // Causal prefill — appends the prompt's K/V, predicts token 0.
        let prefillLogits = m.forwardBlock(
            tokenIds: promptTokens,
            positions: Array(0 ..< promptTokens.count),
            caches: caches, append: true)
        nfe += 1
        var next = argmax(prefillLogits[promptTokens.count - 1])

        var generated: [Int] = []
        generated.reserveCapacity(params.maxNewTokens)
        for i in 0 ..< params.maxNewTokens {
            generated.append(next)
            if let eos, next == eos { break }
            let pos = promptTokens.count + i
            let logits = m.forwardBlock(
                tokenIds: [next], positions: [pos],
                caches: caches, append: true)
            nfe += 1
            next = argmax(logits[0])
        }
        return (generated, nfe)
    }

    private func driveDiffusion(
        promptTokens: [Int],
        params: DiffusionParameters
    )
        -> (tokens: [Int], nfe: Int)
    {
        let m = diffusionEngine()
        let blockLength = params.blockLength
        precondition(
            params.maxNewTokens % blockLength == 0,
            "generateDiffusion: maxNewTokens (\(params.maxNewTokens)) must be a "
                + "multiple of blockLength (\(blockLength))")
        precondition(!promptTokens.isEmpty, "generateDiffusion: prompt is empty")

        let maskId = m.maskTokenId
        let eos = params.stopOnEOS ? m.eosTokenId : nil
        // Cache depth = prompt + all generated tokens + one block of
        // scratch headroom for the denoise forward, clamped to the
        // model's context ceiling (which already folds in
        // LoadOptions.maxContextLength). The diffusion cache is
        // preallocated to this depth (it stages into the free tail), so
        // bounding it here also bounds the up-front footprint.
        let cacheDepth = min(
            m.maxContextWindow, promptTokens.count + params.maxNewTokens + blockLength)
        let caches = m.makeLayerCaches(maxSeq: cacheDepth)
        var nfe = 0

        // Causal prefill — appends the prompt's K/V, seeds the first block.
        let promptPositions = Array(0 ..< promptTokens.count)
        let prefillLogits = m.forwardBlock(
            tokenIds: promptTokens,
            positions: promptPositions,
            caches: caches, append: true)
        nfe += 1
        var nextToken = argmax(prefillLogits[promptTokens.count - 1])

        let numBlocks = params.maxNewTokens / blockLength
        var generated: [Int] = []
        generated.reserveCapacity(params.maxNewTokens)

        for b in 0 ..< numBlocks {
            var block = [Int](repeating: maskId, count: blockLength)
            block[0] = nextToken  // causal-context seed
            let blockStart = promptTokens.count + b * blockLength
            let blockPositions = Array(blockStart ..< blockStart + blockLength)

            let initialMaskCount = block.filter { $0 == maskId }.count
            let transferBudget = Self.numTransferTokens(
                maskCount: initialMaskCount,
                steps: blockLength)

            // Denoise: repeatedly forward the block and commit the
            // highest-confidence masked positions.
            for step in 0 ..< blockLength {
                let isMask = block.map { $0 == maskId }
                if !isMask.contains(true) { break }
                let blockLogits = m.forwardBlock(
                    tokenIds: block,
                    positions: blockPositions,
                    caches: caches, append: false)
                nfe += 1
                let (x0, transfer) = Self.transferIndex(
                    blockLogits: blockLogits, isMask: isMask,
                    numTransfer: transferBudget[step],
                    threshold: params.confidenceThreshold)
                for p in transfer { block[p] = x0[p] }
                if let eos, block.contains(eos) { break }
            }

            generated.append(contentsOf: block)

            // Causal commit — append the finalised block, seed the next.
            let commitLogits = m.forwardBlock(
                tokenIds: block,
                positions: blockPositions,
                caches: caches, append: true)
            nfe += 1
            nextToken = argmax(commitLogits[blockLength - 1])

            if let eos, let idx = generated.firstIndex(of: eos) {
                generated = Array(generated[...idx])
                break
            }
        }

        return (generated, nfe)
    }

    private func driveSelfSpeculative(
        promptTokens: [Int],
        params: DiffusionParameters
    )
        -> (tokens: [Int], nfe: Int)
    {
        let m = diffusionEngine()
        let blockLength = params.blockLength
        precondition(!promptTokens.isEmpty, "generateSelfSpeculative: prompt is empty")

        let maskId = m.maskTokenId
        let eos = params.stopOnEOS ? m.eosTokenId : nil
        // Self-speculation may overshoot maxNewTokens by up to one block
        // (a fully-accepted draft) and stages a block of scratch K/V.
        // Clamped to the model's context ceiling (folds in
        // LoadOptions.maxContextLength); the preallocated cache's
        // up-front footprint is bounded by this depth.
        let cacheDepth = min(
            m.maxContextWindow, promptTokens.count + params.maxNewTokens + 2 * blockLength)
        let caches = m.makeLayerCaches(maxSeq: cacheDepth)
        let rawCaches: [KVCache] = caches.map { $0 as! KVCache }
        var nfe = 0

        // Causal prefill.
        let prefillLogits = m.forwardBlock(
            tokenIds: promptTokens,
            positions: Array(0 ..< promptTokens.count),
            caches: caches, append: true)
        nfe += 1
        var nextToken = argmax(prefillLogits[promptTokens.count - 1])

        var generated: [Int] = [nextToken]

        while generated.count < params.maxNewTokens {
            let cacheLen = rawCaches[0].length
            var block = [Int](repeating: maskId, count: blockLength)
            block[0] = nextToken  // verified seed
            let blockPositions = Array(cacheLen ..< cacheLen + blockLength)

            // Draft phase — bidirectional, single full pass (greedy).
            // `useLora: true` engages the linear_spec_lora drafter when
            // an adapter is attached (a no-op otherwise).
            let draftLogits = m.forwardBlock(
                tokenIds: block,
                positions: blockPositions,
                caches: caches, append: false,
                useLora: true)
            nfe += 1
            for p in 0 ..< blockLength where block[p] == maskId {
                block[p] = argmax(draftLogits[p])
            }

            // Verify phase — causal, appends the whole block to the cache.
            let verifyLogits = m.forwardBlock(
                tokenIds: block,
                positions: blockPositions,
                caches: caches, append: true)
            nfe += 1
            let arTokens = (0 ..< blockLength).map { argmax(verifyLogits[$0]) }

            // Accept the longest prefix of the draft that the AR verifier
            // agrees with, plus one bonus token.
            let outcome = SpeculativeAccept.verify(
                draft: Array(block[1 ..< blockLength]),
                verifierTokens: Array(arTokens[0 ..< blockLength - 1]),
                bonusToken: arTokens[blockLength - 1])

            generated.append(contentsOf: outcome.committedTokens)

            // Roll the KV cache back: verify appended `blockLength`
            // positions; keep only prefix + the committed tokens.
            for c in rawCaches { c.truncate(toLength: cacheLen + outcome.committedCount) }
            nextToken = outcome.bonusToken

            if let eos, let idx = generated.firstIndex(of: eos) {
                generated = Array(generated[...idx])
                break
            }
        }

        if generated.count > params.maxNewTokens {
            generated = Array(generated[..<params.maxNewTokens])
        }
        return (generated, nfe)
    }
}

// ─── Confidence-transfer helpers (ports of the HF reference) ──────────

extension Model {

    /// Even split of `maskCount` masked positions across `steps`
    /// denoising steps, remainder front-loaded. Port of
    /// `_get_num_transfer_tokens`.
    static func numTransferTokens(maskCount: Int, steps: Int) -> [Int] {
        guard steps > 0 else { return [] }
        let base = maskCount / steps
        let remainder = maskCount % steps
        return (0 ..< steps).map { $0 < remainder ? base + 1 : base }
    }

    /// Pick which masked positions to commit this denoising step. Greedy
    /// (temperature 0): the committed token is the argmax; confidence is
    /// its softmax probability. Port of `_get_transfer_index`.
    ///
    /// With a `threshold`, every masked position above it commits (the
    /// single highest-confidence position always commits, so a block
    /// never stalls). Without one, the top-`numTransfer` positions by
    /// confidence commit.
    static func transferIndex(
        blockLogits: [Tensor], isMask: [Bool],
        numTransfer: Int, threshold: Float?
    )
        -> (x0: [Int], transfer: [Int])
    {
        let n = blockLogits.count
        var x0 = [Int](repeating: 0, count: n)
        var confidence = [Float](repeating: -.infinity, count: n)

        for p in 0 ..< n {
            let logits = Sampling.decodeF32(blockLogits[p])
            var bestIdx = 0
            var bestVal = logits[0]
            for i in 1 ..< logits.count where logits[i] > bestVal {
                bestVal = logits[i]
                bestIdx = i
            }
            x0[p] = bestIdx
            if isMask[p] {
                // softmax(logits)[argmax] = 1 / Σ exp(logit_i - logit_max)
                var sum: Float = 0
                for v in logits { sum += Foundation.exp(v - bestVal) }
                confidence[p] = sum > 0 ? 1.0 / sum : 0
            }
        }

        let maskedByConfidence = (0 ..< n)
            .filter { isMask[$0] }
            .sorted { confidence[$0] > confidence[$1] }

        var transfer: [Int] = []
        if let threshold {
            for (rank, p) in maskedByConfidence.enumerated() {
                // Rank 0 always commits so the block keeps progressing.
                if rank == 0 || confidence[p] >= threshold { transfer.append(p) }
            }
        } else {
            let k = min(max(numTransfer, 0), maskedByConfidence.count)
            transfer = Array(maskedByConfidence.prefix(k))
        }
        return (x0, transfer)
    }
}

// ─── Argmax over a logits tensor ─────────────────────────────────────

private func argmax(_ logits: Tensor) -> Int {
    let values = Sampling.decodeF32(logits)
    var bestIdx = 0
    var bestVal = values[0]
    for i in 1 ..< values.count where values[i] > bestVal {
        bestVal = values[i]
        bestIdx = i
    }
    return bestIdx
}
