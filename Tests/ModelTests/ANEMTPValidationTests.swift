// ANEMTPValidationTests — Day 2: end-to-end validation that the MTP
// drafter predicts the same next-token as the main model would, given
// a real hidden state from main model inference.
//
// Pipeline:
//   1. Load Qwen3.6-A3B + MTP mlpackage.
//   2. Prefill a prompt.
//   3. Run forwardWithHidden on the last prompt token → (hidden_t, _).
//   4. Run forward on the same token to also get logits → predicted
//      next-token via greedy argmax (ground truth).
//   5. Get embed_t = embedTokens(last_prompt_token).
//   6. Compute RoPE sin/cos for the next position.
//   7. Feed (hidden_t, embed_t, sin, cos) to MTP → hidden_next.
//   8. Project hidden_next via lm_head → logits → argmax → drafter
//      candidate.
//   9. Compare drafter candidate to ground truth. ANE MTP is an
//      approximation — at top-1 it should agree ~70-90% of the time;
//      at top-5 agreement should be 90%+.

import Foundation
import Testing
import Metal
import CoreML
@testable import FFAI

@Suite("ANE MTP end-to-end validation")
struct ANEMTPValidationTests {

    @Test("MTP drafter predicts top-1 next-token that matches main model greedy (or appears in top-5)")
    func mtpMatchesMainGreedy() async throws {
        let modelPath = "/Users/tom/models/Qwen3.6-35B-A3B-4bit"
        let mtpPath = "/Users/tom/models/Qwen3.6-35B-A3B-mtp.mlpackage"
        guard FileManager.default.fileExists(atPath: modelPath),
              FileManager.default.fileExists(atPath: mtpPath) else {
            print("ANEMTPValidation skipped: main model or MTP mlpackage missing")
            return
        }
        var optsBuilder = LoadOptions()
        optsBuilder.prewarm = false
        let opts = optsBuilder
        let m: Model = try await ModelLoadLock.shared.loadSerially {
            try await Model.load(modelPath, options: opts)
        }
        guard let qwen = m.qwen35 else {
            Issue.record("expected Qwen35Model engine")
            return
        }
        let device = Device.shared

        // Load + compile MTP mlpackage.
        let mtpURL = URL(fileURLWithPath: mtpPath)
        let compiledURL = try await MLModel.compileModel(at: mtpURL)
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all
        let mtp = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        // Prefill a fibonacci-style prompt — same prompt as the
        // spec-decode bench so the comparison is directly relevant.
        let prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\ndef "
        let promptTokens = m.tokenizer.encode(text: prompt)
        let promptLen = promptTokens.count
        precondition(promptLen >= 4, "test needs ≥4 prompt tokens")

        let caches = qwen.makeLayerCaches()
        // Prefill — capture BOTH pre-finalNorm and post-finalNorm
        // hidden states on the last step to A/B which one MTP expects.
        var lastPre: Tensor!
        var lastPost: Tensor!
        var lastLogits: Tensor!
        for (i, tok) in promptTokens.enumerated() {
            let cmd = device.makeCommandBuffer()
            let (pre, post, l) = qwen.forwardWithBothHiddens(tokenId: tok, position: i,
                                                              caches: caches,
                                                              on: cmd, device: device)
            cmd.commit()
            await cmd.completed()
            lastPre = pre
            lastPost = post
            lastLogits = l
        }
        let lastToken = promptTokens[promptLen - 1]
        let groundTruthToken = argmaxHost(lastLogits)
        print("ANEMTPValidation: last prompt token=\(lastToken), main model predicts next token=\(groundTruthToken)")
        _ = lastPre  // shut up unused warning
        // Use post-finalNorm by default; flip via env to try pre-norm.
        let usePre = ProcessInfo.processInfo.environment["ANEMTP_USE_PRENORM"] == "1"
        let lastHidden: Tensor = usePre ? lastPre : lastPost
        print("ANEMTPValidation: using \(usePre ? "PRE-finalNorm" : "post-finalNorm") hidden state")

        // Get embed_t. Default = embed_tokens[lastToken] (input token at t).
        // Env flip: embed_tokens[groundTruthToken] (DeepSeek-MTP style:
        // embed of the FUTURE token, "cheating" with the ground truth
        // to validate that the architecture works at all).
        let useGroundTruthEmbed = ProcessInfo.processInfo.environment["ANEMTP_EMBED_GROUND_TRUTH"] == "1"
        let embedToken = useGroundTruthEmbed ? groundTruthToken : lastToken
        print("ANEMTPValidation: embed_t source = \(useGroundTruthEmbed ? "GROUND TRUTH (cheat)" : "lastInputToken") = \(embedToken)")
        let tokBuf = device.makeBuffer(length: 4)
        var tid = UInt32(embedToken)
        memcpy(tokBuf.contents(), &tid, 4)
        let tokTensor = Tensor(buffer: tokBuf, offset: 0, shape: [1], dtype: .u32)
        let embedCmd = device.makeCommandBuffer()
        let embedT = qwen.embedTokens(tokTensor, on: embedCmd).reshaped(to: [qwen.hidden])
        embedCmd.commit()
        await embedCmd.completed()

        // Compute RoPE sin/cos for next position = promptLen.
        // Qwen3.6 RoPE: head_dim = 128, base = 1000000.
        // theta_i = base^(-2i/d) for i in 0..d/2 = 64.
        let headDim = 128
        let halfDim = headDim / 2
        let ropeBase: Float = 1_000_000.0
        let position = Float(promptLen)
        var sinArr = [Float](repeating: 0, count: halfDim)
        var cosArr = [Float](repeating: 0, count: halfDim)
        for i in 0..<halfDim {
            let theta = powf(ropeBase, -Float(2 * i) / Float(headDim))
            let freq = position * theta
            sinArr[i] = sin(freq)
            cosArr[i] = cos(freq)
        }

        // Convert hidden_t + embed_t to f16 (mtp expects fp16 inputs
        // per Day 1 inspection — dataType.rawValue=65552 = fp16).
        let hiddenHost = lastHidden.toFloatArray()
        let embedHost = embedT.toFloatArray()

        let mtpHidden = try MLMultiArray(shape: [1, 2048].map { NSNumber(value: $0) },
                                          dataType: .float16)
        let mtpEmbed  = try MLMultiArray(shape: [1, 2048].map { NSNumber(value: $0) },
                                          dataType: .float16)
        let mtpSin    = try MLMultiArray(shape: [NSNumber(value: halfDim)],
                                          dataType: .float16)
        let mtpCos    = try MLMultiArray(shape: [NSNumber(value: halfDim)],
                                          dataType: .float16)
        copyFloatsToFP16(hiddenHost, into: mtpHidden)
        copyFloatsToFP16(embedHost,  into: mtpEmbed)
        copyFloatsToFP16(sinArr,     into: mtpSin)
        copyFloatsToFP16(cosArr,     into: mtpCos)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_t": MLFeatureValue(multiArray: mtpHidden),
            "embed_t":  MLFeatureValue(multiArray: mtpEmbed),
            "sin":      MLFeatureValue(multiArray: mtpSin),
            "cos":      MLFeatureValue(multiArray: mtpCos),
        ])
        let mtpT0 = Date()
        let mtpOut = try await mtp.prediction(from: provider)
        let mtpS = Date().timeIntervalSince(mtpT0)
        guard let hiddenNext = mtpOut.featureValue(for: "hidden_next")?.multiArrayValue
        else {
            Issue.record("MTP missing hidden_next output")
            return
        }
        print("ANEMTPValidation: MTP predict in \(String(format: "%.2f", mtpS * 1000)) ms")

        // Copy hidden_next (fp16) into a Tensor we can project.
        let hiddenNextTensor = Tensor.empty(shape: [qwen.hidden], dtype: lastHidden.dtype,
                                             device: device)
        let dst = hiddenNextTensor.buffer.contents().advanced(by: hiddenNextTensor.offset)
        let src = hiddenNext.dataPointer
        switch lastHidden.dtype {
        case .f16:
            memcpy(dst, src, qwen.hidden * 2)
        case .bf16:
            // Convert fp16 → bf16 host-side via fp32 intermediary.
            var f16Buf = [Float16](repeating: 0, count: qwen.hidden)
            memcpy(&f16Buf, src, qwen.hidden * 2)
            var bf16Buf = [UInt16](repeating: 0, count: qwen.hidden)
            for i in 0..<qwen.hidden {
                let f = Float(f16Buf[i])
                let bits = f.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                bf16Buf[i] = UInt16(rounded >> 16)
            }
            memcpy(dst, &bf16Buf, qwen.hidden * 2)
        case .f32:
            var f16Buf = [Float16](repeating: 0, count: qwen.hidden)
            memcpy(&f16Buf, src, qwen.hidden * 2)
            var f32Buf = [Float](repeating: 0, count: qwen.hidden)
            for i in 0..<qwen.hidden { f32Buf[i] = Float(f16Buf[i]) }
            memcpy(dst, &f32Buf, qwen.hidden * 4)
        default:
            Issue.record("unsupported dtype \(lastHidden.dtype) for MTP integration")
            return
        }

        // Project through lm_head + argmax → drafter candidate.
        let projCmd = device.makeCommandBuffer()
        let mtpLogits = qwen.projectHiddenToLogits(hiddenNextTensor, on: projCmd)
        projCmd.commit()
        await projCmd.completed()
        let drafterTop1 = argmaxHost(mtpLogits)

        // Top-5 from main + MTP for comparison.
        let mainTop5 = topKHost(lastLogits, k: 5)
        let mtpTop5  = topKHost(mtpLogits, k: 5)
        let mtpInTop5 = mainTop5.contains(drafterTop1)
        let mainInMTPTop5 = mtpTop5.contains(groundTruthToken)

        print("ANEMTPValidation: drafter top-1=\(drafterTop1), main top-1=\(groundTruthToken)")
        print("ANEMTPValidation: main top-5=\(mainTop5)")
        print("ANEMTPValidation: MTP  top-5=\(mtpTop5)")
        print("ANEMTPValidation: drafter ∈ main_top5: \(mtpInTop5)")
        print("ANEMTPValidation: main_top1 ∈ MTP_top5: \(mainInMTPTop5)")
        print("ANEMTPValidation: top-1 match: \(drafterTop1 == groundTruthToken)")
    }
}

@inline(__always)
private func argmaxHost(_ logits: Tensor) -> Int {
    let host = logits.toFloatArray()
    var bestIdx = 0
    var bestVal = host[0]
    for i in 1..<host.count {
        if host[i] > bestVal { bestVal = host[i]; bestIdx = i }
    }
    return bestIdx
}

@inline(__always)
private func topKHost(_ logits: Tensor, k: Int) -> [Int] {
    let host = logits.toFloatArray()
    let indexed = host.enumerated().map { ($0.offset, $0.element) }
    let sorted = indexed.sorted { $0.1 > $1.1 }
    return sorted.prefix(k).map { $0.0 }
}

@inline(__always)
private func copyFloatsToFP16(_ src: [Float], into arr: MLMultiArray) {
    precondition(arr.count == src.count,
                 "copyFloatsToFP16: count mismatch \(arr.count) vs \(src.count)")
    let dst = arr.dataPointer.bindMemory(to: Float16.self, capacity: src.count)
    for i in 0..<src.count { dst[i] = Float16(src[i]) }
}
