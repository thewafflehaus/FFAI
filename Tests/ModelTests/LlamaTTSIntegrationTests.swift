// Slow integration test: downloads (or hits cache) an Orpheus-style
// LlamaTTS checkpoint and exercises the Llama acoustic backbone +
// Orpheus token protocol. Skipped automatically if the network or the
// checkpoint isn't available — mirrors the other ModelTests suites.
//
// LlamaTTS reuses FFAI's `LlamaModel` engine for the acoustic backbone
// and adds the Orpheus prompt framing + autoregressive SNAC-code decode
// loop. The SNAC neural codec (the waveform-synthesis tail) is a
// separate codec port; this suite verifies the model loads, the prompt
// framing is well-formed, and `generateCodes` emits de-interleaved SNAC
// code planes — the contract the codec consumes.

import Foundation
import Testing
@testable import FFAI

@Suite("LlamaTTS (Orpheus) integration", .serialized)
struct LlamaTTSIntegrationTests {

    /// Load LlamaTTS from the HF cache / network, or return nil with a
    /// printed skip reason.
    private func loadLlamaTTS() async -> LlamaTTSModel? {
        for repoId in [
            "mlx-community/orpheus-3b-0.1-ft-bf16",
            "canopylabs/orpheus-3b-0.1-ft",
        ] {
            do {
                let locator = ModelLocator()
                let dir = try await ModelLoadLock.shared.loadSerially {
                    try await locator.resolve(idOrPath: repoId)
                }
                return try await LlamaTTSModel.load(directory: dir)
            } catch {
                print("LlamaTTS load from \(repoId) skipped: \(error)")
            }
        }
        return nil
    }

    @Test("deinterleave — SNAC code planes have the right shape")
    func deinterleave_planeShapes() {
        // One SNAC frame is 7 interleaved code tokens. Three frames →
        // 21 tokens → layer1 has 3, layer2 has 6, layer3 has 12.
        let tokens = Array(0..<21)
        let planes = LlamaTTSModel.deinterleaveSNACCodes(tokens)
        #expect(planes.count == 3)
        #expect(planes[0].count == 3)
        #expect(planes[1].count == 6)
        #expect(planes[2].count == 12)
        // A partial trailing frame is dropped.
        let partial = LlamaTTSModel.deinterleaveSNACCodes(Array(0..<25))
        #expect(partial[0].count == 3)
    }

    @Test("load — Orpheus checkpoint binds the Llama backbone")
    func loadLlamaTTS_bindsBackbone() async throws {
        guard let model = await loadLlamaTTS() else {
            print("LlamaTTS integration test skipped: checkpoint unavailable")
            return
        }
        #expect(model.backbone.nLayers > 0)
        #expect(model.backbone.vocab > OrpheusTokens.audioTokenOffset)
        #expect(model.sampleRate == 24_000)
    }

    @Test("promptTokens — Orpheus framing is well-formed")
    func promptTokens_framing() async throws {
        guard let model = await loadLlamaTTS() else {
            print("LlamaTTS integration test skipped: checkpoint unavailable")
            return
        }
        let ids = model.promptTokens(text: "Hello there.", voice: "tara")
        // [SOH] ... [EOT][EOH] — start, end-of-text, end-of-human.
        #expect(ids.first == OrpheusTokens.startOfHuman)
        #expect(ids[ids.count - 2] == OrpheusTokens.endOfText)
        #expect(ids.last == OrpheusTokens.endOfHuman)
        #expect(ids.count > 4)
    }

    @Test("generateCodes — decode emits finite SNAC code planes")
    func generateCodes_emitsCodes() async throws {
        guard let model = await loadLlamaTTS() else {
            print("LlamaTTS integration test skipped: checkpoint unavailable")
            return
        }
        // Greedy decode, capped short for test runtime.
        let planes = try model.generateCodes(
            text: "Hi.", voice: "tara", maxFrames: 16, temperature: 0)
        #expect(planes.count == 3)
        // Non-empty: at least one complete SNAC frame was emitted.
        #expect(!planes[0].isEmpty,
                "LlamaTTS produced no SNAC codes")
        // SNAC up-sampling: layer2 is 2×, layer3 is 4× layer1's length.
        #expect(planes[1].count == 2 * planes[0].count)
        #expect(planes[2].count == 4 * planes[0].count)
        print("LlamaTTS generated \(planes[0].count) SNAC frames")
    }
}
