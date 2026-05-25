// SAMAudio integration test — loads the real checkpoint and runs a segment call.
//
// The test gracefully skips when no checkpoint resolves (no network, no HF cache).
// DO NOT run this with `make test-unit` — it belongs to ModelTests which are
// executed with `make test-integration` (serialized, --num-workers 1).
//
// When a checkpoint IS available the test verifies:
//   • Config fields match the published sam-audio-large spec.
//   • The model loads without crashing and populates its weight store.
//   • `segment` returns waveforms with the expected length for the input.

import Foundation
import Testing
@testable import FFAI

@Suite("SAMAudio Integration", .serialized)
struct SAMAudioIntegrationTests {

    @Test("load + segment produces correctly-shaped output")
    func loadAndSegment() async throws {
        let model = try await SAMAudioModel.load(SAMAudio.defaultRepo)

        // Config sanity: large variant.
        #expect(model.config.transformer.dim == 2816)
        #expect(model.config.transformer.nLayers == 22)
        #expect(model.config.transformer.nHeads == 22)
        // Published mlx-community/sam-audio-large-fp16 config ships
        // sample_rate=48000 — the descript-audio-codec downsamples to a
        // 25 Hz latent at 48 kHz × 1920 product-of-encoder-rates.
        #expect(model.config.audioCodec.sampleRate == 48000)
        #expect(model.config.audioCodec.codebookDim == 128)

        // At least some weights were loaded from the checkpoint.
        #expect(model.loadedParameterCount > 0)

        // Build a 1-second synthetic 440 Hz sine waveform at the model's sample rate.
        let sampleRate = model.config.audioCodec.sampleRate
        let nSamples = sampleRate          // 1 second
        let waveform = (0..<nSamples).map { i in
            0.5 * Foundation.sin(2.0 * Float.pi * 440.0 * Float(i) / Float(sampleRate))
        }
        let description = "drums"

        let result = try await model.segment(
            waveform: waveform,
            description: description,
            ode: SAMAudioODEOptions(method: .euler, stepSize: 1.0 / 4.0)
        )

        // One item in, one item out.
        #expect(result.target.count == 1)
        #expect(result.residual.count == 1)

        // Output length matches input (decoder must preserve sample count).
        #expect(result.target[0].count == waveform.count)
        #expect(result.residual[0].count == waveform.count)
    }
}

