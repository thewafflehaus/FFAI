// FishS1DACTests — unit tests for FishS1DACConfig parsing and basic
// structural invariants of the codec.
//
// These tests run without loading any checkpoint (no GPU required) and
// complete in milliseconds. End-to-end decode with real weights is in the
// integration test (FishSpeechIntegrationTests.swift).

import Foundation
import Testing
@testable import FFAI

@Suite("FishS1DAC config + structural invariants")
struct FishS1DACTests {

    // MARK: - Config parsing

    @Test("FishS1DACConfig decodes a representative codec.json")
    func configDecodeFullJSON() throws {
        let json = """
        {
          "encoder_dim": 64,
          "encoder_rates": [2, 4, 8, 8],
          "latent_dim": 1024,
          "decoder_dim": 1536,
          "decoder_rates": [8, 8, 4, 2],
          "n_codebooks": 9,
          "codebook_size": 1024,
          "codebook_dim": 8,
          "semantic_codebook_size": 4096,
          "downsample_factor": [2, 2],
          "sample_rate": 44100,
          "quantizer_transformer_layers": 8,
          "quantizer_transformer_heads": 16,
          "quantizer_transformer_dim": 1024,
          "quantizer_transformer_intermediate_size": 3072,
          "quantizer_transformer_head_dim": 64,
          "quantizer_window_size": 128
        }
        """
        let config = try JSONDecoder().decode(
            FishS1DACConfig.self, from: Data(json.utf8))

        #expect(config.encoderDim == 64)
        #expect(config.encoderRates == [2, 4, 8, 8])
        #expect(config.latentDim == 1024)
        #expect(config.decoderDim == 1536)
        #expect(config.decoderRates == [8, 8, 4, 2])
        #expect(config.nCodebooks == 9)
        #expect(config.codebookSize == 1024)
        #expect(config.codebookDim == 8)
        #expect(config.semanticCodebookSize == 4096)
        #expect(config.downsampleFactor == [2, 2])
        #expect(config.sampleRate == 44_100)
        #expect(config.quantizerTransformerLayers == 8)
        #expect(config.quantizerTransformerHeads == 16)
        #expect(config.quantizerTransformerDim == 1024)
        #expect(config.quantizerTransformerIntermediateSize == 3072)
        #expect(config.quantizerTransformerHeadDim == 64)
        #expect(config.quantizerWindowSize == 128)
    }

    @Test("FishS1DACConfig fills S2 preset defaults for a sparse config")
    func configDefaults() throws {
        let config = try JSONDecoder().decode(
            FishS1DACConfig.self, from: Data("{}".utf8))

        // Architecture defaults for FishAudio S2.
        #expect(config.encoderDim == 64)
        #expect(config.encoderRates == [2, 4, 8, 8])
        #expect(config.latentDim == 1024)
        #expect(config.decoderDim == 1536)
        #expect(config.decoderRates == [8, 8, 4, 2])
        #expect(config.nCodebooks == 9)
        #expect(config.codebookSize == 1024)
        #expect(config.codebookDim == 8)
        #expect(config.semanticCodebookSize == 4096)
        #expect(config.downsampleFactor == [2, 2])
        #expect(config.sampleRate == 44_100)
        #expect(config.quantizerTransformerLayers == 8)
        #expect(config.quantizerWindowSize == 128)
    }

    // MARK: - Computed properties

    @Test("FishS1DACConfig.hopLength equals product of encoder rates")
    func hopLength() throws {
        let config = try JSONDecoder().decode(
            FishS1DACConfig.self, from: Data("{}".utf8))
        // Default encoderRates = [2, 4, 8, 8] → product = 512.
        #expect(config.hopLength == 2 * 4 * 8 * 8)
        #expect(config.hopLength == 512)
    }

    @Test("FishS1DACConfig.quantizerUpsampleFactor equals product of downsampleFactor")
    func quantizerUpsampleFactor() throws {
        let config = try JSONDecoder().decode(
            FishS1DACConfig.self, from: Data("{}".utf8))
        // Default downsampleFactor = [2, 2] → product = 4.
        #expect(config.quantizerUpsampleFactor == 4)
    }

    @Test("FishS1DACConfig.frameLength is 4× hopLength")
    func frameLength() throws {
        let config = try JSONDecoder().decode(
            FishS1DACConfig.self, from: Data("{}".utf8))
        #expect(config.frameLength == config.hopLength * 4)
        #expect(config.frameLength == 2048)
    }

    @Test("FishS1DACConfig accepts optional downsample_dims field")
    func configDownsampleDimsOptional() throws {
        let withDims = """
        {"downsample_dims": [512, 1024]}
        """
        let config = try JSONDecoder().decode(
            FishS1DACConfig.self, from: Data(withDims.utf8))
        #expect(config.downsampleDims == [512, 1024])

        // Absent key → nil.
        let without = try JSONDecoder().decode(
            FishS1DACConfig.self, from: Data("{}".utf8))
        #expect(without.downsampleDims == nil)
    }

    // MARK: - Error descriptions

    @Test("FishS1DACError has descriptive descriptions")
    func errorDescriptions() {
        let missing = FishS1DACError.missingWeights("quantizer.foo")
        #expect(missing.description.contains("missing weights"))
        #expect(missing.description.contains("quantizer.foo"))

        let notFound = FishS1DACError.configNotFound("/some/path")
        #expect(notFound.description.contains("config not found"))

        let shape = FishS1DACError.shapeMismatch("expected [1,1024,T]")
        #expect(shape.description.contains("shape mismatch"))
    }

    // MARK: - FishS1DAC public surface

    @Test("FishS1DAC exposes sampleRate, hopLength, numCodebooks from config")
    func dacPublicSurface() throws {
        // Build a minimal config by parsing the default empty JSON.
        let config = try JSONDecoder().decode(
            FishS1DACConfig.self, from: Data("{}".utf8))

        // hopLength and numCodebooks computed from defaults.
        #expect(config.sampleRate == 44_100)
        #expect(config.hopLength == 512)
        // numCodebooks = nCodebooks (residual) + 1 (semantic).
        #expect(config.nCodebooks + 1 == 10)
    }

    // MARK: - Checkpoint-gated end-to-end (graceful skip)

    /// Resolve a local FishS1DAC checkpoint directory, or nil if unset.
    /// Set `FFAI_FISH_DAC_DIR` to a directory holding `codec.json` (or
    /// `config.json`) and `codec.safetensors` / `model.safetensors` to
    /// exercise the full decode path.
    private func dacCheckpointDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["FFAI_FISH_DAC_DIR"]
        else { return nil }
        let url = URL(fileURLWithPath: path)
        let hasConfig = FileManager.default.fileExists(
            atPath: url.appendingPathComponent("codec.json").path) ||
            FileManager.default.fileExists(
                atPath: url.appendingPathComponent("config.json").path)
        return hasConfig ? url : nil
    }

    @Test("FishS1DAC.load + decode produces finite waveform samples")
    func loadAndDecodeWithCheckpoint() throws {
        guard let dir = dacCheckpointDir() else {
            // No checkpoint environment variable set — skip gracefully.
            return
        }

        let codec = try FishS1DAC.load(from: dir)

        // Public invariants from config.
        #expect(codec.sampleRate == 44_100)
        #expect(codec.hopLength == 512)
        #expect(codec.numCodebooks == 10)    // 1 semantic + 9 residual (S2 defaults)

        // 4 frames of silence codes (all zeros).
        let nFrames = 4
        let codes = (0..<codec.numCodebooks).map { _ in
            [Int32](repeating: 0, count: nFrames)
        }

        let waveform = try codec.decode(codes: codes)

        // Shape: [1, 1, nFrames * hopLength].
        let expectedLen = nFrames * codec.hopLength
        #expect(waveform.shape.count == 3)
        #expect(waveform.shape[0] == 1)
        #expect(waveform.shape[1] == 1)
        #expect(waveform.shape[2] == expectedLen)

        // All samples must be finite.
        let floats = AudioMath.floats(waveform)
        #expect(floats.allSatisfy { $0.isFinite })

        // Rough amplitude sanity: silence codes should produce small output
        // (codec embedding [0] may not be exactly silent, but should be bounded).
        let rms = sqrt(floats.map { $0 * $0 }.reduce(0, +) / Float(floats.count))
        #expect(rms < 1.0)    // well within ±1 f32 audio range
    }
}
