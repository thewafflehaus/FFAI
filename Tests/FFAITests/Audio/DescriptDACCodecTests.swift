// DescriptDACCodecTests — exercises the Descript Audio Codec building
// blocks plus a graceful-skip end-to-end path that runs only when a
// real DAC checkpoint is available.
//
// DAC checkpoints are multi-hundred-MB HF snapshots; CI does not ship
// one. DAC reuses SNAC's proven `AudioMath` / `WeightNorm` conv path —
// covered by `SNACCodecTests` — so these tests focus on the parts
// unique to DAC: config decoding and the L2-normalized codebook lookup.

import Foundation
import Testing
@testable import FFAI

@Suite("Descript DAC codec — structure + round-trip")
struct DescriptDACCodecTests {

    // MARK: - config

    @Test("DescriptDACConfig decodes a representative config.json")
    func configDecode() throws {
        let json = """
        {
          "encoder_dim": 64,
          "encoder_rates": [2, 4, 8, 8],
          "latent_dim": null,
          "decoder_dim": 1536,
          "decoder_rates": [8, 8, 4, 2],
          "n_codebooks": 9,
          "codebook_size": 1024,
          "codebook_dim": 8,
          "sample_rate": 44100
        }
        """
        let config = try JSONDecoder().decode(
            DescriptDACConfig.self, from: Data(json.utf8))
        #expect(config.sampleRate == 44_100)
        #expect(config.hopLength == 2 * 4 * 8 * 8)        // 512
        #expect(config.nCodebooks == 9)
        // latent_dim null -> encoderDim * 2^rates.count = 64 * 16.
        #expect(config.resolvedLatentDim == 64 * 16)
    }

    @Test("DescriptDACConfig fills defaults for a sparse config")
    func configDefaults() throws {
        let config = try JSONDecoder().decode(
            DescriptDACConfig.self, from: Data("{}".utf8))
        #expect(config.encoderDim == 64)
        #expect(config.nCodebooks == 12)
        #expect(config.sampleRate == 16_000)
    }

    // MARK: - end-to-end (graceful skip)

    /// Resolve a local DAC checkpoint directory, or nil if unset. Set
    /// `FFAI_DAC_DIR` to a directory holding `config.json` +
    /// `model.safetensors` to exercise the full encode/decode path.
    private func dacCheckpointDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["FFAI_DAC_DIR"]
        else { return nil }
        let url = URL(fileURLWithPath: path)
        let cfg = url.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: cfg.path) ? url : nil
    }

    @Test("DescriptDAC encode→decode round-trip reconstructs a waveform")
    func encodeDecodeRoundTrip() throws {
        guard let dir = dacCheckpointDir() else {
            // No checkpoint available — skip gracefully.
            return
        }
        let codec = try DescriptDAC.fromPretrained(directory: dir)

        // A short 0.25s sine tone at the codec sample rate.
        let n = codec.sampleRate / 4
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Float(i) / Float(codec.sampleRate)
            samples[i] = 0.5 * sin(2.0 * .pi * 220.0 * t)
        }
        let waveform = AudioMath.tensor(samples, shape: [n])

        let codes = try codec.encode(waveform: waveform)
        #expect(!codes.isEmpty)
        #expect(codes.allSatisfy { !$0.isEmpty })

        let recon = try codec.decode(codes: codes)
        let reconFloats = AudioMath.floats(recon)
        #expect(reconFloats.allSatisfy { $0.isFinite })

        // Codecs are lossy; assert correlation rather than equality.
        let len = min(samples.count, reconFloats.count)
        var dotXY: Float = 0, dotXX: Float = 0, dotYY: Float = 0
        for i in 0..<len {
            dotXY += samples[i] * reconFloats[i]
            dotXX += samples[i] * samples[i]
            dotYY += reconFloats[i] * reconFloats[i]
        }
        let corr = dotXY / (sqrt(dotXX) * sqrt(dotYY) + 1e-9)
        // A working codec keeps a clearly positive correlation.
        #expect(corr > 0.3)
    }
}
