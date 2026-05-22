// BigVGANCodecTests — exercises the BigVGAN vocoder building blocks
// plus a graceful-skip end-to-end path that runs only when a real
// BigVGAN checkpoint is available.
//
// BigVGAN checkpoints are tens-of-MB HF snapshots; CI does not ship
// one. BigVGAN reuses SNAC's proven `WeightNorm` conv path — covered by
// `SNACCodecTests` — so these tests focus on the parts unique to
// BigVGAN: config decoding and the Kaiser-sinc anti-aliasing filter.

import Foundation
import Testing
@testable import FFAI

@Suite("BigVGAN vocoder — structure + decode")
struct BigVGANCodecTests {

    // MARK: - config

    @Test("BigVGANConfig decodes a representative config.json")
    func configDecode() throws {
        let json = """
        {
          "num_mels": 100,
          "upsample_rates": [4, 4, 2, 2, 2, 2],
          "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
          "upsample_initial_channel": 1536,
          "resblock": "1",
          "resblock_kernel_sizes": [3, 7, 11],
          "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
          "activation": "snakebeta",
          "snake_logscale": true
        }
        """
        let config = try JSONDecoder().decode(
            BigVGANConfig.self, from: Data(json.utf8))
        #expect(config.numMels == 100)
        #expect(config.numUpsamples == 6)
        #expect(config.numKernels == 3)
        #expect(config.resblock == .one)
        #expect(config.activation == .snakebeta)
    }

    @Test("BigVGANConfig fills defaults for a sparse config")
    func configDefaults() throws {
        let config = try JSONDecoder().decode(
            BigVGANConfig.self, from: Data("{}".utf8))
        #expect(config.numMels == 100)
        #expect(config.upsampleInitialChannel == 1536)
        #expect(config.useTanhAtFinal == true)
    }

    // MARK: - Kaiser-sinc filter

    @Test("Kaiser-sinc low-pass filter is finite and sum-normalized")
    func kaiserSincNormalized() {
        let filter = BigVGANFilter.kaiserSinc(
            cutoff: 0.25, halfWidth: 0.3, kernelSize: 12)
        #expect(filter.count == 12)
        #expect(filter.allSatisfy { $0.isFinite })
        // The kernel is normalized to sum to 1.
        let sum = filter.reduce(0, +)
        #expect(abs(sum - 1.0) < 1e-4)
    }

    @Test("Kaiser-sinc filter is zero when cutoff is non-positive")
    func kaiserSincZeroCutoff() {
        let filter = BigVGANFilter.kaiserSinc(
            cutoff: 0.0, halfWidth: 0.3, kernelSize: 8)
        #expect(filter.allSatisfy { $0 == 0 })
    }

    // MARK: - end-to-end (graceful skip)

    /// Resolve a local BigVGAN checkpoint directory, or nil if unset.
    /// Set `FFAI_BIGVGAN_DIR` to a directory holding `config.json` +
    /// `*.safetensors` to exercise the full decode path.
    private func bigvganCheckpointDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["FFAI_BIGVGAN_DIR"]
        else { return nil }
        let url = URL(fileURLWithPath: path)
        let cfg = url.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: cfg.path) ? url : nil
    }

    @Test("BigVGAN decode turns a mel into a finite waveform")
    func decodeProducesWaveform() throws {
        guard let dir = bigvganCheckpointDir() else {
            // No checkpoint available — skip gracefully.
            return
        }
        let vocoder = try BigVGAN.fromPretrained(directory: dir)

        // A small synthetic mel feature map [melBins, T].
        let melBins = vocoder.melChannels
        let frames = 32
        var feats = [Float](repeating: 0, count: melBins * frames)
        for ch in 0..<melBins {
            for t in 0..<frames {
                feats[ch * frames + t] =
                    0.1 * sin(Float(t) * 0.07 + Float(ch) * 0.03)
            }
        }
        let melTensor = AudioMath.tensor(feats, shape: [melBins, frames])

        let waveform = try vocoder.decode(mel: melTensor)
        let samples = AudioMath.floats(waveform)
        #expect(!samples.isEmpty)
        #expect(samples.allSatisfy { $0.isFinite })
        // The final tanh / clip keeps the output bounded in [-1, 1].
        #expect(samples.allSatisfy { $0 >= -1.0001 && $0 <= 1.0001 })
    }
}
