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
// DACVAECodecTests — exercises the DACVAE (VAE-style Descript Audio
// Codec) building blocks plus a graceful-skip end-to-end path that runs
// only when a real DACVAE checkpoint is available.
//
// DACVAE checkpoints are multi-hundred-MB HF snapshots; CI does not ship
// one. DACVAE reuses SNAC's proven `AudioMath` / `WeightNorm` conv path
// (covered by `SNACCodecTests`), so these tests focus on the parts
// unique to DACVAE: config decoding and the continuous encode→decode
// round-trip.

import Foundation
import Testing
@testable import FFAI

@Suite("DACVAE codec — structure + round-trip")
struct DACVAECodecTests {

    // MARK: - config

    @Test("DACVAEConfig decodes a representative config.json")
    func configDecode() throws {
        let json = """
        {
          "encoder_dim": 64,
          "encoder_rates": [2, 8, 10, 12],
          "latent_dim": 1024,
          "decoder_dim": 1536,
          "decoder_rates": [12, 10, 8, 2],
          "n_codebooks": 16,
          "codebook_size": 1024,
          "codebook_dim": 128,
          "sample_rate": 48000
        }
        """
        let config = try JSONDecoder().decode(
            DACVAEConfig.self, from: Data(json.utf8))
        #expect(config.sampleRate == 48_000)
        #expect(config.hopLength == 2 * 8 * 10 * 12)   // 1920
        #expect(config.codebookDim == 128)
        #expect(config.latentDim == 1024)
    }

    @Test("DACVAEConfig fills defaults for a sparse config")
    func configDefaults() throws {
        let config = try JSONDecoder().decode(
            DACVAEConfig.self, from: Data("{}".utf8))
        #expect(config.encoderDim == 64)
        #expect(config.nCodebooks == 16)
        #expect(config.sampleRate == 48_000)
        #expect(config.codebookDim == 128)
    }

    // MARK: - end-to-end (graceful skip)

    /// Resolve a local DACVAE checkpoint directory, or nil if unset. Set
    /// `FFAI_DACVAE_DIR` to a directory holding `config.json` +
    /// `model.safetensors` to exercise the full encode/decode path.
    private func dacvaeCheckpointDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["FFAI_DACVAE_DIR"]
        else { return nil }
        let url = URL(fileURLWithPath: path)
        let cfg = url.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: cfg.path) ? url : nil
    }

    @Test("DACVAE encode→decode round-trip reconstructs a waveform")
    func encodeDecodeRoundTrip() throws {
        guard let dir = dacvaeCheckpointDir() else {
            // No checkpoint available — skip gracefully.
            return
        }
        let codec = try DACVAE.fromPretrained(directory: dir)

        // A short 0.25s sine tone at the codec sample rate.
        let n = codec.sampleRate / 4
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Float(i) / Float(codec.sampleRate)
            samples[i] = 0.5 * sin(2.0 * .pi * 220.0 * t)
        }
        let waveform = AudioMath.tensor(samples, shape: [n])

        // Encode → continuous latent [1, codebookDim, T].
        let latents = try codec.encode(waveform: waveform)
        #expect(latents.shape.count == 3)
        #expect(latents.shape[1] == codec.config.codebookDim)
        let latentFloats = AudioMath.floats(latents)
        #expect(latentFloats.allSatisfy { $0.isFinite })

        // Decode → reconstructed waveform.
        let recon = try codec.decode(latents: latents)
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
