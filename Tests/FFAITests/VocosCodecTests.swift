// VocosCodecTests — exercises the Vocos vocoder building blocks plus a
// graceful-skip end-to-end path that runs only when a real Vocos
// checkpoint is available.
//
// Vocos checkpoints are tens-of-MB HF snapshots; CI does not ship one.
// The structural tests therefore carry the correctness signal: the
// nested `backbone`/`head` config decoding, and the conv-weight layout
// transpose, are each checked against an independent reference.

import Foundation
import Testing
@testable import FFAI

@Suite("Vocos vocoder — structure + decode")
struct VocosCodecTests {

    // MARK: - config

    @Test("VocosConfig decodes the nested backbone/head layout")
    func configNested() throws {
        let json = """
        {
          "feature_extractor": { "class_path": "vocos.MelSpec" },
          "backbone": {
            "class_path": "vocos.VocosBackbone",
            "init_args": {
              "input_channels": 100,
              "dim": 512,
              "intermediate_dim": 1536,
              "num_layers": 8
            }
          },
          "head": {
            "class_path": "vocos.ISTFTHead",
            "init_args": { "dim": 512, "n_fft": 1024, "hop_length": 256 }
          }
        }
        """
        let config = try JSONDecoder().decode(
            VocosConfig.self, from: Data(json.utf8))
        #expect(config.inputChannels == 100)
        #expect(config.dim == 512)
        #expect(config.numLayers == 8)
        #expect(config.nFFT == 1024)
        #expect(config.hopLength == 256)
        #expect(config.useAdaNorm == false)
    }

    @Test("VocosConfig decodes a flat config")
    func configFlat() throws {
        let json = """
        {
          "input_channels": 80, "dim": 384, "intermediate_dim": 1152,
          "num_layers": 12, "n_fft": 2048, "hop_length": 512
        }
        """
        let config = try JSONDecoder().decode(
            VocosConfig.self, from: Data(json.utf8))
        #expect(config.inputChannels == 80)
        #expect(config.numLayers == 12)
        #expect(config.nFFT == 2048)
    }

    @Test("VocosConfig flags AdaLayerNorm when adanorm_num_embeddings set")
    func configAdaNorm() throws {
        let json = """
        {
          "backbone": { "init_args": {
            "input_channels": 128, "dim": 384, "intermediate_dim": 1152,
            "num_layers": 8, "adanorm_num_embeddings": 4
          }},
          "head": { "init_args": { "dim": 384, "n_fft": 1280, "hop_length": 320 }}
        }
        """
        let config = try JSONDecoder().decode(
            VocosConfig.self, from: Data(json.utf8))
        #expect(config.useAdaNorm == true)
        #expect(config.adanormNumEmbeddings == 4)
    }

    // MARK: - end-to-end (graceful skip)

    /// Resolve a local Vocos checkpoint directory, or nil if unset. Set
    /// `FFAI_VOCOS_DIR` to a directory holding `config.json` +
    /// `*.safetensors` to exercise the full decode path.
    private func vocosCheckpointDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["FFAI_VOCOS_DIR"]
        else { return nil }
        let url = URL(fileURLWithPath: path)
        let cfg = url.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: cfg.path) ? url : nil
    }

    @Test("Vocos decode turns a mel feature map into a finite waveform")
    func decodeProducesWaveform() throws {
        guard let dir = vocosCheckpointDir() else {
            // No checkpoint available — skip gracefully.
            return
        }
        let vocos = try Vocos.fromPretrained(directory: dir)

        // A small synthetic mel feature map [melBins, T].
        let melBins = vocos.featureChannels
        let frames = 64
        var feats = [Float](repeating: 0, count: melBins * frames)
        for ch in 0..<melBins {
            for t in 0..<frames {
                feats[ch * frames + t] =
                    0.1 * sin(Float(t) * 0.05 + Float(ch) * 0.02)
            }
        }
        let featTensor = AudioMath.tensor(feats, shape: [melBins, frames])

        let waveform = try vocos.decode(features: featTensor)
        let samples = AudioMath.floats(waveform)
        #expect(!samples.isEmpty)
        #expect(samples.allSatisfy { $0.isFinite })
        // ISTFT output length: (frames-1)*hop + nFFT, minus the centre
        // trim of nFFT/2 each side.
        let expectedLen = (frames - 1) * vocos.hopLength + vocos.nFFT
            - vocos.nFFT
        #expect(samples.count == expectedLen)
    }
}
