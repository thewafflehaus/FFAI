import Foundation
import Testing
@testable import FFAI

// Unit tests for the voice-activity-detection (VAD) model families —
// SileroVAD and SmartTurn. These exercise config decoding, the
// published per-branch geometry, weight-key remapping, and the
// `VADModelRegistry` architecture detection. The full checkpoint-load +
// forward path is covered by the ModelTests integration suites; these
// run purely on in-process data so they are fast + offline.
@Suite("VAD models")
struct VADModelTests {

    // ─── SileroVADConfig ─────────────────────────────────────────────

    @Test("SileroVADBranchConfig — published 16 kHz defaults")
    func sileroBranchDefaults16k() {
        let c = SileroVADBranchConfig.default16k
        #expect(c.sampleRate == 16000)
        #expect(c.filterLength == 256)
        #expect(c.hopLength == 128)
        #expect(c.pad == 64)
        #expect(c.cutoff == 129)
        #expect(c.contextSize == 64)
        #expect(c.chunkSize == 512)
    }

    @Test("SileroVADBranchConfig — published 8 kHz defaults")
    func sileroBranchDefaults8k() {
        let c = SileroVADBranchConfig.default8k
        #expect(c.sampleRate == 8000)
        #expect(c.filterLength == 128)
        #expect(c.hopLength == 64)
        #expect(c.pad == 32)
        #expect(c.cutoff == 65)
        #expect(c.contextSize == 32)
        #expect(c.chunkSize == 256)
    }

    @Test("SileroVADConfig — a missing config falls back to defaults")
    func sileroConfigEmptyDecode() {
        let c = SileroVADConfig.decode(from: [:])
        #expect(c.threshold == 0.5)
        #expect(c.minSpeechDurationMs == 250)
        #expect(c.minSilenceDurationMs == 100)
        #expect(c.speechPadMs == 30)
        // Both branches fall back to the published geometry.
        #expect(c.branch16k.chunkSize == 512)
        #expect(c.branch16k.contextSize == 64)
        #expect(c.branch8k.chunkSize == 256)
    }

    @Test("SileroVADConfig — decodes the published silero-vad config.json")
    func sileroConfigFullDecode() {
        // Mirrors `mlx-community/silero-vad`'s config.json verbatim.
        let raw: [String: Any] = [
            "model_type": "silero_vad",
            "threshold": 0.42,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 120,
            "speech_pad_ms": 40,
            "branch_16k": [
                "sample_rate": 16000, "filter_length": 256, "hop_length": 128,
                "pad": 64, "cutoff": 129, "context_size": 64, "chunk_size": 512,
            ],
            "branch_8k": [
                "sample_rate": 8000, "filter_length": 128, "hop_length": 64,
                "pad": 32, "cutoff": 65, "context_size": 32, "chunk_size": 256,
            ],
        ]
        let c = SileroVADConfig.decode(from: raw)
        #expect(c.threshold == 0.42)
        #expect(c.minSpeechDurationMs == 300)
        #expect(c.minSilenceDurationMs == 120)
        #expect(c.speechPadMs == 40)
        #expect(c.branch16k.filterLength == 256)
        #expect(c.branch16k.hopLength == 128)
        #expect(c.branch16k.cutoff == 129)
        #expect(c.branch8k.filterLength == 128)
        #expect(c.branch8k.cutoff == 65)
    }

    @Test("SileroVADConfig — a partial branch block keeps per-field defaults")
    func sileroConfigPartialBranch() {
        // Only `threshold` and a sparse 16k block are present; missing
        // branch fields fall back to the published 16k geometry.
        let raw: [String: Any] = [
            "threshold": 0.6,
            "branch_16k": ["chunk_size": 512],
        ]
        let c = SileroVADConfig.decode(from: raw)
        #expect(c.threshold == 0.6)
        #expect(c.branch16k.chunkSize == 512)
        // Untouched fields keep the default.
        #expect(c.branch16k.cutoff == 129)
        #expect(c.branch16k.contextSize == 64)
    }

    @Test("SileroVADModel.remap — branch-prefixes raw checkpoint keys")
    func sileroRemap() {
        // `vad_16k.` / `vad_8k.` → `branch16k.` / `branch8k.`.
        #expect(SileroVADModel.remap("vad_16k.conv1.weight") == "branch16k.conv1.weight")
        #expect(SileroVADModel.remap("vad_8k.lstm.Wx") == "branch8k.lstm.Wx")
        // `val_*` validation tensors are dropped.
        #expect(SileroVADModel.remap("val_loss") == nil)
        // Unprefixed keys pass through unchanged.
        #expect(SileroVADModel.remap("threshold") == "threshold")
    }

    @Test("SileroVADModel.probsToSegments — threshold + hysteresis smoothing")
    func sileroProbsToSegments() {
        // 40 chunks @ 512 samples / 16 kHz: silence, a speech burst long
        // enough to clear minSpeechDurationMs, then silence again.
        var probs = [Float](repeating: 0.05, count: 40)
        for i in 12..<28 { probs[i] = 0.9 }
        let segments = SileroVADModel.probsToSegments(
            probs, audioLen: 40 * 512, sampleRate: 16000, chunkSize: 512,
            threshold: 0.5, minSpeechDurationMs: 250,
            minSilenceDurationMs: 100, speechPadMs: 30)
        // The burst should yield exactly one speech segment with a
        // positive duration that does not cover the whole clip.
        #expect(segments.count == 1)
        if let s = segments.first {
            #expect(s.durationSeconds > 0)
            #expect(s.startSample < s.endSample)
            #expect(s.endSample <= 40 * 512)
        }
    }

    @Test("SileroVADModel.probsToSegments — all-silence yields no segments")
    func sileroProbsToSegmentsSilence() {
        let probs = [Float](repeating: 0.02, count: 30)
        let segments = SileroVADModel.probsToSegments(
            probs, audioLen: 30 * 512, sampleRate: 16000, chunkSize: 512,
            threshold: 0.5, minSpeechDurationMs: 250,
            minSilenceDurationMs: 100, speechPadMs: 30)
        #expect(segments.isEmpty)
    }

    // ─── SmartTurnConfig ─────────────────────────────────────────────

    @Test("SmartTurnConfig — published smart-turn-v3 defaults")
    func smartTurnConfigDefaults() {
        let c = SmartTurnConfig()
        #expect(c.numMelBins == 80)
        #expect(c.maxSourcePositions == 400)
        #expect(c.dModel == 384)
        #expect(c.encoderAttentionHeads == 6)
        #expect(c.encoderLayers == 4)
        #expect(c.encoderFfnDim == 1536)
        #expect(c.kProjBias == false)
        #expect(c.samplingRate == 16000)
        #expect(c.maxAudioSeconds == 8)
        #expect(c.nFft == 400)
        #expect(c.hopLength == 160)
        #expect(c.normalizeAudio == true)
        #expect(c.threshold == 0.5)
        // headDim must divide evenly — the encoder layer relies on it.
        #expect(c.dModel % c.encoderAttentionHeads == 0)
    }

    @Test("SmartTurnConfig — an empty config falls back to defaults")
    func smartTurnConfigEmptyDecode() {
        let c = SmartTurnConfig.decode(from: [:])
        #expect(c.dModel == 384)
        #expect(c.encoderLayers == 4)
        #expect(c.numMelBins == 80)
        #expect(c.threshold == 0.5)
    }

    @Test("SmartTurnConfig — reads the nested encoder / processor blocks")
    func smartTurnConfigNestedDecode() {
        let raw: [String: Any] = [
            "model_type": "smart_turn_v3",
            "encoder_config": [
                "num_mel_bins": 80, "max_source_positions": 400,
                "d_model": 256, "encoder_attention_heads": 4,
                "encoder_layers": 6, "encoder_ffn_dim": 1024,
                "k_proj_bias": true,
            ],
            "processor_config": [
                "sampling_rate": 16000, "max_audio_seconds": 10,
                "n_fft": 512, "hop_length": 128,
                "normalize_audio": false, "threshold": 0.65,
            ],
        ]
        let c = SmartTurnConfig.decode(from: raw)
        #expect(c.dModel == 256)
        #expect(c.encoderAttentionHeads == 4)
        #expect(c.encoderLayers == 6)
        #expect(c.encoderFfnDim == 1024)
        #expect(c.kProjBias == true)
        #expect(c.maxAudioSeconds == 10)
        #expect(c.nFft == 512)
        #expect(c.hopLength == 128)
        #expect(c.normalizeAudio == false)
        #expect(c.threshold == 0.65)
    }

    @Test("SmartTurnConfig — top-level keys are used when blocks absent")
    func smartTurnConfigFlatDecode() {
        // Some checkpoints store encoder / processor fields at the top
        // level rather than in nested blocks; `decode` falls back to
        // `raw` for both.
        let raw: [String: Any] = [
            "d_model": 512, "encoder_layers": 8, "n_fft": 480,
        ]
        let c = SmartTurnConfig.decode(from: raw)
        #expect(c.dModel == 512)
        #expect(c.encoderLayers == 8)
        #expect(c.nFft == 480)
        // Unspecified fields keep the published default.
        #expect(c.numMelBins == 80)
    }

    @Test("SmartTurnModel.remap — flattens nested head keys, drops val_*")
    func smartTurnRemap() {
        // A leading `inner.` prefix is stripped.
        #expect(SmartTurnModel.remap("inner.encoder.conv1.weight")
                == "encoder.conv1.weight")
        // `pool_attention.N` / `classifier.N` → underscored flat names.
        #expect(SmartTurnModel.remap("pool_attention.0.weight")
                == "pool_attention_0.weight")
        #expect(SmartTurnModel.remap("classifier.6.bias")
                == "classifier_6.bias")
        // `val_*` validation tensors are dropped.
        #expect(SmartTurnModel.remap("val_accuracy") == nil)
        // An ordinary key passes through unchanged.
        #expect(SmartTurnModel.remap("encoder.layer_norm.weight")
                == "encoder.layer_norm.weight")
    }

    // ─── VADModelRegistry detection ──────────────────────────────────

    @Test("AudioModelKind — model_type sets map each VAD architecture")
    func audioModelKindModelTypes() {
        #expect(AudioModelKind.sileroVAD.modelTypes.contains("silero_vad"))
        #expect(AudioModelKind.smartTurn.modelTypes.contains("smart_turn"))
        #expect(AudioModelKind.smartTurn.modelTypes.contains("smart_turn_v3"))
        #expect(AudioModelKind.sortformer.modelTypes.contains("sortformer"))
    }

    @Test("VADModelRegistry.detectKind — dispatches by config model_type")
    func registryDetectKindByConfig() throws {
        let dir = try writeTempConfig(["model_type": "silero_vad"],
                                      named: "some-checkpoint")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .sileroVAD)

        let dir2 = try writeTempConfig(["model_type": "smart_turn_v3"],
                                       named: "another-checkpoint")
        defer { try? FileManager.default.removeItem(at: dir2) }
        #expect(try VADModelRegistry.detectKind(in: dir2) == .smartTurn)
    }

    @Test("VADModelRegistry.detectKind — falls back to directory name")
    func registryDetectKindByName() throws {
        // No config.json at all — detection must fall back to the
        // directory name heuristic.
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("models--mlx-community--smart-turn-v3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .smartTurn)
    }

    @Test("VADModelRegistry.detectKind — rejects an unknown architecture")
    func registryDetectKindUnknown() throws {
        let dir = try writeTempConfig(["model_type": "not_a_vad_model"],
                                      named: "mystery-model")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: AudioModelError.self) {
            _ = try VADModelRegistry.detectKind(in: dir)
        }
    }

    // Write a config.json into a fresh temp directory and return the
    // directory URL.
    private func writeTempConfig(_ config: [String: Any],
                                 named: String) throws -> URL {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("\(named)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }
}
