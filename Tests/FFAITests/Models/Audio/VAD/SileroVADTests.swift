// SileroVADTests — unit tests for the SileroVAD family: branch / model
// config decoding, post-processing logic, the weight-key remap, and the
// VADModelRegistry detection path. All tests run offline, no checkpoint
// required — the full forward path is covered by the integration suite.

import Foundation
import Testing
@testable import FFAI

@Suite("SileroVAD")
struct SileroVADTests {

    // ─── SileroVADBranchConfig defaults ──────────────────────────────────

    @Test("SileroVADBranchConfig — published 16 kHz defaults")
    func branchDefault16k() {
        let b = SileroVADBranchConfig.default16k
        #expect(b.sampleRate == 16_000)
        #expect(b.filterLength > 0)
        #expect(b.hopLength > 0)
        #expect(b.contextSize > 0)
        #expect(b.chunkSize > 0)
    }

    @Test("SileroVADBranchConfig — published 8 kHz defaults")
    func branchDefault8k() {
        let b = SileroVADBranchConfig.default8k
        #expect(b.sampleRate == 8_000)
        #expect(b.filterLength > 0)
        #expect(b.chunkSize > 0)
    }

    // ─── SileroVADConfig decoding ────────────────────────────────────────

    @Test("SileroVADConfig — published defaults")
    func configDefaults() {
        let c = SileroVADConfig()
        #expect(c.threshold == 0.5)
        #expect(c.minSpeechDurationMs == 250)
        #expect(c.minSilenceDurationMs == 100)
        #expect(c.speechPadMs == 30)
        #expect(c.branch16k.sampleRate == 16_000)
        #expect(c.branch8k.sampleRate == 8_000)
    }

    @Test("SileroVADConfig.decode — empty raw falls back to defaults")
    func configEmptyDecode() {
        let c = SileroVADConfig.decode(from: [:])
        #expect(c.threshold == 0.5)
        #expect(c.minSpeechDurationMs == 250)
        #expect(c.minSilenceDurationMs == 100)
        #expect(c.speechPadMs == 30)
    }

    @Test("SileroVADConfig.decode — full config is decoded correctly")
    func configFullDecode() {
        let raw: [String: Any] = [
            "model_type": "silero_vad",
            "threshold": 0.6,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 150,
            "speech_pad_ms": 40,
        ]
        let c = SileroVADConfig.decode(from: raw)
        #expect(abs(c.threshold - 0.6) < 1e-5)
        #expect(c.minSpeechDurationMs == 300)
        #expect(c.minSilenceDurationMs == 150)
        #expect(c.speechPadMs == 40)
    }

    @Test("SileroVADConfig.decode — partial config keeps per-field defaults")
    func configPartialDecode() {
        let raw: [String: Any] = ["threshold": 0.7]
        let c = SileroVADConfig.decode(from: raw)
        #expect(abs(c.threshold - 0.7) < 1e-5)
        // Untouched fields keep the published default.
        #expect(c.minSpeechDurationMs == 250)
    }

    // ─── Weight-key remap ───────────────────────────────────────────────

    @Test("SileroVADModel.remap — vad_16k.* maps to branch16k.*")
    func remap16k() {
        let key = "vad_16k.stft_conv.weight"
        let mapped = SileroVADModel.remap(key)
        #expect(mapped == "branch16k.stft_conv.weight")
    }

    @Test("SileroVADModel.remap — vad_8k.* maps to branch8k.*")
    func remap8k() {
        let key = "vad_8k.conv1.weight"
        let mapped = SileroVADModel.remap(key)
        #expect(mapped == "branch8k.conv1.weight")
    }

    @Test("SileroVADModel.remap — val_* keys are dropped")
    func remapDropsValidation() {
        let mapped = SileroVADModel.remap("val_acc")
        #expect(mapped == nil)
    }

    @Test("SileroVADModel.remap — unknown keys pass through unchanged")
    func remapPassThrough() {
        let mapped = SileroVADModel.remap("threshold")
        #expect(mapped == "threshold")
    }

    // ─── probsToSegments ────────────────────────────────────────────────

    @Test("SileroVADModel.probsToSegments — all-silence yields no segments")
    func probsToSegmentsSilence() {
        let probs = [Float](repeating: 0.02, count: 30)
        let segs = SileroVADModel.probsToSegments(
            probs, audioLen: 30 * 512, sampleRate: 16_000,
            chunkSize: 512, threshold: 0.5,
            minSpeechDurationMs: 250, minSilenceDurationMs: 100,
            speechPadMs: 30)
        #expect(segs.isEmpty)
    }

    @Test("SileroVADModel.probsToSegments — speech burst yields one segment")
    func probsToSegmentsBurst() {
        // 30 chunks × 512 samples / 16 kHz = ~960 ms total. Frames 5..25
        // (20 chunks = 640 ms) are speech above threshold — well past
        // minSpeechDurationMs=250 ms.
        var probs = [Float](repeating: 0.02, count: 30)
        for i in 5..<25 { probs[i] = 0.95 }
        let segs = SileroVADModel.probsToSegments(
            probs, audioLen: 30 * 512, sampleRate: 16_000,
            chunkSize: 512, threshold: 0.5,
            minSpeechDurationMs: 250, minSilenceDurationMs: 100,
            speechPadMs: 30)
        #expect(segs.count == 1)
        if let s = segs.first {
            #expect(s.startSample < s.endSample)
            #expect(s.durationSeconds > 0)
        }
    }

    // ─── Error stringification ───────────────────────────────────────────

    @Test("SileroVADError.description — unsupportedSampleRate carries the rate")
    func errorDescriptionSampleRate() {
        let err = SileroVADError.unsupportedSampleRate(48_000)
        #expect(err.description.contains("48000"))
    }

    @Test("SileroVADError.description — missingWeight carries the key")
    func errorDescriptionMissingWeight() {
        let err = SileroVADError.missingWeight("branch16k.conv1.weight")
        #expect(err.description.contains("conv1.weight"))
    }

    // ─── VADModelRegistry detection ─────────────────────────────────────

    @Test("AudioModelKind.sileroVAD — modelTypes contains silero_vad")
    func audioModelKindSilero() {
        #expect(AudioModelKind.sileroVAD.modelTypes.contains("silero_vad"))
    }

    @Test("VADModelRegistry.detectKind — recognizes silero_vad model_type")
    func registryDetectKindSilero() throws {
        let dir = try writeTempConfig(["model_type": "silero_vad"],
                                      named: "silero-vad-checkpoint")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .sileroVAD)
    }

    @Test("VADModelRegistry.detectKind — falls back to directory name for silero")
    func registryDetectKindSileroByName() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent(
            "models--mlx-community--silero-vad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .sileroVAD)
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

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
