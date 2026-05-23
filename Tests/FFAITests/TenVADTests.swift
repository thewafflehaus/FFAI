import Foundation
import Testing
@testable import FFAI

// Unit tests for TenVAD — config decoding, post-processing logic, and
// VADModelRegistry detection. These run purely on in-process data (no
// checkpoint, no network) so they are fast + offline.
//
// The full checkpoint-load + forward path is covered by
// TenVADIntegrationTests in ModelTests; that suite requires the
// `TEN-framework/ten-vad` snapshot and is disabled pending a
// safetensors-format mlx-community conversion (see TenVAD.swift header
// for details).
@Suite("TenVAD")
struct TenVADTests {

    // ─── TenVADConfig decoding ───────────────────────────────────────

    @Test("TenVADConfig — published defaults match reference values")
    func configDefaults() {
        let c = TenVADConfig()
        // Published defaults from the TEN-VAD upstream (aed_st.h).
        #expect(c.hopSize == 256)
        #expect(c.threshold == 0.5)
        #expect(c.minSpeechDurationMs == 250)
        #expect(c.minSilenceDurationMs == 100)
        #expect(c.speechPadMs == 30)
    }

    @Test("TenVADConfig — empty config falls back to defaults")
    func configEmptyDecode() {
        let c = TenVADConfig.decode(from: [:])
        #expect(c.hopSize == 256)
        #expect(c.threshold == 0.5)
        #expect(c.minSpeechDurationMs == 250)
        #expect(c.minSilenceDurationMs == 100)
        #expect(c.speechPadMs == 30)
    }

    @Test("TenVADConfig — full config.json is decoded correctly")
    func configFullDecode() {
        let raw: [String: Any] = [
            "model_type": "ten_vad",
            "hop_size": 256,
            "threshold": 0.6,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 150,
            "speech_pad_ms": 50,
        ]
        let c = TenVADConfig.decode(from: raw)
        #expect(c.hopSize == 256)
        #expect(c.threshold == 0.6)
        #expect(c.minSpeechDurationMs == 300)
        #expect(c.minSilenceDurationMs == 150)
        #expect(c.speechPadMs == 50)
    }

    @Test("TenVADConfig — partial config keeps per-field defaults")
    func configPartialDecode() {
        let raw: [String: Any] = ["threshold": 0.7]
        let c = TenVADConfig.decode(from: raw)
        #expect(c.threshold == 0.7)
        // Untouched fields keep the published default.
        #expect(c.hopSize == 256)
        #expect(c.minSpeechDurationMs == 250)
    }

    // ─── TenVADModel.probsToSegments ────────────────────────────────

    @Test("TenVADModel.probsToSegments — speech burst yields one segment")
    func probsToSegmentsOneBurst() {
        // 60 frames @ 256 samples / 16 kHz: silence, then a ~1.5s burst
        // of speech above threshold, then silence again.
        var probs = [Float](repeating: 0.05, count: 60)
        for i in 15..<50 { probs[i] = 0.85 }
        let segments = TenVADModel.probsToSegments(
            probs, audioLen: 60 * 256, sampleRate: 16000,
            hopSize: 256, threshold: 0.5,
            minSpeechDurationMs: 250, minSilenceDurationMs: 100,
            speechPadMs: 30)
        #expect(segments.count == 1)
        if let s = segments.first {
            #expect(s.durationSeconds > 0)
            #expect(s.startSample < s.endSample)
            #expect(s.endSample <= 60 * 256)
        }
    }

    @Test("TenVADModel.probsToSegments — all-silence yields no segments")
    func probsToSegmentsSilence() {
        let probs = [Float](repeating: 0.02, count: 50)
        let segments = TenVADModel.probsToSegments(
            probs, audioLen: 50 * 256, sampleRate: 16000,
            hopSize: 256, threshold: 0.5,
            minSpeechDurationMs: 250, minSilenceDurationMs: 100,
            speechPadMs: 30)
        #expect(segments.isEmpty)
    }

    @Test("TenVADModel.probsToSegments — all-speech yields one segment")
    func probsToSegmentsAllSpeech() {
        let probs = [Float](repeating: 0.9, count: 40)
        let audioLen = 40 * 256
        let segments = TenVADModel.probsToSegments(
            probs, audioLen: audioLen, sampleRate: 16000,
            hopSize: 256, threshold: 0.5,
            minSpeechDurationMs: 250, minSilenceDurationMs: 100,
            speechPadMs: 30)
        #expect(segments.count == 1)
        if let s = segments.first {
            #expect(s.startSample == 0)
            #expect(s.endSample <= audioLen)
        }
    }

    @Test("TenVADModel.probsToSegments — two bursts with sufficient gap yield two segments")
    func probsToSegmentsTwoBursts() {
        // Each burst is 25 frames × 256 samples / 16000 Hz = 400ms,
        // which clears minSpeechDurationMs=250ms. The gap between bursts
        // is 10 frames × 256 / 16000 = 160ms > minSilenceDurationMs=100ms.
        var probs = [Float](repeating: 0.02, count: 100)
        for i in 5..<30 { probs[i] = 0.9 }   // burst 1: 25 frames, 400ms
        for i in 40..<65 { probs[i] = 0.9 }   // burst 2: 25 frames, 400ms
        let segments = TenVADModel.probsToSegments(
            probs, audioLen: 100 * 256, sampleRate: 16000,
            hopSize: 256, threshold: 0.5,
            minSpeechDurationMs: 250, minSilenceDurationMs: 100,
            speechPadMs: 0)
        // Both bursts are 400ms > minSpeechDurationMs=250ms, gap=160ms.
        #expect(segments.count == 2)
    }

    // ─── VADModelRegistry detection ─────────────────────────────────

    @Test("AudioModelKind — tenVAD model_type set contains expected strings")
    func audioModelKindTenVAD() {
        let types = AudioModelKind.tenVAD.modelTypes
        #expect(types.contains("ten_vad"))
        #expect(types.contains("ten-vad"))
        #expect(types.contains("tenvad"))
    }

    @Test("VADModelRegistry.detectKind — recognizes ten_vad model_type")
    func registryDetectKindTenVAD() throws {
        let dir = try writeTempConfig(["model_type": "ten_vad"],
                                      named: "ten-vad-checkpoint")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .tenVAD)
    }

    @Test("VADModelRegistry.detectKind — recognizes ten-vad model_type")
    func registryDetectKindTenVADDash() throws {
        let dir = try writeTempConfig(["model_type": "ten-vad"],
                                      named: "ten-vad-checkpoint-2")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .tenVAD)
    }

    @Test("VADModelRegistry.detectKind — falls back to directory name for ten-vad")
    func registryDetectKindTenVADByName() throws {
        // No config.json — detection falls back to directory name.
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("models--TEN-framework--ten-vad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .tenVAD)
    }

    // ─── TenVADModel.frameworkBinary ────────────────────────────────

    @Test("TenVADModel.frameworkBinary — returns nil for an empty snapshot")
    func frameworkBinaryMissing() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-framework-\(UUID().uuidString)")
        // Don't create the directory — frameworkBinary should return nil.
        #expect(TenVADModel.frameworkBinary(in: dir) == nil)
    }

    @Test("TenVADModel.frameworkBinary — finds canonical framework path")
    func frameworkBinaryFindsCanonical() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("fake-ten-vad-\(UUID().uuidString)")
        let binaryDir = dir.appendingPathComponent("lib/macOS/ten_vad.framework/Versions/A")
        try FileManager.default.createDirectory(at: binaryDir, withIntermediateDirectories: true)
        let binaryURL = binaryDir.appendingPathComponent("ten_vad")
        // Create a placeholder file.
        try Data().write(to: binaryURL)
        defer { try? FileManager.default.removeItem(at: dir) }
        let found = TenVADModel.frameworkBinary(in: dir)
        #expect(found == binaryURL)
    }

    @Test("TenVADModel.frameworkBinary — finds flat fallback binary")
    func frameworkBinaryFindsFlatFallback() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("fake-ten-vad-flat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let binaryURL = dir.appendingPathComponent("ten_vad")
        try Data().write(to: binaryURL)
        defer { try? FileManager.default.removeItem(at: dir) }
        let found = TenVADModel.frameworkBinary(in: dir)
        #expect(found == binaryURL)
    }

    // ─── TenVADModel.loadFromDirectory — missing framework ──────────

    @Test("TenVADModel.loadFromDirectory — throws when framework binary is absent")
    func loadFromDirectoryMissingFramework() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("empty-ten-vad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: TenVADError.self) {
            _ = try TenVADModel.loadFromDirectory(dir)
        }
    }

    // ─── Helpers ────────────────────────────────────────────────────

    /// Write a minimal config.json into a fresh temp directory.
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
