// SmartTurnTests — unit tests for the SmartTurn endpoint / turn
// detection family: config decoding, registry detection, and error
// surface. All tests run offline, no checkpoint required.

import Foundation
import Testing
@testable import FFAI

@Suite("SmartTurn")
struct SmartTurnTests {

    // ─── SmartTurnConfig ─────────────────────────────────────────────────

    @Test("SmartTurnConfig — published defaults match upstream")
    func configDefaults() {
        let c = SmartTurnConfig()
        // Encoder hyper-parameters (smart_turn_v3 defaults).
        #expect(c.numMelBins == 80)
        #expect(c.maxSourcePositions == 400)
        #expect(c.dModel == 384)
        #expect(c.encoderAttentionHeads == 6)
        #expect(c.encoderLayers == 4)
        #expect(c.encoderFfnDim == 1536)
        #expect(c.kProjBias == false)
        // Processor hyper-parameters.
        #expect(c.samplingRate == 16_000)
        #expect(c.maxAudioSeconds == 8)
        #expect(c.nFft == 400)
        #expect(c.hopLength == 160)
        #expect(c.normalizeAudio == true)
        #expect(c.threshold == 0.5)
    }

    @Test("SmartTurnConfig.decode — empty raw falls back to defaults")
    func configEmptyDecode() {
        let c = SmartTurnConfig.decode(from: [:])
        #expect(c.dModel == 384)
        #expect(c.encoderLayers == 4)
        #expect(c.samplingRate == 16_000)
        #expect(c.threshold == 0.5)
    }

    @Test("SmartTurnConfig.decode — top-level keys override defaults")
    func configTopLevelOverrides() {
        let raw: [String: Any] = [
            "model_type": "smart_turn_v3",
            "d_model": 512,
            "encoder_layers": 6,
            "threshold": 0.6,
        ]
        let c = SmartTurnConfig.decode(from: raw)
        #expect(c.dModel == 512)
        #expect(c.encoderLayers == 6)
        #expect(abs(c.threshold - 0.6) < 1e-5)
    }

    @Test("SmartTurnConfig.decode — nested encoder_config and processor_config decode")
    func configNestedDecode() {
        let raw: [String: Any] = [
            "encoder_config": [
                "d_model": 768,
                "encoder_attention_heads": 12,
                "encoder_layers": 8,
            ] as [String: Any],
            "processor_config": [
                "sampling_rate": 24_000,
                "threshold": 0.7,
            ] as [String: Any],
        ]
        let c = SmartTurnConfig.decode(from: raw)
        #expect(c.dModel == 768)
        #expect(c.encoderAttentionHeads == 12)
        #expect(c.encoderLayers == 8)
        #expect(c.samplingRate == 24_000)
        #expect(abs(c.threshold - 0.7) < 1e-5)
    }

    // ─── Error stringification ───────────────────────────────────────────

    @Test("SmartTurnError.description — missingWeight carries the key")
    func errorDescriptionMissingWeight() {
        let err = SmartTurnError.missingWeight("encoder.conv1.weight")
        #expect(err.description.contains("SmartTurn"))
        #expect(err.description.contains("conv1.weight"))
    }

    @Test("SmartTurnError.description — invalidAudio carries the reason")
    func errorDescriptionInvalidAudio() {
        let err = SmartTurnError.invalidAudio("empty waveform")
        #expect(err.description.contains("SmartTurn"))
        #expect(err.description.contains("empty"))
    }

    // ─── VADModelRegistry detection ─────────────────────────────────────

    @Test("AudioModelKind.smartTurn — modelTypes contains expected strings")
    func audioModelKindSmartTurn() {
        let types = AudioModelKind.smartTurn.modelTypes
        #expect(types.contains("smart_turn"))
        #expect(types.contains("smart_turn_v3"))
        #expect(types.contains("smart-turn"))
    }

    @Test("VADModelRegistry.detectKind — recognizes smart_turn model_type")
    func registryDetectKindSmartTurn() throws {
        let dir = try writeTempConfig(["model_type": "smart_turn"],
                                      named: "smart-turn-checkpoint")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .smartTurn)
    }

    @Test("VADModelRegistry.detectKind — recognizes smart_turn_v3 model_type")
    func registryDetectKindSmartTurnV3() throws {
        let dir = try writeTempConfig(["model_type": "smart_turn_v3"],
                                      named: "smart-turn-v3-checkpoint")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .smartTurn)
    }

    @Test("VADModelRegistry.detectKind — falls back to directory name for smart-turn")
    func registryDetectKindSmartTurnByName() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent(
            "models--pipecat-ai--smart-turn-v3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .smartTurn)
    }

    // ─── Loader rejection paths ─────────────────────────────────────────

    @Test("SmartTurnModel.loadFromDirectory — throws when snapshot is empty")
    func loadFromDirectoryEmpty() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("empty-smart-turn-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: Error.self) {
            _ = try SmartTurnModel.loadFromDirectory(dir)
        }
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
