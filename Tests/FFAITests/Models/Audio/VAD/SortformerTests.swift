import Foundation
import Testing
@testable import FFAI

// Unit tests for the Sortformer speaker-diarization family — config
// decoding, published architecture defaults, segment post-processing,
// and the `VADModelRegistry` dispatch. Full checkpoint-load + forward is
// covered by the ModelTests integration suite; these run purely on
// in-process data so they are fast + offline.
//
// Note: `SortformerModel.remapAndTranspose` requires `[String: Tensor]`
// (GPU-resident SafeTensors), so it cannot be exercised here without a
// real GPU device; its correctness is verified by the integration suite.
@Suite("Sortformer")
struct SortformerTests {

    // ─── SortformerFCConfig ──────────────────────────────────────────

    @Test("SortformerFCConfig — published diar_streaming_sortformer_4spk defaults")
    func fcConfigDefaults() {
        let c = SortformerFCConfig(from: [:])
        #expect(c.hiddenSize == 512)
        // Default numLayers from the checkpoint is 18 (the default fallback value).
        #expect(c.numHeads == 8)
        #expect(c.intermediateSize == 2048)
        #expect(c.numMelBins == 80)
        #expect(c.convKernelSize == 9)
        #expect(c.subsamplingFactor == 8)
        #expect(c.subsamplingConvChannels == 256)
        #expect(c.subsamplingConvKernelSize == 3)
        #expect(c.subsamplingConvStride == 2)
        // headDim should divide evenly — required by self-attention.
        #expect(c.hiddenSize % c.numHeads == 0)
    }

    @Test("SortformerFCConfig — reads custom values from config dict")
    func fcConfigCustomDecode() {
        let raw: [String: Any] = [
            "hidden_size": 256,
            "num_hidden_layers": 12,
            "num_attention_heads": 4,
            "intermediate_size": 1024,
            "num_mel_bins": 128,
            "conv_kernel_size": 31,
            "subsampling_factor": 4,
            "subsampling_conv_channels": 128,
            "subsampling_conv_kernel_size": 5,
            "subsampling_conv_stride": 1,
        ]
        let c = SortformerFCConfig(from: raw)
        #expect(c.hiddenSize == 256)
        #expect(c.numLayers == 12)
        #expect(c.numHeads == 4)
        #expect(c.intermediateSize == 1024)
        #expect(c.numMelBins == 128)
        #expect(c.convKernelSize == 31)
        #expect(c.subsamplingFactor == 4)
        #expect(c.subsamplingConvChannels == 128)
        #expect(c.subsamplingConvKernelSize == 5)
        #expect(c.subsamplingConvStride == 1)
    }

    // ─── SortformerTFConfig ──────────────────────────────────────────

    @Test("SortformerTFConfig — published tf_encoder defaults")
    func tfConfigDefaults() {
        let c = SortformerTFConfig(from: [:])
        #expect(c.dModel == 192)
        #expect(c.numLayers == 18)
        #expect(c.numHeads == 8)
        #expect(c.ffnDim == 768)
        #expect(c.layerNormEps == 1e-5)
        #expect(c.maxPositions == 1500)
        #expect(c.kProjBias == false)
        // headDim must divide dModel.
        #expect(c.dModel % c.numHeads == 0)
    }

    @Test("SortformerTFConfig — reads nested tf_encoder_config block")
    func tfConfigNestedDecode() {
        let raw: [String: Any] = [
            "d_model": 384,
            "encoder_layers": 6,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 1536,
            "layer_norm_eps": 1e-6,
            "max_source_positions": 1000,
            "k_proj_bias": true,
        ]
        let c = SortformerTFConfig(from: raw)
        #expect(c.dModel == 384)
        #expect(c.numLayers == 6)
        #expect(c.numHeads == 4)
        #expect(c.ffnDim == 1536)
        #expect(abs(c.layerNormEps - 1e-6) < 1e-10)
        #expect(c.maxPositions == 1000)
        #expect(c.kProjBias == true)
    }

    // ─── SortformerModulesConfig ─────────────────────────────────────

    @Test("SortformerModulesConfig — published modules defaults")
    func modulesConfigDefaults() {
        let c = SortformerModulesConfig(from: [:])
        #expect(c.numSpeakers == 4)
        #expect(c.fcDModel == 512)
        #expect(c.tfDModel == 192)
        #expect(c.subsamplingFactor == 8)
    }

    @Test("SortformerModulesConfig — reads custom values")
    func modulesConfigDecode() {
        let raw: [String: Any] = [
            "num_speakers": 2,
            "fc_d_model": 256,
            "tf_d_model": 128,
            "subsampling_factor": 4,
        ]
        let c = SortformerModulesConfig(from: raw)
        #expect(c.numSpeakers == 2)
        #expect(c.fcDModel == 256)
        #expect(c.tfDModel == 128)
        #expect(c.subsamplingFactor == 4)
    }

    // ─── SortformerProcessorConfig ───────────────────────────────────

    @Test("SortformerProcessorConfig — published processor defaults")
    func processorConfigDefaults() {
        let c = SortformerProcessorConfig(from: [:])
        #expect(c.featureSize == 80)
        #expect(c.sampleRate == 16000)
        #expect(c.hopLength == 160)
        #expect(c.nFft == 512)
        #expect(c.winLength == 400)
        #expect(abs(c.preemphasis - 0.97) < 1e-5)
    }

    @Test("SortformerProcessorConfig — reads custom processor_config block")
    func processorConfigDecode() {
        let raw: [String: Any] = [
            "feature_size": 128,
            "sampling_rate": 8000,
            "hop_length": 80,
            "n_fft": 256,
            "win_length": 200,
            "preemphasis": 0.95,
        ]
        let c = SortformerProcessorConfig(from: raw)
        #expect(c.featureSize == 128)
        #expect(c.sampleRate == 8000)
        #expect(c.hopLength == 80)
        #expect(c.nFft == 256)
        #expect(c.winLength == 200)
        #expect(abs(c.preemphasis - 0.95) < 1e-5)
    }

    // ─── SortformerConfig ────────────────────────────────────────────

    @Test("SortformerConfig — empty dict yields nested defaults")
    func topLevelConfigEmptyDecode() {
        let c = SortformerConfig(from: [:])
        #expect(c.numSpeakers == 4)
        #expect(c.fcEncoder.hiddenSize == 512)
        #expect(c.tfEncoder.dModel == 192)
        #expect(c.modules.numSpeakers == 4)
        #expect(c.processor.featureSize == 80)
    }

    @Test("SortformerConfig — reads nested config blocks")
    func topLevelConfigNestedDecode() {
        let raw: [String: Any] = [
            "model_type": "sortformer",
            "num_speakers": 2,
            "fc_encoder_config": [
                "hidden_size": 256,
                "num_hidden_layers": 8,
                "num_mel_bins": 128,
            ] as [String: Any],
            "tf_encoder_config": [
                "d_model": 128,
                "encoder_layers": 4,
            ] as [String: Any],
            "modules_config": [
                "num_speakers": 2,
                "fc_d_model": 256,
                "tf_d_model": 128,
            ] as [String: Any],
            "processor_config": [
                "feature_size": 128,
                "sampling_rate": 16000,
            ] as [String: Any],
        ]
        let c = SortformerConfig(from: raw)
        #expect(c.numSpeakers == 2)
        #expect(c.fcEncoder.hiddenSize == 256)
        #expect(c.fcEncoder.numLayers == 8)
        #expect(c.fcEncoder.numMelBins == 128)
        #expect(c.tfEncoder.dModel == 128)
        #expect(c.tfEncoder.numLayers == 4)
        #expect(c.modules.numSpeakers == 2)
        #expect(c.modules.fcDModel == 256)
        #expect(c.modules.tfDModel == 128)
        #expect(c.processor.featureSize == 128)
    }

    // ─── VADModelRegistry — Sortformer dispatch ──────────────────────

    @Test("AudioModelKind.sortformer — model_type set contains expected strings")
    func sortformerModelTypeSet() {
        let kinds = AudioModelKind.sortformer.modelTypes
        #expect(kinds.contains("sortformer"))
        #expect(kinds.contains("diar_sortformer"))
    }

    @Test("VADModelRegistry.detectKind — dispatches sortformer by config model_type")
    func registryDetectSortformerByConfig() throws {
        let dir = try writeTempConfig(["model_type": "sortformer"],
                                     named: "test-sortformer")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .sortformer)
    }

    @Test("VADModelRegistry.detectKind — dispatches diar_sortformer by config model_type")
    func registryDetectDiarSortformerByConfig() throws {
        let dir = try writeTempConfig(["model_type": "diar_sortformer"],
                                     named: "test-diar-sortformer")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .sortformer)
    }

    @Test("VADModelRegistry.detectKind — falls back to sortformer by directory name")
    func registryDetectSortformerByName() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent(
            "sortformer-4spk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir,
                                               withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .sortformer)
    }

    @Test("VADModelRegistry.detectKind — falls back to sortformer by diar in name")
    func registryDetectDiarByName() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent(
            "diar_streaming_4spk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir,
                                               withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .sortformer)
    }

    // ─── SortformerModel.probsToSegments ────────────────────────────

    @Test("SortformerModel.probsToSegments — single-speaker burst → one segment")
    func probsToSegmentsBurst() {
        // Build a [T, numSpeakers] probability matrix: speaker 0 is active
        // during frames 10-24, all others are silent.
        let T = 40
        let numSpeakers = 4
        var probs = [[Float]](repeating: [Float](repeating: 0.05, count: numSpeakers),
                             count: T)
        for t in 10..<25 { probs[t][0] = 0.85 }

        // frameStride = hop * subsamplingFactor = 160 * 8 = 1280 samples @ 16kHz
        let frameDuration = Float(1280) / Float(16000)
        let segments = SortformerModel.probsToSegments(
            probs, frameDuration: frameDuration, threshold: 0.5)

        // Speaker 0 should produce exactly one segment.
        let s0segs = segments.filter { $0.speaker == 0 }
        #expect(s0segs.count == 1)
        if let seg = s0segs.first {
            #expect(seg.durationSeconds > 0)
            #expect(seg.startSeconds < seg.endSeconds)
            // Segment should not span the whole clip.
            let clipDur = Double(frameDuration) * Double(T)
            #expect(seg.durationSeconds < clipDur)
        }

        // Speakers 1-3 should be silent.
        for spk in 1..<numSpeakers {
            #expect(segments.filter { $0.speaker == spk }.isEmpty)
        }
    }

    @Test("SortformerModel.probsToSegments — all-silence yields no segments")
    func probsToSegmentsAllSilence() {
        let T = 20
        let numSpeakers = 4
        let probs = [[Float]](
            repeating: [Float](repeating: 0.02, count: numSpeakers), count: T)
        let segments = SortformerModel.probsToSegments(
            probs, frameDuration: Float(1280) / Float(16000), threshold: 0.5)
        #expect(segments.isEmpty)
    }

    @Test("SortformerModel.probsToSegments — empty input yields no segments")
    func probsToSegmentsEmpty() {
        let segments = SortformerModel.probsToSegments(
            [], frameDuration: Float(1280) / Float(16000), threshold: 0.5)
        #expect(segments.isEmpty)
    }

    @Test("SortformerModel.probsToSegments — multi-speaker overlap")
    func probsToSegmentsMultiSpeaker() {
        // Two speakers active simultaneously in the middle.
        let T = 30
        let numSpeakers = 2
        var probs = [[Float]](repeating: [Float](repeating: 0.02, count: numSpeakers),
                             count: T)
        for t in 5..<20 { probs[t][0] = 0.9 }
        for t in 10..<25 { probs[t][1] = 0.85 }

        let frameDuration = Float(1280) / Float(16000)
        let segments = SortformerModel.probsToSegments(
            probs, frameDuration: frameDuration, threshold: 0.5)

        // Each speaker should produce one segment.
        #expect(segments.filter { $0.speaker == 0 }.count == 1)
        #expect(segments.filter { $0.speaker == 1 }.count == 1)
        // Both speakers should have positive-duration segments.
        for seg in segments {
            #expect(seg.durationSeconds > 0)
        }
    }

    // Write a config.json into a fresh temp directory and return the
    // directory URL.
    private func writeTempConfig(_ config: [String: Any],
                                 named: String) throws -> URL {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("\(named)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir,
                                               withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }
}
