import Foundation
import Testing
@testable import FFAI

// Unit tests for FireRedVAD — config decoding, CMVN parameters, the
// post-processing pipeline, weight-key conventions, and registry detection.
// Run offline, no checkpoint required.
//
// The full checkpoint-load + forward path is covered by
// FireRedVADIntegrationTests (ModelTests), which requires the
// `FireRedTeam/FireRedVAD` snapshot and is disabled pending a
// safetensors-format mlx-community conversion.
@Suite("FireRedVAD")
struct FireRedVADTests {

    // ─── FireRedVADConfig decoding ────────────────────────────────────

    @Test("FireRedVADConfig — published defaults match upstream FireRedVadConfig")
    func configDefaults() {
        let c = FireRedVADConfig()
        // DFSMN architecture (verified from VAD/model.pth.tar args namespace).
        #expect(c.numBlocks == 8)
        #expect(c.numDnnLayers == 1)
        #expect(c.hiddenSize == 256)
        #expect(c.projSize == 128)
        #expect(c.lookbackOrder == 20)
        #expect(c.lookbackStride == 1)
        #expect(c.lookaheadOrder == 20)
        #expect(c.lookaheadStride == 1)
        #expect(c.idim == 80)
        #expect(c.odim == 1)
        // Audio front-end.
        #expect(c.numMelBins == 80)
        #expect(c.frameLengthSamples == 400)  // 25 ms at 16 kHz
        #expect(c.frameShiftSamples == 160)   // 10 ms at 16 kHz
        // Post-processing defaults (from FireRedVadConfig dataclass).
        #expect(c.smoothWindowSize == 5)
        #expect(c.speechThreshold == 0.4)
        #expect(c.minSpeechFrame == 20)
        #expect(c.maxSpeechFrame == 2000)
        #expect(c.minSilenceFrame == 20)
        #expect(c.mergeSilenceFrame == 0)
        #expect(c.extendSpeechFrame == 0)
    }

    @Test("FireRedVADConfig — empty config falls back to defaults")
    func configEmptyDecode() {
        let c = FireRedVADConfig.decode(from: [:])
        #expect(c.hiddenSize == 256)
        #expect(c.projSize == 128)
        #expect(c.speechThreshold == 0.4)
        #expect(c.minSpeechFrame == 20)
    }

    @Test("FireRedVADConfig — full config.json is decoded correctly")
    func configFullDecode() {
        let raw: [String: Any] = [
            "model_type": "firered_vad",
            "num_blocks": 8,
            "hidden_size": 256,
            "proj_size": 128,
            "lookback_order": 20,
            "lookahead_order": 20,
            "smooth_window_size": 3,
            "speech_threshold": 0.5,
            "min_speech_frame": 15,
            "max_speech_frame": 1000,
            "min_silence_frame": 10,
        ]
        let c = FireRedVADConfig.decode(from: raw)
        #expect(c.numBlocks == 8)
        #expect(c.hiddenSize == 256)
        #expect(c.projSize == 128)
        #expect(c.smoothWindowSize == 3)
        #expect(c.speechThreshold == 0.5)
        #expect(c.minSpeechFrame == 15)
        #expect(c.maxSpeechFrame == 1000)
        #expect(c.minSilenceFrame == 10)
    }

    @Test("FireRedVADConfig — partial config keeps per-field defaults")
    func configPartialDecode() {
        let raw: [String: Any] = ["speech_threshold": 0.6]
        let c = FireRedVADConfig.decode(from: raw)
        #expect(c.speechThreshold == 0.6)
        // Untouched fields keep the published default.
        #expect(c.numBlocks == 8)
        #expect(c.minSpeechFrame == 20)
    }

    // ─── CMVN parameters ─────────────────────────────────────────────

    @Test("FireRedCMVN.default — has 80 means and 80 inv-stds")
    func cmvnDefaultDimension() {
        let cmvn = FireRedCMVN.default
        #expect(cmvn.dim == 80)
        #expect(cmvn.mean.count == 80)
        #expect(cmvn.invStd.count == 80)
    }

    @Test("FireRedCMVN.default — mean values are in the expected Kaldi fbank range")
    func cmvnDefaultMeanRange() {
        // The CMVN means were derived from the `cmvn.ark` shipped in
        // `FireRedTeam/FireRedVAD/VAD/`. Kaldi log-fbank values for 16 kHz
        // speech across 80 bins typically lie in [9, 17] after CMVN stats
        // computed over a large corpus.
        let mean = FireRedCMVN.defaultMean
        for (i, m) in mean.enumerated() {
            #expect(m > 8 && m < 18,
                    "mean[\(i)] = \(m) — expected Kaldi log-fbank range (8, 18)")
        }
    }

    @Test("FireRedCMVN.default — inv-std values are positive")
    func cmvnDefaultInvStdPositive() {
        for (i, s) in FireRedCMVN.defaultInvStd.enumerated() {
            #expect(s > 0, "invStd[\(i)] = \(s) must be positive")
        }
    }

    @Test("FireRedCMVN — apply normalises a flat feature row to near-zero mean")
    func cmvnApply() {
        let cmvn = FireRedCMVN.default
        // Use the mean itself as input — after CMVN this row should be
        // all zeros (mean - mean) * invStd = 0.
        var features: [[Float]] = [cmvn.mean]
        cmvn.apply(&features)
        for v in features[0] {
            #expect(abs(v) < 1e-4, "CMVN of mean row should be ~0, got \(v)")
        }
    }

    // ─── Post-processing ─────────────────────────────────────────────

    @Test("FireRedVADPostprocessor.smooth — windowSize=1 is identity")
    func smoothIdentity() {
        let probs: [Float] = [0.1, 0.9, 0.3, 0.7, 0.5]
        let smoothed = FireRedVADPostprocessor.smooth(probs, windowSize: 1)
        #expect(smoothed == probs)
    }

    @Test("FireRedVADPostprocessor.smooth — causal average across window")
    func smoothAverage() {
        let probs: [Float] = [0.0, 0.0, 1.0, 1.0, 1.0]
        let smoothed = FireRedVADPostprocessor.smooth(probs, windowSize: 3)
        // Frame 0: [0.0] → 0.0
        // Frame 1: [0.0, 0.0] → 0.0
        // Frame 2: [0.0, 0.0, 1.0] → 1/3
        // Frame 3: [0.0, 1.0, 1.0] → 2/3
        // Frame 4: [1.0, 1.0, 1.0] → 1.0
        #expect(abs(smoothed[0] - 0.0) < 1e-4)
        #expect(abs(smoothed[1] - 0.0) < 1e-4)
        #expect(abs(smoothed[2] - 1.0/3.0) < 1e-4)
        #expect(abs(smoothed[3] - 2.0/3.0) < 1e-4)
        #expect(abs(smoothed[4] - 1.0) < 1e-4)
    }

    @Test("FireRedVADPostprocessor.stateMachineDecisions — burst above threshold triggers speech")
    func stateMachineSpeechBurst() {
        // 60 frames: 5 silence, 30 speech (> minSpeechFrame=20), 25 silence.
        // The 30-frame burst exceeds minSpeechFrame so it trips SPEECH state.
        var binary = [Int](repeating: 0, count: 60)
        for i in 5..<35 { binary[i] = 1 }
        let decisions = FireRedVADPostprocessor.stateMachineDecisions(
            binary, minSpeechFrame: 20, minSilenceFrame: 20)
        // At least some frames should be marked speech.
        let speechCount = decisions.filter { $0 == 1 }.count
        #expect(speechCount > 0)
    }

    @Test("FireRedVADPostprocessor.stateMachineDecisions — short burst below minSpeechFrame is rejected")
    func stateMachineShortBurstRejected() {
        // 20 frames: 5 silence, 5 speech, 10 silence.
        var binary = [Int](repeating: 0, count: 20)
        for i in 5..<10 { binary[i] = 1 }
        let decisions = FireRedVADPostprocessor.stateMachineDecisions(
            binary, minSpeechFrame: 20, minSilenceFrame: 20)
        // 5-frame burst < minSpeechFrame=20 → no speech.
        let speechCount = decisions.filter { $0 == 1 }.count
        #expect(speechCount == 0)
    }

    @Test("FireRedVADPostprocessor.process — all-silence yields no segments")
    func processAllSilence() {
        let probs = [Float](repeating: 0.05, count: 100)
        let config = FireRedVADConfig()
        let segments = FireRedVADPostprocessor.process(
            probs: probs, config: config,
            audioDurationSeconds: 1.0, sampleRate: 16000)
        #expect(segments.isEmpty)
    }

    @Test("FireRedVADPostprocessor.process — long speech burst yields one segment")
    func processOneBurst() {
        // 400 frames @ 10ms = 4s audio. Frames 50..350 (3s) are speech —
        // well above minSpeechFrame=20, so the burst clears the state machine.
        var probs = [Float](repeating: 0.05, count: 400)
        for i in 50..<350 { probs[i] = 0.95 }
        let config = FireRedVADConfig()
        let segments = FireRedVADPostprocessor.process(
            probs: probs, config: config,
            audioDurationSeconds: 4.0, sampleRate: 16000)
        #expect(segments.count == 1)
        if let s = segments.first {
            #expect(s.startSample < s.endSample)
            #expect(s.durationSeconds > 0)
        }
    }

    @Test("FireRedVADPostprocessor.process — all-speech yields one segment spanning the clip")
    func processAllSpeech() {
        let probs = [Float](repeating: 0.95, count: 200)
        let config = FireRedVADConfig()
        let segments = FireRedVADPostprocessor.process(
            probs: probs, config: config,
            audioDurationSeconds: 2.0, sampleRate: 16000)
        #expect(segments.count == 1)
    }

    // ─── VADModelRegistry detection ──────────────────────────────────

    @Test("AudioModelKind — fireRedVAD modelTypes contains expected strings")
    func audioModelKindFireRedVAD() {
        let types = AudioModelKind.fireRedVAD.modelTypes
        #expect(types.contains("firered_vad"))
        #expect(types.contains("firered-vad"))
        #expect(types.contains("fireredvad"))
    }

    @Test("VADModelRegistry.detectKind — recognizes firered_vad model_type")
    func registryDetectKindFireRedVAD() throws {
        let dir = try writeTempConfig(["model_type": "firered_vad"],
                                      named: "firered-vad-checkpoint")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .fireRedVAD)
    }

    @Test("VADModelRegistry.detectKind — recognizes firered-vad model_type (dash variant)")
    func registryDetectKindFireRedVADDash() throws {
        let dir = try writeTempConfig(["model_type": "firered-vad"],
                                      named: "firered-vad-checkpoint-2")
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .fireRedVAD)
    }

    @Test("VADModelRegistry.detectKind — falls back to directory name for FireRedVAD")
    func registryDetectKindFireRedVADByName() throws {
        // No config.json — detection must fall back to the directory name.
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent(
            "models--FireRedTeam--FireRedVAD-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try VADModelRegistry.detectKind(in: dir) == .fireRedVAD)
    }

    @Test("FireRedVADModel.loadFromDirectory — throws missingWeight when pth.tar is absent")
    func loadFromDirectoryMissingCheckpoint() throws {
        let base = FileManager.default.temporaryDirectory
        let dir = base.appendingPathComponent("empty-fireredvad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: Error.self) {
            _ = try FireRedVADModel.loadFromDirectory(dir)
        }
    }

    // ─── FSMN forward sanity ─────────────────────────────────────────

    @Test("FireRedFSMN — all-zero weights produce identity (residual only)")
    func fsmnZeroWeights() {
        // With all-zero filter weights, the FSMN memory is exactly the
        // residual input — so forward should return the input unchanged.
        let P = 4; let N1 = 3; let N2 = 3; let T = 10
        let zeros = [Float](repeating: 0, count: P * N1)
        let fsmn = FireRedFSMN(lookbackWeight: zeros, lookaheadWeight: zeros,
                               P: P, N1: N1, S1: 1, N2: N2, S2: 1)
        let input = (0..<(T * P)).map { Float($0) * 0.1 }
        let output = fsmn.forward(input, T: T)
        // Output should equal input (zero filters → zero convolution → residual only).
        for i in input.indices {
            #expect(abs(output[i] - input[i]) < 1e-5,
                    "FSMN zero-weight: output[\(i)] \(output[i]) != input \(input[i])")
        }
    }

    @Test("FireRedFSMN — output shape matches input shape")
    func fsmnOutputShape() {
        let P = 8; let N1 = 5; let N2 = 5; let T = 20
        let lb = [Float](repeating: 0.01, count: P * N1)
        let la = [Float](repeating: 0.01, count: P * N2)
        let fsmn = FireRedFSMN(lookbackWeight: lb, lookaheadWeight: la,
                               P: P, N1: N1, S1: 1, N2: N2, S2: 1)
        let input = [Float](repeating: 1.0, count: T * P)
        let output = fsmn.forward(input, T: T)
        #expect(output.count == T * P)
    }

    // ─── Helpers ─────────────────────────────────────────────────────

    /// Write a minimal `config.json` into a fresh temp directory.
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
