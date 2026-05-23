// Slow integration test: downloads (or hits cache) the TEN-framework/ten-vad
// checkpoint and runs voice-activity detection on a real speech clip.
//
// TEN-VAD is a native C library distributed as a pre-compiled
// ten_vad.framework (macOS) inside the TEN-framework/ten-vad HuggingFace
// repo. The weights are embedded in the compiled binary (via an embedded
// ONNX model at src/onnx_model/ten-vad.onnx); there is no
// mlx-community safetensors conversion as of 2026-05-22.
//
// DISABLED REASON:
//   The TEN-framework/ten-vad HF repo requires a non-trivial download
//   (~15MB for the macOS framework), and the macOS framework binary
//   shipped by TEN-framework is unsigned / quarantined on first use,
//   which would cause a Gatekeeper dialog in CI environments.
//   Additionally, dlopen of a freshly downloaded framework requires
//   removing the quarantine attribute (`xattr -d com.apple.quarantine`),
//   which is out of scope for an automated test runner.
//
//   To enable this suite:
//   1. Download TEN-framework/ten-vad manually:
//      `python3 -c "from huggingface_hub import snapshot_download; snapshot_download('TEN-framework/ten-vad')"`
//   2. Remove quarantine from the framework:
//      `xattr -dr com.apple.quarantine ~/.cache/huggingface/hub/models--TEN-framework--ten-vad/`
//   3. Remove the `.disabled` trait from the @Suite below.
//
// This suite is intentionally assertive: a load failure FAILS rather
// than silently skipping. The contract is "the TenVAD checkpoint loads
// and actually detects speech in AudioFixtures.clean001Waveform()".

import Foundation
import Testing
@testable import FFAI

// DISABLED: ten_vad.framework requires quarantine removal before dlopen
// can succeed in an automated environment. See header comment for
// re-enable steps.
@Suite("TenVAD integration", .disabled("ten_vad.framework is unsigned — run xattr -dr com.apple.quarantine on the snapshot before enabling"), .serialized)
struct TenVADIntegrationTests {

    /// The TEN-VAD HuggingFace repo id (ONNX + native macOS framework).
    ///
    /// No mlx-community safetensors conversion exists as of 2026-05-22.
    /// If one is published (e.g. `mlx-community/TEN-VAD`), update this
    /// constant and remove the `.disabled` trait above.
    static let repoId = "TEN-framework/ten-vad"

    /// Load the TEN-VAD model through `VADModelRegistry`, holding the
    /// global model-load lock.
    private func loadTenVAD() async throws -> TenVADModel {
        let loaded = try await ModelLoadLock.shared.loadSerially {
            try await VADModelRegistry.fromPretrained(Self.repoId)
        }
        guard case .tenVAD(let model) = loaded else {
            Issue.record("VADModelRegistry resolved \(Self.repoId) to \(loaded.kind), expected .tenVAD")
            throw TenVADError.createFailed
        }
        return model
    }

    @Test("load + detect — produces a finite probability stream on a real speech clip")
    func loadAndDetect() async throws {
        let model = try await loadTenVAD()

        // Config geometry from the published defaults.
        #expect(model.config.hopSize == 256)
        #expect(model.config.threshold >= 0 && model.config.threshold <= 1)

        let waveform = try AudioFixtures.clean001Waveform()
        let output = try model.detect(audio: waveform, sampleRate: 16000)

        // Probability stream must be finite and in [0, 1].
        #expect(output.isWellFormed)
        #expect(!output.probabilities.isEmpty)
        #expect(output.frameStrideSamples == model.config.hopSize)
        #expect(output.sampleRate == 16000)

        // Frame count: ceil(waveformLen / hopSize).
        let expectedFrames = (waveform.count + model.config.hopSize - 1)
            / model.config.hopSize
        #expect(output.probabilities.count == expectedFrames)

        // The "Sure, I can help you with that." clip (clean_001.wav,
        // ~1.85 s of speech) must produce at least one detected segment
        // and a max probability above a generous floor.
        let maxProb = output.probabilities.max() ?? 0
        #expect(maxProb > 0.3,
                "max speech probability \(maxProb) — TenVAD should detect speech in clean_001.wav")
        #expect(!output.segments.isEmpty)

        // Total detected speech must be positive and not span the whole clip.
        let clipDuration = Double(waveform.count) / 16000.0
        #expect(output.totalSpeechSeconds > 0)
        #expect(output.totalSpeechSeconds < clipDuration)

        print("TenVAD detect: \(output.probabilities.count) frames, "
              + "maxProb=\(maxProb), segments=\(output.segments.count), "
              + "totalSpeech=\(output.totalSpeechSeconds)s")
    }

    @Test("detect — empty clip yields an empty, well-formed result")
    func emptyClip() async throws {
        let model = try await loadTenVAD()
        let output = try model.detect(audio: [], sampleRate: 16000)
        #expect(output.probabilities.isEmpty)
        #expect(output.segments.isEmpty)
        #expect(output.isWellFormed)
    }

    @Test("detect — unsupported sample rate is rejected")
    func unsupportedSampleRate() async throws {
        let model = try await loadTenVAD()
        #expect(throws: TenVADError.self) {
            _ = try model.detect(audio: [0, 0, 0], sampleRate: 44100)
        }
    }

    @Test("registry dispatch — resolves TEN-framework/ten-vad to .tenVAD kind")
    func registryDispatch() async throws {
        let loaded = try await ModelLoadLock.shared.loadSerially {
            try await VADModelRegistry.fromPretrained(Self.repoId)
        }
        #expect(loaded.kind == .tenVAD)
    }
}
