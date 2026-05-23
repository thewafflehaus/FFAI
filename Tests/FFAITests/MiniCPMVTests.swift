// MiniCPM-V 4.6 — pure-logic unit coverage. The GPU-bound encoder
// machinery (vit_merger + merger window cross-attention) is exercised
// by `Tests/ModelTests/MiniCPMVIntegrationTests.swift`.

import Foundation
import Metal
import Testing

@testable import FFAI

// ─── Family registration ─────────────────────────────────────────────

@Suite("MiniCPM-V 4.6 — family registration")
struct MiniCPMVRegistrationTests {

    @Test("MiniCPMV4_6 owns the minicpmv4_6 model_type + ConditionalGeneration arch")
    func familyStrings() {
        #expect(MiniCPMV4_6.modelTypes.contains("minicpmv4_6"))
        #expect(MiniCPMV4_6.architectures.contains(
            "MiniCPMV4_6ForConditionalGeneration"))
    }

    @Test("VL registry recognises MiniCPMV4_6ForConditionalGeneration")
    func vlRegistryRecognition() {
        #expect(VisionLanguageArchitectures.architectures.contains(
            "MiniCPMV4_6ForConditionalGeneration"))
    }

    @Test("default image_token_id matches the checkpoint chat template")
    func defaultImageTokenId() {
        #expect(MiniCPMV4_6.defaultImageTokenId == 248056)
    }
}

// ─── Position-embedding bilinear interpolation ───────────────────────

@Suite("MiniCPM-V 4.6 — position-embedding interpolation")
struct MiniCPMVPosEmbInterpTests {

    /// Identity case: target side equals stored side → returned tensor
    /// is the original (no work, no precision drift).
    @Test("storedSide == targetSide is a pass-through")
    func identityPath() {
        let device = Device.shared
        let hidden = 4
        let side = 3
        let vals: [Float] = (0..<(side * side * hidden)).map { Float($0) }
        let src = Tensor.empty(shape: [side * side, hidden], dtype: .f32,
                               device: device)
        src.copyIn(from: vals)
        let out = interpolatePositionEmbedding(
            src, storedSide: side, targetSide: side, hidden: hidden,
            device: device)
        #expect(out.toArray(as: Float.self) == vals)
    }

    /// Sanity check the bilinear resample at a coarse grid: every output
    /// position should equal `f(srcX, srcY)` for a linear function
    /// `f(x, y) = ax + by + c` over (srcX, srcY) sample coords. Bilinear
    /// is exact for linear functions, so the resample must be too.
    @Test("bilinear resample is exact on a linear field")
    func bilinearExactOnLinearField() {
        let device = Device.shared
        let storedSide = 8, targetSide = 4, hidden = 1
        // f(srcX, srcY) = 2·srcX + 3·srcY + 1.
        var src = [Float](repeating: 0, count: storedSide * storedSide)
        for y in 0..<storedSide {
            for x in 0..<storedSide {
                src[y * storedSide + x] = 2 * Float(x) + 3 * Float(y) + 1
            }
        }
        let srcT = Tensor.empty(shape: [storedSide * storedSide, hidden],
                                dtype: .f32, device: device)
        srcT.copyIn(from: src)

        let out = interpolatePositionEmbedding(
            srcT, storedSide: storedSide, targetSide: targetSide,
            hidden: hidden, device: device)
        let got = out.toArray(as: Float.self)

        // Same half-pixel-centered sampling the impl uses.
        let scale = Float(storedSide) / Float(targetSide)
        for ty in 0..<targetSide {
            let srcY = (Float(ty) + 0.5) * scale - 0.5
            for tx in 0..<targetSide {
                let srcX = (Float(tx) + 0.5) * scale - 0.5
                let expected = 2 * srcX + 3 * srcY + 1
                let actual = got[ty * targetSide + tx]
                #expect(abs(actual - expected) < 1e-4,
                        "(ty=\(ty), tx=\(tx)): expected \(expected), got \(actual)")
            }
        }
    }

    /// Constant field stays constant — independent of grid sizes.
    @Test("constant field resamples to the same constant")
    func constantFieldPreserved() {
        let device = Device.shared
        let hidden = 3
        let src = Tensor.empty(shape: [7 * 7, hidden], dtype: .f32,
                               device: device)
        src.copyIn(from: [Float](repeating: 1.5, count: 7 * 7 * hidden))
        let out = interpolatePositionEmbedding(
            src, storedSide: 7, targetSide: 3, hidden: hidden,
            device: device)
        for v in out.toArray(as: Float.self) {
            #expect(abs(v - 1.5) < 1e-5)
        }
    }
}
