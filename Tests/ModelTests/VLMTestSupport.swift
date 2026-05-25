// Shared helpers for the vision-language model integration tests.
//
// Every VLM family's "image + text" test feeds the same real photograph
// — `Resources/dog.jpeg`, a golden retriever — and asserts the decoded
// caption mentions a dog. Centralizing the image load here keeps the six
// family test files in lock-step and gives a single, precise failure
// message if the fixture goes missing.
//
// The video helpers below decode `Resources/cat.mp4` to ordered RGB
// frames via AVFoundation. They're consumed by:
//   • single-frame VLM tests today (`catVideoFirstFrame()`) — exercises
//     the AVFoundation extraction path on every VLM run
//   • the future multi-frame `encode(frames:)` integration tests
//     (`catVideoFrames(maxFrames:)`) — tracked under #142 (the video-
//     inference port across Qwen2VL/Qwen25VL/Qwen3VL/Qwen3VLMoe/
//     MiniCPMV/SmolVLM2/QwenOmni; FFAI's vision towers currently take
//     one image and replicate it `temporal_patch_size` times)

import AVFoundation
import CoreGraphics
import Foundation
import Testing
@testable import FFAI

enum VLMTestSupport {

    /// Absolute path to a file under `Tests/ModelTests/Resources/`.
    ///
    /// `Tests/ModelTests/` has no SwiftPM resource bundle, so the fixture
    /// is resolved relative to this source file's location at compile
    /// time via `#filePath` — robust to the working directory the test
    /// runner happens to use.
    static func resourceURL(_ name: String,
                            file: StaticString = #filePath) -> URL {
        URL(fileURLWithPath: "\(file)")
            .deletingLastPathComponent()
            .appendingPathComponent("Resources")
            .appendingPathComponent(name)
    }

    /// Load the shared golden-retriever test photo as an `RGBImage`.
    /// Fails the calling test (rather than skipping) if the fixture is
    /// missing or undecodable — the image is checked into the repo, so a
    /// miss is a real breakage.
    static func dogImage() throws -> RGBImage {
        let url = resourceURL("dog.jpeg")
        do {
            return try RGBImage.load(contentsOf: url)
        } catch {
            let message = "VLM tests: failed to load dog.jpeg fixture: \(error)"
            Issue.record(Comment(rawValue: message))
            throw error
        }
    }

    /// Load the dog fixture, resize to `targetSize × targetSize`, and
    /// return a planar **CHW** float array — the layout PaliGemma /
    /// SigLIP-style towers consume directly. No mean/std normalization is
    /// applied here (each caller picks its own); pixels are in `[0,1]`.
    ///
    /// `count = 3 * targetSize * targetSize`. CPU-only — the dog image is
    /// small, this helper is dwarfed by the rest of the vision encoder
    /// cost, so a simple bilinear resize is fine.
    static func dogImageCHW(targetSize: Int) throws -> [Float] {
        let img = try dogImage()
        let resized = ImagePreprocessing.resize(
            img, targetW: targetSize, targetH: targetSize)
        var planar = [Float](repeating: 0, count: 3 * targetSize * targetSize)
        let plane = targetSize * targetSize
        for y in 0..<targetSize {
            for x in 0..<targetSize {
                let srcBase = (y * targetSize + x) * 3
                let dstBase = y * targetSize + x
                planar[0 * plane + dstBase] = resized.pixels[srcBase]
                planar[1 * plane + dstBase] = resized.pixels[srcBase + 1]
                planar[2 * plane + dstBase] = resized.pixels[srcBase + 2]
            }
        }
        return planar
    }

    /// Same as `dogImageCHW(targetSize:)` but applies per-channel
    /// normalization in the same pass. Use this when the family expects
    /// CLIP-mean (`[0.48145466, 0.4578275, 0.40821073]`) /
    /// CLIP-std (`[0.26862954, 0.26130258, 0.27577711]`) or its own
    /// ImageNormalization preset.
    static func dogImageCHWNormalized(
        targetSize: Int, normalization: ImageNormalization
    ) throws -> [Float] {
        var pixels = try dogImageCHW(targetSize: targetSize)
        let plane = targetSize * targetSize
        let means = [normalization.mean.0, normalization.mean.1, normalization.mean.2]
        let stds  = [normalization.std.0,  normalization.std.1,  normalization.std.2]
        for c in 0..<3 {
            let m = means[c], s = stds[c]
            for i in (c * plane)..<((c + 1) * plane) {
                pixels[i] = (pixels[i] - m) / s
            }
        }
        return pixels
    }

    // MARK: - Video fixtures (cat.mp4)

    /// Decode the first frame of `Resources/cat.mp4` as an `RGBImage`.
    /// Used by every VLM today (one frame fed through the existing
    /// single-image `encode(image:)` path); will be superseded by
    /// `catVideoFrames(...)` once the multi-frame video-inference port
    /// (task #142) lands per-model.
    static func catVideoFirstFrame() throws -> RGBImage {
        let frames = try catVideoFrames(maxFrames: 1)
        guard let first = frames.first else {
            throw VLMTestSupportError.videoDecodeFailed(
                "cat.mp4 decoded to zero frames")
        }
        return first
    }

    /// Decode up to `maxFrames` evenly-spaced frames from
    /// `Resources/cat.mp4` as `[RGBImage]` in display order. AVFoundation
    /// reads cleanly from h.264 / HEVC / ProRes mp4s on Apple Silicon
    /// without any extra framework dependency. Asks for `maxFrames`
    /// timestamps spaced over the asset's duration; clips to actual
    /// frame count if the clip is shorter than requested.
    ///
    /// Fails the calling test if AVFoundation can't decode the asset —
    /// the file is checked in, so a miss is a real breakage (e.g. the
    /// fixture got truncated / replaced).
    static func catVideoFrames(maxFrames: Int) throws -> [RGBImage] {
        precondition(maxFrames > 0, "maxFrames must be positive")
        let url = resourceURL("cat.mp4")
        let asset = AVURLAsset(url: url)
        let durationSecs = CMTimeGetSeconds(asset.duration)
        guard durationSecs.isFinite, durationSecs > 0 else {
            throw VLMTestSupportError.videoDecodeFailed(
                "cat.mp4 has invalid duration \(durationSecs)")
        }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        // Tight tolerance so the timestamps land on real frames, not
        // interpolated approximations — relevant for the multi-frame
        // path once the video-inference port (#142) ships.
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let stamps: [CMTime] = (0..<maxFrames).map { i in
            // Evenly spaced — frame 0 at t=0, frame N-1 just before
            // duration. For maxFrames=1 this picks t=0 (the natural
            // "thumbnail" frame).
            let t = maxFrames == 1
                ? 0.0
                : durationSecs * Double(i) / Double(maxFrames)
            return CMTime(seconds: t, preferredTimescale: 600)
        }

        var frames: [RGBImage] = []
        frames.reserveCapacity(stamps.count)
        for stamp in stamps {
            do {
                let cg = try generator.copyCGImage(at: stamp, actualTime: nil)
                frames.append(try rgbImage(from: cg))
            } catch {
                // Skip frames AVFoundation can't decode; only throw if
                // we end up with no frames at all.
                continue
            }
        }
        guard !frames.isEmpty else {
            throw VLMTestSupportError.videoDecodeFailed(
                "cat.mp4 — AVAssetImageGenerator returned no frames")
        }
        return frames
    }

    /// Convert a CoreGraphics image to FFAI's `RGBImage` (planar-friendly
    /// RGB Floats in [0,1]). Drops the alpha channel if present.
    private static func rgbImage(from cg: CGImage) throws -> RGBImage {
        let w = cg.width
        let h = cg.height
        // Render into a fresh 32-bit RGBA buffer so we don't have to
        // guess the source colorspace / alpha layout.
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        let bytesPerRow = w * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = raw.withUnsafeMutableBufferPointer({ ptr in
            CGContext(data: ptr.baseAddress,
                      width: w, height: h, bitsPerComponent: 8,
                      bytesPerRow: bytesPerRow,
                      space: colorSpace, bitmapInfo: bitmapInfo)
        }) else {
            throw VLMTestSupportError.videoDecodeFailed(
                "CGContext init failed for \(w)x\(h)")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        // Pack as RGB Float [0,1] row-major: pixels[(y*w+x)*3 + c].
        var pixels = [Float](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            for x in 0..<w {
                let srcBase = (y * w + x) * 4
                let dstBase = (y * w + x) * 3
                pixels[dstBase + 0] = Float(raw[srcBase + 0]) / 255.0
                pixels[dstBase + 1] = Float(raw[srcBase + 1]) / 255.0
                pixels[dstBase + 2] = Float(raw[srcBase + 2]) / 255.0
            }
        }
        return RGBImage(width: w, height: h, pixels: pixels)
    }

    /// Assert a decoded VLM caption actually describes the dog photo.
    /// Mirrors the mlx-swift-lm VLM benchmark check: a lowercased
    /// substring match on "dog" — wide enough to accept "dog", "Golden
    /// Retriever" → no, but "puppy"/"canine" callers add their own — the
    /// single-word "dog" anchor is the agreed contract.
    static func expectMentionsDog(_ text: String, label: String,
                                  sourceLocation: SourceLocation = #_sourceLocation) {
        let lowered = text.lowercased()
        #expect(lowered.contains("dog"),
                "\(label): caption should mention a dog — got: \(text)",
                sourceLocation: sourceLocation)
    }

    /// Assert a decoded VLM caption describes the cat video frame.
    /// Accepts either "cat" or "kitten" — model verbosity varies.
    static func expectMentionsCat(_ text: String, label: String,
                                  sourceLocation: SourceLocation = #_sourceLocation) {
        let lowered = text.lowercased()
        #expect(lowered.contains("cat") || lowered.contains("kitten"),
                "\(label): caption should mention a cat — got: \(text)",
                sourceLocation: sourceLocation)
    }
}

enum VLMTestSupportError: Error, CustomStringConvertible {
    case videoDecodeFailed(String)

    var description: String {
        switch self {
        case .videoDecodeFailed(let msg):
            return "VLMTestSupport video decode failed: \(msg)"
        }
    }
}
