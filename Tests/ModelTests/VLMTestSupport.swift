// Shared helpers for the vision-language model integration tests.
//
// Every VLM family's "image + text" test feeds the same real photograph
// — `Resources/dog.jpeg`, a golden retriever — and asserts the decoded
// caption mentions a dog. Centralizing the image load here keeps the six
// family test files in lock-step and gives a single, precise failure
// message if the fixture goes missing.

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
}
