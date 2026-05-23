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
