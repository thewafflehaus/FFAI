// swift-tools-version: 6.1
//
// FFAI — Fucking Fast Apple Inference
//
// Apple Silicon LLM inference library built on pre-compiled Metal kernels
// generated from the metaltile Rust DSL.
// See planning/plan.md and planning/architecture.md.

import PackageDescription

let package = Package(
    name: "FFAI",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "FFAI",
            targets: ["FFAI"]
        ),
        .library(
            name: "MetalTileSwift",
            targets: ["MetalTileSwift"]
        ),
        .executable(
            name: "ffai",
            targets: ["FFAICLI"]
        ),
    ],
    dependencies: [
        // Hugging Face Hub client for model snapshot download/cache.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        // Tokenizer loading (AutoTokenizer.from(modelFolder:)).
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // CLI argument parsing for ffai.
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        // Pre-compiled Metal kernels + Swift dispatch wrappers.
        // Resources are produced at build time by metaltile-emit (Rust bin
        // in the sibling metaltile workspace) and live under Resources/.
        // Generated/ contains the typed Swift wrappers, also produced by
        // metaltile-emit.
        .target(
            name: "MetalTileSwift",
            path: "Sources/MetalTileSwift",
            resources: [
                .copy("Resources"),
            ]
            // TODO: add MetalTileEmitPlugin once SPM build plugin lands
            // (Phase 0 deliverable — see planning/plan.md).
        ),

        // Main inference library: Tensor, Module system, model families,
        // KV cache, sampling, generate loop, HF download, tokenizer.
        .target(
            name: "FFAI",
            dependencies: [
                "MetalTileSwift",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/FFAI"
        ),

        // CLI: ffai --model <id-or-path> --prompt "..."
        .executableTarget(
            name: "FFAICLI",
            dependencies: [
                "FFAI",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/FFAICLI"
        ),

        // Tests
        .testTarget(
            name: "MetalTileSwiftTests",
            dependencies: ["MetalTileSwift"],
            path: "Tests/MetalTileSwiftTests"
        ),
        .testTarget(
            name: "FFAITests",
            dependencies: ["FFAI"],
            path: "Tests/FFAITests"
        ),
        .testTarget(
            name: "ModelTests",
            dependencies: ["FFAI"],
            path: "Tests/ModelTests",
            resources: [
                // Golden fixtures captured from mlx-lm. See
                // Tools/capture-fixtures.py and planning/plan.md Phase 0
                // testing reference convention.
                .copy("../Fixtures"),
            ]
        ),
    ]
)
