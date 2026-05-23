// ANEMTPInspectTests — Day 1: load the Qwen3.6-A3B MTP drafter mlpackage
// via Core ML, print its input/output schema, run a sample inference,
// and bench latency. This is the foundation for ANEMTPDrafter (the
// real Drafter conformance wrapper that lands on top).
//
// Once this test passes + reports the model signature, we know:
//   * Whether the mlpackage takes token IDs OR hidden states as input
//   * What it outputs (next-token logits, top-k, embedding)
//   * Inference latency on ANE (target < 5 ms/predict for spec decode
//     to win — at 1.5 ms/predict the win compounds with verify cost
//     of 77 ms at T=2 to give ~2× decode tps at 90% acceptance)

import Foundation
import Testing
import CoreML

@Suite("ANE MTP drafter inspect")
struct ANEMTPInspectTests {

    @Test("Load Qwen3.6-A3B MTP mlpackage and print signature")
    func loadAndPrintSchema() async throws {
        let path = "/Users/tom/models/Qwen3.6-35B-A3B-mtp.mlpackage"
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            print("ANEMTPInspect skipped: \(path) not found")
            return
        }

        // Try compile + load via Core ML.
        let config = MLModelConfiguration()
        config.computeUnits = .all  // CPU + GPU + ANE — let Core ML pick.

        let t0 = Date()
        let compiledURL = try await MLModel.compileModel(at: url)
        let compileS = Date().timeIntervalSince(t0)
        print("MTP mlpackage compiled in \(String(format: "%.2f", compileS))s → \(compiledURL.path)")

        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        let desc = model.modelDescription

        print("=== Inputs ===")
        for (name, descIn) in desc.inputDescriptionsByName {
            let typeStr: String
            switch descIn.type {
            case .invalid: typeStr = "invalid"
            case .double: typeStr = "double"
            case .int64: typeStr = "int64"
            case .string: typeStr = "string"
            case .image: typeStr = "image"
            case .multiArray:
                let m = descIn.multiArrayConstraint
                typeStr = "multiArray shape=\(m?.shape ?? []) dtype=\(m?.dataType.rawValue ?? 0)"
            case .dictionary: typeStr = "dictionary"
            case .sequence: typeStr = "sequence"
            case .state: typeStr = "state"
            @unknown default: typeStr = "unknown"
            }
            print("  \(name): \(typeStr)")
        }

        print("=== Outputs ===")
        for (name, descOut) in desc.outputDescriptionsByName {
            let typeStr: String
            switch descOut.type {
            case .multiArray:
                let m = descOut.multiArrayConstraint
                typeStr = "multiArray shape=\(m?.shape ?? []) dtype=\(m?.dataType.rawValue ?? 0)"
            case .int64: typeStr = "int64"
            case .double: typeStr = "double"
            default: typeStr = "other(\(descOut.type.rawValue))"
            }
            print("  \(name): \(typeStr)")
        }

        // Metadata block — author, description, version, training metadata.
        let meta = desc.metadata
        for (k, v) in meta {
            print("  metadata[\(k.rawValue)]: \(v)")
        }

        // Sanity: dummy zero-fill any int64 1-D input + small zero array
        // for any multiArray, and see if predict succeeds + how fast.
        var dummyInputs: [String: MLFeatureValue] = [:]
        for (name, descIn) in desc.inputDescriptionsByName {
            switch descIn.type {
            case .multiArray:
                guard let m = descIn.multiArrayConstraint else { continue }
                let shape = m.shape.map { $0.intValue }
                guard !shape.isEmpty else { continue }
                let dtype = m.dataType
                let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dtype)
                // Zero-fill — implementation depends on dtype but
                // calloc-style works for int32/float32 because
                // MLMultiArray underlying buffer is contiguous.
                let bytes: Int
                switch dtype {
                case .int32: bytes = arr.count * 4
                case .float32: bytes = arr.count * 4
                case .float16: bytes = arr.count * 2
                case .double: bytes = arr.count * 8
                @unknown default: bytes = arr.count * 4
                }
                memset(arr.dataPointer, 0, bytes)
                dummyInputs[name] = MLFeatureValue(multiArray: arr)
            case .int64:
                dummyInputs[name] = MLFeatureValue(int64: 0)
            default: break
            }
        }
        guard !dummyInputs.isEmpty else {
            print("ANEMTPInspect: no compatible inputs found — manual schema work needed")
            return
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: dummyInputs)
        // Warm + bench.
        let prediction = try await model.prediction(from: provider)
        for outputName in desc.outputDescriptionsByName.keys {
            if let val = prediction.featureValue(for: outputName) {
                print("Output \(outputName): \(val.type) — \(val.multiArrayValue?.shape ?? [])")
            }
        }

        let iters = 20
        let warmCount = 5
        for _ in 0..<warmCount { _ = try await model.prediction(from: provider) }
        var times: [Double] = []
        for _ in 0..<iters {
            let t = Date()
            _ = try await model.prediction(from: provider)
            times.append(Date().timeIntervalSince(t))
        }
        let med = times.sorted()[iters / 2] * 1000
        let mn = times.min()! * 1000
        let mx = times.max()! * 1000
        print("ANE MTP predict latency: median=\(String(format: "%.2f", med))ms, min=\(String(format: "%.2f", mn))ms, max=\(String(format: "%.2f", mx))ms over \(iters) iters")
    }
}
