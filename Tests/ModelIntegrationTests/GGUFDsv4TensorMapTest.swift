// Copyright 2026 Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Tensor-map introspection over the DSv4-Flash GGUF — dumps every
// tensor name + ggml type + shape so the DeepSeekV4Model loader can
// be written against real names, not guesses. Gated on the same
// `FFAI_DSV4_GGUF_PATH` env var as the rest of the integration suite;
// printed to stdout when `FFAI_DSV4_DUMP_TENSOR_MAP=1` is also set,
// so the test stays silent during normal CI.

import Foundation
import Testing

@testable import FFAI

@Suite("GGUF DSv4 tensor-map introspection", .serialized)
struct GGUFDsv4TensorMapTest {

    private var modelPath: String? {
        let env = ProcessInfo.processInfo.environment["FFAI_DSV4_GGUF_PATH"]
            ?? NSString("~/models/ds4-model").expandingTildeInPath
        return FileManager.default.fileExists(atPath: env) ? env : nil
    }

    @Test("Dump tensor map (FFAI_DSV4_DUMP_TENSOR_MAP=1 to print)")
    func dumpTensorMap() throws {
        guard let dir = modelPath else {
            print("GGUFDsv4TensorMapTest: skipping (no model at FFAI_DSV4_GGUF_PATH)")
            return
        }
        let bundle = try GGUFTensorBundle(directory: URL(fileURLWithPath: dir))
        let infos = bundle.reader.tensorInfos
        // Always assert the loader sees a non-trivial tensor count so
        // the test fails loudly if the GGUF path is wrong.
        #expect(infos.count > 100, "DSv4 GGUF must have >100 tensors; saw \(infos.count)")

        guard ProcessInfo.processInfo.environment["FFAI_DSV4_DUMP_TENSOR_MAP"] == "1" else { return }

        // Group by `blk.N` prefix + non-block tensors so the output is
        // readable; sort for deterministic diff.
        let sorted = infos.sorted { $0.name < $1.name }
        var perLayer: [Int: [GGUFTensorInfo]] = [:]
        var nonBlock: [GGUFTensorInfo] = []
        for info in sorted {
            if let layer = parseLayer(info.name) {
                perLayer[layer, default: []].append(info)
            } else {
                nonBlock.append(info)
            }
        }
        print("== DSv4 tensor map ==")
        print("Total tensors:", infos.count)
        print("Non-block tensors:", nonBlock.count)
        for info in nonBlock {
            print(formatRow(info))
        }
        // Print only layer 0 + 1 + 2 + 42 (= last). These cover the
        // four attention regimes (full, full, CSA, full).
        let interesting = [0, 1, 2, 42]
        for layer in interesting {
            guard let tensors = perLayer[layer] else { continue }
            print("--- blk.\(layer) (\(tensors.count) tensors) ---")
            for info in tensors {
                print(formatRow(info))
            }
        }
        print("Layers found:", perLayer.keys.sorted())
        print("Per-layer tensor counts:",
            perLayer.keys.sorted().map { "\($0):\(perLayer[$0]!.count)" }.joined(separator: " "))
    }

    /// Parse `blk.N.…` → `N`, else nil.
    private func parseLayer(_ name: String) -> Int? {
        let parts = name.split(separator: ".")
        guard parts.count >= 2, parts[0] == "blk", let n = Int(parts[1]) else { return nil }
        return n
    }

    private func formatRow(_ info: GGUFTensorInfo) -> String {
        let shape = info.dimensions.map { String($0) }.joined(separator: "×")
        return "  \(info.name)  type=\(info.type)  shape=\(shape)"
    }
}
