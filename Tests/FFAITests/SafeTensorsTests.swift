import Foundation
import Testing
@testable import FFAI

@Suite("SafeTensors")
struct SafeTensorsTests {
    /// Build a minimal valid safetensors file in `directory` containing one
    /// f32 tensor named `tensorName` with the given values + shape.
    static func writeSyntheticFile(directory: URL, tensorName: String,
                                   shape: [Int], values: [Float],
                                   filename: String = "model.safetensors") throws -> URL
    {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let dataLen = values.count * MemoryLayout<Float>.size
        let header = #"{"\#(tensorName)":{"dtype":"F32","shape":\#(shape.description),"data_offsets":[0,\#(dataLen)]}}"#
        let headerBytes = Array(header.utf8)
        var headerLen = UInt64(headerBytes.count)

        var data = Data()
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(contentsOf: headerBytes)
        values.withUnsafeBufferPointer { buf in
            data.append(contentsOf: UnsafeRawBufferPointer(buf))
        }

        let url = directory.appendingPathComponent(filename)
        try data.write(to: url)
        return url
    }

    @Test("load synthetic single-tensor file")
    func loadSingleFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let values: [Float] = [1, 2, 3, 4, 5, 6]
        let url = try Self.writeSyntheticFile(
            directory: dir, tensorName: "alpha.weight",
            shape: [2, 3], values: values
        )

        let f = try SafeTensorsFile(url: url)
        let t = try f.tensor(named: "alpha.weight")
        #expect(t.shape == [2, 3])
        #expect(t.dtype == .f32)
        #expect(t.toArray(as: Float.self) == values)
    }

    @Test("missing tensor throws")
    func missingTensor() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = try Self.writeSyntheticFile(
            directory: dir, tensorName: "present", shape: [1], values: [1]
        )
        let f = try SafeTensorsFile(url: url)
        do {
            _ = try f.tensor(named: "absent")
            Issue.record("expected error")
        } catch let e as SafeTensorsError {
            if case .missingTensor = e { /* ok */ } else {
                Issue.record("expected .missingTensor, got \(e)")
            }
        }
    }

    @Test("file-not-found throws")
    func fileNotFound() {
        do {
            _ = try SafeTensorsFile(url: URL(fileURLWithPath: "/__nope__/missing.safetensors"))
            Issue.record("expected error")
        } catch let e as SafeTensorsError {
            if case .fileNotFound = e { /* ok */ } else {
                Issue.record("expected .fileNotFound, got \(e)")
            }
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test("bundle picks up single .safetensors via fallback glob")
    func bundleSingleFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let values: [Float] = [42, 43, 44]
        _ = try Self.writeSyntheticFile(
            directory: dir, tensorName: "x", shape: [3], values: values
        )

        let bundle = try SafeTensorsBundle(directory: dir)
        #expect(bundle.has("x"))
        let t = try bundle.tensor(named: "x")
        #expect(t.toArray(as: Float.self) == values)
        #expect(bundle.allKeys == ["x"])
    }

    @Test("sharded bundle reads model.safetensors.index.json")
    func bundleSharded() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-st-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Two shards
        _ = try Self.writeSyntheticFile(
            directory: dir, tensorName: "a", shape: [1], values: [10],
            filename: "model-00001-of-00002.safetensors"
        )
        _ = try Self.writeSyntheticFile(
            directory: dir, tensorName: "b", shape: [1], values: [20],
            filename: "model-00002-of-00002.safetensors"
        )
        // Index file
        let index = """
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}
        """
        try index.write(to: dir.appendingPathComponent("model.safetensors.index.json"),
                        atomically: true, encoding: .utf8)

        let bundle = try SafeTensorsBundle(directory: dir)
        #expect(try bundle.tensor(named: "a").toArray(as: Float.self) == [10])
        #expect(try bundle.tensor(named: "b").toArray(as: Float.self) == [20])
        #expect(bundle.allKeys.sorted() == ["a", "b"])
    }
}
