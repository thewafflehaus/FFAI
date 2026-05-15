// Cover the SafeTensors error paths that the happy-path tests don't reach.

import Foundation
import Testing
@testable import FFAI

@Suite("SafeTensors errors")
struct SafeTensorsErrorTests {
    static func writeRaw(_ bytes: [UInt8], filename: String = "model.safetensors") throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-st-err-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(filename)
        try Data(bytes).write(to: url)
        return url
    }

    @Test("header smaller than 8 bytes throws")
    func headerTooSmall() throws {
        let url = try Self.writeRaw([0x01, 0x02, 0x03])
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        do {
            _ = try SafeTensorsFile(url: url)
            Issue.record("expected throw")
        } catch let e as SafeTensorsError {
            if case .headerTooSmall = e { /* ok */ } else {
                Issue.record("expected .headerTooSmall, got \(e)")
            }
        }
    }

    @Test("malformed JSON header throws")
    func malformedJSON() throws {
        var bytes = [UInt8]()
        let payload = "not json"
        var len = UInt64(payload.utf8.count)
        withUnsafeBytes(of: &len) { bytes.append(contentsOf: $0) }
        bytes.append(contentsOf: payload.utf8)
        let url = try Self.writeRaw(bytes)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        do {
            _ = try SafeTensorsFile(url: url)
            Issue.record("expected throw")
        } catch let e as SafeTensorsError {
            if case .headerJSONMalformed = e { /* ok */ } else {
                Issue.record("expected .headerJSONMalformed, got \(e)")
            }
        }
    }

    @Test("header entry missing required field throws")
    func malformedEntry() throws {
        var bytes = [UInt8]()
        let payload = #"{"x": {"dtype": "F32"}}"#  // missing shape, data_offsets
        var len = UInt64(payload.utf8.count)
        withUnsafeBytes(of: &len) { bytes.append(contentsOf: $0) }
        bytes.append(contentsOf: payload.utf8)
        let url = try Self.writeRaw(bytes)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        do {
            _ = try SafeTensorsFile(url: url)
            Issue.record("expected throw")
        } catch let e as SafeTensorsError {
            if case .headerEntryMalformed = e { /* ok */ } else {
                Issue.record("expected .headerEntryMalformed, got \(e)")
            }
        }
    }

    @Test("unsupported dtype throws")
    func unsupportedDType() throws {
        var bytes = [UInt8]()
        let payload = #"{"x": {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]}}"#
        var len = UInt64(payload.utf8.count)
        withUnsafeBytes(of: &len) { bytes.append(contentsOf: $0) }
        bytes.append(contentsOf: payload.utf8)
        // Pad with 8 bytes of fake tensor data
        bytes.append(contentsOf: [UInt8](repeating: 0, count: 8))
        let url = try Self.writeRaw(bytes)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        do {
            _ = try SafeTensorsFile(url: url)
            Issue.record("expected throw")
        } catch let e as SafeTensorsError {
            if case .unsupportedDType = e { /* ok */ } else {
                Issue.record("expected .unsupportedDType, got \(e)")
            }
        }
    }

    @Test("error descriptions mention key info")
    func errorDescriptions() {
        let cases: [SafeTensorsError] = [
            .fileNotFound(URL(fileURLWithPath: "/x")),
            .headerTooSmall,
            .headerJSONMalformed,
            .headerEntryMalformed("foo"),
            .unsupportedDType("F64", key: "bar"),
            .mmapFailed(URL(fileURLWithPath: "/y")),
            .mtlBufferFailed,
            .missingTensor("baz"),
        ]
        for e in cases {
            #expect(!String(describing: e).isEmpty)
        }
    }

    @Test("bundle missing tensor throws")
    func bundleMissingTensor() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-st-bundle-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try SafeTensorsTests.writeSyntheticFile(
            directory: dir, tensorName: "present", shape: [1], values: [1]
        )
        let bundle = try SafeTensorsBundle(directory: dir)
        do {
            _ = try bundle.tensor(named: "absent")
            Issue.record("expected throw")
        } catch let e as SafeTensorsError {
            if case .missingTensor = e { /* ok */ } else {
                Issue.record("got \(e)")
            }
        }
    }
}
