// BenchTests — BenchMethod enum, BenchRow construction, and the
// BenchmarkWriter markdown + JSON sidecar round-trip. End-to-end
// benchmark execution is exercised by the CLI smoke (`ffai bench
// --method simple --model <small> --prompt ...`).

import Foundation
import Testing
@testable import FFAI

@Suite("Bench")
struct BenchTests {

    // MARK: - BenchMethod enum

    @Test("BenchMethod rawValues are stable")
    func methodRawValues() {
        #expect(BenchMethod.simple.rawValue == "simple")
        #expect(BenchMethod.summarization.rawValue == "summarization")
        #expect(BenchMethod.wikitext2.rawValue == "wikitext2")
        #expect(BenchMethod.niah.rawValue == "niah")
        #expect(BenchMethod.multiTurn.rawValue == "multi-turn")
        #expect(BenchMethod.toolCalling.rawValue == "tool-calling")
        #expect(BenchMethod.ngramSpot.rawValue == "ngram-spot")
        #expect(BenchMethod.ngramSweep.rawValue == "ngram-sweep")
        #expect(BenchMethod.ngramSweepSummary.rawValue == "ngram-sweep-summary")
        #expect(BenchMethod.vision.rawValue == "vision")
    }

    @Test("BenchMethod isImplemented matches today's reality")
    func methodImplementation() {
        #expect(BenchMethod.simple.isImplemented)
        #expect(BenchMethod.summarization.isImplemented)
        #expect(BenchMethod.wikitext2.isImplemented)
        #expect(BenchMethod.niah.isImplemented == false)
        #expect(BenchMethod.multiTurn.isImplemented == false)
        #expect(BenchMethod.toolCalling.isImplemented == false)
        #expect(BenchMethod.ngramSpot.isImplemented == false)
        #expect(BenchMethod.vision.isImplemented == false)
    }

    @Test("Every method has a non-empty description")
    func methodDescriptions() {
        for m in BenchMethod.allCases {
            #expect(!m.description.isEmpty, "missing description for \(m.rawValue)")
        }
    }

    @Test("Unimplemented methods name a dependency; implemented ones don't")
    func methodDependencies() {
        for m in BenchMethod.allCases {
            if m.isImplemented {
                #expect(m.dependency == nil)
            } else {
                #expect(m.dependency != nil, "missing dependency note for \(m.rawValue)")
            }
        }
    }

    // MARK: - BenchRow

    private func makeStats() -> GenerationStats {
        GenerationStats(
            promptTokens: 5, generatedTokens: 16, contextSize: 4096,
            prefillTimeS: 0.1, decodeTimeS: 1.0, timeToFirstTokenMs: 100,
            steadyTokensPerSecond: 18.0,
            baselineGPUBytes: 1_000_000_000,
            postPrefillGPUBytes: 1_100_000_000,
            postDecodeGPUBytes: 1_120_000_000,
            prefillPeakGPUBytes: 1_150_000_000,
            decodePeakGPUBytes: 1_125_000_000,
            wiredTicketBytes: 16 * 1024 * 1024 * 1024,
            weightsBytes: 800_000_000,
            kvCacheAllocatedBytes: 64 * 1024 * 1024,
            kvCacheUsedBytes: 12 * 1024 * 1024,
            thinkPerplexity: nil, genPerplexity: 5.5,
            thinkKLDivergence: nil, genKLDivergence: 0.04,
            thinkTokenCount: nil, genTokenCount: nil
        )
    }

    @Test("BenchRow.init(stats:) maps the right fields")
    func rowFromStats() {
        let row = BenchRow(model: "demo/foo", method: "simple",
                           quantization: "4bit",
                           stats: makeStats(), outputPreview: "hello",
                           genPerplexity: 5.5, genKLDivergence: 0.04)
        #expect(row.model == "demo/foo")
        #expect(row.method == "simple")
        #expect(row.quantization == "4bit")
        #expect(row.contextSize == 4096)
        #expect(row.promptTokens == 5)
        #expect(row.generatedTokens == 16)
        #expect(row.peakGPUBytes == 1_150_000_000)
        #expect(row.kvCacheUsedBytes == 12 * 1024 * 1024)
        #expect(row.weightsBytes == 800_000_000)
        #expect(row.genPerplexity == 5.5)
        #expect(row.genKLDivergence == 0.04)
        #expect(row.outputPreview == "hello")
    }

    // MARK: - BenchmarkWriter

    @Test("Writer creates report directory + markdown + JSON sidecar")
    func writerRoundTrip() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-bench-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let writer = BenchmarkWriter(reportDirectory: tempDir, chipSlug: "test-chip")
        let row = BenchRow(model: "demo/foo", method: "simple",
                           quantization: "4bit",
                           stats: makeStats(), outputPreview: "hello world",
                           genPerplexity: nil, genKLDivergence: nil)
        let urls = try writer.append(row, date: Date(timeIntervalSince1970: 0))

        // Date 1970-01-01 → file stem `test-chip-1970-01-01`.
        #expect(urls.markdown.lastPathComponent == "test-chip-1970-01-01.md")
        #expect(urls.sidecar.lastPathComponent == ".test-chip-1970-01-01.state.json")
        #expect(FileManager.default.fileExists(atPath: urls.markdown.path))
        #expect(FileManager.default.fileExists(atPath: urls.sidecar.path))

        // Markdown contains the row.
        let md = try String(contentsOf: urls.markdown, encoding: .utf8)
        #expect(md.contains("demo/foo"))
        #expect(md.contains("simple"))
        #expect(md.contains("4bit"))
        #expect(md.contains("hello world"))

        // Sidecar JSON decodes back to a BenchReport with one row.
        let data = try Data(contentsOf: urls.sidecar)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let report = try decoder.decode(BenchReport.self, from: data)
        #expect(report.chip == "test-chip")
        #expect(report.rows.count == 1)
        #expect(report.rows.first?.model == "demo/foo")
    }

    @Test("Writer appends rows across multiple calls in the same day")
    func writerAppends() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ffai-bench-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let writer = BenchmarkWriter(reportDirectory: tempDir, chipSlug: "test-chip")
        let row1 = BenchRow(model: "demo/foo", method: "simple",
                            quantization: "bf16",
                            stats: makeStats(), outputPreview: "a",
                            genPerplexity: nil, genKLDivergence: nil)
        let row2 = BenchRow(model: "demo/bar", method: "wikitext2",
                            quantization: "4bit",
                            stats: makeStats(), outputPreview: nil,
                            genPerplexity: 5.5, genKLDivergence: 0.04)
        let date = Date(timeIntervalSince1970: 0)
        _ = try writer.append(row1, date: date)
        let urls = try writer.append(row2, date: date)

        let data = try Data(contentsOf: urls.sidecar)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let report = try decoder.decode(BenchReport.self, from: data)
        #expect(report.rows.count == 2)
        #expect(report.rows.map(\.method) == ["simple", "wikitext2"])
    }

    @Test("dateStamp formats yyyy-MM-dd in UTC")
    func dateStamp() {
        let s = BenchmarkWriter.dateStamp(Date(timeIntervalSince1970: 0))
        #expect(s == "1970-01-01")
    }

    @Test("formatBytes scales MB / GB cleanly")
    func formatBytesScales() {
        #expect(BenchmarkWriter.formatBytes(0).contains("0.0 MB"))
        #expect(BenchmarkWriter.formatBytes(512 * 1024 * 1024).contains("MB"))
        #expect(BenchmarkWriter.formatBytes(2 * 1024 * 1024 * 1024).contains("GB"))
    }

    // MARK: - BenchRunnerError

    @Test("BenchRunnerError descriptions name the missing piece")
    func runnerErrorDescriptions() {
        let e1 = BenchRunnerError.notImplemented(method: .niah,
                                                 dependency: "sliding-window mask")
        #expect(String(describing: e1).contains("niah"))
        #expect(String(describing: e1).contains("sliding-window mask"))

        let e2 = BenchRunnerError.missingPrompt
        #expect(String(describing: e2).contains("--prompt"))

        let e3 = BenchRunnerError.wikitext2CorpusMissing(URL(fileURLWithPath: "/x/y"))
        #expect(String(describing: e3).contains("--wikitext2-corpus"))

        let e4 = BenchRunnerError.kldRequiresReferenceModel
        #expect(String(describing: e4).contains("--ref-model"))
    }
}
