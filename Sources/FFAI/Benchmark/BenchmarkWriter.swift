// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// BenchmarkWriter — append a row to the day's benchmark report
// (markdown table + JSON sidecar). Mirrors mlx-swift-lm's
// BenchmarkWriter so analysis scripts work cross-repo.
//
// Layout under `--report-dir`:
//
//   {chip-slug}-{YYYY-MM-DD}.md             ← human-readable table
//   .{chip-slug}-{YYYY-MM-DD}.state.json    ← structured sidecar
//
// We render the markdown deterministically from the JSON sidecar
// every append (so manual edits to the .md don't survive — that's
// intentional; the sidecar is the source of truth).

import Foundation

public struct BenchRow: Sendable, Codable, Equatable {
    public let model: String
    public let method: String
    public let quantization: String?
    public let contextSize: Int
    public let promptTokens: Int
    public let prefillTokensPerSecond: Double
    public let decodeTokensPerSecond: Double
    public let steadyTokensPerSecond: Double?
    public let timeToFirstTokenMs: Double
    public let generatedTokens: Int
    public let baselineGPUBytes: Int
    public let peakGPUBytes: Int
    public let kvCacheUsedBytes: Int
    public let weightsBytes: Int
    public let wiredTicketBytes: Int
    public let genPerplexity: Double?
    public let genKLDivergence: Double?
    public let outputPreview: String?

    public init(model: String, method: String, quantization: String?,
                contextSize: Int, promptTokens: Int,
                prefillTokensPerSecond: Double, decodeTokensPerSecond: Double,
                steadyTokensPerSecond: Double?, timeToFirstTokenMs: Double,
                generatedTokens: Int,
                baselineGPUBytes: Int, peakGPUBytes: Int,
                kvCacheUsedBytes: Int, weightsBytes: Int, wiredTicketBytes: Int,
                genPerplexity: Double?, genKLDivergence: Double?,
                outputPreview: String?) {
        self.model = model; self.method = method
        self.quantization = quantization
        self.contextSize = contextSize; self.promptTokens = promptTokens
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.steadyTokensPerSecond = steadyTokensPerSecond
        self.timeToFirstTokenMs = timeToFirstTokenMs
        self.generatedTokens = generatedTokens
        self.baselineGPUBytes = baselineGPUBytes
        self.peakGPUBytes = peakGPUBytes
        self.kvCacheUsedBytes = kvCacheUsedBytes
        self.weightsBytes = weightsBytes
        self.wiredTicketBytes = wiredTicketBytes
        self.genPerplexity = genPerplexity
        self.genKLDivergence = genKLDivergence
        self.outputPreview = outputPreview
    }
}

public struct BenchReport: Sendable, Codable {
    public var chip: String
    public var systemRAMBytes: Int
    public var osVersion: String
    public var createdAt: Date
    public var rows: [BenchRow]

    public init(chip: String, systemRAMBytes: Int, osVersion: String,
                createdAt: Date = Date(), rows: [BenchRow] = []) {
        self.chip = chip
        self.systemRAMBytes = systemRAMBytes
        self.osVersion = osVersion
        self.createdAt = createdAt
        self.rows = rows
    }
}

public enum BenchmarkWriterError: Error, CustomStringConvertible {
    case directoryCreateFailed(URL, any Error)
    case writeFailed(URL, any Error)
    case decodeFailed(URL, any Error)

    public var description: String {
        switch self {
        case .directoryCreateFailed(let url, let e):
            return "Bench: failed to create \(url.path): \(e)"
        case .writeFailed(let url, let e):
            return "Bench: failed to write \(url.path): \(e)"
        case .decodeFailed(let url, let e):
            return "Bench: failed to decode \(url.path): \(e)"
        }
    }
}

public struct BenchmarkWriter: Sendable {
    public let reportDirectory: URL
    public let chipSlug: String

    public init(reportDirectory: URL, chipSlug: String? = nil) {
        self.reportDirectory = reportDirectory
        self.chipSlug = chipSlug ?? Self.detectChipSlug()
    }

    /// Append `row` to today's report. Idempotent against the date —
    /// multiple appends in the same day grow the same files.
    public func append(_ row: BenchRow,
                       date: Date = Date()) throws -> (markdown: URL, sidecar: URL) {
        try ensureDirectory()
        let stem = "\(chipSlug)-\(Self.dateStamp(date))"
        let mdURL = reportDirectory.appendingPathComponent("\(stem).md")
        let jsonURL = reportDirectory.appendingPathComponent(".\(stem).state.json")

        var report: BenchReport
        if let data = try? Data(contentsOf: jsonURL) {
            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                report = try decoder.decode(BenchReport.self, from: data)
            } catch {
                throw BenchmarkWriterError.decodeFailed(jsonURL, error)
            }
        } else {
            report = BenchReport(
                chip: chipSlug,
                systemRAMBytes: Self.detectSystemRAM(),
                osVersion: Self.detectOSVersion(),
                createdAt: date
            )
        }
        report.rows.append(row)

        // Write JSON sidecar.
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: jsonURL, options: .atomic)
        } catch {
            throw BenchmarkWriterError.writeFailed(jsonURL, error)
        }

        // Re-render markdown deterministically from the sidecar.
        do {
            try Self.renderMarkdown(report: report).write(to: mdURL, atomically: true,
                                                          encoding: .utf8)
        } catch {
            throw BenchmarkWriterError.writeFailed(mdURL, error)
        }

        return (mdURL, jsonURL)
    }

    // MARK: - Helpers

    private func ensureDirectory() throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: reportDirectory.path) {
            do {
                try fm.createDirectory(at: reportDirectory, withIntermediateDirectories: true)
            } catch {
                throw BenchmarkWriterError.directoryCreateFailed(reportDirectory, error)
            }
        }
    }

    public static func dateStamp(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        return f.string(from: date)
    }

    public static func detectChipSlug() -> String {
        // Best-effort. `sysctl -n machdep.cpu.brand_string` is the
        // canonical source; fall back to a generic label if it fails.
        let task = Process()
        task.launchPath = "/usr/sbin/sysctl"
        task.arguments = ["-n", "machdep.cpu.brand_string"]
        let pipe = Pipe()
        task.standardOutput = pipe
        do { try task.run() } catch { return "apple-silicon" }
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let raw = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if raw.isEmpty { return "apple-silicon" }
        return raw
            .lowercased()
            .replacingOccurrences(of: "apple ", with: "")
            .replacingOccurrences(of: " ", with: "-")
    }

    public static func detectSystemRAM() -> Int {
        Int(ProcessInfo.processInfo.physicalMemory)
    }

    public static func detectOSVersion() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    public static func renderMarkdown(report: BenchReport) -> String {
        var out = ""
        out += "# FFAI Bench — \(report.chip)\n\n"
        out += "- System RAM: \(formatBytes(report.systemRAMBytes))\n"
        out += "- OS: \(report.osVersion)\n"
        out += "- Created: \(report.createdAt.ISO8601Format())\n\n"
        out += "| Model | Method | Quant | Ctx | Prompt | Prefill tok/s | Decode tok/s | Steady tok/s | TTFT (ms) | Gen tokens | Baseline GPU | Peak GPU | KV used | Weights | Gen PPL | Gen KLD | Sample |\n"
        out += "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|\n"
        for r in report.rows {
            out += "| \(r.model) | \(r.method) | \(r.quantization ?? "-") | \(r.contextSize) | \(r.promptTokens) | "
            out += String(format: "%.2f | %.2f | %@ | %.2f | %d | %@ | %@ | %@ | %@ | %@ | %@ | %@ |\n",
                          r.prefillTokensPerSecond,
                          r.decodeTokensPerSecond,
                          r.steadyTokensPerSecond.map { String(format: "%.2f", $0) } ?? "-",
                          r.timeToFirstTokenMs,
                          r.generatedTokens,
                          formatBytes(r.baselineGPUBytes),
                          formatBytes(r.peakGPUBytes),
                          formatBytes(r.kvCacheUsedBytes),
                          formatBytes(r.weightsBytes),
                          r.genPerplexity.map { String(format: "%.3f", $0) } ?? "-",
                          r.genKLDivergence.map { String(format: "%.4f", $0) } ?? "-",
                          (r.outputPreview ?? "").replacingOccurrences(of: "|", with: "\\|"))
        }
        return out
    }

    public static func formatBytes(_ bytes: Int) -> String {
        let mb = Double(bytes) / 1024.0 / 1024.0
        if mb >= 1024 { return String(format: "%.2f GB", mb / 1024) }
        return String(format: "%.1f MB", mb)
    }
}

public extension BenchRow {
    /// Construct a BenchRow from a `GenerationStats` + identifying
    /// fields. The bench harness always has the stats already; this
    /// keeps the call site one line.
    init(model: String, method: String, quantization: String?,
         stats: GenerationStats, outputPreview: String?,
         genPerplexity: Double? = nil, genKLDivergence: Double? = nil) {
        self.init(
            model: model, method: method, quantization: quantization,
            contextSize: stats.contextSize, promptTokens: stats.promptTokens,
            prefillTokensPerSecond: stats.prefillTokensPerSecond,
            decodeTokensPerSecond: stats.decodeTokensPerSecond,
            steadyTokensPerSecond: stats.steadyTokensPerSecond,
            timeToFirstTokenMs: stats.timeToFirstTokenMs,
            generatedTokens: stats.generatedTokens,
            baselineGPUBytes: stats.baselineGPUBytes,
            peakGPUBytes: stats.peakGPUBytes,
            kvCacheUsedBytes: stats.kvCacheUsedBytes,
            weightsBytes: stats.weightsBytes,
            wiredTicketBytes: stats.wiredTicketBytes,
            genPerplexity: genPerplexity,
            genKLDivergence: genKLDivergence,
            outputPreview: outputPreview
        )
    }
}
