// VADModelRegistry — entry point for the voice-activity-detection
// model families.
//
// The causal-LM families (Llama / Qwen3 / Mamba2) flow through
// `ModelRegistry` → `Model`: tokenizer in, tokens out. The STT/TTS
// audio families flow through `AudioModelRegistry`. The VAD family
// (SileroVAD / SmartTurn / Sortformer) has a fundamentally different
// contract — audio waveform in, speech-probability stream out — so it
// gets its own registry rather than being forced through the
// `LanguageModel` protocol.
//
// Each VAD family file owns its own loader (`loadFromDirectory` /
// `fromPretrained`); this registry is the canonical place to discover
// which audio architectures are in tree and to dispatch a checkpoint
// to the right one by `model_type`.

import Foundation

// ─── Audio architecture kinds ────────────────────────────────────────

/// The audio architectures FFAI can load. Used to dispatch a checkpoint
/// directory to the right family loader.
public enum AudioModelKind: String, Sendable, CaseIterable {
    /// SileroVAD — streaming voice-activity detection.
    case sileroVAD
    /// SmartTurn — conversational endpoint / turn detection.
    case smartTurn
    /// Sortformer — multi-speaker diarization.
    case sortformer
    /// TenVAD — TEN-framework lightweight VAD (native macOS framework).
    case tenVAD

    /// The `model_type` strings (from `config.json`) that map to this
    /// architecture.
    var modelTypes: Set<String> {
        switch self {
        case .sileroVAD:  return ["silero_vad"]
        case .smartTurn:  return ["smart_turn", "smart_turn_v3", "smart-turn"]
        case .sortformer: return ["sortformer", "diar_sortformer"]
        case .tenVAD:     return ["ten_vad", "ten-vad", "tenvad"]
        }
    }
}

public enum AudioModelError: Error, CustomStringConvertible {
    case unknownArchitecture(String)
    case missingConfig(URL)

    public var description: String {
        switch self {
        case .unknownArchitecture(let m):
            return "VADModelRegistry: unrecognized audio model_type: \(m)"
        case .missingConfig(let url):
            return "VADModelRegistry: config.json not found at \(url.path)"
        }
    }
}

// ─── A loaded audio model (type-erased) ──────────────────────────────

/// A loaded audio model. The concrete value is one of the VAD family
/// models; switch on `kind` (or downcast `model`) to use it.
///
/// Cases are added as each family is ported; Sortformer lands in a
/// later commit.
public enum LoadedVADModel: @unchecked Sendable {
    case sileroVAD(SileroVADModel)
    case smartTurn(SmartTurnModel)
    case sortformer(SortformerModel)
    case tenVAD(TenVADModel)

    public var kind: AudioModelKind {
        switch self {
        case .sileroVAD:  return .sileroVAD
        case .smartTurn:  return .smartTurn
        case .sortformer: return .sortformer
        case .tenVAD:     return .tenVAD
        }
    }
}

// ─── Registry ────────────────────────────────────────────────────────

public enum VADModelRegistry {
    /// Inspect a checkpoint directory's `config.json` and return its
    /// audio architecture kind. Falls back to filename heuristics if the
    /// config lacks a recognizable `model_type`.
    public static func detectKind(in directory: URL) throws -> AudioModelKind {
        let configURL = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            // Prefer `model_type`, then `architecture`.
            let candidates = [raw["model_type"] as? String,
                              raw["architecture"] as? String].compactMap { $0?.lowercased() }
            for candidate in candidates {
                for kind in AudioModelKind.allCases where kind.modelTypes.contains(candidate) {
                    return kind
                }
            }
        }
        // Fallback: infer from the directory name.
        let name = directory.lastPathComponent.lowercased()
        if name.contains("silero") { return .sileroVAD }
        if name.contains("smart-turn") || name.contains("smart_turn") { return .smartTurn }
        if name.contains("sortformer") || name.contains("diar") { return .sortformer }
        if name.contains("ten-vad") || name.contains("ten_vad") || name.contains("tenvad") {
            return .tenVAD
        }
        throw AudioModelError.unknownArchitecture(directory.lastPathComponent)
    }

    /// Load an audio model from a local snapshot directory, dispatching
    /// to the right family loader by detected architecture.
    public static func loadFromDirectory(_ directory: URL,
                                         device: Device = .shared) throws -> LoadedVADModel {
        let kind = try detectKind(in: directory)
        switch kind {
        case .sileroVAD:
            return .sileroVAD(try SileroVADModel.loadFromDirectory(directory, device: device))
        case .smartTurn:
            return .smartTurn(try SmartTurnModel.loadFromDirectory(directory, device: device))
        case .sortformer:
            return .sortformer(try SortformerModel.loadFromDirectory(directory, device: device))
        case .tenVAD:
            return .tenVAD(try TenVADModel.loadFromDirectory(directory, device: device))
        }
    }

    /// Download (or hit cache) an audio checkpoint from HuggingFace and
    /// load it.
    public static func fromPretrained(_ idOrPath: String,
                                      device: Device = .shared) async throws -> LoadedVADModel {
        let dir = try await ModelLocator().resolve(idOrPath: idOrPath)
        return try loadFromDirectory(dir, device: device)
    }
}
