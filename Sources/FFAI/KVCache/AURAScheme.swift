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
// AURAScheme — bit-width recipe for AURA-compressed KV cache.
//
// AURA's quality knob is *asymmetric K/V*: K precision dominates
// quality (softmax exponentiates score perturbations) while V
// precision matters less (V-aggregation is a linear weighted sum).
// The shipping recipes in mlx-swift-lm use names like `aura4v2`
// (4-bit K, 2-bit V) — `aura{kb}v{vb}` (or `aura{kb}` when the K
// and V bit widths match).
//
// See `papers/aura-compression-algorithm.md` §2.5 (engineering
// addition 3) and Tom Turney's `asymmetric-kv-compression.md` for
// the empirical justification for the K-heavy K/V split.

import Foundation

/// AURA bit-width recipe for a KV cache.
public struct AURAScheme: Sendable, Equatable, Hashable {
    /// Number of bits per coordinate for keys. Must be one of
    /// `AURACodebook.supportedBits` (2 / 3 / 4 / 8).
    public let keyBits: Int

    /// Number of bits per coordinate for values. Must be one of
    /// `AURACodebook.supportedBits` (2 / 3 / 4 / 8).
    public let valueBits: Int

    public init(keyBits: Int, valueBits: Int) {
        precondition(
            AURACodebook.supportedBits.contains(keyBits),
            "AURAScheme: keyBits=\(keyBits) not in \(AURACodebook.supportedBits.sorted())")
        precondition(
            AURACodebook.supportedBits.contains(valueBits),
            "AURAScheme: valueBits=\(valueBits) not in \(AURACodebook.supportedBits.sorted())")
        self.keyBits = keyBits
        self.valueBits = valueBits
    }

    /// Canonical name: `aura{kb}v{vb}` (or `aura{kb}` when symmetric).
    public var name: String {
        keyBits == valueBits ? "aura\(keyBits)" : "aura\(keyBits)v\(valueBits)"
    }

    /// Stability-first default — symmetric 4-bit K/V. Matches the
    /// session-plan "locked decisions" §1 entry: when `--kv-cache aura`
    /// is passed without a suffix, fall back to `aura4v4`.
    public static let `default` = AURAScheme(keyBits: 4, valueBits: 4)

    /// Production recipe from `aura-compression-algorithm.md` §2.5 —
    /// 4-bit K + 2-bit V. Roughly 5× compression vs fp16 with
    /// near-baseline quality on tested attention-only models.
    public static let aura4v2 = AURAScheme(keyBits: 4, valueBits: 2)

    /// Production K-protected recipe — 8-bit K + 4-bit V. Matches
    /// canonical TQ+'s `q8_0-K + turbo4-V` shape; on Qwen3-0.6B-4bit
    /// the FFAI KLD harness measures mean_kld=0.029 + same-top=89%
    /// (vs aura4v4's 1.24 / 47%, a 43× quality improvement at 50%
    /// size cost). The K-side precision is what dominates attention
    /// quality (softmax exponentiates K-score errors); V can be
    /// aggressive cheaply.
    public static let aura8v4 = AURAScheme(keyBits: 8, valueBits: 4)

    /// Sibling of `aura8v4` — 8-bit K + 2-bit V. Tightest size at
    /// preserved K precision.
    public static let aura8v2 = AURAScheme(keyBits: 8, valueBits: 2)

    /// Auto-asymmetric-policy resolver. Mirrors canonical TQ+'s
    /// `TURBO_AUTO_ASYMMETRIC` env behavior: when the model has a
    /// high GQA fan-out (gqaFactor ≥ 6), shared K rows get
    /// "amplified" by the softmax across many Q heads — small K
    /// quantization errors compound across the GQA group. The
    /// production fix is to keep K at the highest available precision
    /// (8-bit Lloyd-Max in AURA-land, q8_0 in canonical TQ+).
    ///
    /// Behavior:
    ///   - If `gqaFactor < 6`, return `requested` unchanged.
    ///   - If `gqaFactor ≥ 6` and `requested.keyBits < 8`, return a
    ///     scheme with keyBits bumped to 8 (V untouched).
    ///   - If `gqaFactor ≥ 6` and `requested.keyBits == 8`, return
    ///     `requested` unchanged (already protected).
    ///
    /// Pure resolver — always applies the policy when conditions are
    /// met. **The policy itself is not opt-in here**; the opt-in lives
    /// at the call site (model loaders gate this on
    /// `FFAI_AURA_AUTO_ASYM=1`, and a per-load `LoadOptions` flag will
    /// replace the env knob in a follow-up). Tests + future API
    /// callers that want the canonical TQ+ behaviour can invoke this
    /// directly without env coupling.
    ///
    /// Canonical-source mapping: TURBO_AUTO_ASYMMETRIC in
    /// `~/local_llms/llama.cpp/src/llama-kv-cache.cpp`. Threshold = 6
    /// matches the llama.cpp implementation.
    public static func autoAsymmetric(
        requested: AURAScheme, gqaFactor: Int
    ) -> AURAScheme {
        if gqaFactor < 6 { return requested }
        if requested.keyBits >= 8 { return requested }
        return AURAScheme(keyBits: 8, valueBits: requested.valueBits)
    }

    /// True when the caller has opted into the auto-asymmetric policy
    /// via `FFAI_AURA_AUTO_ASYM=1`. Read once at module load. Default
    /// OFF — Eric's "no magic by default" stance: the caller must
    /// explicitly request the policy.
    public static let autoAsymmetricOptedIn: Bool = {
        ProcessInfo.processInfo.environment["FFAI_AURA_AUTO_ASYM"] == "1"
    }()

    /// Parse a CLI / config string. Accepts:
    ///
    /// - `aura` — the stability-first default (aura4v4).
    /// - `aura{kb}` — symmetric, e.g. `aura4` → keyBits=4, valueBits=4.
    /// - `aura{kb}v{vb}` — asymmetric, e.g. `aura4v2` → 4-bit K, 2-bit V.
    ///
    /// Returns `nil` if the string isn't a valid AURA scheme.
    public static func parse(_ s: String) -> AURAScheme? {
        let lower = s.lowercased()
        if lower == "aura" { return .default }
        guard lower.hasPrefix("aura") else { return nil }
        let rest = String(lower.dropFirst("aura".count))
        if let vIdx = rest.firstIndex(of: "v") {
            let kPart = String(rest[..<vIdx])
            let vPart = String(rest[rest.index(after: vIdx)...])
            guard let kb = Int(kPart), let vb = Int(vPart),
                AURACodebook.supportedBits.contains(kb),
                AURACodebook.supportedBits.contains(vb)
            else { return nil }
            return AURAScheme(keyBits: kb, valueBits: vb)
        } else {
            guard let bits = Int(rest), AURACodebook.supportedBits.contains(bits)
            else { return nil }
            return AURAScheme(keyBits: bits, valueBits: bits)
        }
    }
}
