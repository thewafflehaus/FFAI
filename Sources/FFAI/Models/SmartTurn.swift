// SmartTurn — conversational endpoint / turn-detection model.
//
// SmartTurn (v3) decides whether a spoken utterance has reached a
// natural turn boundary. It is a Whisper-style audio encoder followed
// by attention pooling and a small MLP classifier emitting a single
// "turn complete" probability:
//
//   log-mel → conv1 (GELU) → conv2 (GELU, /2) → +pos_emb
//     → N transformer encoder layers (pre-LN) → final layer_norm
//     → attention pool → classifier MLP → sigmoid
//
// The model is ~8M params; like SileroVAD it runs on the CPU via
// `VADCompute` (see that file's header for the rationale). Unlike a
// per-frame VAD, SmartTurn emits one utterance-level probability, so it
// returns a `VADEndpointOutput` rather than a probability stream.
//
// Checkpoint: `mlx-community/smart-turn-v3`. Conv weights ship in MLX
// `[outC, K, inC]` layout; we transpose to PyTorch `[outC, inC, K]` at
// load time. Some checkpoints prefix keys with `inner.`; that prefix is
// stripped, and `pool_attention.N` / `classifier.N` are flattened to
// `pool_attention_N` / `classifier_N`.

import Foundation

// ─── Errors ──────────────────────────────────────────────────────────

public enum SmartTurnError: Error, CustomStringConvertible {
    case missingWeight(String)
    case invalidAudio(String)

    public var description: String {
        switch self {
        case .missingWeight(let w): return "SmartTurn: required weight missing: \(w)"
        case .invalidAudio(let m):  return "SmartTurn: \(m)"
        }
    }
}

// ─── Config ──────────────────────────────────────────────────────────

/// SmartTurn encoder + processor configuration.
public struct SmartTurnConfig: Sendable {
    // Encoder geometry.
    public let numMelBins: Int
    public let maxSourcePositions: Int
    public let dModel: Int
    public let encoderAttentionHeads: Int
    public let encoderLayers: Int
    public let encoderFfnDim: Int
    public let kProjBias: Bool
    // Audio processor.
    public let samplingRate: Int
    public let maxAudioSeconds: Int
    public let nFft: Int
    public let hopLength: Int
    public let normalizeAudio: Bool
    public let threshold: Float

    public init(numMelBins: Int = 80, maxSourcePositions: Int = 400,
                dModel: Int = 384, encoderAttentionHeads: Int = 6,
                encoderLayers: Int = 4, encoderFfnDim: Int = 1536,
                kProjBias: Bool = false,
                samplingRate: Int = 16000, maxAudioSeconds: Int = 8,
                nFft: Int = 400, hopLength: Int = 160,
                normalizeAudio: Bool = true, threshold: Float = 0.5) {
        self.numMelBins = numMelBins
        self.maxSourcePositions = maxSourcePositions
        self.dModel = dModel
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderLayers = encoderLayers
        self.encoderFfnDim = encoderFfnDim
        self.kProjBias = kProjBias
        self.samplingRate = samplingRate
        self.maxAudioSeconds = maxAudioSeconds
        self.nFft = nFft
        self.hopLength = hopLength
        self.normalizeAudio = normalizeAudio
        self.threshold = threshold
    }

    /// Decode from a HuggingFace `config.json` dictionary. Reads the
    /// nested `encoder_config` / `processor_config` blocks if present,
    /// else top-level keys, else published defaults.
    public static func decode(from raw: [String: Any]) -> SmartTurnConfig {
        let enc = (raw["encoder_config"] as? [String: Any]) ?? raw
        let proc = (raw["processor_config"] as? [String: Any]) ?? raw
        func i(_ d: [String: Any], _ k: String, _ fb: Int) -> Int { (d[k] as? Int) ?? fb }
        func b(_ d: [String: Any], _ k: String, _ fb: Bool) -> Bool { (d[k] as? Bool) ?? fb }
        func f(_ d: [String: Any], _ k: String, _ fb: Float) -> Float {
            (d[k] as? NSNumber)?.floatValue ?? fb
        }
        return SmartTurnConfig(
            numMelBins: i(enc, "num_mel_bins", 80),
            maxSourcePositions: i(enc, "max_source_positions", 400),
            dModel: i(enc, "d_model", 384),
            encoderAttentionHeads: i(enc, "encoder_attention_heads", 6),
            encoderLayers: i(enc, "encoder_layers", 4),
            encoderFfnDim: i(enc, "encoder_ffn_dim", 1536),
            kProjBias: b(enc, "k_proj_bias", false),
            samplingRate: i(proc, "sampling_rate", 16000),
            maxAudioSeconds: i(proc, "max_audio_seconds", 8),
            nFft: i(proc, "n_fft", 400),
            hopLength: i(proc, "hop_length", 160),
            normalizeAudio: b(proc, "normalize_audio", true),
            threshold: f(proc, "threshold", 0.5))
    }
}

// ─── One Whisper-style encoder layer ─────────────────────────────────

/// Pre-LN transformer encoder layer (Whisper convention):
/// `x + Attn(LN(x))`, then `x + FC2(GELU(FC1(LN(x))))`.
struct SmartTurnEncoderLayer {
    let selfAttnLayerNorm: VADLayerNorm
    let qProj, kProj, vProj, outProj: VADLinear
    let finalLayerNorm: VADLayerNorm
    let fc1, fc2: VADLinear
    let numHeads: Int
    let headDim: Int

    /// Run the layer over a `[seqLen, dModel]` sequence.
    func forward(_ x: [Float], seqLen: Int, dModel: Int) -> [Float] {
        // Self-attention sub-block.
        let normed = selfAttnLayerNorm.applyRows(x, rows: seqLen)
        let q = qProj.applyRows(normed, rows: seqLen)
        let k = kProj.applyRows(normed, rows: seqLen)
        let v = vProj.applyRows(normed, rows: seqLen)
        // Whisper scales by sqrt(headDim) (divides QKᵀ).
        let scale = Float(headDim).squareRoot()
        let attn = vadMultiHeadAttention(
            q: q, k: k, v: v, seqLen: seqLen,
            numHeads: numHeads, headDim: headDim, scale: scale)
        let attnOut = outProj.applyRows(attn, rows: seqLen)
        var h = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { h[i] = x[i] + attnOut[i] }

        // Feed-forward sub-block.
        let normed2 = finalLayerNorm.applyRows(h, rows: seqLen)
        let mid = VADMath.gelu(fc1.applyRows(normed2, rows: seqLen))
        let ff = fc2.applyRows(mid, rows: seqLen)
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { out[i] = h[i] + ff[i] }
        return out
    }
}

// ─── SmartTurn model ─────────────────────────────────────────────────

/// Loaded SmartTurn model. Audio-in / endpoint-probability-out — a VAD
/// family, reached via `VADModelRegistry`, not `ModelRegistry`.
public final class SmartTurnModel: @unchecked Sendable {
    public let config: SmartTurnConfig

    // Encoder front-end convs.
    let conv1: VADConv1d   // [numMelBins → dModel], K=3, pad=1
    let conv2: VADConv1d   // [dModel → dModel], K=3, stride=2, pad=1
    let positionEmbedding: [Float]   // [maxSourcePositions, dModel]
    let layers: [SmartTurnEncoderLayer]
    let encoderLayerNorm: VADLayerNorm
    // Attention pooling + classifier head.
    let poolAttn0: VADLinear   // [dModel → 256]
    let poolAttn2: VADLinear   // [256 → 1]
    let classifier0: VADLinear // [dModel → 256]
    let classifier1: VADLayerNorm // over 256
    let classifier4: VADLinear // [256 → 64]
    let classifier6: VADLinear // [64 → 1]
    // Cached mel front-end.
    let melWindow: [Float]
    let melFilterbank: [Float]

    init(config: SmartTurnConfig,
         conv1: VADConv1d, conv2: VADConv1d, positionEmbedding: [Float],
         layers: [SmartTurnEncoderLayer], encoderLayerNorm: VADLayerNorm,
         poolAttn0: VADLinear, poolAttn2: VADLinear,
         classifier0: VADLinear, classifier1: VADLayerNorm,
         classifier4: VADLinear, classifier6: VADLinear) {
        self.config = config
        self.conv1 = conv1
        self.conv2 = conv2
        self.positionEmbedding = positionEmbedding
        self.layers = layers
        self.encoderLayerNorm = encoderLayerNorm
        self.poolAttn0 = poolAttn0
        self.poolAttn2 = poolAttn2
        self.classifier0 = classifier0
        self.classifier1 = classifier1
        self.classifier4 = classifier4
        self.classifier6 = classifier6
        // Whisper window: nFft samples; center-padded if needed.
        self.melWindow = VADAudioFrontend.hannWindow(size: config.nFft)
        self.melFilterbank = VADAudioFrontend.melFilterbank(
            sampleRate: config.samplingRate, nFft: config.nFft, nMels: config.numMelBins)
    }

    // ─── Audio → log-mel features ────────────────────────────────────

    /// Build the `[numMelBins, targetFrames]` log-mel feature block the
    /// encoder expects, matching mlx-audio-swift's
    /// `prepareInputFeatures`: power STFT → mel → log10 → max-clamp →
    /// `(x+4)/4` rescale, then trim/left-pad to a fixed frame count.
    func melFeatures(audio: [Float]) -> (values: [Float], numMels: Int, numFrames: Int) {
        // Pad / trim to maxAudioSeconds, then optional normalize.
        var samples = audio
        let maxSamples = config.maxAudioSeconds * config.samplingRate
        if samples.count > maxSamples {
            samples = Array(samples[(samples.count - maxSamples)...])
        } else if samples.count < maxSamples {
            samples = [Float](repeating: 0, count: maxSamples - samples.count) + samples
        }
        if config.normalizeAudio, !samples.isEmpty {
            let mean = samples.reduce(0, +) / Float(samples.count)
            var variance: Float = 0
            for x in samples { let d = x - mean; variance += d * d }
            variance /= Float(samples.count)
            let std = max(variance.squareRoot(), 1e-7)
            for i in samples.indices { samples[i] = (samples[i] - mean) / std }
        }

        let (power, numFrames, nBins) = VADAudioFrontend.powerSpectrogram(
            samples, window: melWindow, nFft: config.nFft, hopLength: config.hopLength)
        var mel = VADAudioFrontend.applyMelFilterbank(
            power: power, numFrames: numFrames, nBins: nBins,
            filterbank: melFilterbank, nMels: config.numMelBins)

        // log10 with floor, max-clamp, (x+4)/4 — Whisper-style.
        var maxVal = -Float.greatestFiniteMagnitude
        for i in mel.indices {
            mel[i] = log10f(max(mel[i], 1e-10))
            if mel[i] > maxVal { maxVal = mel[i] }
        }
        let floor = maxVal - 8
        for i in mel.indices {
            mel[i] = (max(mel[i], floor) + 4) / 4
        }

        // mel is [numFrames, numMels]; trim/left-pad frames to the
        // target so the encoder sees a fixed length.
        let targetFrames = config.maxAudioSeconds * config.samplingRate / config.hopLength
        let nMels = config.numMelBins
        var framed = mel
        var frames = numFrames
        if frames > targetFrames {
            framed = Array(mel[((frames - targetFrames) * nMels)...])
            frames = targetFrames
        } else if frames < targetFrames {
            let padFrames = targetFrames - frames
            framed = [Float](repeating: 0, count: padFrames * nMels) + mel
            frames = targetFrames
        }
        return (framed, nMels, frames)
    }

    // ─── Forward ─────────────────────────────────────────────────────

    /// Run the encoder + classifier and return the raw "turn complete"
    /// probability in `[0, 1]`.
    func probability(forMelFeatures mel: [Float], numMels: Int, numFrames: Int) -> Float {
        // Encoder expects channel-major [numMels, numFrames]; `mel` is
        // frame-major [numFrames, numMels]. Transpose.
        var chMajor = [Float](repeating: 0, count: numMels * numFrames)
        for f in 0..<numFrames {
            for m in 0..<numMels {
                chMajor[m * numFrames + f] = mel[f * numMels + m]
            }
        }

        // conv1 (GELU) — [numMels → dModel], K=3, pad=1, stride=1.
        var (h, hLen) = conv1.apply(chMajor, inLength: numFrames)
        h = VADMath.gelu(h)
        // conv2 (GELU) — [dModel → dModel], K=3, stride=2, pad=1.
        (h, hLen) = conv2.apply(h, inLength: hLen)
        h = VADMath.gelu(h)

        let dModel = config.dModel
        // h is channel-major [dModel, hLen]; transpose to seq-major
        // [hLen, dModel] and add positional embeddings. The position
        // table has `maxSourcePositions` rows; clamp longer sequences.
        var x = [Float](repeating: 0, count: hLen * dModel)
        for t in 0..<hLen {
            let hasPos = t < config.maxSourcePositions
            for d in 0..<dModel {
                let pos = hasPos ? positionEmbedding[t * dModel + d] : 0
                x[t * dModel + d] = h[d * hLen + t] + pos
            }
        }

        // Transformer encoder stack.
        for layer in layers {
            x = layer.forward(x, seqLen: hLen, dModel: dModel)
        }
        x = encoderLayerNorm.applyRows(x, rows: hLen)

        // Attention pooling: scores = poolAttn2(tanh(poolAttn0(x))),
        // softmax over time, weighted sum.
        let pa0 = VADMath.tanhActivation(poolAttn0.applyRows(x, rows: hLen))
        let pa2 = poolAttn2.applyRows(pa0, rows: hLen)   // [hLen, 1]
        var weights = pa2
        VADMath.softmaxInPlace(&weights, range: 0..<hLen)
        var pooled = [Float](repeating: 0, count: dModel)
        for t in 0..<hLen {
            let w = weights[t]
            let base = t * dModel
            for d in 0..<dModel { pooled[d] += w * x[base + d] }
        }

        // Classifier MLP: Linear → LayerNorm → GELU → Linear → GELU →
        // Linear → sigmoid.
        var c = classifier0.apply(pooled)
        c = classifier1.apply(c)
        c = VADMath.gelu(c)
        c = classifier4.apply(c)
        c = VADMath.gelu(c)
        let logit = classifier6.apply(c)[0]
        return VADMath.sigmoid(logit)
    }

    /// Predict whether `audio` reached a conversational endpoint.
    public func predictEndpoint(audio: [Float],
                                threshold: Float? = nil) -> VADEndpointOutput {
        let (mel, nMels, nFrames) = melFeatures(audio: audio)
        let prob = probability(forMelFeatures: mel, numMels: nMels, numFrames: nFrames)
        let thr = threshold ?? config.threshold
        return VADEndpointOutput(probability: prob, prediction: prob > thr ? 1 : 0)
    }

    // ─── Loading ─────────────────────────────────────────────────────

    /// Normalize a raw checkpoint key to the flat name this loader uses:
    /// strip a leading `inner.`, flatten `pool_attention.N` /
    /// `classifier.N` to underscored names, drop `val_*` tensors.
    static func remap(_ key: String) -> String? {
        if key.hasPrefix("val_") { return nil }
        var out = key
        if out.hasPrefix("inner.") { out.removeFirst("inner.".count) }
        for n in 0...6 {
            out = out.replacingOccurrences(of: "pool_attention.\(n).",
                                           with: "pool_attention_\(n).")
            out = out.replacingOccurrences(of: "classifier.\(n).",
                                           with: "classifier_\(n).")
        }
        return out
    }

    /// Load a SmartTurn checkpoint from a local snapshot directory.
    public static func loadFromDirectory(_ directory: URL,
                                         device: Device = .shared) throws -> SmartTurnModel {
        var config = SmartTurnConfig()
        let configURL = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let raw = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            config = SmartTurnConfig.decode(from: raw)
        }

        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        // Build remapped name → (tensor, shape) table.
        var table: [String: (floats: [Float], shape: [Int])] = [:]
        for key in bundle.allKeys {
            guard let mapped = remap(key) else { continue }
            let t = try bundle.tensor(named: key)
            table[mapped] = (t.toFloatArray(), t.shape)
        }
        func get(_ name: String) throws -> [Float] {
            guard let e = table[name] else { throw SmartTurnError.missingWeight(name) }
            return e.floats
        }
        func getOpt(_ name: String) -> [Float]? { table[name]?.floats }
        func shape(_ name: String) -> [Int]? { table[name]?.shape }

        let d = config.dModel
        let nMels = config.numMelBins

        // Conv weights: checkpoints ship MLX `[outC, K, inC]`; transpose
        // to PyTorch `[outC, inC, K]` that VADConv1d expects.
        func convWeight(_ name: String, outC: Int, inC: Int, k: Int) throws -> [Float] {
            let raw = try get(name)
            precondition(raw.count == outC * inC * k, "SmartTurn: \(name) count mismatch")
            // Detect layout from stored shape: if [outC, K, inC] transpose,
            // if already [outC, inC, K] keep.
            if let s = shape(name), s.count == 3, s[1] == k, s[2] == inC {
                var out = [Float](repeating: 0, count: raw.count)
                for o in 0..<outC {
                    for kk in 0..<k {
                        for ic in 0..<inC {
                            // src [o, kk, ic] → dst [o, ic, kk]
                            out[(o * inC + ic) * k + kk] = raw[(o * k + kk) * inC + ic]
                        }
                    }
                }
                return out
            }
            return raw
        }

        let conv1 = VADConv1d(
            weight: try convWeight("encoder.conv1.weight", outC: d, inC: nMels, k: 3),
            bias: getOpt("encoder.conv1.bias"),
            inChannels: nMels, outChannels: d, kernelSize: 3, padding: 1)
        let conv2 = VADConv1d(
            weight: try convWeight("encoder.conv2.weight", outC: d, inC: d, k: 3),
            bias: getOpt("encoder.conv2.bias"),
            inChannels: d, outChannels: d, kernelSize: 3, stride: 2, padding: 1)

        let posEmb = try get("encoder.embed_positions.weight")

        // Linear weights ship as PyTorch `[out, in]` — matches VADLinear.
        func linear(_ base: String, inF: Int, outF: Int, requireBias: Bool = true) throws -> VADLinear {
            let w = try get("\(base).weight")
            // Some fc weights may ship transposed; detect by shape.
            var weight = w
            if let s = shape("\(base).weight"), s.count == 2, s[0] == inF, s[1] == outF, inF != outF {
                // Stored [in, out] → transpose to [out, in].
                var t = [Float](repeating: 0, count: w.count)
                for o in 0..<outF { for i in 0..<inF { t[o * inF + i] = w[i * outF + o] } }
                weight = t
            }
            return VADLinear(weight: weight,
                             bias: requireBias ? try get("\(base).bias") : getOpt("\(base).bias"),
                             inFeatures: inF, outFeatures: outF)
        }
        func layerNorm(_ base: String, dim: Int) throws -> VADLayerNorm {
            VADLayerNorm(weight: try get("\(base).weight"), bias: try get("\(base).bias"), dim: dim)
        }

        let heads = config.encoderAttentionHeads
        let headDim = d / heads
        var layers: [SmartTurnEncoderLayer] = []
        layers.reserveCapacity(config.encoderLayers)
        for i in 0..<config.encoderLayers {
            let p = "encoder.layers.\(i)"
            layers.append(SmartTurnEncoderLayer(
                selfAttnLayerNorm: try layerNorm("\(p).self_attn_layer_norm", dim: d),
                qProj: try linear("\(p).self_attn.q_proj", inF: d, outF: d),
                kProj: try linear("\(p).self_attn.k_proj", inF: d, outF: d,
                                  requireBias: config.kProjBias),
                vProj: try linear("\(p).self_attn.v_proj", inF: d, outF: d),
                outProj: try linear("\(p).self_attn.out_proj", inF: d, outF: d),
                finalLayerNorm: try layerNorm("\(p).final_layer_norm", dim: d),
                fc1: try linear("\(p).fc1", inF: d, outF: config.encoderFfnDim),
                fc2: try linear("\(p).fc2", inF: config.encoderFfnDim, outF: d),
                numHeads: heads, headDim: headDim))
        }

        return SmartTurnModel(
            config: config, conv1: conv1, conv2: conv2,
            positionEmbedding: posEmb, layers: layers,
            encoderLayerNorm: try layerNorm("encoder.layer_norm", dim: d),
            poolAttn0: try linear("pool_attention_0", inF: d, outF: 256),
            poolAttn2: try linear("pool_attention_2", inF: 256, outF: 1),
            classifier0: try linear("classifier_0", inF: d, outF: 256),
            classifier1: try layerNorm("classifier_1", dim: 256),
            classifier4: try linear("classifier_4", inF: 256, outF: 64),
            classifier6: try linear("classifier_6", inF: 64, outF: 1))
    }

    /// Download (or hit cache) a SmartTurn checkpoint and load it.
    public static func fromPretrained(_ idOrPath: String,
                                      device: Device = .shared) async throws -> SmartTurnModel {
        let dir = try await ModelLocator().resolve(idOrPath: idOrPath)
        return try loadFromDirectory(dir, device: device)
    }
}
