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
// GPU correctness / smoke tests for `Ops.*` wrappers that don't fit
// neatly into the elementwise / SDPA / dequant suites: blit/cast/fused
// activations, KV cache append + round-trip, GDN/Mamba prep+chunk +
// fused mixer norms, SDPA prefill-MMA, MoE BM=8 / scalar M=1 variants,
// unpermute, dynamic-M dequant GEMM.
//
// Each test follows the canonical OpsTests.swift pattern:
//   autoreleasepool { … runAndWait { cb in Ops.foo(…, on: cb) } … }
//
// For ops with an obvious cheap CPU reference we assert numerical
// correctness; for ops with deep multi-tensor shape contracts we
// allocate production-realistic inputs and assert the output buffer
// has the right element count plus all-finite (no NaN / Inf) values,
// which still exercises the kernel dispatch path end-to-end.

import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

@Suite("Ops — special-path wrappers")
struct OpsSpecialPathTests {

    // MARK: - Element-wise blit / cast / fused

    @Test("copy f32 — blit duplicates src element-for-element")
    func copyF32() {
        autoreleasepool {
            let src = Tensor.empty(shape: [6], dtype: .f32)
            src.copyIn(from: [Float(1), -2, 3, -4, 5, -6])
            let dst = Tensor.empty(shape: [6], dtype: .f32)
            dst.zero()
            runAndWait { cb in Ops.copy(src, into: dst, on: cb) }
            #expect(dst.toArray(as: Float.self) == [1, -2, 3, -4, 5, -6])
        }
    }

    @Test("castToF32 f16 — promotes half-precision values to fp32")
    func castToF32FromF16() {
        autoreleasepool {
            let src = Tensor.empty(shape: [4], dtype: .f16)
            src.copyIn(from: [Float16(0.5), -1.25, 2, 0])
            let dst = Tensor.empty(shape: [4], dtype: .f32)
            dst.zero()
            runAndWait { cb in Ops.castToF32(src, into: dst, on: cb) }
            let got = dst.toArray(as: Float.self)
            // f16 representable exactly for these values.
            #expect(abs(got[0] - 0.5) < 1e-6)
            #expect(abs(got[1] - -1.25) < 1e-6)
            #expect(abs(got[2] - 2.0) < 1e-6)
            #expect(abs(got[3]) < 1e-6)
        }
    }

    @Test("siluCastToF32 f16 — fused silu + promotion matches silu(x)")
    func siluCastToF32FromF16() {
        autoreleasepool {
            let src = Tensor.empty(shape: [4], dtype: .f16)
            src.copyIn(from: [Float16(0), 1, -1, 2])
            let dst = Tensor.empty(shape: [4], dtype: .f32)
            dst.zero()
            runAndWait { cb in Ops.siluCastToF32(src, into: dst, on: cb) }
            let r = dst.toArray(as: Float.self)
            // silu(x) = x * sigmoid(x). bf16/f16 input narrows accuracy.
            #expect(abs(r[0]) < 1e-3)
            #expect(abs(r[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 5e-3)
            #expect(abs(r[2] - Float(-1.0 / (1.0 + exp(1.0)))) < 5e-3)
            #expect(abs(r[3] - Float(2.0 / (1.0 + exp(-2.0)))) < 5e-3)
        }
    }

    @Test("swiglu f32 — out[i] = silu(gate[i]) * up[i]")
    func swigluF32() {
        autoreleasepool {
            let gate = Tensor.empty(shape: [4], dtype: .f32)
            let up = Tensor.empty(shape: [4], dtype: .f32)
            gate.copyIn(from: [Float(0), 1, -1, 2])
            up.copyIn(from: [Float(3), 5, 7, 11])
            var out: Tensor!
            runAndWait { cb in out = Ops.swiglu(gate: gate, up: up, on: cb) }
            let r = out.toArray(as: Float.self)
            // silu(0)=0 → 0 * 3 = 0; silu(1) ≈ 0.7311 → * 5 ≈ 3.656; etc.
            #expect(abs(r[0]) < 1e-5)
            #expect(abs(r[1] - 5 * Float(1.0 / (1.0 + exp(-1.0)))) < 1e-3)
            #expect(abs(r[2] - 7 * Float(-1.0 / (1.0 + exp(1.0)))) < 1e-3)
            #expect(abs(r[3] - 11 * Float(2.0 / (1.0 + exp(-2.0)))) < 1e-3)
        }
    }

    @Test("sigmoidScalarFMA f32 — out = base + sigmoid(gate) * value")
    func sigmoidScalarFMAF32() {
        autoreleasepool {
            let gate = Tensor.empty(shape: [1], dtype: .f32)
            gate.copyIn(from: [Float(0)])  // sigmoid(0) = 0.5
            let value = Tensor.empty(shape: [4], dtype: .f32)
            value.copyIn(from: [Float(2), 4, 6, 8])
            let base = Tensor.empty(shape: [4], dtype: .f32)
            base.copyIn(from: [Float(10), 20, 30, 40])
            let out = Tensor.empty(shape: [4], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.sigmoidScalarFMA(gate: gate, value: value, base: base,
                                     into: out, on: cb)
            }
            // sigmoid(0) = 0.5 → out[i] = base[i] + 0.5 * value[i]
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 11) < 1e-4)
            #expect(abs(r[1] - 22) < 1e-4)
            #expect(abs(r[2] - 33) < 1e-4)
            #expect(abs(r[3] - 44) < 1e-4)
        }
    }

    // MARK: - KV cache append

    @Test("kvCacheUpdate f32 — writes one row into [nKV, maxSeq, headDim]")
    func kvCacheUpdateF32() {
        autoreleasepool {
            let nKV = 2, headDim = 4, maxSeq = 3
            let src = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            src.copyIn(from: [Float(1), 2, 3, 4,   // head 0
                              5, 6, 7, 8])         // head 1
            let cache = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            cache.zero()
            runAndWait { cb in
                Ops.kvCacheUpdate(src: src, into: cache,
                                  nKVHeads: nKV, headDim: headDim,
                                  maxSeq: maxSeq, position: 1, on: cb)
            }
            // Expect head 0 row 1 = [1,2,3,4], head 1 row 1 = [5,6,7,8],
            // all other slots still zero.
            let got = cache.toArray(as: Float.self)
            // head 0
            #expect(got[0..<4] == [0, 0, 0, 0])           // row 0
            #expect(Array(got[4..<8]) == [1, 2, 3, 4])    // row 1
            #expect(got[8..<12] == [0, 0, 0, 0])          // row 2
            // head 1
            #expect(got[12..<16] == [0, 0, 0, 0])         // row 0
            #expect(Array(got[16..<20]) == [5, 6, 7, 8])  // row 1
            #expect(got[20..<24] == [0, 0, 0, 0])         // row 2
        }
    }

    // MARK: - KV quantization round-trips
    //
    // For each precision we run quantize → bulk-dequant on the same row
    // and expect the recovered value to be close to the original
    // (affine quantization is approximate, so we use coarse tolerances).
    // Each round-trip exercises both the encode and decode kernel.

    @Test("quantizeKVInt8 + bulkDequantKVInt8 — round-trip recovers input")
    func quantizeBulkDequantKVInt8RoundTrip() {
        autoreleasepool {
            let nKV = 1, headDim = 32, maxSeq = 4, groupSize = 32
            // src has known per-element values; one row only is written.
            let srcVals: [Float] = (0..<headDim).map { Float($0) * 0.05 - 0.7 }
            let src = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            src.copyIn(from: srcVals)
            // int8 packs 4 values per uint32.
            let packs = headDim / 4
            let groups = headDim / groupSize
            let w = Tensor.empty(shape: [nKV, maxSeq, packs], dtype: .u32)
            let s = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            let b = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            w.zero(); s.zero(); b.zero()

            let pos = 2
            runAndWait { cb in
                Ops.quantizeKVInt8(src: src,
                                   weights: w, scales: s, biases: b,
                                   nKVHeads: nKV, headDim: headDim,
                                   maxSeq: maxSeq, groupSize: groupSize,
                                   position: pos, on: cb)
            }
            // Bulk-dequant into a working buffer of the same layout as
            // the cache (`[nKV, maxSeq, headDim]`) for `pos+1` positions
            // so the written slot is included.
            let working = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            working.zero()
            runAndWait { cb in
                Ops.bulkDequantKVInt8(weights: w, scales: s, biases: b,
                                      into: working,
                                      nKVHeads: nKV, headDim: headDim,
                                      maxSeq: maxSeq, groupSize: groupSize,
                                      nPositions: pos + 1, on: cb)
            }
            let got = working.toArray(as: Float.self)
            // Recovered slice for (head 0, pos 2) lives at offset
            // `pos * headDim`.
            for i in 0..<headDim {
                let want = srcVals[i]
                let recovered = got[pos * headDim + i]
                // int8 affine quant ≈ src/255 ≈ 0.005 abs tolerance.
                #expect(abs(recovered - want) < 0.01,
                        "i=\(i): got \(recovered) vs \(want)")
            }
        }
    }

    @Test("quantizeKVInt4 + bulkDequantKVInt4 — round-trip stays in tolerance")
    func quantizeBulkDequantKVInt4RoundTrip() {
        autoreleasepool {
            let nKV = 1, headDim = 32, maxSeq = 4, groupSize = 32
            let srcVals: [Float] = (0..<headDim).map { Float($0) * 0.05 - 0.7 }
            let src = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            src.copyIn(from: srcVals)
            // int4 packs 8 values per uint32.
            let packs = headDim / 8
            let groups = headDim / groupSize
            let w = Tensor.empty(shape: [nKV, maxSeq, packs], dtype: .u32)
            let s = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            let b = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            w.zero(); s.zero(); b.zero()

            let pos = 1
            runAndWait { cb in
                Ops.quantizeKVInt4(src: src,
                                   weights: w, scales: s, biases: b,
                                   nKVHeads: nKV, headDim: headDim,
                                   maxSeq: maxSeq, groupSize: groupSize,
                                   position: pos, on: cb)
            }
            let working = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            working.zero()
            runAndWait { cb in
                Ops.bulkDequantKVInt4(weights: w, scales: s, biases: b,
                                      into: working,
                                      nKVHeads: nKV, headDim: headDim,
                                      maxSeq: maxSeq, groupSize: groupSize,
                                      nPositions: pos + 1, on: cb)
            }
            let got = working.toArray(as: Float.self)
            // int4 affine quant: range/15 ≈ 0.11 step → 0.06 abs tolerance.
            for i in 0..<headDim {
                let want = srcVals[i]
                let recovered = got[pos * headDim + i]
                #expect(abs(recovered - want) < 0.07,
                        "i=\(i): got \(recovered) vs \(want)")
            }
        }
    }

    @Test("quantizeKVAffine + bulkDequantKVAffine — bits=8 dispatch round-trips")
    func quantizeBulkDequantKVAffineBits8() {
        autoreleasepool {
            let nKV = 1, headDim = 32, maxSeq = 2, groupSize = 32
            let srcVals: [Float] = (0..<headDim).map { Float($0) * 0.04 - 0.5 }
            let src = Tensor.empty(shape: [nKV, headDim], dtype: .f32)
            src.copyIn(from: srcVals)
            let packs = headDim / 4
            let groups = headDim / groupSize
            let w = Tensor.empty(shape: [nKV, maxSeq, packs], dtype: .u32)
            let s = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            let b = Tensor.empty(shape: [nKV, maxSeq, groups], dtype: .f32)
            w.zero(); s.zero(); b.zero()

            runAndWait { cb in
                Ops.quantizeKVAffine(src: src,
                                     weights: w, scales: s, biases: b,
                                     nKVHeads: nKV, headDim: headDim,
                                     maxSeq: maxSeq, groupSize: groupSize,
                                     position: 0, bits: 8, on: cb)
            }
            let working = Tensor.empty(shape: [nKV, maxSeq, headDim], dtype: .f32)
            working.zero()
            runAndWait { cb in
                Ops.bulkDequantKVAffine(weights: w, scales: s, biases: b,
                                        into: working,
                                        nKVHeads: nKV, headDim: headDim,
                                        maxSeq: maxSeq, groupSize: groupSize,
                                        nPositions: 1, bits: 8, on: cb)
            }
            let got = working.toArray(as: Float.self)
            for i in 0..<headDim {
                #expect(abs(got[i] - srcVals[i]) < 0.01,
                        "i=\(i): got \(got[i]) vs \(srcVals[i])")
            }
        }
    }

    // MARK: - Fused mixer / GDN prep / chunk recurrence

    @Test("gatedMixerNorm f32 — exercises kernel without NaN / shape drift")
    func gatedMixerNormSmoke() {
        autoreleasepool {
            // Shapes pinned to the kernel's contract:
            //   y, z, out : [Hv, Dv] (Dv % 4 == 0)
            //   w         : [Dv]
            //   epsBuf    : [1] f32
            let hv = 2, dv = 8
            let y = Tensor.empty(shape: [hv, dv], dtype: .f32)
            let yVals: [Float] = (0..<(hv * dv)).map { Float($0) * 0.05 + 0.1 }
            y.copyIn(from: yVals)
            let z = Tensor.empty(shape: [hv, dv], dtype: .f32)
            z.copyIn(from: (0..<(hv * dv)).map { Float($0) * 0.03 - 0.2 })
            let weight = Tensor.empty(shape: [dv], dtype: .f32)
            weight.copyIn(from: (0..<dv).map { _ in Float(1) })
            let epsBuf = Tensor.empty(shape: [1], dtype: .f32)
            epsBuf.copyIn(from: [Float(1e-5)])
            let out = Tensor.empty(shape: [hv, dv], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.gatedMixerNorm(y: y, z: z, weight: weight, epsBuf: epsBuf,
                                   into: out,
                                   numValueHeads: hv, valueHeadDim: dv, on: cb)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == hv * dv)
            for v in got { #expect(v.isFinite, "non-finite output: \(v)") }
        }
    }

    @Test("gatedDeltaPrepStep f32 — exercises fused prep + recurrence dispatch")
    func gatedDeltaPrepStepSmoke() {
        autoreleasepool {
            // dk and dv must be multiples of 32; hv % hk == 0.
            let b = 1, dk = 32, dv = 32, hv = 2, hk = 1
            // convOut layout: [B, 2·Hk·Dk + Hv·Dv].
            let convOutLen = 2 * hk * dk + hv * dv
            let convOut = Tensor.empty(shape: [b, convOutLen], dtype: .f32)
            convOut.copyIn(from: (0..<convOutLen).map { Float($0) * 0.01 })
            let aLog = Tensor.empty(shape: [hv], dtype: .f32)
            aLog.copyIn(from: (0..<hv).map { _ in Float(-0.5) })
            let dtBias = Tensor.empty(shape: [hv], dtype: .f32)
            dtBias.copyIn(from: (0..<hv).map { _ in Float(0.0) })
            // aRaw / bRaw are [B, Hv] (or [Hv] for B=1, same buffer).
            let aRaw = Tensor.empty(shape: [hv], dtype: .f32)
            aRaw.copyIn(from: (0..<hv).map { _ in Float(0.1) })
            let bRaw = Tensor.empty(shape: [hv], dtype: .f32)
            bRaw.copyIn(from: (0..<hv).map { _ in Float(0.2) })
            let qNorm = Tensor.empty(shape: [hk * dk], dtype: .f32)
            qNorm.copyIn(from: (0..<(hk * dk)).map { _ in Float(1) })
            let kNorm = Tensor.empty(shape: [hk * dk], dtype: .f32)
            kNorm.copyIn(from: (0..<(hk * dk)).map { _ in Float(1) })
            // GDN state shape per GDNStateCache: [Hv, Dv, Dk].
            let stateIn = Tensor.empty(shape: [hv, dv, dk], dtype: .f32)
            stateIn.zero()
            let stateOut = Tensor.empty(shape: [hv, dv, dk], dtype: .f32)
            stateOut.zero()
            // y for B=1 is [Hv, Dv].
            let y = Tensor.empty(shape: [hv, dv], dtype: .f32)
            y.zero()
            runAndWait { cb in
                Ops.gatedDeltaPrepStep(
                    convOut: convOut, aLog: aLog, dtBias: dtBias,
                    aRaw: aRaw, bRaw: bRaw,
                    qNormWeight: qNorm, kNormWeight: kNorm,
                    stateIn: stateIn, stateOut: stateOut, y: y,
                    batchSize: b, dk: dk, dv: dv, hv: hv, hk: hk, on: cb)
            }
            let yVals = y.toArray(as: Float.self)
            #expect(yVals.count == hv * dv)
            for v in yVals { #expect(v.isFinite, "y has non-finite: \(v)") }
            let stateVals = stateOut.toArray(as: Float.self)
            for v in stateVals { #expect(v.isFinite, "state has non-finite: \(v)") }
        }
    }

    @Test("gatedDeltaChunk f32 — multi-token recurrence sweep stays finite")
    func gatedDeltaChunkSmoke() {
        // TODO: needs production-shape correctness reference — math
        // matches `mt_gated_delta_step` over `T` tokens but the
        // CPU oracle requires the full Gated Delta Net recurrence to be
        // reimplemented in Swift. Smoke-test: assert dispatch finishes
        // and outputs are finite / correctly sized.
        autoreleasepool {
            let tSteps = 2, hk = 1, hv = 1, dk = 32, dv = 32
            let q = Tensor.empty(shape: [tSteps, hk, dk], dtype: .f32)
            q.copyIn(from: (0..<(tSteps * hk * dk)).map { Float($0) * 0.01 })
            let k = Tensor.empty(shape: [tSteps, hk, dk], dtype: .f32)
            k.copyIn(from: (0..<(tSteps * hk * dk)).map { Float($0) * 0.02 })
            let v = Tensor.empty(shape: [tSteps, hv, dv], dtype: .f32)
            v.copyIn(from: (0..<(tSteps * hv * dv)).map { Float($0) * 0.03 })
            let g = Tensor.empty(shape: [tSteps, hv], dtype: .f32)
            g.copyIn(from: (0..<(tSteps * hv)).map { _ in Float(0.9) })
            let beta = Tensor.empty(shape: [tSteps, hv], dtype: .f32)
            beta.copyIn(from: (0..<(tSteps * hv)).map { _ in Float(0.3) })
            let stateIn = Tensor.empty(shape: [hv, dv, dk], dtype: .f32)
            stateIn.zero()
            let stateOut = Tensor.empty(shape: [hv, dv, dk], dtype: .f32)
            stateOut.zero()
            let y = Tensor.empty(shape: [tSteps, hv, dv], dtype: .f32)
            y.zero()
            let tLen = Tensor.empty(shape: [1], dtype: .u32)
            tLen.copyIn(from: [UInt32(tSteps)])
            runAndWait { cb in
                Ops.gatedDeltaChunk(
                    q: q, k: k, v: v, g: g, beta: beta,
                    stateIn: stateIn, into: y, stateOut: stateOut,
                    tLen: tLen,
                    numKeyHeads: hk, numValueHeads: hv,
                    keyHeadDim: dk, valueHeadDim: dv, on: cb)
            }
            let yVals = y.toArray(as: Float.self)
            #expect(yVals.count == tSteps * hv * dv)
            for value in yVals { #expect(value.isFinite, "y non-finite: \(value)") }
            let stateVals = stateOut.toArray(as: Float.self)
            for value in stateVals { #expect(value.isFinite, "state non-finite: \(value)") }
        }
    }

    // MARK: - SDPA prefill MMA

    @Test("sdpaPrefillMma f32 — dispatch over 32-aligned qLen produces finite out")
    func sdpaPrefillMmaSmoke() {
        // TODO: needs production-shape correctness reference — a CPU
        // softmax-attention oracle exists for sdpaDecode; we could
        // extend it to T queries but at the small shapes the metaltile
        // GPU correctness test already validates against. Smoke-test:
        // dispatch a 32-aligned qLen and assert the output is finite.
        autoreleasepool {
            let nQHeads = 4, nKVHeads = 2, headDim = 64, qLen = 32, kLen = 32
            let q = Tensor.empty(shape: [nQHeads, qLen, headDim], dtype: .f32)
            q.copyIn(from: (0..<(nQHeads * qLen * headDim))
                .map { Float($0 % 11) * 0.01 })
            let k = Tensor.empty(shape: [nKVHeads, kLen, headDim], dtype: .f32)
            k.copyIn(from: (0..<(nKVHeads * kLen * headDim))
                .map { Float($0 % 13) * 0.01 })
            let v = Tensor.empty(shape: [nKVHeads, kLen, headDim], dtype: .f32)
            v.copyIn(from: (0..<(nKVHeads * kLen * headDim))
                .map { Float($0 % 17) * 0.01 })
            let scale = 1.0 / Float(headDim).squareRoot()
            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaPrefillMma(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                    qLen: qLen, kLen: kLen, scale: scale, on: cb)
            }
            #expect(out.shape == [nQHeads, qLen, headDim])
            let got = out.toArray(as: Float.self)
            #expect(got.count == nQHeads * qLen * headDim)
            for v in got { #expect(v.isFinite, "non-finite: \(v)") }
        }
    }

    // MARK: - MoE batched-prefill variants

    @Test("moeGatherDequantGemmInt4Bm8 f32 — dispatch on BM=8 tile shape")
    func moeBm8Smoke() {
        // TODO: needs production-shape correctness reference — the
        // kernel's BM=8 tile is exercised at decode top-K; the BM=16
        // canonical path (moeGatherDequantGemmInt4) has a full
        // correctness test in MoEBgemmBm64MppTests.swift. Smoke-test
        // asserts the dispatch produces correctly-sized finite output.
        autoreleasepool {
            let nExperts = 2
            let mTotal = 8, nOut = 32, kIn = 32
            let groupSize = 32
            // weight packed: [nExperts, nOut, kIn/8] u32; one expert per
            // row is selected via `indices` (CSR offsets aren't used by
            // the BM=8 dispatcher — see Ops.moeGatherDequantGemmInt4Bm8).
            let packs = kIn / 8
            let weight = Tensor.empty(shape: [nExperts, nOut, packs], dtype: .u32)
            weight.zero()
            let groups = kIn / groupSize
            let scales = Tensor.empty(shape: [nExperts, nOut, groups], dtype: .f32)
            scales.copyIn(from: (0..<(nExperts * nOut * groups))
                .map { Float($0) * 0.01 + 0.1 })
            let biases = Tensor.empty(shape: [nExperts, nOut, groups], dtype: .f32)
            biases.copyIn(from: (0..<(nExperts * nOut * groups))
                .map { Float($0) * -0.005 })
            let indices = Tensor.empty(shape: [mTotal], dtype: .u32)
            indices.copyIn(from: (0..<mTotal).map { UInt32($0 % nExperts) })
            let input = Tensor.empty(shape: [mTotal, kIn], dtype: .f32)
            input.copyIn(from: (0..<(mTotal * kIn)).map { Float($0) * 0.001 })
            let out = Tensor.empty(shape: [mTotal, nOut], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.moeGatherDequantGemmInt4Bm8(
                    input: input, weight: weight, scales: scales, biases: biases,
                    indices: indices,
                    mTotal: mTotal, nOut: nOut, kIn: kIn, groupSize: groupSize,
                    on: cb, into: out)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == mTotal * nOut)
            for v in got { #expect(v.isFinite, "non-finite: \(v)") }
        }
    }

    @Test("moeGatherDequantGemmInt4M1 f32 — scalar T=1 dispatch finishes finite")
    func moeM1Smoke() {
        // TODO: needs production-shape correctness reference — the
        // canonical `moeGatherDequantGemmInt4` test in
        // MoEBgemmBm64MppTests.swift covers the cooperative path; this
        // scalar `m1` variant has the same math but a per-element
        // simd_sum reduction. Smoke-test asserts dispatch produces
        // correctly-sized finite output.
        autoreleasepool {
            let nExperts = 2
            let tRows = 1, mOut = 32, kIn = 32, groupSize = 32
            let packs = kIn / 8
            let weight = Tensor.empty(shape: [nExperts, mOut, packs], dtype: .u32)
            weight.zero()
            let groups = kIn / groupSize
            let scales = Tensor.empty(shape: [nExperts, mOut, groups], dtype: .f32)
            scales.copyIn(from: (0..<(nExperts * mOut * groups))
                .map { Float($0) * 0.01 + 0.1 })
            let biases = Tensor.empty(shape: [nExperts, mOut, groups], dtype: .f32)
            biases.copyIn(from: (0..<(nExperts * mOut * groups))
                .map { Float($0) * -0.005 })
            // CSR expertOffsets: tRows rows total mapped expert 0..0
            // (all rows route to expert 0). Layout: [n_experts + 1].
            let expertOffsets = Tensor.empty(shape: [nExperts + 1], dtype: .u32)
            expertOffsets.copyIn(from: [UInt32(0), UInt32(tRows), UInt32(tRows)])
            let x = Tensor.empty(shape: [tRows, kIn], dtype: .f32)
            x.copyIn(from: (0..<(tRows * kIn)).map { Float($0) * 0.01 })
            let out = Tensor.empty(shape: [tRows, mOut], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.moeGatherDequantGemmInt4M1(
                    x, weight, scales, biases, expertOffsets,
                    tRows, mOut, kIn, nExperts, groupSize,
                    cb, out)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == tRows * mOut)
            for v in got { #expect(v.isFinite, "non-finite: \(v)") }
        }
    }

    @Test("moeUnpermute f32 — weighted scatter-sum produces correct shape")
    func moeUnpermuteSmoke() {
        // TODO: needs production-shape correctness reference — the math
        // is a per-token gather + weighted sum across top-K expert
        // outputs, validated in production MoE forward tests. Smoke
        // here asserts dispatch produces correctly-sized output that
        // matches a hand-computed two-token case.
        autoreleasepool {
            let nRows = 2, hidden = 4, k = 2
            // expertOutputs: [nRows·k, hidden] — easy values per slot.
            let expertOutputs = Tensor.empty(shape: [nRows * k, hidden], dtype: .f32)
            expertOutputs.copyIn(from: [
                Float(1), 1, 1, 1,      // row 0 slot 0 → at pos 0
                Float(2), 2, 2, 2,      // row 0 slot 1 → at pos 1
                Float(3), 3, 3, 3,      // row 1 slot 0 → at pos 2
                Float(4), 4, 4, 4,      // row 1 slot 1 → at pos 3
            ])
            // Identity permutation: slot (row, k) lives at position
            // row*k + k.
            let invPerm = Tensor.empty(shape: [nRows, k], dtype: .u32)
            invPerm.copyIn(from: [UInt32(0), 1, 2, 3])
            // Equal-weight combine.
            let weights = Tensor.empty(shape: [nRows, k], dtype: .f32)
            weights.copyIn(from: [Float(0.5), 0.5, 0.5, 0.5])
            let out = Tensor.empty(shape: [nRows, hidden], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.moeUnpermute(
                    expertOutputs: expertOutputs,
                    invPerm: invPerm, topKWeights: weights,
                    into: out,
                    nRows: nRows, hidden: hidden, k: k, on: cb)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == nRows * hidden)
            // Expected: row 0 = 0.5*1 + 0.5*2 = 1.5; row 1 = 0.5*3 + 0.5*4 = 3.5
            for c in 0..<hidden {
                #expect(abs(got[c] - 1.5) < 1e-4, "row0 col\(c): \(got[c])")
                #expect(abs(got[hidden + c] - 3.5) < 1e-4, "row1 col\(c): \(got[hidden + c])")
            }
        }
    }

    // MARK: - Dynamic-M dequant GEMM

    @Test("dequantGemmDynamicM f32 — T=32 aligned fast path runs finite")
    func dequantGemmDynamicMSmoke() {
        // TODO: needs production-shape correctness reference — the
        // canonical 4-bit dequant + matmul oracle is exercised at the
        // GEMV scale in QuantizedOpsTests.swift; the dynamic-M kernel
        // shares the same dequant math but tiles across M. Smoke-test
        // dispatches the 32-aligned fast path and asserts output is
        // correctly sized and finite.
        autoreleasepool {
            let t = 32, nOut = 32, kIn = 32, groupSize = 32
            let packs = kIn / 8
            let weight = Tensor.empty(shape: [nOut, packs], dtype: .u32)
            // Non-zero quantized payload so the dequant path produces
            // varying outputs.
            weight.copyIn(from: (0..<(nOut * packs)).map { UInt32($0 + 1) })
            let groups = kIn / groupSize
            let scales = Tensor.empty(shape: [nOut, groups], dtype: .f32)
            scales.copyIn(from: (0..<(nOut * groups))
                .map { Float($0) * 0.01 + 0.05 })
            let biases = Tensor.empty(shape: [nOut, groups], dtype: .f32)
            biases.copyIn(from: (0..<(nOut * groups))
                .map { Float($0) * -0.005 })
            let input = Tensor.empty(shape: [t, kIn], dtype: .f32)
            input.copyIn(from: (0..<(t * kIn)).map { Float($0) * 0.001 })
            let out = Tensor.empty(shape: [t, nOut], dtype: .f32)
            out.zero()
            runAndWait { cb in
                Ops.dequantGemmDynamicM(
                    input: input,
                    weight: weight, scales: scales, biases: biases,
                    t: t, nOut: nOut, kIn: kIn, groupSize: groupSize,
                    on: cb, device: .shared, into: out)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == t * nOut)
            for v in got { #expect(v.isFinite, "non-finite: \(v)") }
        }
    }
}
