import Foundation
import Metal
import Testing
@testable import FFAI

// Vision Op wrappers — conv2d, patchEmbed, rope2D. Each kernel is a
// Grid3D one-thread-per-output kernel from metaltile-std/src/ffai/.
// Tests assert the GPU output against a CPU reference implementation at
// production-realistic shapes.
@Suite("OpsVision")
struct OpsVisionTests {

    // ─── conv2d ──────────────────────────────────────────────────────

    /// CPU reference for NCHW conv2d with OIHW weight + per-out-ch bias.
    private func cpuConv2d(
        input: [Float], weight: [Float], bias: [Float],
        batch: Int, inCh: Int, inH: Int, inW: Int,
        outCh: Int, kh: Int, kw: Int,
        strideH: Int, strideW: Int, padH: Int, padW: Int
    ) -> [Float] {
        let outH = (inH + 2 * padH - kh) / strideH + 1
        let outW = (inW + 2 * padW - kw) / strideW + 1
        var out = [Float](repeating: 0, count: batch * outCh * outH * outW)
        for n in 0..<batch {
            for oc in 0..<outCh {
                for oh in 0..<outH {
                    for ow in 0..<outW {
                        var acc = bias[oc]
                        for ic in 0..<inCh {
                            for ky in 0..<kh {
                                let ih = oh * strideH + ky - padH
                                guard ih >= 0 && ih < inH else { continue }
                                for kx in 0..<kw {
                                    let iw = ow * strideW + kx - padW
                                    guard iw >= 0 && iw < inW else { continue }
                                    let inIdx = ((n * inCh + ic) * inH + ih) * inW + iw
                                    let wIdx = ((oc * inCh + ic) * kh + ky) * kw + kx
                                    acc += input[inIdx] * weight[wIdx]
                                }
                            }
                        }
                        let oIdx = ((n * outCh + oc) * outH + oh) * outW + ow
                        out[oIdx] = acc
                    }
                }
            }
        }
        return out
    }

    @Test("conv2d f32 — 3-channel 8x8 image, patch16-style 2x2 stride 2")
    func conv2dF32() {
        autoreleasepool {
            let batch = 1, inCh = 3, inH = 8, inW = 8
            let outCh = 4, kh = 2, kw = 2, strideH = 2, strideW = 2

            var inputData = [Float](repeating: 0, count: batch * inCh * inH * inW)
            for i in inputData.indices { inputData[i] = Float(i % 7) * 0.1 - 0.3 }
            var weightData = [Float](repeating: 0, count: outCh * inCh * kh * kw)
            for i in weightData.indices { weightData[i] = Float((i % 5) - 2) * 0.07 }
            let biasData = (0..<outCh).map { Float($0) * 0.01 }

            let input = Tensor.empty(shape: [batch, inCh, inH, inW], dtype: .f32)
            input.copyIn(from: inputData)
            let weight = Tensor.empty(shape: [outCh, inCh, kh, kw], dtype: .f32)
            weight.copyIn(from: weightData)
            let bias = Tensor.empty(shape: [outCh], dtype: .f32)
            bias.copyIn(from: biasData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.conv2d(input: input, weight: weight, bias: bias,
                                 strideH: strideH, strideW: strideW, on: cb)
            }
            #expect(out.shape == [batch, outCh, 4, 4])
            let got = out.toArray(as: Float.self)
            let want = cpuConv2d(
                input: inputData, weight: weightData, bias: biasData,
                batch: batch, inCh: inCh, inH: inH, inW: inW,
                outCh: outCh, kh: kh, kw: kw,
                strideH: strideH, strideW: strideW, padH: 0, padW: 0)
            #expect(got.count == want.count)
            for (g, w) in zip(got, want) { #expect(abs(g - w) < 1e-3) }
        }
    }

    @Test("conv2d f16 — patch14-style projection, 3ch 28x28 -> 14x14")
    func conv2dF16Patch14() {
        autoreleasepool {
            let batch = 1, inCh = 3, inH = 28, inW = 28
            let outCh = 8, kh = 14, kw = 14, stride = 14

            var inputData = [Float](repeating: 0, count: batch * inCh * inH * inW)
            for i in inputData.indices { inputData[i] = Float(i % 11) * 0.03 - 0.15 }
            var weightData = [Float](repeating: 0, count: outCh * inCh * kh * kw)
            for i in weightData.indices { weightData[i] = Float((i % 9) - 4) * 0.01 }
            let biasData = (0..<outCh).map { Float($0) * 0.05 }

            let input = Tensor.empty(shape: [batch, inCh, inH, inW], dtype: .f16)
            input.copyIn(from: inputData.map { Float16($0) })
            let weight = Tensor.empty(shape: [outCh, inCh, kh, kw], dtype: .f16)
            weight.copyIn(from: weightData.map { Float16($0) })
            let bias = Tensor.empty(shape: [outCh], dtype: .f16)
            bias.copyIn(from: biasData.map { Float16($0) })

            var out: Tensor!
            runAndWait { cb in
                out = Ops.conv2d(input: input, weight: weight, bias: bias,
                                 strideH: stride, strideW: stride, on: cb)
            }
            #expect(out.shape == [batch, outCh, 2, 2])
            let got = out.toFloatArray()
            let want = cpuConv2d(
                input: inputData, weight: weightData, bias: biasData,
                batch: batch, inCh: inCh, inH: inH, inW: inW,
                outCh: outCh, kh: kh, kw: kw,
                strideH: stride, strideW: stride, padH: 0, padW: 0)
            for (g, w) in zip(got, want) { #expect(abs(g - w) < 0.2) }
        }
    }

    @Test("conv2d f32 — padded conv contributes zero on out-of-range reads")
    func conv2dF32Padded() {
        autoreleasepool {
            let batch = 1, inCh = 1, inH = 4, inW = 4
            let outCh = 1, kh = 3, kw = 3, stride = 1, pad = 1

            let inputData = (0..<16).map { Float($0) }
            let weightData = [Float](repeating: 1, count: 9)
            let biasData: [Float] = [0]

            let input = Tensor.empty(shape: [batch, inCh, inH, inW], dtype: .f32)
            input.copyIn(from: inputData)
            let weight = Tensor.empty(shape: [outCh, inCh, kh, kw], dtype: .f32)
            weight.copyIn(from: weightData)
            let bias = Tensor.empty(shape: [outCh], dtype: .f32)
            bias.copyIn(from: biasData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.conv2d(input: input, weight: weight, bias: bias,
                                 strideH: stride, strideW: stride,
                                 padH: pad, padW: pad, on: cb)
            }
            #expect(out.shape == [batch, outCh, 4, 4])
            let got = out.toArray(as: Float.self)
            let want = cpuConv2d(
                input: inputData, weight: weightData, bias: biasData,
                batch: batch, inCh: inCh, inH: inH, inW: inW,
                outCh: outCh, kh: kh, kw: kw,
                strideH: stride, strideW: stride, padH: pad, padW: pad)
            for (g, w) in zip(got, want) { #expect(abs(g - w) < 1e-3) }
        }
    }

    // ─── patch_embed ─────────────────────────────────────────────────

    /// CPU reference for the fused unfold + linear-projection patch embed.
    private func cpuPatchEmbed(
        image: [Float], weight: [Float], bias: [Float],
        inCh: Int, inH: Int, inW: Int, patchH: Int, patchW: Int, hidden: Int
    ) -> [Float] {
        let patchesH = inH / patchH
        let patchesW = inW / patchW
        let numPatches = patchesH * patchesW
        let patchDim = inCh * patchH * patchW
        var out = [Float](repeating: 0, count: numPatches * hidden)
        for patch in 0..<numPatches {
            let py0 = (patch / patchesW) * patchH
            let px0 = (patch % patchesW) * patchW
            for h in 0..<hidden {
                var acc = bias[h]
                for ic in 0..<inCh {
                    for py in 0..<patchH {
                        for px in 0..<patchW {
                            let imgIdx = (ic * inH + (py0 + py)) * inW + (px0 + px)
                            let wIdx = h * patchDim + ic * patchH * patchW + py * patchW + px
                            acc += image[imgIdx] * weight[wIdx]
                        }
                    }
                }
                out[patch * hidden + h] = acc
            }
        }
        return out
    }

    @Test("patchEmbed f32 — 3ch 16x16 image, 4x4 patches into hidden 12")
    func patchEmbedF32() {
        autoreleasepool {
            let inCh = 3, inH = 16, inW = 16, patchH = 4, patchW = 4, hidden = 12
            let patchDim = inCh * patchH * patchW

            var imageData = [Float](repeating: 0, count: inCh * inH * inW)
            for i in imageData.indices { imageData[i] = Float(i % 13) * 0.02 - 0.13 }
            var weightData = [Float](repeating: 0, count: hidden * patchDim)
            for i in weightData.indices { weightData[i] = Float((i % 7) - 3) * 0.015 }
            let biasData = (0..<hidden).map { Float($0) * 0.01 }

            let image = Tensor.empty(shape: [inCh, inH, inW], dtype: .f32)
            image.copyIn(from: imageData)
            let weight = Tensor.empty(shape: [hidden, patchDim], dtype: .f32)
            weight.copyIn(from: weightData)
            let bias = Tensor.empty(shape: [hidden], dtype: .f32)
            bias.copyIn(from: biasData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.patchEmbed(image: image, weight: weight, bias: bias,
                                     patchH: patchH, patchW: patchW, on: cb)
            }
            #expect(out.shape == [16, hidden])  // (16/4)^2 = 16 patches
            let got = out.toArray(as: Float.self)
            let want = cpuPatchEmbed(
                image: imageData, weight: weightData, bias: biasData,
                inCh: inCh, inH: inH, inW: inW,
                patchH: patchH, patchW: patchW, hidden: hidden)
            #expect(got.count == want.count)
            for (g, w) in zip(got, want) { #expect(abs(g - w) < 1e-3) }
        }
    }

    @Test("patchEmbed bf16 — patch14 ViT stem shape")
    func patchEmbedBf16() {
        autoreleasepool {
            let inCh = 3, inH = 28, inW = 42, patchH = 14, patchW = 14, hidden = 16
            let patchDim = inCh * patchH * patchW

            var imageData = [Float](repeating: 0, count: inCh * inH * inW)
            for i in imageData.indices { imageData[i] = Float(i % 17) * 0.01 - 0.08 }
            var weightData = [Float](repeating: 0, count: hidden * patchDim)
            for i in weightData.indices { weightData[i] = Float((i % 11) - 5) * 0.004 }
            let biasData = (0..<hidden).map { Float($0) * 0.02 }

            let image = Tensor.empty(shape: [inCh, inH, inW], dtype: .bf16)
            image.copyIn(from: imageData.map { bf16Bits($0) })
            let weight = Tensor.empty(shape: [hidden, patchDim], dtype: .bf16)
            weight.copyIn(from: weightData.map { bf16Bits($0) })
            let bias = Tensor.empty(shape: [hidden], dtype: .bf16)
            bias.copyIn(from: biasData.map { bf16Bits($0) })

            var out: Tensor!
            runAndWait { cb in
                out = Ops.patchEmbed(image: image, weight: weight, bias: bias,
                                     patchH: patchH, patchW: patchW, on: cb)
            }
            #expect(out.shape == [6, hidden])  // (28/14)*(42/14) = 2*3 = 6
            let got = out.toFloatArray()
            let want = cpuPatchEmbed(
                image: imageData, weight: weightData, bias: biasData,
                inCh: inCh, inH: inH, inW: inW,
                patchH: patchH, patchW: patchW, hidden: hidden)
            for (g, w) in zip(got, want) { #expect(abs(g - w) < 0.3) }
        }
    }

    // ─── rope_2d ─────────────────────────────────────────────────────

    /// CPU reference for 2D vision RoPE.
    private func cpuRope2D(
        qk: [Float], positions: [UInt32],
        nTokens: Int, nHeads: Int, headDim: Int, thetaBase: Float
    ) -> [Float] {
        let halfDim = headDim / 2
        let quarterDim = headDim / 4
        var out = qk
        for token in 0..<nTokens {
            let row = Float(positions[token * 2])
            let col = Float(positions[token * 2 + 1])
            for head in 0..<nHeads {
                let base = (token * nHeads + head) * headDim
                for j in 0..<quarterDim {
                    let invFreq = exp2(-2.0 * Float(j) * log2(thetaBase) / Float(halfDim))
                    let cosR = cos(row * invFreq), sinR = sin(row * invFreq)
                    let cosC = cos(col * invFreq), sinC = sin(col * invFreq)
                    let xr1 = qk[base + j], xr2 = qk[base + j + quarterDim]
                    out[base + j] = xr1 * cosR - xr2 * sinR
                    out[base + j + quarterDim] = xr1 * sinR + xr2 * cosR
                    let xc1 = qk[base + halfDim + j]
                    let xc2 = qk[base + halfDim + j + quarterDim]
                    out[base + halfDim + j] = xc1 * cosC - xc2 * sinC
                    out[base + halfDim + j + quarterDim] = xc1 * sinC + xc2 * cosC
                }
            }
        }
        return out
    }

    @Test("rope2D f32 — 6 tokens, 4 heads, head_dim 64")
    func rope2DF32() {
        autoreleasepool {
            let nTokens = 6, nHeads = 4, headDim = 64
            let thetaBase: Float = 10_000

            var qkData = [Float](repeating: 0, count: nTokens * nHeads * headDim)
            for i in qkData.indices { qkData[i] = Float(i % 19) * 0.05 - 0.45 }
            // (row, col) over a 2x3 patch grid.
            var positions = [UInt32]()
            for t in 0..<nTokens {
                positions.append(UInt32(t / 3))
                positions.append(UInt32(t % 3))
            }

            let qk = Tensor.empty(shape: [nTokens, nHeads, headDim], dtype: .f32)
            qk.copyIn(from: qkData)
            let pos = Tensor.empty(shape: [nTokens, 2], dtype: .u32)
            pos.copyIn(from: positions)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.rope2D(qk, positions: pos, nTokens: nTokens,
                                 nHeads: nHeads, headDim: headDim,
                                 thetaBase: thetaBase, on: cb)
            }
            #expect(out.shape == [nTokens, nHeads, headDim])
            let got = out.toArray(as: Float.self)
            let want = cpuRope2D(qk: qkData, positions: positions,
                                 nTokens: nTokens, nHeads: nHeads,
                                 headDim: headDim, thetaBase: thetaBase)
            #expect(got.count == want.count)
            for (g, w) in zip(got, want) { #expect(abs(g - w) < 1e-3) }
        }
    }

    @Test("rope2D f16 — coherent rotation at production head_dim 128")
    func rope2DF16() {
        autoreleasepool {
            let nTokens = 9, nHeads = 8, headDim = 128
            let thetaBase: Float = 10_000

            var qkData = [Float](repeating: 0, count: nTokens * nHeads * headDim)
            for i in qkData.indices { qkData[i] = Float(i % 23) * 0.02 - 0.22 }
            var positions = [UInt32]()
            for t in 0..<nTokens {
                positions.append(UInt32(t / 3))
                positions.append(UInt32(t % 3))
            }

            let qk = Tensor.empty(shape: [nTokens, nHeads, headDim], dtype: .f16)
            qk.copyIn(from: qkData.map { Float16($0) })
            let pos = Tensor.empty(shape: [nTokens, 2], dtype: .u32)
            pos.copyIn(from: positions)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.rope2D(qk, positions: pos, nTokens: nTokens,
                                 nHeads: nHeads, headDim: headDim,
                                 thetaBase: thetaBase, on: cb)
            }
            #expect(out.shape == [nTokens, nHeads, headDim])
            let got = out.toFloatArray()
            let want = cpuRope2D(qk: qkData, positions: positions,
                                 nTokens: nTokens, nHeads: nHeads,
                                 headDim: headDim, thetaBase: thetaBase)
            for (g, w) in zip(got, want) { #expect(abs(g - w) < 5e-2) }
        }
    }

    // ─── layer_norm ──────────────────────────────────────────────────

    @Test("layerNorm f32 — multi-row normalization with scale + shift")
    func layerNormF32() {
        autoreleasepool {
            let nRows = 3, rowSize = 8
            var xData = [Float](repeating: 0, count: nRows * rowSize)
            for i in xData.indices { xData[i] = Float(i % 5) * 0.5 - 1.0 }
            let weightData = (0..<rowSize).map { Float($0) * 0.1 + 1.0 }
            let biasData = (0..<rowSize).map { Float($0) * 0.05 }

            let x = Tensor.empty(shape: [nRows, rowSize], dtype: .f32)
            x.copyIn(from: xData)
            let weight = Tensor.empty(shape: [rowSize], dtype: .f32)
            weight.copyIn(from: weightData)
            let bias = Tensor.empty(shape: [rowSize], dtype: .f32)
            bias.copyIn(from: biasData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.layerNorm(x, weight: weight, bias: bias, eps: 1e-5,
                                    nRows: nRows, rowSize: rowSize, on: cb)
            }
            #expect(out.shape == [nRows, rowSize])
            let got = out.toArray(as: Float.self)
            // CPU reference.
            for r in 0..<nRows {
                let row = Array(xData[(r * rowSize)..<((r + 1) * rowSize)])
                let mean = row.reduce(0, +) / Float(rowSize)
                let varc = row.map { ($0 - mean) * ($0 - mean) }
                    .reduce(0, +) / Float(rowSize)
                let inv = 1 / (varc + 1e-5).squareRoot()
                for c in 0..<rowSize {
                    let want = (row[c] - mean) * inv * weightData[c] + biasData[c]
                    #expect(abs(got[r * rowSize + c] - want) < 1e-3)
                }
            }
        }
    }

    // ─── helpers ─────────────────────────────────────────────────────

    /// Round a Float to its bf16 bit pattern (top 16 bits of the f32).
    private func bf16Bits(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(rounded >> 16)
    }
}
