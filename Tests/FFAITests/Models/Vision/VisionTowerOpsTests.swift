// VisionTowerOps — unit tests for the module-internal weight-prep
// helpers shared by every dynamic-resolution vision tower (Qwen 2-VL /
// Qwen 2.5-VL / Qwen 3-VL / Gemma 4-VL / Nemotron-VL / Pixtral).
//
// All helpers are CPU-side load-time ops (bias broadcast + zero-pad +
// Conv3d → 2-D GEMM repack) — they touch tensors once at load, not in
// the hot forward path. The tests below exercise each helper on a
// hand-built input and assert the exact output (no GPU dispatch needed
// beyond the `Ops.add` inside `addRowBias`).

import Foundation
import Metal
import Testing

@testable import FFAI

// ─── Constants ────────────────────────────────────────────────────────

@Suite("VisionTowerOps Constants")
struct VisionTowerOpsConstantsTests {

    /// The K-tile width is the contract between caller-side patch-embed
    /// alignment and the `mt_gemm` kernel. 16 is the metaltile minimum;
    /// changing this silently breaks every dynamic-resolution tower, so
    /// pin it.
    @Test("gemmKTileWidth is 16")
    func kTileWidthPinned() {
        #expect(gemmKTileWidth == 16)
    }
}

// ─── Bias broadcast ───────────────────────────────────────────────────

@Suite("VisionTowerOps addRowBias")
struct VisionTowerOpsAddRowBiasTests {

    /// Broadcast-add a constant bias row across N tiled rows. Output[r,c]
    /// must equal `x[r,c] + bias[c]` for every cell.
    @Test("broadcasts [rowSize] bias across nRows rows")
    func broadcastsBias() {
        let device = Device.shared
        let nRows = 3, rowSize = 4
        // x = row-major [0..11].
        let xVals: [Float] = (0..<(nRows * rowSize)).map { Float($0) }
        let x = ImagePreprocessing.makeTensor(
            from: xVals, shape: [nRows, rowSize], dtype: .f32, device: device)
        // bias = [10, 20, 30, 40] — distinct per column so we can detect
        // column-order mistakes.
        let biasVals: [Float] = [10, 20, 30, 40]
        let bias = ImagePreprocessing.makeTensor(
            from: biasVals, shape: [rowSize], dtype: .f32, device: device)

        let cmd = device.makeCommandBuffer()
        let out = addRowBias(x, bias: bias, nRows: nRows,
                             rowSize: rowSize, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let got = out.toArray(as: Float.self)
        for r in 0..<nRows {
            for c in 0..<rowSize {
                let expected = xVals[r * rowSize + c] + biasVals[c]
                let actual = got[r * rowSize + c]
                #expect(abs(actual - expected) < 1e-5,
                        "(r=\(r), c=\(c)): expected \(expected), got \(actual)")
            }
        }
    }
}

// ─── Linear padding ───────────────────────────────────────────────────

@Suite("VisionTowerOps padLinearRows")
struct VisionTowerOpsPadRowsTests {

    /// outOld == toRows → returned `Linear` is the input identity (no
    /// pad work, no shape change, bias untouched).
    @Test("identity when outOld == toRows")
    func identityPassThrough() {
        let device = Device.shared
        let w = ImagePreprocessing.makeTensor(
            from: [1, 2, 3, 4], shape: [2, 2], dtype: .f32, device: device)
        let b = ImagePreprocessing.makeTensor(
            from: [10, 20], shape: [2], dtype: .f32, device: device)
        let lin = Linear(weight: w, bias: b)
        let out = padLinearRows(lin, toRows: 2, device: device)
        #expect(out.weight.shape == [2, 2])
        #expect(out.weight.toArray(as: Float.self) == [1, 2, 3, 4])
        #expect(out.bias?.toArray(as: Float.self) == [10, 20])
    }

    /// Pad `[outOld, inDim]` → `[toRows, inDim]` zero-extends the trailing
    /// rows (and the bias) with zero. Existing rows + bias entries are
    /// preserved bit-exact at their original indices.
    @Test("zero-extends rows and bias when toRows > outOld")
    func zeroExtendsRowsAndBias() {
        let device = Device.shared
        // weight = [[1, 2], [3, 4]] (outOld=2, inDim=2). Pad to outNew=4.
        let w = ImagePreprocessing.makeTensor(
            from: [1, 2, 3, 4], shape: [2, 2], dtype: .f32, device: device)
        let b = ImagePreprocessing.makeTensor(
            from: [10, 20], shape: [2], dtype: .f32, device: device)
        let lin = Linear(weight: w, bias: b)

        let out = padLinearRows(lin, toRows: 4, device: device)
        #expect(out.weight.shape == [4, 2])
        #expect(out.weight.toArray(as: Float.self) ==
                [1, 2, 3, 4, 0, 0, 0, 0])
        #expect(out.bias?.shape == [4])
        #expect(out.bias?.toArray(as: Float.self) == [10, 20, 0, 0])
    }
}

@Suite("VisionTowerOps padLinearCols")
struct VisionTowerOpsPadColsTests {

    /// inOld == toCols → returned `Linear` is the input identity.
    @Test("identity when inOld == toCols")
    func identityPassThrough() {
        let device = Device.shared
        let w = ImagePreprocessing.makeTensor(
            from: [1, 2, 3, 4], shape: [2, 2], dtype: .f32, device: device)
        let lin = Linear(weight: w, bias: nil)
        let out = padLinearCols(lin, toCols: 2, device: device)
        #expect(out.weight.shape == [2, 2])
        #expect(out.weight.toArray(as: Float.self) == [1, 2, 3, 4])
    }

    /// Pad `[outDim, inOld]` → `[outDim, toCols]` zero-extends the
    /// trailing columns. Existing columns stay at their original index;
    /// bias is unchanged (input-col padding contributes nothing to the
    /// dot product, so bias shape is the row count, not col count).
    @Test("zero-extends trailing columns; bias unchanged")
    func zeroExtendsCols() {
        let device = Device.shared
        // weight = [[1, 2], [3, 4]] (outDim=2, inOld=2). Pad to inNew=4.
        let w = ImagePreprocessing.makeTensor(
            from: [1, 2, 3, 4], shape: [2, 2], dtype: .f32, device: device)
        let b = ImagePreprocessing.makeTensor(
            from: [10, 20], shape: [2], dtype: .f32, device: device)
        let lin = Linear(weight: w, bias: b)

        let out = padLinearCols(lin, toCols: 4, device: device)
        #expect(out.weight.shape == [2, 4])
        // Row 0: 1, 2, 0, 0. Row 1: 3, 4, 0, 0.
        #expect(out.weight.toArray(as: Float.self) ==
                [1, 2, 0, 0, 3, 4, 0, 0])
        // Bias is the unmodified original (input-col pad doesn't touch it).
        #expect(out.bias?.shape == [2])
        #expect(out.bias?.toArray(as: Float.self) == [10, 20])
    }
}

@Suite("VisionTowerOps padLinearColsTo")
struct VisionTowerOpsPadColsTensorTests {

    /// inOld == toCols → returned tensor is the input (no work).
    @Test("identity when inOld == toCols")
    func identityPassThrough() {
        let device = Device.shared
        let w = ImagePreprocessing.makeTensor(
            from: [1, 2, 3, 4], shape: [2, 2], dtype: .f32, device: device)
        let out = padLinearColsTo(w, toCols: 2, device: device)
        #expect(out.shape == [2, 2])
        #expect(out.toArray(as: Float.self) == [1, 2, 3, 4])
    }

    /// Tensor-overload: same operation as the `Linear` form but on a raw
    /// weight tensor — used by Gemma 4-VL's patch-embed Conv2d reshape
    /// path that doesn't carry a `Linear`.
    @Test("zero-extends trailing columns on a raw tensor")
    func zeroExtendsCols() {
        let device = Device.shared
        let w = ImagePreprocessing.makeTensor(
            from: [1, 2, 3, 4], shape: [2, 2], dtype: .f32, device: device)
        let out = padLinearColsTo(w, toCols: 4, device: device)
        #expect(out.shape == [2, 4])
        #expect(out.toArray(as: Float.self) ==
                [1, 2, 0, 0, 3, 4, 0, 0])
    }
}

// ─── Conv3d patch-embed reshape ───────────────────────────────────────

@Suite("VisionTowerOps flattenPatchEmbed")
struct VisionTowerOpsFlattenPatchEmbedTests {

    /// MLX channel-last layout: input `[hidden, tP, py, px, inCh]` with
    /// in-channels ≤ 4 (the layout-detection sentinel). Output column
    /// order is `(((t·inCh + ch)·p + py)·p + px)` for each output row.
    /// We use a tiny shape so every cell is hand-traceable.
    @Test("MLX channel-last layout repacks into 2-D GEMM weight")
    func mlxChannelLastLayout() {
        let device = Device.shared
        // hidden=1, tP=1, p=2, inCh=2 → patchDim = 1·2·2·2 = 8.
        // Pad to 16 so the K-tile padding columns also get checked.
        let hidden = 1, tP = 1, p = 2, inCh = 2
        let patchDim = tP * p * p * inCh    // 8
        let patchDimPadded = 16
        // Walk the source contiguously so each cell is obvious.
        let srcVals: [Float] =
            (0..<(hidden * tP * p * p * inCh)).map { Float($0) }
        let w = ImagePreprocessing.makeTensor(
            from: srcVals, shape: [hidden, tP, p, p, inCh], dtype: .f32,
            device: device)

        let out = flattenPatchEmbed(
            w, hidden: hidden, patchDim: patchDim,
            patchDimPadded: patchDimPadded, device: device)
        #expect(out.shape == [hidden, patchDimPadded])
        let got = out.toArray(as: Float.self)

        // Reproduce the column index the impl uses, in MLX layout order.
        // Source index s = ((((o·tP + t)·p + py)·p + px)·inCh + ch).
        // Dest column col = (((t·inCh + ch)·p + py)·p + px).
        var expected = [Float](repeating: 0, count: hidden * patchDimPadded)
        for o in 0..<hidden {
            for t in 0..<tP {
                for py in 0..<p {
                    for px in 0..<p {
                        for ch in 0..<inCh {
                            let s = ((((o * tP + t) * p + py) * p + px) * inCh + ch)
                            let col = (((t * inCh + ch) * p + py) * p + px)
                            expected[o * patchDimPadded + col] = srcVals[s]
                        }
                    }
                }
            }
        }
        #expect(got == expected)
        // Trailing pad columns are zero.
        for col in patchDim..<patchDimPadded {
            #expect(got[col] == 0,
                    "pad column \(col) should be zero, got \(got[col])")
        }
    }

    /// PyTorch channel-first layout: input `[hidden, inCh, tP, py, px]`
    /// with in-channels large enough to trip the detector (impl checks
    /// `shape[4] <= 4` for MLX layout; here the trailing dim is `p` and
    /// inCh is on dim 1, so we need `p > 4` to land in the
    /// channel-first branch). Use p=5 to be unambiguous.
    @Test("PyTorch channel-first layout repacks into 2-D GEMM weight")
    func pytorchChannelFirstLayout() {
        let device = Device.shared
        // hidden=1, inCh=2, tP=1, p=5 → patchDim = 1·5·5·2 = 50.
        // Detector trips on shape[4] (= p = 5) > 4 → channel-first branch.
        let hidden = 1, inCh = 2, tP = 1, p = 5
        let patchDim = tP * p * p * inCh    // 50
        let patchDimPadded = 64             // next K-tile multiple ≥ 50
        let srcVals: [Float] =
            (0..<(hidden * inCh * tP * p * p)).map { Float($0) }
        let w = ImagePreprocessing.makeTensor(
            from: srcVals, shape: [hidden, inCh, tP, p, p], dtype: .f32,
            device: device)

        let out = flattenPatchEmbed(
            w, hidden: hidden, patchDim: patchDim,
            patchDimPadded: patchDimPadded, device: device)
        #expect(out.shape == [hidden, patchDimPadded])
        let got = out.toArray(as: Float.self)

        // PyTorch layout source index: ((((o·inCh + ch)·tP + t)·p + py)·p + px).
        var expected = [Float](repeating: 0, count: hidden * patchDimPadded)
        for o in 0..<hidden {
            for ch in 0..<inCh {
                for t in 0..<tP {
                    for py in 0..<p {
                        for px in 0..<p {
                            let s = ((((o * inCh + ch) * tP + t) * p + py) * p + px)
                            let col = (((t * inCh + ch) * p + py) * p + px)
                            expected[o * patchDimPadded + col] = srcVals[s]
                        }
                    }
                }
            }
        }
        #expect(got == expected)
        // Trailing pad columns are zero.
        for col in patchDim..<patchDimPadded {
            #expect(got[col] == 0,
                    "pad column \(col) should be zero, got \(got[col])")
        }
    }
}
