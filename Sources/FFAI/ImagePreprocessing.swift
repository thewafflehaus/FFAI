// ImagePreprocessing — CPU-side image resize / normalize / patchify for
// vision-language models.
//
// Every VLM vision encoder consumes a fixed-layout pixel tensor: an
// image resized to the model's expected resolution, channel-normalized
// with the checkpoint's per-channel mean/std, and laid out NCHW. This
// file does that on the CPU — the cost is negligible next to the
// transformer forward, and keeping it CPU-side avoids a Metal round-trip
// for what is fundamentally a one-shot setup step (Phase 6.5 spec:
// "CPU initially; Metal later if it shows up in profiles").
//
// The output is a `Tensor` in the model's activation dtype, ready to
// hand straight to `Ops.conv2d` / `Ops.patchEmbed`.

import Foundation

/// A decoded RGB image in planar CPU memory: `pixels` is row-major
/// `[height, width, 3]` with values already in `[0, 1]`.
public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    /// Interleaved RGB, `height * width * 3` floats in `[0, 1]`.
    public let pixels: [Float]

    public init(width: Int, height: Int, pixels: [Float]) {
        precondition(pixels.count == width * height * 3,
                     "RGBImage: pixel count \(pixels.count) != w*h*3 \(width * height * 3)")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// A solid-color test image — handy for unit tests and as a
    /// deterministic stand-in when no real image is supplied.
    public static func solid(width: Int, height: Int,
                             r: Float, g: Float, b: Float) -> RGBImage {
        var px = [Float](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            px[i * 3] = r; px[i * 3 + 1] = g; px[i * 3 + 2] = b
        }
        return RGBImage(width: width, height: height, pixels: px)
    }
}

/// Normalization constants — per-channel mean / std. The defaults are
/// the CLIP / SigLIP values most VLM checkpoints ship.
public struct ImageNormalization: Sendable {
    public let mean: (Float, Float, Float)
    public let std: (Float, Float, Float)

    public init(mean: (Float, Float, Float), std: (Float, Float, Float)) {
        self.mean = mean
        self.std = std
    }

    /// SigLIP / Gemma-VL normalization — mean 0.5, std 0.5 per channel
    /// (maps `[0,1]` pixels to `[-1,1]`).
    public static let siglip = ImageNormalization(
        mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))

    /// CLIP / Qwen-VL normalization — ImageNet-derived per-channel
    /// statistics OpenAI's CLIP preprocessing established.
    public static let clip = ImageNormalization(
        mean: (0.48145466, 0.4578275, 0.40821073),
        std: (0.26862954, 0.26130258, 0.27577711))
}

public enum ImagePreprocessing {

    /// Resize `image` to `targetW × targetH` with bilinear interpolation.
    /// Vision encoders expect a fixed input resolution (SigLIP 224 / 896,
    /// CLIP 336, …); this is the resampling step.
    public static func resize(_ image: RGBImage,
                              targetW: Int, targetH: Int) -> RGBImage {
        precondition(targetW > 0 && targetH > 0,
                     "ImagePreprocessing.resize: target dims must be positive")
        if image.width == targetW && image.height == targetH { return image }

        var out = [Float](repeating: 0, count: targetW * targetH * 3)
        // Map each output pixel center back into the input grid. The
        // half-pixel offset keeps the sampling centered, matching the
        // `align_corners = false` convention HF image processors use.
        let scaleX = Float(image.width) / Float(targetW)
        let scaleY = Float(image.height) / Float(targetH)
        for oy in 0..<targetH {
            let srcY = (Float(oy) + 0.5) * scaleY - 0.5
            let y0 = max(0, min(image.height - 1, Int(srcY.rounded(.down))))
            let y1 = min(image.height - 1, y0 + 1)
            let wy = max(0, min(1, srcY - Float(y0)))
            for ox in 0..<targetW {
                let srcX = (Float(ox) + 0.5) * scaleX - 0.5
                let x0 = max(0, min(image.width - 1, Int(srcX.rounded(.down))))
                let x1 = min(image.width - 1, x0 + 1)
                let wx = max(0, min(1, srcX - Float(x0)))
                for c in 0..<3 {
                    let p00 = image.pixels[(y0 * image.width + x0) * 3 + c]
                    let p01 = image.pixels[(y0 * image.width + x1) * 3 + c]
                    let p10 = image.pixels[(y1 * image.width + x0) * 3 + c]
                    let p11 = image.pixels[(y1 * image.width + x1) * 3 + c]
                    let top = p00 * (1 - wx) + p01 * wx
                    let bot = p10 * (1 - wx) + p11 * wx
                    out[(oy * targetW + ox) * 3 + c] = top * (1 - wy) + bot * wy
                }
            }
        }
        return RGBImage(width: targetW, height: targetH, pixels: out)
    }

    /// Resize + per-channel normalize, producing an NCHW float tensor
    /// `[1, 3, targetH, targetW]` in `dtype` — the canonical layout
    /// `Ops.conv2d` consumes (and, reshaped to `[3, h, w]`, what
    /// `Ops.patchEmbed` consumes).
    ///
    /// `pixel = (pixel - mean[c]) / std[c]`, applied per channel after
    /// the resize.
    public static func preprocess(
        _ image: RGBImage,
        targetW: Int, targetH: Int,
        normalization: ImageNormalization,
        dtype: DType,
        device: Device = .shared
    ) -> Tensor {
        let resized = resize(image, targetW: targetW, targetH: targetH)
        let means = [normalization.mean.0, normalization.mean.1, normalization.mean.2]
        let stds = [normalization.std.0, normalization.std.1, normalization.std.2]

        // Interleaved RGB → planar NCHW, normalizing in the same pass.
        var planar = [Float](repeating: 0, count: 3 * targetH * targetW)
        let plane = targetH * targetW
        for y in 0..<targetH {
            for x in 0..<targetW {
                let srcBase = (y * targetW + x) * 3
                for c in 0..<3 {
                    let v = (resized.pixels[srcBase + c] - means[c]) / stds[c]
                    planar[c * plane + y * targetW + x] = v
                }
            }
        }
        return makeTensor(from: planar, shape: [1, 3, targetH, targetW],
                          dtype: dtype, device: device)
    }

    /// Patchify a normalized planar `[3, h, w]` float array into a flat
    /// `[num_patches, patch_dim]` tensor — each row is one
    /// `3 × patchH × patchW` patch flattened in `(c, py, px)` order,
    /// matching the `patch_embed` weight-column convention.
    ///
    /// This is the explicit-unfold path; the fused `Ops.patchEmbed`
    /// kernel does the same gather internally, so most callers go
    /// straight through `Ops.patchEmbed` and never need this. It exists
    /// for encoders that materialize patches (and for testing).
    public static func patchify(
        planar: [Float], channels: Int, height: Int, width: Int,
        patchH: Int, patchW: Int, dtype: DType, device: Device = .shared
    ) -> Tensor {
        precondition(planar.count == channels * height * width,
                     "ImagePreprocessing.patchify: array size mismatch")
        precondition(height % patchH == 0 && width % patchW == 0,
                     "ImagePreprocessing.patchify: image not divisible by patch")
        let patchesH = height / patchH
        let patchesW = width / patchW
        let numPatches = patchesH * patchesW
        let patchDim = channels * patchH * patchW
        let plane = height * width

        var out = [Float](repeating: 0, count: numPatches * patchDim)
        for ph in 0..<patchesH {
            for pw in 0..<patchesW {
                let patch = ph * patchesW + pw
                var col = 0
                for c in 0..<channels {
                    for py in 0..<patchH {
                        let row = (ph * patchH + py) * width + pw * patchW
                        for px in 0..<patchW {
                            out[patch * patchDim + col] = planar[c * plane + row + px]
                            col += 1
                        }
                    }
                }
            }
        }
        return makeTensor(from: out, shape: [numPatches, patchDim],
                          dtype: dtype, device: device)
    }

    /// Build a `Tensor` of `shape` / `dtype` from a `[Float]` source,
    /// converting to the storage dtype. Centralizes the f32 / f16 / bf16
    /// host-side conversion every preprocessing path shares.
    static func makeTensor(from values: [Float], shape: [Int],
                           dtype: DType, device: Device) -> Tensor {
        let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
        switch dtype {
        case .f32:
            t.copyIn(from: values)
        case .f16:
            t.copyIn(from: values.map { Float16($0) })
        case .bf16:
            t.copyIn(from: values.map { v -> UInt16 in
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            })
        default:
            fatalError("ImagePreprocessing: unsupported dtype \(dtype)")
        }
        return t
    }
}
