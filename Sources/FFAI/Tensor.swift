// Tensor — handle to a region of GPU memory + shape + dtype.
//
// Phase 2: contiguous row-major only (no strides). Slicing returns a
// Tensor that points at the same MTLBuffer with an updated byte offset.

import Foundation
import Metal

public struct Tensor: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset into `buffer` where this tensor's data begins.
    public let offset: Int
    public let shape: [Int]
    public let dtype: DType

    public init(buffer: MTLBuffer, offset: Int = 0, shape: [Int], dtype: DType) {
        self.buffer = buffer
        self.offset = offset
        self.shape = shape
        self.dtype = dtype
    }

    public var elementCount: Int { shape.reduce(1, *) }
    public var byteCount: Int { elementCount * dtype.byteSize }

    /// Allocate a new contiguous tensor. Caller-owned buffer.
    public static func empty(shape: [Int], dtype: DType, device: Device = .shared) -> Tensor {
        let count = shape.reduce(1, *)
        let bytes = count * dtype.byteSize
        return Tensor(buffer: device.makeBuffer(length: bytes), offset: 0, shape: shape, dtype: dtype)
    }

    /// Reshape (no copy). Element count must match.
    public func reshaped(to newShape: [Int]) -> Tensor {
        let newCount = newShape.reduce(1, *)
        precondition(newCount == elementCount,
                     "reshape mismatch: \(shape) (\(elementCount)) -> \(newShape) (\(newCount))")
        return Tensor(buffer: buffer, offset: offset, shape: newShape, dtype: dtype)
    }

    /// Slice along dim 0: rows in `[start, start + count)`. Inner dims
    /// preserved. Equivalent to `tensor[start:start+count]` in numpy.
    public func slicedRows(start: Int, count: Int) -> Tensor {
        precondition(!shape.isEmpty, "cannot slice scalar tensor")
        precondition(start + count <= shape[0], "row slice out of bounds")
        let innerElems = shape.dropFirst().reduce(1, *)
        let rowBytes = innerElems * dtype.byteSize
        var newShape = shape
        newShape[0] = count
        return Tensor(
            buffer: buffer,
            offset: offset + start * rowBytes,
            shape: newShape,
            dtype: dtype
        )
    }

    // ─── Host I/O helpers (use sparingly; CPU↔GPU sync points) ───────

    /// Read all elements as a typed array. Storage-shared assumed.
    public func toArray<T>(as type: T.Type) -> [T] {
        let count = elementCount
        let ptr = buffer.contents().advanced(by: offset).bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Overwrite contents from a typed array. Storage-shared assumed.
    public func copyIn<T>(from array: [T]) {
        precondition(array.count == elementCount, "copyIn count mismatch")
        let dst = buffer.contents().advanced(by: offset)
        array.withUnsafeBufferPointer { src in
            dst.copyMemory(from: UnsafeRawPointer(src.baseAddress!),
                           byteCount: elementCount * MemoryLayout<T>.stride)
        }
    }

    /// Zero the entire tensor (storage-shared).
    public func zero() {
        memset(buffer.contents().advanced(by: offset), 0, byteCount)
    }

    /// Read all elements as `[Float]`, converting from the tensor's
    /// dtype. Handles `f32` / `f16` / `bf16` — the three floating dtypes
    /// FFAI activations use. CPU↔GPU sync point: the caller must ensure
    /// any pending GPU writes to this tensor have completed.
    public func toFloatArray() -> [Float] {
        switch dtype {
        case .f32:
            return toArray(as: Float.self)
        case .f16:
            return toArray(as: Float16.self).map { Float($0) }
        case .bf16:
            // bf16 is the top 16 bits of an IEEE-754 f32; reconstitute
            // by shifting back into the high half of a 32-bit pattern.
            return toArray(as: UInt16.self).map {
                Float(bitPattern: UInt32($0) << 16)
            }
        case .i32:
            return toArray(as: Int32.self).map { Float($0) }
        case .u32:
            return toArray(as: UInt32.self).map { Float($0) }
        case .i8:
            return toArray(as: Int8.self).map { Float($0) }
        case .u8:
            return toArray(as: UInt8.self).map { Float($0) }
        case .i64:
            return toArray(as: Int64.self).map { Float($0) }
        case .u64:
            return toArray(as: UInt64.self).map { Float($0) }
        }
    }

    /// Build a tensor of `shape` whose every element equals `value`,
    /// stored in `dtype`. Used to broadcast a CPU scalar into an
    /// element-wise op (e.g. an MoE combine weight). Only the floating
    /// dtypes are supported.
    public static func filled(_ value: Float, shape: [Int], dtype: DType,
                              device: Device = .shared) -> Tensor {
        let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
        let n = t.elementCount
        switch dtype {
        case .f32:
            t.copyIn(from: [Float](repeating: value, count: n))
        case .f16:
            t.copyIn(from: [Float16](repeating: Float16(value), count: n))
        case .bf16:
            // Round-to-nearest by adding the bf16 rounding bias before
            // truncating the low 16 bits of the f32 bit pattern.
            let bits = value.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            t.copyIn(from: [UInt16](repeating: UInt16(rounded >> 16), count: n))
        default:
            fatalError("Tensor.filled: unsupported dtype \(dtype)")
        }
        return t
    }
}
