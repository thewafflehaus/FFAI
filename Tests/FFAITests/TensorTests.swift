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
import Testing

@testable import FFAI

@Suite("Tensor")
struct TensorTests {
    @Test("empty allocates correct byte length")
    func empty() {
        let t = Tensor.empty(shape: [4, 8], dtype: .f32)
        #expect(t.shape == [4, 8])
        #expect(t.dtype == .f32)
        #expect(t.elementCount == 32)
        #expect(t.byteCount == 128)
        #expect(t.offset == 0)
        #expect(t.buffer.length >= 128)
    }

    @Test("reshape preserves element count and storage")
    func reshape() {
        let t = Tensor.empty(shape: [2, 6], dtype: .f32)
        let reshaped = t.reshaped(to: [3, 4])
        #expect(reshaped.shape == [3, 4])
        #expect(reshaped.elementCount == 12)
        #expect(reshaped.buffer === t.buffer)
        #expect(reshaped.offset == t.offset)
    }

    @Test("toArray / copyIn round-trip")
    func roundTrip() {
        let t = Tensor.empty(shape: [5], dtype: .f32)
        let values: [Float] = [1.5, -2.5, 0, 4.25, 100.0]
        t.copyIn(from: values)
        #expect(t.toArray(as: Float.self) == values)
    }

    @Test("zero clears bytes")
    func zero() {
        let t = Tensor.empty(shape: [3], dtype: .f32)
        t.copyIn(from: [Float(1), Float(2), Float(3)])
        t.zero()
        #expect(t.toArray(as: Float.self) == [0, 0, 0])
    }

    @Test("slicedRows updates offset and shape, shares buffer")
    func slicedRows() {
        let t = Tensor.empty(shape: [4, 3], dtype: .f32)
        let values: [Float] = (0 ..< 12).map { Float($0) }
        t.copyIn(from: values)

        let slice = t.slicedRows(start: 1, count: 2)
        #expect(slice.shape == [2, 3])
        #expect(slice.buffer === t.buffer)
        #expect(slice.offset == 1 * 3 * MemoryLayout<Float>.size)
        #expect(slice.toArray(as: Float.self) == [3, 4, 5, 6, 7, 8])
    }

    @Test("toFloatArray converts every floating dtype")
    func toFloatArrayConvertsFloatingDtypes() {
        let f32 = Tensor.empty(shape: [3], dtype: .f32)
        f32.copyIn(from: [Float(1.5), -2.25, 3])
        #expect(f32.toFloatArray() == [1.5, -2.25, 3])

        let f16 = Tensor.empty(shape: [3], dtype: .f16)
        f16.copyIn(from: [Float16(1.5), -2.25, 3])
        #expect(f16.toFloatArray() == [1.5, -2.25, 3])

        // bf16: top 16 bits of an f32 — 1.5 / -2.25 / 3 are exact.
        let bf16 = Tensor.empty(shape: [3], dtype: .bf16)
        bf16.copyIn(
            from: [
                Float(1.5).bitPattern, Float(-2.25).bitPattern,
                Float(3).bitPattern,
            ].map { UInt16($0 >> 16) })
        #expect(bf16.toFloatArray() == [1.5, -2.25, 3])
    }

    @Test("filled broadcasts a scalar across every floating dtype")
    func filledBroadcastsScalar() {
        let f32 = Tensor.filled(2.5, shape: [4], dtype: .f32)
        #expect(f32.toFloatArray() == [2.5, 2.5, 2.5, 2.5])

        let f16 = Tensor.filled(-1.25, shape: [2], dtype: .f16)
        #expect(f16.toFloatArray() == [-1.25, -1.25])

        // 0.5 is exactly representable in bf16; round-trip is exact.
        let bf16 = Tensor.filled(0.5, shape: [3], dtype: .bf16)
        #expect(bf16.toFloatArray() == [0.5, 0.5, 0.5])
    }
}
