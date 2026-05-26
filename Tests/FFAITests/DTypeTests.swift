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
import Foundation
import Testing

@testable import FFAI

@Suite("DType")
struct DTypeTests {
    @Test("byteSize matches expected")
    func byteSizes() {
        #expect(DType.f32.byteSize == 4)
        #expect(DType.f16.byteSize == 2)
        #expect(DType.bf16.byteSize == 2)
        #expect(DType.i32.byteSize == 4)
        #expect(DType.u32.byteSize == 4)
        #expect(DType.i8.byteSize == 1)
        #expect(DType.u8.byteSize == 1)
    }

    @Test("fromSafeTensors recognizes all supported types")
    func fromSafeTensors() {
        #expect(DType.fromSafeTensors("F32") == .f32)
        #expect(DType.fromSafeTensors("F16") == .f16)
        #expect(DType.fromSafeTensors("BF16") == .bf16)
        #expect(DType.fromSafeTensors("I32") == .i32)
        #expect(DType.fromSafeTensors("U32") == .u32)
        #expect(DType.fromSafeTensors("I8") == .i8)
        #expect(DType.fromSafeTensors("U8") == .u8)
        // case-insensitive
        #expect(DType.fromSafeTensors("f32") == .f32)
        #expect(DType.fromSafeTensors("bf16") == .bf16)
        // unknown
        #expect(DType.fromSafeTensors("F64") == nil)
        #expect(DType.fromSafeTensors("nonsense") == nil)
    }

    @Test("kernelSuffix matches raw value")
    func kernelSuffix() {
        #expect(DType.f32.kernelSuffix == "f32")
        #expect(DType.f16.kernelSuffix == "f16")
        #expect(DType.bf16.kernelSuffix == "bf16")
    }

    @Test("Codable round-trip via raw value")
    func codable() throws {
        let original: DType = .bf16
        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(DType.self, from: encoded)
        #expect(decoded == original)
    }
}
