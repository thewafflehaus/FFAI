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

@Suite("Device")
struct DeviceTests {
    @Test("shared device + queue available")
    func shared() {
        let d = Device.shared
        #expect(d.mtlDevice.name.count > 0)
        // commandQueue is non-optional after construction
        _ = d.commandQueue
    }

    @Test("makeBuffer allocates requested length")
    func makeBuffer() {
        let buf = Device.shared.makeBuffer(length: 1024)
        #expect(buf.length >= 1024)
    }

    @Test("makeCommandBuffer returns a usable buffer")
    func makeCommandBuffer() {
        let cb = Device.shared.makeCommandBuffer()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
    }
}
