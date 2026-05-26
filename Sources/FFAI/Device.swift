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
// Device — singleton wrapper over the system MTLDevice + a default
// command queue. Exposes a single Sendable handle that all of FFAI uses
// to allocate buffers and submit work.

import Foundation
import Metal
import MetalTileSwift

public final class Device: @unchecked Sendable {
    public let mtlDevice: MTLDevice
    public let commandQueue: MTLCommandQueue

    public static let shared: Device = {
        // Reuse the same MTLDevice + queue MetalTileSwift uses, so PSOs
        // and buffers are guaranteed compatible.
        let lib = MetalTileLibrary.shared
        return Device(mtlDevice: lib.device, commandQueue: lib.commandQueue)
    }()

    public init(mtlDevice: MTLDevice, commandQueue: MTLCommandQueue) {
        self.mtlDevice = mtlDevice
        self.commandQueue = commandQueue
    }

    /// Allocate a fresh shared-storage MTLBuffer of the given byte length.
    public func makeBuffer(length: Int) -> MTLBuffer {
        guard let buf = mtlDevice.makeBuffer(length: length, options: .storageModeShared) else {
            fatalError("Device.makeBuffer(length: \(length)) returned nil")
        }
        return buf
    }

    /// Make a new MTLCommandBuffer.
    public func makeCommandBuffer() -> MTLCommandBuffer {
        guard let cb = commandQueue.makeCommandBuffer() else {
            fatalError("Device.makeCommandBuffer() returned nil")
        }
        return cb
    }
}
