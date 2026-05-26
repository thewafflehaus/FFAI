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
// ThinkingSplitTests — format enum + Split data model + the
// tokenizer-free ChatML scanner (`splitChatML(tokens:openMarker:closeMarker:)`).
// Tokenizer-driven format detection runs through the integration tests
// against real models.

import Foundation
import Testing
@testable import FFAI

@Suite("ThinkingSplit")
struct ThinkingSplitTests {

    @Test("ThinkingFormat allCases includes every documented format")
    func allFormats() {
        let all = ThinkingFormat.allCases
        #expect(all.contains(.none))
        #expect(all.contains(.chatML))
        #expect(all.contains(.harmony))
        #expect(all.contains(.gemmaChannel))
        #expect(all.count == 4)
    }

    @Test("ThinkingFormat rawValues are stable")
    func rawValues() {
        #expect(ThinkingFormat.none.rawValue == "none")
        #expect(ThinkingFormat.chatML.rawValue == "chatML")
        #expect(ThinkingFormat.harmony.rawValue == "harmony")
        #expect(ThinkingFormat.gemmaChannel.rawValue == "gemmaChannel")
    }

    @Test("ChatML scanner partitions on the supplied marker ids")
    func chatMLPartition() {
        let stream = [50, 100, 1, 2, 3, 101, 7, 8, 9]
        let split = ThinkingSplit.splitChatML(tokens: stream,
                                              openMarker: 100,
                                              closeMarker: 101)
        #expect(split != nil)
        if let s = split {
            #expect(Array(s.thinkTokens) == [1, 2, 3])
            #expect(Array(s.genTokens) == [7, 8, 9])
            #expect(s.format == .chatML)
        }
    }

    @Test("ChatML scanner returns nil when open marker absent")
    func chatMLNoOpen() {
        #expect(ThinkingSplit.splitChatML(tokens: [1, 2, 3],
                                          openMarker: 100,
                                          closeMarker: 101) == nil)
    }

    @Test("ChatML scanner returns nil when block never closes")
    func chatMLPartialBlock() {
        let stream = [100, 1, 2, 3]
        #expect(ThinkingSplit.splitChatML(tokens: stream,
                                          openMarker: 100,
                                          closeMarker: 101) == nil)
    }

    @Test("Empty think segment yields an empty thinkTokens slice")
    func chatMLEmptyThink() {
        let stream = [100, 101, 5, 6]
        let split = ThinkingSplit.splitChatML(tokens: stream,
                                              openMarker: 100,
                                              closeMarker: 101)
        #expect(split?.thinkTokens.isEmpty == true)
        #expect(Array(split?.genTokens ?? []) == [5, 6])
    }
}
