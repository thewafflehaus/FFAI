// ChatTemplateTests — type shape (ChatMessage / ChatTemplateOptions /
// ReasoningEffort / ChatTemplateError). End-to-end render-through-the-
// tokenizer is exercised by the integration tests with real models.

import Foundation
import Testing
@testable import FFAI

@Suite("ChatTemplate types")
struct ChatTemplateTests {

    @Test("ChatMessage.Role rawValues are stable")
    func roleRawValues() {
        #expect(ChatMessage.Role.system.rawValue == "system")
        #expect(ChatMessage.Role.user.rawValue == "user")
        #expect(ChatMessage.Role.assistant.rawValue == "assistant")
        #expect(ChatMessage.Role.tool.rawValue == "tool")
    }

    @Test("ChatMessage round-trips through asTemplateMessage")
    func asTemplateMessage() {
        let m1 = ChatMessage(role: .user, content: "Hi")
        let dict1 = m1.asTemplateMessage
        #expect(dict1["role"] as? String == "user")
        #expect(dict1["content"] as? String == "Hi")
        #expect(dict1["thinking"] == nil)

        let m2 = ChatMessage(role: .assistant, content: "Hello",
                             thinking: "<reasoning here>")
        let dict2 = m2.asTemplateMessage
        #expect(dict2["thinking"] as? String == "<reasoning here>")
    }

    @Test("ChatMessage Equatable")
    func messageEquatable() {
        let a = ChatMessage(role: .user, content: "Hi")
        let b = ChatMessage(role: .user, content: "Hi")
        #expect(a == b)
        let c = ChatMessage(role: .user, content: "Hi", thinking: "x")
        #expect(a != c)
    }

    @Test("ChatTemplateOptions defaults")
    func optionsDefaults() {
        let o = ChatTemplateOptions()
        #expect(o.addGenerationPrompt == true)
        #expect(o.enableThinking == false)
        #expect(o.reasoningEffort == nil)
        #expect(o.maxLength == nil)
        #expect(o.truncation == false)
        #expect(o.extraContext.isEmpty)
    }

    @Test("ChatTemplateOptions custom init + Equatable on typed fields")
    func optionsCustomInit() {
        let o = ChatTemplateOptions(
            addGenerationPrompt: false,
            enableThinking: true,
            reasoningEffort: .high,
            maxLength: 4096,
            truncation: true
        )
        #expect(o.addGenerationPrompt == false)
        #expect(o.enableThinking == true)
        #expect(o.reasoningEffort == .high)
        #expect(o.maxLength == 4096)
        #expect(o.truncation == true)

        let same = ChatTemplateOptions(
            addGenerationPrompt: false, enableThinking: true,
            reasoningEffort: .high, maxLength: 4096, truncation: true
        )
        #expect(o == same)

        let different = ChatTemplateOptions(enableThinking: true,
                                            reasoningEffort: .low)
        #expect(o != different)
    }

    @Test("ReasoningEffort rawValues align with Harmony convention")
    func reasoningEffort() {
        #expect(ReasoningEffort.low.rawValue == "low")
        #expect(ReasoningEffort.medium.rawValue == "medium")
        #expect(ReasoningEffort.high.rawValue == "high")
    }

    @Test("ChatTemplateError descriptions are non-empty")
    func errorDescriptions() {
        let e1 = ChatTemplateError.noTemplateOnTokenizer
        #expect(!String(describing: e1).isEmpty)
        struct Boom: Error {}
        let e2 = ChatTemplateError.renderFailed(Boom())
        #expect(String(describing: e2).contains("Boom"))
    }
}
