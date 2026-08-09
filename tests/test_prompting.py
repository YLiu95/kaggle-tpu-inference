from __future__ import annotations

import unittest

from ktpu.prompting import build_messages, tokenize_messages


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        self.calls.append({"messages": messages, "tokenize": tokenize, **kwargs})
        if tokenize:
            return [10, 11, 12, 13]
        return "<rendered>"


class BatchEncodingLike(dict):
    pass


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        if tokenize:
            return BatchEncodingLike(input_ids=[1, 2, 3])
        return "<rendered>"


class PromptingTests(unittest.TestCase):
    def test_system_and_user_messages(self) -> None:
        messages = build_messages("hello", "be concise")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_chat_template_is_rendered_and_tokenized_before_launch(self) -> None:
        tokenizer = FakeTokenizer()
        info = tokenize_messages(
            tokenizer, build_messages("hello"), enable_thinking=True
        )
        self.assertEqual(info.rendered_prompt, "<rendered>")
        self.assertEqual(info.input_tokens, 4)
        self.assertEqual(len(tokenizer.calls), 2)
        self.assertTrue(tokenizer.calls[0]["add_generation_prompt"])
        self.assertTrue(tokenizer.calls[0]["enable_thinking"])

    def test_batch_encoding_mapping_is_supported(self) -> None:
        info = tokenize_messages(MappingTokenizer(), build_messages("hello"))
        self.assertEqual(info.input_ids, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
