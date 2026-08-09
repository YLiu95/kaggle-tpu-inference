from __future__ import annotations

import json
import unittest

from ktpu.streaming import (
    StreamChunk,
    chunks_from_payload,
    iter_sse_payloads,
    stream_chat,
)


def event(value: dict[str, object]) -> list[str]:
    return [f"data: {json.dumps(value)}", ""]


class FakeResponse:
    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        return iter(self.lines)


class FakeClient:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.request_json = None

    def stream(self, method: str, url: str, *, json: dict[str, object]):
        self.request_json = json
        self.method = method
        self.url = url
        return FakeResponse(self.lines)

    def close(self) -> None:
        pass


class StreamingTests(unittest.TestCase):
    def test_sse_and_reasoning_parser(self) -> None:
        lines = [
            *event(
                {
                    "choices": [
                        {"delta": {"reasoning_content": "think", "content": None}}
                    ]
                }
            ),
            *event({"choices": [{"delta": {"content": "answer"}}]}),
            "data: [DONE]",
            "",
        ]
        payloads = list(iter_sse_payloads(lines))
        chunks = [
            chunk for payload in payloads for chunk in chunks_from_payload(payload)
        ]
        self.assertEqual(
            chunks,
            [StreamChunk("reasoning", "think"), StreamChunk("response", "answer")],
        )

    def test_stream_chat_tracks_usage_and_callbacks(self) -> None:
        lines = [
            *event(
                {
                    "choices": [
                        {"delta": {"reasoning_content": "A", "content": None}}
                    ]
                }
            ),
            *event({"choices": [{"delta": {"content": "B"}}]}),
            *event({"choices": [], "usage": {"completion_tokens": 7}}),
            "data: [DONE]",
            "",
        ]
        chunks: list[StreamChunk] = []
        counts: list[int] = []
        first: list[float] = []
        client = FakeClient(lines)
        result = stream_chat(
            base_url="http://127.0.0.1:8000",
            model="model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=20,
            temperature=0.0,
            enable_thinking=True,
            on_chunk=chunks.append,
            on_first_token=first.append,
            on_token_count=counts.append,
            count_tokens=len,
            client=client,  # type: ignore[arg-type]
        )
        self.assertEqual(result.reasoning, "A")
        self.assertEqual(result.content, "B")
        self.assertEqual(result.completion_tokens, 7)
        self.assertEqual(counts[-1], 7)
        self.assertEqual(len(first), 1)
        self.assertTrue(client.request_json["stream"])
        self.assertEqual(
            client.request_json["chat_template_kwargs"], {"enable_thinking": True}
        )


if __name__ == "__main__":
    unittest.main()

