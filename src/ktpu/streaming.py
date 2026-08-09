from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import httpx

from ktpu.errors import StreamingError


@dataclass(frozen=True)
class StreamChunk:
    channel: str
    text: str


@dataclass(frozen=True)
class StreamResult:
    reasoning: str
    content: str
    completion_tokens: int
    ttft_s: float | None
    elapsed_s: float

    @property
    def tokens_per_second(self) -> float | None:
        if self.completion_tokens <= 0 or self.elapsed_s <= 0:
            return None
        generation_time = max(
            self.elapsed_s - (self.ttft_s or 0.0),
            1e-9,
        )
        return self.completion_tokens / generation_time


def iter_sse_payloads(lines: Iterable[str]) -> Iterator[dict[str, object]]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines.clear()
                if payload == "[DONE]":
                    return
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise StreamingError(f"Invalid SSE JSON: {payload[:200]}") from exc
                if isinstance(value, dict):
                    yield value
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            value = json.loads(payload)
            if isinstance(value, dict):
                yield value


def chunks_from_payload(payload: dict[str, object]) -> Iterator[StreamChunk]:
    if payload.get("error"):
        raise StreamingError(f"Server returned an error: {payload['error']}")
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("thinking")
        )
        if isinstance(reasoning, str) and reasoning:
            yield StreamChunk("reasoning", reasoning)
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield StreamChunk("response", content)


def stream_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    on_chunk: Callable[[StreamChunk], None],
    on_first_token: Callable[[float], None] | None = None,
    on_token_count: Callable[[int], None] | None = None,
    count_tokens: Callable[[str], int] | None = None,
    client: httpx.Client | None = None,
) -> StreamResult:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    owns_client = client is None
    http_client = client or httpx.Client(timeout=None)
    started = time.monotonic()
    first_at: float | None = None
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    completion_tokens = 0
    usage_received = False
    try:
        with http_client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=payload
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = response.read().decode("utf-8", "replace")
                raise StreamingError(
                    f"Inference request failed ({response.status_code}): {body[:1000]}"
                ) from exc
            for event in iter_sse_payloads(response.iter_lines()):
                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                    usage_received = True
                    if on_token_count is not None:
                        on_token_count(completion_tokens)
                for chunk in chunks_from_payload(event):
                    if first_at is None:
                        first_at = time.monotonic()
                        if on_first_token is not None:
                            on_first_token(first_at - started)
                    if chunk.channel == "reasoning":
                        reasoning_parts.append(chunk.text)
                    else:
                        content_parts.append(chunk.text)
                    on_chunk(chunk)
                    if count_tokens is not None and not usage_received:
                        completion_tokens = count_tokens(
                            "".join(reasoning_parts) + "".join(content_parts)
                        )
                        if on_token_count is not None:
                            on_token_count(completion_tokens)
    except httpx.HTTPError as exc:
        raise StreamingError(f"Inference connection failed: {exc}") from exc
    finally:
        if owns_client:
            http_client.close()
    elapsed = time.monotonic() - started
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    if not (reasoning.strip() or content.strip()):
        raise StreamingError("The server completed without meaningful text output.")
    return StreamResult(
        reasoning=reasoning,
        content=content,
        completion_tokens=completion_tokens,
        ttft_s=(first_at - started) if first_at is not None else None,
        elapsed_s=elapsed,
    )
