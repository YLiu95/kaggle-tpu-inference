"""Incremental detokenisation + Gemma-4 reasoning/answer channel splitting."""

from __future__ import annotations

THINK_OPEN = "<|channel>"
THINK_CLOSE = "<channel|>"


class IncrementalDetokenizer:
    """Turns a stream of token ids into text deltas (handles multi-token characters)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.text = ""

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        delta = text[len(self.text) :]
        self.text = text
        return delta


class ChannelSplitter:
    """Splits the stream into ``reasoning`` and ``answer`` segments.

    Gemma-4 wraps its chain of thought in the single-token markers ``<|channel>`` (id 100)
    and ``<channel|>`` (id 101), immediately followed by the literal channel name
    ``thought``. Switching on ids is exact; the text markers are a fallback.
    """

    def __init__(self, open_id: int | None = None, close_id: int | None = None):
        self.open_id = open_id
        self.close_id = close_id
        self.mode = "answer"
        self.buf = ""
        self._drop_channel_name = False

    def feed_token(self, token_id: int, delta: str) -> list[tuple[str, str]]:
        if self.open_id is None:
            return self.feed(delta)
        if token_id == self.open_id:
            self.mode = "reasoning"
            self._drop_channel_name = True
            return []
        if token_id == self.close_id:
            self.mode = "answer"
            return []
        if self._drop_channel_name:
            self._drop_channel_name = False
            if delta.strip() in ("thought", ""):
                return []
        return [(self.mode, delta)] if delta else []

    def feed(self, delta: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        self.buf += delta
        while True:
            marker = THINK_OPEN if self.mode == "answer" else THINK_CLOSE
            idx = self.buf.find(marker)
            if idx >= 0:
                if idx:
                    out.append((self.mode, self.buf[:idx]))
                self.buf = self.buf[idx + len(marker) :]
                self.mode = "reasoning" if self.mode == "answer" else "answer"
                continue
            keep = 0
            for k in range(1, min(len(marker), len(self.buf)) + 1):
                if self.buf.endswith(marker[:k]):
                    keep = k
            if len(self.buf) > keep:
                out.append((self.mode, self.buf[: len(self.buf) - keep]))
                self.buf = self.buf[len(self.buf) - keep :]
            return out

    def flush(self) -> list[tuple[str, str]]:
        if not self.buf:
            return []
        out = [(self.mode, self.buf)]
        self.buf = ""
        return out
