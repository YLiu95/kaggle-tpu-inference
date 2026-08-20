"""Incremental detokenisation + Gemma-4 reasoning/answer channel splitting."""

from __future__ import annotations

THINK_OPEN = "<|channel>thought"
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
    """Splits the raw stream into ``reasoning`` and ``answer`` segments.

    Gemma-4 emits its chain of thought between ``<|channel>thought`` and ``<channel|>``.
    Markers may straddle a token boundary, so partial suffixes are buffered.
    """

    def __init__(self):
        self.buf = ""
        self.mode = "answer"

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
