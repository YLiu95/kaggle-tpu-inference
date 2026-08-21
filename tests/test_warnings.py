"""End-to-end event-stream tests for the limit warnings, with a stubbed engine.

``generate_events`` is the single place both the daemon and the in-process path build
their event stream, so exercising it here covers the warnings the user actually sees
without loading 57 GiB of weights.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma4_tpu.session import generate_events  # noqa: E402
from gemma4_tpu.ui import print_warning  # noqa: E402


class FakeConfig:
    eos_token_ids = (1, 106)
    enable_moe_block = False
    num_experts = 0
    top_k_experts = 0
    hidden_size = 5376
    vocab_size = 262144
    num_hidden_layers = 4
    layer_types = ("sliding_attention",) * 3 + ("full_attention",)


class FakeTokenizer:
    """Deterministic 1-char-per-token tokenizer over the printable ASCII range."""

    def apply_chat_template(self, messages, **kw):
        text = " ".join(m["content"] for m in messages)
        return {"input_ids": [[2] + [ord(c) % 1000 + 200 for c in text]]}

    def encode(self, text, add_special_tokens=False):
        return [12345, 6789]  # not a single token -> no channel markers

    def decode(self, ids, **kw):
        return "".join("x" for _ in ids)

    def convert_ids_to_tokens(self, ids):
        return ["x" for _ in ids]


class FakeEngine:
    def __init__(self, max_len, max_output_tokens, emit, stop_at_end=False,
                 safe_prompt_tokens=None):
        self.cfg = FakeConfig()
        self.max_len = max_len
        self.max_output_tokens = max_output_tokens
        self.safe_prompt_tokens = max_len if safe_prompt_tokens is None else safe_prompt_tokens
        self.n_devices = 8
        self.param_bytes = 61 * 10**9
        self.active_params = 30 * 10**9
        self.cache_bytes = 6 * 2**30
        self.decode_step_seconds = 0.009
        self.hbm = {"total_bytes_per_chip": 12.8 * 2**30}
        self._emit = emit
        self._stop_at_end = stop_at_end
        self.seen_max_new_tokens = None

    def generate(self, prompt_ids, max_new_tokens, **kw):
        self.seen_max_new_tokens = max_new_tokens
        n = min(self._emit, max_new_tokens)
        yield {"kind": "prefill", "token": 300, "ttft": 0.05,
               "prompt_tokens": len(prompt_ids), "t": time.perf_counter()}
        for i in range(1, n):
            yield {"kind": "decode", "token": 300 + i, "index": i,
                   "t": time.perf_counter(), "total_tokens": i + 1}
        stopped = self._stop_at_end or n < max_new_tokens
        yield {"kind": "finish", "stopped_naturally": stopped,
               "generated": n, "allowed": max_new_tokens}


def events_for(prompt, max_new_tokens, max_len, max_output=None, emit=5, stop=True,
               safe_prompt_tokens=None):
    engine = FakeEngine(max_len, max_output or max_len, emit, stop_at_end=stop,
                        safe_prompt_tokens=safe_prompt_tokens)
    info = {"kind": "info", "model": "google/gemma-4-31B-it", "model_key": "31b",
            "model_max_context": 30720, "num_experts": 0, "top_k_experts": 0}
    req = {"prompt": prompt, "max_new_tokens": max_new_tokens, "think": False}
    return engine, list(generate_events(engine, FakeTokenizer(), None, req, info))


def kinds(evs):
    return [e["kind"] for e in evs]


def warning_codes(evs):
    return [e["code"] for e in evs if e["kind"] == "warning"]


def test_normal_request_emits_no_warning():
    _, evs = events_for("hello world", 768, 16384)
    assert warning_codes(evs) == []
    assert "info" in kinds(evs) and evs[-1]["kind"] == "done"


def test_oversized_output_request_warns_and_is_clamped():
    engine, evs = events_for("hello", 50000, 16384)
    codes = warning_codes(evs)
    assert "request_exceeds_max_new_tokens" in codes
    assert "output_truncated_by_context" in codes
    assert engine.seen_max_new_tokens < 16384
    for e in evs:
        if e["kind"] == "warning":
            assert e["remedy"], "every warning must tell the user how to raise the limit"


def test_prompt_longer_than_context_aborts_with_a_warning():
    long_prompt = "a" * 3000
    engine, evs = events_for(long_prompt, 128, 512)
    assert warning_codes(evs) == ["prompt_exceeds_context"]
    assert engine.seen_max_new_tokens is None, "must not attempt to generate"
    assert kinds(evs)[-1] == "done"
    assert "token" not in kinds(evs)
    w = next(e for e in evs if e["kind"] == "warning")
    assert "--max-len" in w["remedy"]
    assert "30720" in w["remedy"], "should point at the model's real ceiling"


def test_long_prompt_warns_that_prefill_may_not_fit():
    engine, evs = events_for("a" * 3000, 128, 16384, safe_prompt_tokens=1024)
    assert "prompt_prefill_memory_risk" in warning_codes(evs)
    assert engine.seen_max_new_tokens == 128, "it is a warning, not a refusal"


def test_prefill_warning_is_not_emitted_when_the_prompt_already_failed():
    _, evs = events_for("a" * 3000, 128, 512, safe_prompt_tokens=64)
    assert warning_codes(evs) == ["prompt_exceeds_context"]


def test_hitting_the_output_budget_warns_after_generation():
    _, evs = events_for("hi", 5, 16384, emit=5, stop=False)
    assert warning_codes(evs) == ["hit_max_new_tokens"]
    warn_index = kinds(evs).index("warning")
    assert warn_index > kinds(evs).index("prefill"), "warning comes after the tokens"
    assert "--max-new-tokens" in next(e for e in evs if e["kind"] == "warning")["remedy"]


def test_natural_stop_produces_no_completion_warning():
    _, evs = events_for("hi", 5, 16384, emit=3, stop=True)
    assert warning_codes(evs) == []


def test_info_event_reports_the_effective_limits():
    _, evs = events_for("hi", 900, 16384)
    info = next(e for e in evs if e["kind"] == "info")
    assert info["requested_max_new_tokens"] == 900
    assert info["allowed_max_new_tokens"] == 900
    assert info["max_output_tokens"] == 16384


def test_warnings_render_without_crashing():
    from rich.console import Console

    _, evs = events_for("hello", 50000, 16384)
    console = Console(file=open(os.devnull, "w"), force_terminal=False)
    for e in evs:
        if e["kind"] == "warning":
            print_warning(console, e)
