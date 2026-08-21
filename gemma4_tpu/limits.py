"""Context-window / output-length limits and the warnings shown when a request exceeds them.

Two independent budgets:

``context_tokens``  the resident KV cache the daemon allocated at start-up. Changing it
                    means reallocating the cache and recompiling, i.e. a daemon restart.
``max_new_tokens``  how many tokens this single request may generate. Changing it is
                    just a flag on the next run.

Every function here is pure so the behaviour can be unit-tested without a TPU
(``tests/test_limits.py``).
"""

from __future__ import annotations

import dataclasses
import os

# Ceiling for the default (26B-A4B) configuration. Per-model values live in
# :mod:`gemma4_tpu.models`; these stay as the module-level fallbacks.
MAX_OUTPUT_TOKENS = 32768
DEFAULT_CONTEXT_TOKENS = 32768

HOW_TO_RAISE_CONTEXT = (
    "raise the resident context with `bash serve.sh restart --max-len N` "
    "(or `GEMMA4_MAX_LEN=N bash serve.sh restart`); with `--local`, pass `--max-len N` "
    "to run.sh. A larger cache costs KV-bytes/token x N per chip and one recompile."
)
HOW_TO_RAISE_OUTPUT = "raise the output budget with `--max-new-tokens N` on run.sh."


def env_max_len(default: int) -> int:
    """``GEMMA4_MAX_LEN`` overrides the per-model default resident context."""
    raw = os.environ.get("GEMMA4_MAX_LEN", "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def allowed_output_tokens(
    requested: int,
    prompt_tokens: int,
    context_tokens: int,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> int:
    """Largest output length that fits, given the prompt and the resident cache."""
    available = max(0, context_tokens - prompt_tokens)
    return min(max(1, requested), max_output_tokens, available)


@dataclasses.dataclass(frozen=True)
class LimitWarning:
    """A single user-facing limit warning."""

    code: str
    message: str
    remedy: str
    detail: dict = dataclasses.field(default_factory=dict)

    def to_event(self) -> dict:
        return {
            "kind": "warning",
            "code": self.code,
            "message": self.message,
            "remedy": self.remedy,
            **self.detail,
        }


def check_request(
    requested: int,
    prompt_tokens: int,
    context_tokens: int,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    model_max_context: int | None = None,
) -> tuple[int, list[LimitWarning]]:
    """Return ``(allowed_max_new_tokens, warnings)`` for one request.

    ``allowed`` is 0 when the prompt alone does not fit, in which case callers must not
    generate.
    """
    warnings: list[LimitWarning] = []
    ceiling = max(model_max_context or context_tokens, context_tokens)

    if prompt_tokens >= context_tokens:
        headroom = ""
        if ceiling > context_tokens:
            headroom = (
                f" This model can hold up to {ceiling:,} tokens on the v5e-8, so "
                f"`--max-len {ceiling}` would fit this prompt."
            )
        warnings.append(
            LimitWarning(
                code="prompt_exceeds_context",
                message=(
                    f"your input is {prompt_tokens:,} tokens but the resident context "
                    f"window is {context_tokens:,} tokens, so there is no room to generate"
                ),
                remedy=HOW_TO_RAISE_CONTEXT + headroom,
                detail={
                    "prompt_tokens": prompt_tokens,
                    "context_tokens": context_tokens,
                    "model_max_context": ceiling,
                },
            )
        )
        return 0, warnings

    allowed = allowed_output_tokens(requested, prompt_tokens, context_tokens, max_output_tokens)

    if requested > max_output_tokens:
        warnings.append(
            LimitWarning(
                code="request_exceeds_max_new_tokens",
                message=(
                    f"you asked for {requested:,} new tokens but the per-request cap is "
                    f"{max_output_tokens:,}"
                ),
                remedy=(
                    "the cap is MAX_OUTPUT_TOKENS in gemma4_tpu/limits.py and can never "
                    "exceed the resident context window, so " + HOW_TO_RAISE_CONTEXT
                ),
                detail={"requested": requested, "max_output_tokens": max_output_tokens},
            )
        )

    if allowed < min(requested, max_output_tokens):
        headroom = ""
        if ceiling > context_tokens:
            needed = min(prompt_tokens + requested, ceiling)
            headroom = (
                f" Serving this model with `--max-len {needed}` (ceiling {ceiling:,}) "
                f"would grant the full request."
            )
        warnings.append(
            LimitWarning(
                code="output_truncated_by_context",
                message=(
                    f"prompt {prompt_tokens:,} + requested {requested:,} = "
                    f"{prompt_tokens + requested:,} tokens exceeds the "
                    f"{context_tokens:,}-token context window, so max-new-tokens was "
                    f"reduced to {allowed:,}"
                ),
                remedy=HOW_TO_RAISE_CONTEXT + headroom,
                detail={
                    "requested": requested,
                    "allowed": allowed,
                    "prompt_tokens": prompt_tokens,
                    "context_tokens": context_tokens,
                    "model_max_context": ceiling,
                },
            )
        )
    return allowed, warnings


def check_completion(generated: int, allowed: int, stopped_naturally: bool) -> list[LimitWarning]:
    """Warn when generation was cut off by the budget instead of by a stop token."""
    if stopped_naturally or generated < allowed:
        return []
    return [
        LimitWarning(
            code="hit_max_new_tokens",
            message=(
                f"generation stopped after {generated:,} tokens because it hit the "
                f"max-new-tokens budget, not an end-of-turn token, so the answer is truncated"
            ),
            remedy=HOW_TO_RAISE_OUTPUT + " If it is already at the context limit, " +
            HOW_TO_RAISE_CONTEXT,
            detail={"generated": generated, "allowed": allowed},
        )
    ]
