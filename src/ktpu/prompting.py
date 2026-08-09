from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ktpu.errors import KtpuError


@dataclass(frozen=True)
class PromptInfo:
    messages: list[dict[str, str]]
    rendered_prompt: str
    input_ids: list[int]

    @property
    def input_tokens(self) -> int:
        return len(self.input_ids)


def build_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _normalize_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise KtpuError("Tokenizer returned an unexpected input_ids value.")
    return [int(item) for item in value]


def tokenize_messages(
    tokenizer: object,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = True,
) -> PromptInfo:
    kwargs = {
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        rendered = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=False, **kwargs
        )
        tokenized = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=True, **kwargs
        )
    except Exception as exc:
        raise KtpuError(f"Could not render/tokenize the chat prompt: {exc}") from exc
    return PromptInfo(
        messages=messages,
        rendered_prompt=str(rendered),
        input_ids=_normalize_ids(tokenized),
    )


def load_and_tokenize_prompt(
    model_id: str,
    prompt: str,
    *,
    system: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    enable_thinking: bool = True,
) -> tuple[object, PromptInfo]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise KtpuError(f"Could not load tokenizer for {model_id!r}: {exc}") from exc
    messages = build_messages(prompt, system)
    return tokenizer, tokenize_messages(
        tokenizer, messages, enable_thinking=enable_thinking
    )
