from __future__ import annotations

import math
from dataclasses import dataclass

from ktpu.constants import (
    DEFAULT_CONTEXT_BUCKET,
    DEFAULT_HEADROOM_FRACTION,
    DEFAULT_KV_OVERHEAD_FRACTION,
    DEFAULT_WEIGHT_OVERHEAD_FRACTION,
    GIB,
)
from ktpu.errors import SizingError
from ktpu.model import ModelProfile


@dataclass(frozen=True)
class SizingResult:
    hbm_total_bytes: int
    hbm_in_use_bytes: int
    weights_bytes: int
    weight_overhead_bytes: int
    runtime_headroom_bytes: int
    kv_budget_bytes: int
    kv_bytes_per_token: int
    effective_kv_bytes_per_token: int
    model_limit: int
    calculated_safe_context: int
    user_context_cap: int | None
    applied_context: int
    input_tokens: int
    safe_output_tokens: int
    user_output_cap: int | None
    applied_output_tokens: int


def _positive_cap(name: str, value: int | None) -> int | None:
    if value is not None and value <= 0:
        raise SizingError(f"{name} must be greater than zero.")
    return value


def calculate_limits(
    profile: ModelProfile,
    *,
    hbm_total_bytes: int,
    hbm_in_use_bytes: int,
    input_tokens: int,
    context_cap: int | None = None,
    output_cap: int | None = None,
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    weight_overhead_fraction: float = DEFAULT_WEIGHT_OVERHEAD_FRACTION,
    kv_overhead_fraction: float = DEFAULT_KV_OVERHEAD_FRACTION,
    context_bucket: int = DEFAULT_CONTEXT_BUCKET,
) -> SizingResult:
    context_cap = _positive_cap("Context cap", context_cap)
    output_cap = _positive_cap("Output cap", output_cap)
    if hbm_total_bytes <= 0:
        raise SizingError("No TPU HBM was detected.")
    if input_tokens < 0:
        raise SizingError("Input token count cannot be negative.")
    if not (0 <= headroom_fraction < 1):
        raise SizingError("Headroom fraction must be between zero and one.")
    runtime_headroom = max(int(hbm_total_bytes * headroom_fraction), 8 * GIB)
    weight_overhead = int(profile.weights_bytes * weight_overhead_fraction)
    kv_budget = (
        hbm_total_bytes
        - max(0, hbm_in_use_bytes)
        - profile.weights_bytes
        - weight_overhead
        - runtime_headroom
    )
    if kv_budget <= 0:
        raise SizingError(
            "Model weights plus runtime headroom exceed available TPU HBM."
        )
    effective_kv = max(
        1, math.ceil(profile.kv_bytes_per_token * (1 + kv_overhead_fraction))
    )
    raw_context = kv_budget // effective_kv
    safe_context = min(profile.model_limit, int(raw_context))
    if context_bucket > 1 and safe_context >= context_bucket:
        safe_context = (safe_context // context_bucket) * context_bucket
    if safe_context <= 0:
        raise SizingError("Calculated safe context is zero.")
    applied_context = min(
        safe_context, context_cap if context_cap is not None else safe_context
    )
    if input_tokens >= applied_context:
        raise SizingError(
            f"Rendered prompt has {input_tokens} tokens but the applied context is "
            f"{applied_context}; reduce the prompt or increase safe capacity."
        )
    safe_output = applied_context - input_tokens
    applied_output = min(
        safe_output, output_cap if output_cap is not None else safe_output
    )
    return SizingResult(
        hbm_total_bytes=hbm_total_bytes,
        hbm_in_use_bytes=max(0, hbm_in_use_bytes),
        weights_bytes=profile.weights_bytes,
        weight_overhead_bytes=weight_overhead,
        runtime_headroom_bytes=runtime_headroom,
        kv_budget_bytes=kv_budget,
        kv_bytes_per_token=profile.kv_bytes_per_token,
        effective_kv_bytes_per_token=effective_kv,
        model_limit=profile.model_limit,
        calculated_safe_context=safe_context,
        user_context_cap=context_cap,
        applied_context=applied_context,
        input_tokens=input_tokens,
        safe_output_tokens=safe_output,
        user_output_cap=output_cap,
        applied_output_tokens=applied_output,
    )

