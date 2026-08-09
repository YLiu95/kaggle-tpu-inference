from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ktpu.errors import SizingError


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    weights_bytes: int
    model_limit: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype_bytes: int
    kv_bytes_per_token: int
    dtype_name: str
    revision: str | None = None


def _first(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("text_config", "language_config", "decoder"):
        value = config.get(key)
        if isinstance(value, dict):
            return value
    return config


def _dtype_size(dtype: object) -> tuple[str, int]:
    name = str(dtype or "bfloat16").lower().replace("torch.", "")
    if name in {"float32", "fp32"}:
        return name, 4
    if name in {"float64", "fp64"}:
        return name, 8
    if name in {"int8", "uint8", "float8", "fp8"}:
        return name, 1
    if name in {"int4", "uint4", "fp4"}:
        # KV cache is not ordinarily stored below one byte even for 4-bit weights.
        return name, 1
    return name, 2


def profile_from_config_dict(
    model_id: str,
    config: dict[str, Any],
    weights_bytes: int,
    revision: str | None = None,
) -> ModelProfile:
    text = _text_config(config)
    model_limit = int(
        _first(
            text,
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "seq_length",
        )
        or _first(config, "max_position_embeddings", "model_max_length")
        or 0
    )
    layers = int(_first(text, "num_hidden_layers", "n_layer", "num_layers") or 0)
    kv_heads = int(
        _first(text, "num_key_value_heads", "multi_query_group_num")
        or _first(text, "num_attention_heads", "n_head")
        or 0
    )
    attention_heads = int(_first(text, "num_attention_heads", "n_head") or kv_heads)
    hidden_size = int(_first(text, "hidden_size", "n_embd", "d_model") or 0)
    head_dim = int(_first(text, "head_dim", "attention_head_dim") or 0)
    if not head_dim and hidden_size and attention_heads:
        head_dim = hidden_size // attention_heads
    dtype_name, dtype_bytes = _dtype_size(
        _first(text, "kv_cache_dtype", "dtype", "torch_dtype")
        or _first(config, "dtype", "torch_dtype")
    )
    missing = [
        name
        for name, value in (
            ("model context limit", model_limit),
            ("hidden layers", layers),
            ("KV heads", kv_heads),
            ("attention head dimension", head_dim),
            ("weight size", weights_bytes),
        )
        if value <= 0
    ]
    if missing:
        raise SizingError(
            f"Model metadata is incomplete ({', '.join(missing)}); refusing to guess."
        )
    # Conservative full-attention estimate. Hybrid/sliding attention can consume less,
    # but sizing to the upper bound is safer across vLLM cache implementations.
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    return ModelProfile(
        model_id=model_id,
        weights_bytes=int(weights_bytes),
        model_limit=model_limit,
        num_hidden_layers=layers,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
        dtype_name=dtype_name,
        revision=revision,
    )


def _local_weight_size(path: Path) -> int:
    safetensors = list(path.rglob("*.safetensors"))
    files = safetensors or list(path.rglob("*.bin"))
    return sum(item.stat().st_size for item in files if item.is_file())


def _remote_weight_size(model_info: object) -> int:
    siblings = getattr(model_info, "siblings", None) or []
    safetensors = [
        item
        for item in siblings
        if str(getattr(item, "rfilename", "")).endswith(".safetensors")
    ]
    candidates = safetensors or [
        item
        for item in siblings
        if str(getattr(item, "rfilename", "")).endswith((".bin", ".pt"))
    ]
    total = sum(int(getattr(item, "size", 0) or 0) for item in candidates)
    if total:
        return total
    metadata = getattr(model_info, "safetensors", None)
    return int(getattr(metadata, "total", 0) or 0)


def load_model_profile(
    model_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> ModelProfile:
    """Fetch only configuration and file metadata; this does not load model weights."""
    try:
        from transformers import AutoConfig

        config_object = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        config = config_object.to_dict()
    except Exception as exc:
        raise SizingError(
            f"Could not read model configuration for {model_id!r}: {exc}"
        ) from exc

    local_path = Path(model_id).expanduser()
    resolved_revision = revision
    if local_path.exists():
        weights_bytes = _local_weight_size(local_path)
    else:
        try:
            from huggingface_hub import HfApi

            token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGING_FACE_HUB_TOKEN"
            )
            info = HfApi(token=token).model_info(
                model_id, revision=revision, files_metadata=True
            )
            weights_bytes = _remote_weight_size(info)
            resolved_revision = str(getattr(info, "sha", "") or revision or "") or None
        except Exception as exc:
            raise SizingError(
                "Could not obtain model weight metadata. Check network/model access "
                f"for {model_id!r}: {exc}"
            ) from exc
    return profile_from_config_dict(
        model_id=model_id,
        config=config,
        weights_bytes=weights_bytes,
        revision=resolved_revision,
    )

