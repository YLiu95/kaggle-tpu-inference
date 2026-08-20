"""Streaming safetensors -> sharded JAX device arrays loader."""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Callable

import jax
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding

from .config import TextConfig
from .model import FULL, SLIDING, param_shapes, param_specs

_TORCH_TO_NP = {"bfloat16": ml_dtypes.bfloat16, "float16": np.float16, "float32": np.float32}


def _to_numpy(t) -> np.ndarray:
    import torch

    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.contiguous().numpy()


def checkpoint_key_map(cfg: TextConfig, prefix: str = "model.language_model.") -> dict[str, str]:
    """checkpoint tensor name -> flat parameter name used by :mod:`model`."""
    lm = prefix
    out = {lm + "embed_tokens.weight": "embed", lm + "norm.weight": "norm"}
    simple = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ]
    for i in range(cfg.num_hidden_layers):
        src = f"{lm}layers.{i}."
        dst = f"layers.{i}."
        for name in simple:
            out[src + name + ".weight"] = dst + name
        out[src + "layer_scalar"] = dst + "layer_scalar"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            out[src + "self_attn." + name + ".weight"] = dst + name
        for name in ("gate_proj", "up_proj", "down_proj"):
            out[src + "mlp." + name + ".weight"] = dst + "mlp." + name
        out[src + "router.proj.weight"] = dst + "router.proj"
        out[src + "router.scale"] = dst + "router.scale"
        out[src + "router.per_expert_scale"] = dst + "router.per_expert_scale"
        out[src + "experts.gate_up_proj"] = dst + "experts.gate_up_proj"
        out[src + "experts.down_proj"] = dst + "experts.down_proj"
    return out


def _reshape(cfg: TextConfig, flat_name: str, arr: np.ndarray) -> np.ndarray:
    target = param_shapes(cfg)[flat_name]
    if arr.shape == target:
        return arr
    if arr.size != int(np.prod(target)):
        raise ValueError(f"{flat_name}: checkpoint shape {arr.shape} != expected {target}")
    return arr.reshape(target)


def params_from_arrays(cfg: TextConfig, get, names, mesh: Mesh, prefix: str = "model.language_model."):
    """Build sharded params from any ``name -> numpy array`` accessor."""
    shapes = param_shapes(cfg)
    specs = param_specs(cfg)
    key_map = checkpoint_key_map(cfg, prefix)
    params: dict[str, jax.Array] = {}
    for ck in names:
        flat = key_map.get(ck)
        if flat is None or flat not in shapes or flat in params:
            continue
        arr = _reshape(cfg, flat, get(ck))
        params[flat] = jax.device_put(arr, NamedSharding(mesh, specs[flat]))
        del arr
    return params


def load_params(
    cfg: TextConfig,
    model_dir: str,
    mesh: Mesh,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, jax.Array]:
    shapes = param_shapes(cfg)
    specs = param_specs(cfg)
    key_map = checkpoint_key_map(cfg)

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {model_dir}")

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        wanted_by_file: dict[str, list[str]] = {}
        for ck in key_map:
            if ck in weight_map:
                wanted_by_file.setdefault(os.path.join(model_dir, weight_map[ck]), []).append(ck)
    else:
        wanted_by_file = {f: list(key_map) for f in files}

    from safetensors import safe_open

    params: dict[str, jax.Array] = {}
    total = len(shapes)
    done = 0
    for path, keys in wanted_by_file.items():
        with safe_open(path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            for ck in keys:
                if ck not in available:
                    continue
                flat = key_map[ck]
                if flat not in shapes:
                    continue
                arr = _reshape(cfg, flat, _to_numpy(f.get_tensor(ck)))
                params[flat] = jax.device_put(arr, NamedSharding(mesh, specs[flat]))
                del arr
                done += 1
                if progress:
                    progress(done, total, flat)

    missing = sorted(set(shapes) - set(params))
    if missing:
        raise KeyError(f"missing {len(missing)} params, e.g. {missing[:5]}")
    return params


def param_bytes(cfg: TextConfig) -> int:
    return sum(2 * int(np.prod(s)) for s in param_shapes(cfg).values())


def active_params_per_token(cfg: TextConfig) -> int:
    """Parameters actually touched when decoding one token (for MFU/MBU accounting)."""
    h = cfg.hidden_size
    total = 0
    for lt in cfg.layer_types:
        d = cfg.head_dim_for(lt)
        kv = cfg.kv_heads_for(lt)
        total += cfg.num_attention_heads * d * h  # q
        total += kv * d * h  # k
        if not cfg.k_eq_v_for(lt):
            total += kv * d * h  # v
        total += cfg.num_attention_heads * d * h  # o
        total += 3 * cfg.intermediate_size * h
        if cfg.enable_moe_block:
            total += cfg.num_experts * h  # router
            total += cfg.top_k_experts * 3 * cfg.moe_intermediate_size * h
    total += cfg.vocab_size * h  # lm_head (tied)
    return total
