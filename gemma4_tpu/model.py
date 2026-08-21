"""JAX/SPMD implementation of the Gemma-4 text decoder for TPU v5e.

Handles both checkpoint families with one code path: dense models (31B, 12B) simply have
``enable_moe_block=False``, which drops the router/expert branch and leaves the Megatron
dense MLP.

Sharding strategy (1-D mesh named ``tp`` over all 8 chips, activations replicated):

* sliding layers  -> q/k/v/o sharded over the KV heads (q heads follow their KV head)
* full layers     -> q/o sharded over the query groups, KV replicated (too few KV heads
                     to split: 2 on the 26B-A4B, 4 on the 31B)
* dense MLP       -> Megatron column/row split over ``intermediate_size``
* MoE experts     -> split over ``moe_intermediate_size`` so every chip keeps all 128
                     experts but only 1/8 of each expert's hidden dim. This keeps the
                     per-token expert gather local (no cross-chip all-to-all) and turns
                     the combine into one small all-reduce.
* embeddings      -> split over ``hidden_size`` (lookup is local, lm_head becomes an
                     all-reduce over the contracted dim)
"""

from __future__ import annotations

import math
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import TextConfig

Pytree = Any
FULL = "full_attention"
SLIDING = "sliding_attention"
NEG_INF = np.float32(-1e30)

# Benchmark-only ablations (see bench.py); read lazily so bench can set them post-import.
def ablate(name: str) -> bool:
    return name in os.environ.get("GEMMA4_ABLATE", "").split(",")


# --------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------
def rms_norm(x: jax.Array, weight: jax.Array | None, eps: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    out = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)
    if weight is not None:
        out = out * weight.astype(jnp.float32)
    return out.astype(x.dtype)


def gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x, approximate=True)


def rope_inv_freq(head_dim: int, theta: float, partial_rotary_factor: float = 1.0) -> np.ndarray:
    """Mirrors transformers' default + ``proportional`` rope init (NoPE tail = zero freq)."""
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_dim))
    nope = head_dim // 2 - rope_angles
    if nope > 0:
        inv = np.concatenate([inv, np.zeros(nope, dtype=np.float64)])
    return inv.astype(np.float32)


def rope_tables(positions: jax.Array, inv_freq: jax.Array) -> tuple[jax.Array, jax.Array]:
    freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """x: [B, T, ..., D]; cos/sin: [T, D] (broadcast over the head axes)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    shape = (1, cos.shape[0]) + (1,) * (x.ndim - 3) + (d,)
    c = cos.reshape(shape).astype(x.dtype)
    s = sin.reshape(shape).astype(x.dtype)
    return x * c + rotated * s


# --------------------------------------------------------------------------------------
# parameter layout / sharding
# --------------------------------------------------------------------------------------
def param_shapes(cfg: TextConfig) -> dict[str, tuple[int, ...]]:
    """Canonical (already reshaped) shapes of every parameter, keyed by flat name."""
    h = cfg.hidden_size
    shapes: dict[str, tuple[int, ...]] = {"embed": (cfg.vocab_size, h), "norm": (h,)}
    for i, lt in enumerate(cfg.layer_types):
        d = cfg.head_dim_for(lt)
        kv = cfg.kv_heads_for(lt)
        g = cfg.num_attention_heads // kv
        p = f"layers.{i}."
        shapes[p + "input_layernorm"] = (h,)
        shapes[p + "post_attention_layernorm"] = (h,)
        shapes[p + "pre_feedforward_layernorm"] = (h,)
        shapes[p + "post_feedforward_layernorm"] = (h,)
        shapes[p + "layer_scalar"] = (1,)
        shapes[p + "q_proj"] = (kv, g, d, h)
        shapes[p + "k_proj"] = (kv, d, h)
        if not cfg.k_eq_v_for(lt):
            shapes[p + "v_proj"] = (kv, d, h)
        shapes[p + "o_proj"] = (h, kv, g, d)
        shapes[p + "q_norm"] = (d,)
        shapes[p + "k_norm"] = (d,)
        shapes[p + "mlp.gate_proj"] = (cfg.intermediate_size, h)
        shapes[p + "mlp.up_proj"] = (cfg.intermediate_size, h)
        shapes[p + "mlp.down_proj"] = (h, cfg.intermediate_size)
        if cfg.enable_moe_block:
            shapes[p + "pre_feedforward_layernorm_2"] = (h,)
            shapes[p + "post_feedforward_layernorm_1"] = (h,)
            shapes[p + "post_feedforward_layernorm_2"] = (h,)
            shapes[p + "router.proj"] = (cfg.num_experts, h)
            shapes[p + "router.scale"] = (h,)
            shapes[p + "router.per_expert_scale"] = (cfg.num_experts,)
            shapes[p + "experts.gate_up_proj"] = (cfg.num_experts, 2, cfg.moe_intermediate_size, h)
            shapes[p + "experts.down_proj"] = (cfg.num_experts, h, cfg.moe_intermediate_size)
    return shapes


def param_specs(cfg: TextConfig) -> dict[str, P]:
    known = param_shapes(cfg)
    specs: dict[str, P] = {"embed": P(None, "tp"), "norm": P(None)}
    for i, lt in enumerate(cfg.layer_types):
        p = f"layers.{i}."
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
            "layer_scalar",
            "q_norm",
            "k_norm",
            "router.scale",
            "router.per_expert_scale",
        ):
            specs[p + name] = P(None)
        specs[p + "router.proj"] = P(None, None)
        if lt == SLIDING:
            specs[p + "q_proj"] = P("tp", None, None, None)
            specs[p + "k_proj"] = P("tp", None, None)
            specs[p + "v_proj"] = P("tp", None, None)
            specs[p + "o_proj"] = P(None, "tp", None, None)
        else:  # only 2 KV heads -> replicate KV, shard the 8 query groups
            specs[p + "q_proj"] = P(None, "tp", None, None)
            specs[p + "k_proj"] = P(None, None, None)
            specs[p + "v_proj"] = P(None, None, None)
            specs[p + "o_proj"] = P(None, None, "tp", None)
        specs[p + "mlp.gate_proj"] = P("tp", None)
        specs[p + "mlp.up_proj"] = P("tp", None)
        specs[p + "mlp.down_proj"] = P(None, "tp")
        specs[p + "experts.gate_up_proj"] = P(None, None, "tp", None)
        specs[p + "experts.down_proj"] = P(None, None, "tp")
    return {k: v for k, v in specs.items() if k in known}


def make_mesh() -> Mesh:
    devs = np.array(jax.devices())
    return Mesh(devs.reshape(devs.size), ("tp",))


def validate_sharding(cfg: TextConfig, n_devices: int) -> None:
    """Fail loudly (and early) if this checkpoint cannot be split over ``n_devices``.

    Every axis the 1-D ``tp`` mesh splits must divide evenly; XLA would otherwise pad
    silently and waste HBM, or reject the sharding minutes into a load.
    """
    problems = []

    def need(value: int, what: str) -> None:
        if value % n_devices:
            problems.append(f"{what}={value} is not divisible by {n_devices}")

    need(cfg.hidden_size, "hidden_size")
    need(cfg.intermediate_size, "intermediate_size")
    if cfg.enable_moe_block:
        need(cfg.moe_intermediate_size, "moe_intermediate_size")
    for lt in set(cfg.layer_types):
        if lt == SLIDING:
            need(cfg.kv_heads_for(lt), "num_key_value_heads (sliding layers)")
        else:
            need(cfg.query_groups_for(lt), "query groups per KV head (full layers)")
    if problems:
        raise ValueError(
            f"checkpoint cannot be sharded over {n_devices} chips: " + "; ".join(problems)
        )


def prefill_attention_bytes_per_chip(cfg: TextConfig, n_devices: int, t: int) -> int:
    """Peak bytes one layer's attention scores need during a ``t``-token prefill.

    ``scores`` is materialised as ``[b, kv, g, t, t]`` float32 and ``probs`` as bf16, so
    the cost is quadratic in the prompt length and is what actually limits how long a
    prompt can be - long before the KV cache does. Full-attention layers are the worst
    case because their KV is replicated, so every chip carries all the query groups.
    """
    worst = 0
    for lt in set(cfg.layer_types):
        kv = cfg.kv_heads_for(lt)
        g = cfg.num_attention_heads // kv
        heads_per_chip = kv * g if lt == FULL else max(1, kv // n_devices) * g
        worst = max(worst, heads_per_chip * t * t * 6)  # float32 scores + bf16 probs
    return worst


def safe_prompt_tokens(
    cfg: TextConfig, n_devices: int, free_bytes: float, bucket: int = 256
) -> int:
    """Largest prefill bucket whose attention working set fits in ``free_bytes``."""
    if free_bytes <= 0:
        return 0
    t = bucket
    while prefill_attention_bytes_per_chip(cfg, n_devices, t + bucket) <= free_bytes:
        t += bucket
    return t


def hbm_estimate(cfg: TextConfig, n_devices: int, batch: int, max_len: int) -> dict[str, float]:
    """Bytes-per-chip of weights + KV cache, plus the resulting HBM fraction.

    Sliding layers shard their KV over the chips; full layers replicate it (too few KV
    heads to split), which is why the two are accounted separately.
    """
    weights = sum(2 * math.prod(s) for s in param_shapes(cfg).values()) / n_devices
    kv_sharded = 0
    kv_replicated = 0
    for lt in cfg.layer_types:
        per_layer = 2 * 2 * batch * max_len * cfg.kv_heads_for(lt) * cfg.head_dim_for(lt)
        if lt == SLIDING:
            kv_sharded += per_layer
        else:
            kv_replicated += per_layer
    cache = kv_sharded / n_devices + kv_replicated
    return {
        "weights_bytes_per_chip": float(weights),
        "cache_bytes_per_chip": float(cache),
        "total_bytes_per_chip": float(weights + cache),
        "kv_bytes_per_token_per_chip": float(cache / max(1, max_len * batch)),
    }


# --------------------------------------------------------------------------------------
# KV cache (one array per layer: avoids copying a giant stacked buffer on every layer)
# --------------------------------------------------------------------------------------
def cache_specs(cfg: TextConfig) -> list[P]:
    return [
        P(None, None, "tp", None) if lt == SLIDING else P(None, None, None, None)
        for lt in cfg.layer_types
    ]


def cache_shapes(cfg: TextConfig, batch: int, max_len: int) -> list[tuple[int, ...]]:
    return [(batch, max_len, cfg.kv_heads_for(lt), cfg.head_dim_for(lt)) for lt in cfg.layer_types]


def init_cache(cfg: TextConfig, mesh: Mesh, batch: int, max_len: int, dtype=jnp.bfloat16) -> dict:
    out: dict[str, list] = {"k": [], "v": []}
    for shape, spec in zip(cache_shapes(cfg, batch, max_len), cache_specs(cfg)):
        sh = NamedSharding(mesh, spec)
        zeros = jax.jit(lambda s=shape: jnp.zeros(s, dtype), out_shardings=sh)()
        out["k"].append(zeros)
        out["v"].append(jax.jit(lambda s=shape: jnp.zeros(s, dtype), out_shardings=sh)())
    return out


def cache_bytes(cfg: TextConfig, batch: int, max_len: int) -> int:
    return sum(2 * 2 * math.prod(s) for s in cache_shapes(cfg, batch, max_len))


# --------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------
def _attention(cfg, lp, x, layer_type, cos, sin, cache_k, cache_v, write_at, mask, kv_len):
    eps = cfg.rms_norm_eps
    q = jnp.einsum("bth,kgdh->btkgd", x, lp["q_proj"])
    q = apply_rope(rms_norm(q, lp["q_norm"], eps), cos, sin)

    k_raw = jnp.einsum("bth,kdh->btkd", x, lp["k_proj"])
    v_raw = k_raw if cfg.k_eq_v_for(layer_type) else jnp.einsum("bth,kdh->btkd", x, lp["v_proj"])
    k = apply_rope(rms_norm(k_raw, lp["k_norm"], eps), cos, sin)
    v = rms_norm(v_raw, None, eps)

    cache_k = jax.lax.dynamic_update_slice(cache_k, k.astype(cache_k.dtype), (0, write_at, 0, 0))
    cache_v = jax.lax.dynamic_update_slice(cache_v, v.astype(cache_v.dtype), (0, write_at, 0, 0))
    kk = cache_k[:, :kv_len]
    vv = cache_v[:, :kv_len]

    scores = jnp.einsum("btkgd,bskd->bkgts", q, kk).astype(jnp.float32)
    scores = jnp.where(mask, scores, NEG_INF)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    ctx = jnp.einsum("bkgts,bskd->btkgd", probs, vv)
    out = jnp.einsum("btkgd,hkgd->bth", ctx, lp["o_proj"])
    return out, cache_k, cache_v


def _dense_mlp(lp: dict, x: jax.Array) -> jax.Array:
    gate = jnp.einsum("bth,ih->bti", x, lp["mlp.gate_proj"])
    up = jnp.einsum("bth,ih->bti", x, lp["mlp.up_proj"])
    return jnp.einsum("bti,hi->bth", gelu(gate) * up, lp["mlp.down_proj"])


def _router(cfg: TextConfig, lp: dict, flat: jax.Array):
    x = rms_norm(flat, None, cfg.rms_norm_eps)
    x = x * lp["router.scale"] * jnp.asarray(cfg.hidden_size**-0.5, x.dtype)
    scores = jnp.einsum("nh,eh->ne", x, lp["router.proj"]).astype(jnp.float32)
    probs = jax.nn.softmax(scores, axis=-1)
    top_w, top_i = jax.lax.top_k(probs, cfg.top_k_experts)
    top_w = top_w / jnp.sum(top_w, axis=-1, keepdims=True)
    top_w = top_w * lp["router.per_expert_scale"].astype(jnp.float32)[top_i]
    return top_w, top_i


def _moe(cfg: TextConfig, lp: dict, flat: jax.Array, dense: bool) -> jax.Array:
    """flat: [N, H], already normalised by pre_feedforward_layernorm_2."""
    top_w, top_i = _router(cfg, lp, flat)
    gate_up = lp["experts.gate_up_proj"]  # [E, 2, F, H]
    down = lp["experts.down_proj"]  # [E, H, F]

    if dense:
        # Prefill: evaluate every expert for every token. One big matmul beats a huge
        # dynamic gather; routing weights zero out the unselected experts.
        w_full = jnp.einsum(
            "nk,nke->ne", top_w, jax.nn.one_hot(top_i, cfg.num_experts, dtype=jnp.float32)
        )
        gu = jnp.einsum("nh,ecfh->necf", flat, gate_up)
        act = gelu(gu[:, :, 0]) * gu[:, :, 1]
        act = act * w_full[:, :, None].astype(act.dtype)
        return jnp.einsum("nef,ehf->nh", act, down)

    mode = os.environ.get("GEMMA4_MOE_DECODE", "slice")
    if mode == "take":
        # One fused gather of the top-k slices; simple, but XLA materialises the copy.
        gu_w = jnp.take(gate_up, top_i, axis=0)  # [N, K, 2, F, H]
        gu = jnp.einsum("nh,nkcfh->nkcf", flat, gu_w)
        act = gelu(gu[:, :, 0]) * gu[:, :, 1]
        act = act * top_w[:, :, None].astype(act.dtype)
        dn_w = jnp.take(down, top_i, axis=0)  # [N, K, H, F]
        return jnp.einsum("nkf,nkhf->nh", act, dn_w)

    # Default decode path: one dynamic-slice per selected expert. Each slice reads only
    # that expert's shard from HBM, unlike jnp.take which XLA materialises in full.
    n_tokens = flat.shape[0]
    h_dim = flat.shape[1]

    def expert_out(n: int, k):
        e = top_i[n, k]
        gu_w = jax.lax.dynamic_index_in_dim(gate_up, e, axis=0, keepdims=False)
        gu = jnp.einsum("h,cfh->cf", flat[n], gu_w)
        a = gelu(gu[0]) * gu[1] * top_w[n, k].astype(gu.dtype)
        dn_w = jax.lax.dynamic_index_in_dim(down, e, axis=0, keepdims=False)
        return jnp.einsum("f,hf->h", a, dn_w)

    rows = []
    for n in range(n_tokens):
        if mode == "loop":
            # Rolled: 8x smaller HLO, which matters because tracing/lowering 30 unrolled
            # layers is most of the cold-start time.
            rows.append(
                jax.lax.fori_loop(
                    0,
                    cfg.top_k_experts,
                    lambda k, acc, n=n: acc + expert_out(n, k).astype(jnp.float32),
                    jnp.zeros((h_dim,), jnp.float32),
                ).astype(flat.dtype)
            )
        else:
            acc = None
            for k in range(cfg.top_k_experts):
                part = expert_out(n, k)
                acc = part if acc is None else acc + part
            rows.append(acc)
    return jnp.stack(rows, axis=0)


def _layer(cfg, lp, x, layer_type, cos, sin, ck, cv, write_at, mask, kv_len, dense_moe, repl):
    eps = cfg.rms_norm_eps
    residual = x
    if not ablate("attn"):
        h = rms_norm(x, lp["input_layernorm"], eps)
        h, ck, cv = _attention(cfg, lp, h, layer_type, cos, sin, ck, cv, write_at, mask, kv_len)
        h = rms_norm(repl(h), lp["post_attention_layernorm"], eps)
        x = residual + h

    residual = x
    if ablate("mlp"):
        h = x
    else:
        h = repl(_dense_mlp(lp, rms_norm(x, lp["pre_feedforward_layernorm"], eps)))

    if cfg.enable_moe_block and not ablate("moe"):
        h1 = rms_norm(h, lp["post_feedforward_layernorm_1"], eps)
        b, t, hid = residual.shape
        h2 = rms_norm(residual.reshape(-1, hid), lp["pre_feedforward_layernorm_2"], eps)
        h2 = repl(_moe(cfg, lp, h2, dense_moe or ablate("densemoe")).reshape(b, t, hid))
        h2 = rms_norm(h2, lp["post_feedforward_layernorm_2"], eps)
        h = h1 + h2

    h = rms_norm(h, lp["post_feedforward_layernorm"], eps)
    x = residual + h
    return x * lp["layer_scalar"].astype(x.dtype), ck, cv


# --------------------------------------------------------------------------------------
# full forward
# --------------------------------------------------------------------------------------
def _layer_params(params: dict, i: int) -> dict:
    prefix = f"layers.{i}."
    return {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}


def forward(
    cfg: TextConfig,
    mesh: Mesh,
    params: dict,
    tokens: jax.Array,
    positions: jax.Array,
    cache: dict,
    write_at,
    masks: dict,
    kv_len: int,
    dense_moe: bool,
    last_index=None,
):
    """Returns (logits, updated cache)."""
    repl3 = NamedSharding(mesh, P(None, None, None))
    shard_h = NamedSharding(mesh, P(None, None, "tp"))

    def repl(v):
        return jax.lax.with_sharding_constraint(
            v, NamedSharding(mesh, P(*([None] * v.ndim)))
        )

    embed = params["embed"]
    x = repl(jnp.take(embed, tokens, axis=0))
    x = x * jnp.asarray(np.float32(cfg.embed_scale)).astype(jnp.bfloat16)

    inv_sl = jnp.asarray(rope_inv_freq(cfg.head_dim, cfg.rope_theta_sliding))
    inv_fu = jnp.asarray(
        rope_inv_freq(cfg.global_head_dim, cfg.rope_theta_full, cfg.rope_partial_rotary_factor_full)
    )
    rope = {SLIDING: rope_tables(positions, inv_sl), FULL: rope_tables(positions, inv_fu)}

    ks, vs = list(cache["k"]), list(cache["v"])
    for i, lt in enumerate(cfg.layer_types):
        cos, sin = rope[lt]
        x, ks[i], vs[i] = _layer(
            cfg, _layer_params(params, i), x, lt, cos, sin, ks[i], vs[i],
            write_at, masks[lt], kv_len, dense_moe, repl,
        )

    x = rms_norm(x, params["norm"], cfg.rms_norm_eps)
    if last_index is not None:
        x = jax.lax.dynamic_slice_in_dim(x, last_index, 1, axis=1)

    if ablate("lmhead"):
        logits = jnp.zeros(x.shape[:2] + (cfg.vocab_size,), jnp.float32) + x[..., :1]
        return logits, {"k": ks, "v": vs}

    logits = jnp.einsum("bth,vh->btv", jax.lax.with_sharding_constraint(x, shard_h), embed)
    logits = jax.lax.with_sharding_constraint(logits, repl3).astype(jnp.float32)
    if cfg.final_logit_softcapping:
        cap = jnp.float32(cfg.final_logit_softcapping)
        logits = jnp.tanh(logits / cap) * cap
    return logits, {"k": ks, "v": vs}


# --------------------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------------------
def prefill_masks(cfg: TextConfig, t: int, prompt_len) -> dict:
    q = jnp.arange(t)[:, None]
    kv = jnp.arange(t)[None, :]
    valid = (kv <= q) & (kv < prompt_len)
    return {
        FULL: valid[None, None, None],
        SLIDING: (valid & (kv > q - cfg.sliding_window))[None, None, None],
    }


def decode_masks(cfg: TextConfig, max_len: int, pos) -> dict:
    kv = jnp.arange(max_len)[None, :]
    valid = kv <= pos
    return {
        FULL: valid[None, None, None],
        SLIDING: (valid & (kv > pos - cfg.sliding_window))[None, None, None],
    }


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------
def sample(logits: jax.Array, key: jax.Array, temperature, top_k: int, top_p) -> jax.Array:
    """logits: [V] float32 -> scalar int32."""
    greedy = jnp.argmax(logits, axis=-1)
    t = jnp.maximum(temperature, 1e-6)
    vals, idx = jax.lax.top_k(logits / t, top_k)
    probs = jax.nn.softmax(vals, axis=-1)
    csum = jnp.cumsum(probs, axis=-1)
    probs = jnp.where((csum - probs) < top_p, probs, 0.0)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    sampled = idx[jax.random.categorical(key, jnp.log(probs + 1e-20))]
    return jnp.where(temperature <= 0.0, greedy, sampled).astype(jnp.int32)
