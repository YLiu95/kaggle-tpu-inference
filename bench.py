#!/usr/bin/env python3
"""Decode-latency microbenchmark / ablation harness."""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ.setdefault("HF_HOME", "/root/hf_cache")

import jax
import jax.numpy as jnp

CACHE_DIR = os.environ.get("GEMMA4_XLA_CACHE", "/root/.cache/gemma4_jax")
os.makedirs(CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma4_tpu.engine import Engine  # noqa: E402

MODEL_DIR = os.environ.get(
    "GEMMA4_MODEL_DIR",
    "/root/hf_cache/hub/models--google--gemma-4-26B-A4B-it/snapshots/"
    "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--ablate", default="", help="comma list: moe,attn,mlp,lmhead,densemoe")
    args = ap.parse_args()

    if args.ablate:
        os.environ["GEMMA4_ABLATE"] = args.ablate

    t0 = time.time()
    eng = Engine(MODEL_DIR, max_len=args.max_len)
    print(f"load {time.time() - t0:.1f}s  weights {eng.param_bytes / 2**30:.1f} GiB", flush=True)

    ct = eng.compile_all()
    print(f"compile prefill {ct['prefill_compile_s']:.1f}s decode {ct['decode_compile_s']:.1f}s", flush=True)

    key = jax.random.PRNGKey(0)
    temp, top_p = eng._put(0.0, jnp.float32), eng._put(1.0, jnp.float32)
    toks = jax.device_put(jnp.zeros((1, 256), jnp.int32), eng._repl)

    t0 = time.perf_counter()
    tok, eng.cache, pos = eng.prefill_fn(256)(eng.params, toks, eng.cache, eng._put(64), key, temp, top_p)
    jax.block_until_ready(tok)
    print(f"prefill(256 padded, 64 real) {1000 * (time.perf_counter() - t0):.1f} ms", flush=True)

    decode = eng.decode_fn()
    for _ in range(4):
        tok, eng.cache, pos = decode(eng.params, tok, eng.cache, pos, key, temp, top_p)
    jax.block_until_ready(tok)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        tok, eng.cache, pos = decode(eng.params, tok, eng.cache, pos, key, temp, top_p)
    jax.block_until_ready(tok)
    dt = time.perf_counter() - t0
    per = dt / args.steps
    hbm_per_dev = (2 * eng.active_params) / eng.n_devices
    print(
        f"decode {1000 * per:.2f} ms/token  {1 / per:.2f} tok/s  "
        f"(weights-only implied BW {hbm_per_dev / per / 1e9:.0f} GB/s per chip)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
