"""Prefill/decode engine: jitted SPMD steps plus an async-dispatch streaming loop."""

from __future__ import annotations

import time
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from . import model as M
from .config import TextConfig, load_text_config
from .weights import active_params_per_token, load_params, param_bytes

PREFILL_BUCKET = 256


def _round_up(x: int, m: int) -> int:
    return max(m, ((x + m - 1) // m) * m)


class Engine:
    def __init__(
        self,
        model_dir: str,
        max_len: int = 4096,
        batch: int = 1,
        top_k: int = 64,
        progress=None,
    ):
        self.model_dir = model_dir
        self.cfg: TextConfig = load_text_config(model_dir)
        self.max_len = max_len
        self.batch = batch
        self.top_k = top_k
        self.mesh = M.make_mesh()
        self.n_devices = int(self.mesh.devices.size)
        self._repl = NamedSharding(self.mesh, P())

        t0 = time.time()
        self.params = load_params(self.cfg, model_dir, self.mesh, progress=progress)
        jax.block_until_ready(list(self.params.values()))
        self.load_seconds = time.time() - t0

        self.cache = M.init_cache(self.cfg, self.mesh, batch, max_len)
        self._prefill_fns: dict[int, object] = {}
        self._decode = None
        self.param_bytes = param_bytes(self.cfg)
        self.cache_bytes = M.cache_bytes(self.cfg, batch, max_len)
        self.active_params = active_params_per_token(self.cfg)
        self.decode_step_seconds = 0.0

    # ---------------------------------------------------------------- jitted steps
    def _build_prefill(self, t: int):
        cfg, mesh, top_k = self.cfg, self.mesh, self.top_k

        def fn(params, tokens, cache, prompt_len, key, temperature, top_p):
            masks = M.prefill_masks(cfg, t, prompt_len)
            logits, cache = M.forward(
                cfg, mesh, params, tokens, jnp.arange(t), cache, 0, masks,
                kv_len=t, dense_moe=True, last_index=prompt_len - 1,
            )
            tok = M.sample(logits[0, 0], jax.random.fold_in(key, 0), temperature, top_k, top_p)
            return tok.reshape(1, 1), cache, prompt_len

        return jax.jit(fn, donate_argnums=(2,))

    def _build_decode(self):
        cfg, mesh, max_len, top_k = self.cfg, self.mesh, self.max_len, self.top_k

        def fn(params, token, cache, pos, key, temperature, top_p):
            masks = M.decode_masks(cfg, max_len, pos)
            logits, cache = M.forward(
                cfg, mesh, params, token, pos.reshape(1), cache, pos, masks,
                kv_len=max_len, dense_moe=False, last_index=None,
            )
            tok = M.sample(logits[0, 0], jax.random.fold_in(key, pos), temperature, top_k, top_p)
            return tok.reshape(1, 1), cache, pos + 1

        return jax.jit(fn, donate_argnums=(2,))

    def prefill_fn(self, t: int):
        if t not in self._prefill_fns:
            self._prefill_fns[t] = self._build_prefill(t)
        return self._prefill_fns[t]

    def decode_fn(self):
        if self._decode is None:
            self._decode = self._build_decode()
        return self._decode

    # ---------------------------------------------------------------- helpers
    def reset_cache(self):
        self.cache = M.init_cache(self.cfg, self.mesh, self.batch, self.max_len)

    def _put(self, value, dtype=jnp.int32):
        return jax.device_put(jnp.asarray(value, dtype), self._repl)

    def compile_all(self, bucket: int = PREFILL_BUCKET) -> dict[str, float]:
        """Warm up XLA compilation for one prefill bucket + the decode step.

        Inputs must carry exactly the shardings used by :meth:`generate`, otherwise jit
        specialises again and the warm-up is wasted.
        """
        out = {}
        key = jax.random.PRNGKey(0)
        temp, top_p = self._put(0.0, jnp.float32), self._put(1.0, jnp.float32)
        toks = jax.device_put(jnp.zeros((self.batch, bucket), jnp.int32), self._repl)

        t0 = time.perf_counter()
        tok, self.cache, pos = self.prefill_fn(bucket)(
            self.params, toks, self.cache, self._put(4), key, temp, top_p
        )
        jax.block_until_ready(tok)
        out["prefill_compile_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        tok, self.cache, pos = self.decode_fn()(
            self.params, tok, self.cache, pos, key, temp, top_p
        )
        jax.block_until_ready(tok)
        out["decode_compile_s"] = time.perf_counter() - t0

        # Blocking device time for a single decode step: used as the reference for the
        # measured TensorCore-busy percentage (libtpu's duty-cycle gRPC is dead on Kaggle).
        n = 8
        t0 = time.perf_counter()
        for _ in range(n):
            tok, self.cache, pos = self.decode_fn()(
                self.params, tok, self.cache, pos, key, temp, top_p
            )
        jax.block_until_ready(tok)
        self.decode_step_seconds = (time.perf_counter() - t0) / n
        out["decode_step_s"] = self.decode_step_seconds

        self.reset_cache()
        return out

    # ---------------------------------------------------------------- generation
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: int = 0,
        stop_ids: tuple[int, ...] | None = None,
        lookahead: int = 3,
    ) -> Iterator[dict]:
        """Yields one event per generated token.

        Decode steps are dispatched ``lookahead`` steps ahead of the host read so the TPU
        never idles waiting for Python; tokens are fed back as device arrays.
        """
        stop = set(stop_ids if stop_ids is not None else self.cfg.eos_token_ids)
        p_len = len(prompt_ids)
        if p_len + max_new_tokens > self.max_len:
            raise ValueError(
                f"prompt({p_len}) + max_new({max_new_tokens}) exceeds max_len={self.max_len}"
            )
        bucket = min(_round_up(p_len, PREFILL_BUCKET), self.max_len)

        self.reset_cache()
        toks = np.zeros((self.batch, bucket), np.int32)
        toks[0, :p_len] = prompt_ids
        key = jax.random.PRNGKey(seed)
        temp_d = self._put(temperature, jnp.float32)
        top_p_d = self._put(top_p, jnp.float32)

        t_start = time.perf_counter()
        tok, self.cache, pos = self.prefill_fn(bucket)(
            self.params, jax.device_put(jnp.asarray(toks), self._repl), self.cache,
            self._put(p_len), key, temp_d, top_p_d,
        )
        first = int(jax.device_get(tok)[0, 0])
        ttft = time.perf_counter() - t_start
        yield {
            "kind": "prefill",
            "token": first,
            "ttft": ttft,
            "prompt_tokens": p_len,
            "prefill_bucket": bucket,
            "t": time.perf_counter(),
        }
        if first in stop:
            return

        decode = self.decode_fn()
        pending: list[jax.Array] = []
        emitted = 1
        stopped = False

        def flush(keep: int):
            nonlocal emitted, stopped
            while len(pending) > keep:
                arr = pending.pop(0)
                value = int(jax.device_get(arr)[0, 0])
                emitted += 1
                yield {
                    "kind": "decode",
                    "token": value,
                    "index": emitted - 1,
                    "t": time.perf_counter(),
                    "total_tokens": emitted,
                }
                if value in stop:
                    stopped = True
                    return

        for _ in range(max_new_tokens - 1):
            tok, self.cache, pos = decode(self.params, tok, self.cache, pos, key, temp_d, top_p_d)
            pending.append(tok)
            for ev in flush(lookahead):
                yield ev
            if stopped:
                return
        for ev in flush(0):
            yield ev
