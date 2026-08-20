"""Numerical parity check: tiny random Gemma-4 in HF/torch vs. the JAX TPU path.

Uses the same MoE / sliding+full attention / k==v / softcap structure as the real
26B-A4B checkpoint, just small enough to run on CPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "tpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import torch

from gemma4_tpu import model as M
from gemma4_tpu.config import load_text_config
from gemma4_tpu.weights import _to_numpy, params_from_arrays

LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]


def build_hf():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    cfg = Gemma4TextConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=32,
        global_head_dim=64,
        num_global_key_value_heads=2,
        sliding_window=8,
        layer_types=LAYER_TYPES,
        num_experts=16,
        top_k_experts=4,
        moe_intermediate_size=64,
        enable_moe_block=True,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        use_bidirectional_attention="vision",
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    cfg._experts_implementation = "eager"  # CPU has no aten::_grouped_mm
    model = Gemma4ForCausalLM(cfg)
    for p in model.parameters():
        with torch.no_grad():
            p.normal_(0.0, 0.05)
    model = model.to(torch.bfloat16).eval()
    return cfg, model


def main() -> int:
    hf_cfg, hf_model = build_hf()

    with tempfile.TemporaryDirectory() as td:
        raw = {"text_config": hf_cfg.to_dict(), "eos_token_id": [1]}
        with open(os.path.join(td, "config.json"), "w") as f:
            json.dump(raw, f, default=str)
        cfg = load_text_config(td)

    sd = hf_model.state_dict()
    arrays = {k: _to_numpy(v.detach()) for k, v in sd.items()}
    mesh = M.make_mesh()
    with mesh:
        params = params_from_arrays(cfg, arrays.__getitem__, list(arrays), mesh, prefix="model.")

    missing = set(M.param_shapes(cfg)) - set(params)
    assert not missing, f"missing params: {sorted(missing)[:10]}"

    seq = 24
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(1, seq)).astype(np.int32)

    with torch.no_grad():
        ref = hf_model(input_ids=torch.tensor(ids, dtype=torch.long)).logits.float().numpy()

    bucket = 32
    padded = np.zeros((1, bucket), np.int32)
    padded[0, :seq] = ids[0]
    cache = M.init_cache(cfg, mesh, 1, bucket)

    def run(params, tokens, cache, prompt_len):
        masks = M.prefill_masks(cfg, bucket, prompt_len)
        logits, cache = M.forward(
            cfg, mesh, params, tokens, jnp.arange(bucket), cache, 0, masks,
            kv_len=bucket, dense_moe=True, last_index=None,
        )
        return logits, cache

    logits, cache = jax.jit(run, static_argnums=())(
        params, jnp.asarray(padded), cache, jnp.int32(seq)
    )
    got = np.asarray(jax.device_get(logits))[0, :seq]
    exp = ref[0, :seq]

    err = np.abs(got - exp)
    rel = err.max() / max(np.abs(exp).max(), 1e-6)
    top1 = (got.argmax(-1) == exp.argmax(-1)).mean()
    corr = np.corrcoef(got.ravel(), exp.ravel())[0, 1]
    print(f"prefill  max|d|={err.max():.4f}  rel={rel:.4f}  top1-match={top1:.3f}  corr={corr:.6f}")

    # sparse (top-k gather) decode path must match the dense prefill path
    dec_cache = M.init_cache(cfg, mesh, 1, bucket)
    pref_len = seq - 1
    pad2 = np.zeros((1, bucket), np.int32)
    pad2[0, :pref_len] = ids[0, :pref_len]
    _, dec_cache = jax.jit(run)(params, jnp.asarray(pad2), dec_cache, jnp.int32(pref_len))

    def step(params, token, cache, pos):
        masks = M.decode_masks(cfg, bucket, pos)
        return M.forward(
            cfg, mesh, params, token, pos.reshape(1), cache, pos, masks,
            kv_len=bucket, dense_moe=False, last_index=None,
        )

    dlogits, _ = jax.jit(step)(
        params, jnp.asarray(ids[:, pref_len : pref_len + 1]), dec_cache, jnp.int32(pref_len)
    )
    dgot = np.asarray(jax.device_get(dlogits))[0, 0]
    dexp = ref[0, seq - 1]
    derr = np.abs(dgot - dexp)
    print(
        f"decode   max|d|={derr.max():.4f}  "
        f"argmax {'OK' if dgot.argmax() == dexp.argmax() else 'MISMATCH'}  "
        f"corr={np.corrcoef(dgot, dexp)[0, 1]:.6f}"
    )

    ok = corr > 0.999 and top1 > 0.9 and np.corrcoef(dgot, dexp)[0, 1] > 0.999
    print("PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
