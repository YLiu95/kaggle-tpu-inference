"""Numerical parity: tiny random Gemma-4 models in HF/torch vs. this JAX SPMD stack.

Two architectures are checked, both reduced to CPU-runnable sizes but keeping every
structural feature that is easy to get silently wrong:

``moe``    the 26B-A4B shape - MoE block, 8 KV heads sliding / 2 global, k == v, softcap
``dense``  the 31B shape     - no MoE, 16 KV heads sliding / 4 global (so full-attention
                               layers shard 8 query groups instead of KV heads)

Run directly (``python3 tests/test_parity.py``) or under pytest. The mesh is 8 devices
either way, so the sharding, the collectives and the ``k == v`` / partial-RoPE paths are
all exercised.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_platform = os.environ.get("GEMMA4_TEST_PLATFORM", "cpu")
os.environ.setdefault("JAX_PLATFORMS", _platform)
if _platform == "cpu" and "xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    ).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gemma4_tpu import model as M  # noqa: E402
from gemma4_tpu.config import load_text_config  # noqa: E402
from gemma4_tpu.weights import _to_numpy, params_from_arrays  # noqa: E402

LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]

ARCHS = {
    # 26B-A4B shape: MoE, 8 sliding KV heads, 2 global KV heads (16 heads -> 8 groups)
    "moe": dict(
        hidden_size=256,
        intermediate_size=128,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        head_dim=32,
        global_head_dim=64,
        enable_moe_block=True,
        num_experts=16,
        top_k_experts=4,
        moe_intermediate_size=64,
    ),
    # 31B shape: dense MLP, 16 sliding KV heads, 4 global KV heads (32 heads -> 8 groups)
    "dense": dict(
        hidden_size=256,
        intermediate_size=128,
        num_attention_heads=32,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        head_dim=32,
        global_head_dim=64,
        enable_moe_block=False,
        num_experts=None,
        top_k_experts=None,
        moe_intermediate_size=None,
    ),
}


def build_hf(arch: str):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    cfg = Gemma4TextConfig(
        vocab_size=1024,
        num_hidden_layers=len(LAYER_TYPES),
        sliding_window=8,
        layer_types=LAYER_TYPES,
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
        **ARCHS[arch],
    )
    torch.manual_seed(0)
    cfg._experts_implementation = "eager"  # CPU has no aten::_grouped_mm
    model = Gemma4ForCausalLM(cfg)
    for p in model.parameters():
        with torch.no_grad():
            p.normal_(0.0, 0.05)
    return cfg, model.to(torch.bfloat16).eval()


def run_parity(arch: str) -> dict:
    hf_cfg, hf_model = build_hf(arch)

    with tempfile.TemporaryDirectory() as td:
        raw = {"text_config": hf_cfg.to_dict(), "eos_token_id": [1]}
        with open(os.path.join(td, "config.json"), "w") as f:
            json.dump(raw, f, default=str)
        cfg = load_text_config(td)

    assert cfg.enable_moe_block == ARCHS[arch]["enable_moe_block"]

    sd = hf_model.state_dict()
    arrays = {k: _to_numpy(v.detach()) for k, v in sd.items()}
    mesh = M.make_mesh()
    M.validate_sharding(cfg, int(mesh.devices.size))
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

    logits, _ = jax.jit(run)(params, jnp.asarray(padded), cache, jnp.int32(seq))
    got = np.asarray(jax.device_get(logits))[0, :seq]
    exp = ref[0, :seq]

    err = np.abs(got - exp)
    top1 = float((got.argmax(-1) == exp.argmax(-1)).mean())
    corr = float(np.corrcoef(got.ravel(), exp.ravel())[0, 1])

    # decode path (sparse top-k MoE gather / single-token dense MLP) must match prefill
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
    dcorr = float(np.corrcoef(dgot, dexp)[0, 1])

    print(
        f"[{arch}] prefill max|d|={err.max():.4f} top1-match={top1:.3f} corr={corr:.6f}  "
        f"decode max|d|={np.abs(dgot - dexp).max():.4f} "
        f"argmax={'OK' if dgot.argmax() == dexp.argmax() else 'MISMATCH'} corr={dcorr:.6f}",
        flush=True,
    )
    return {"corr": corr, "top1": top1, "decode_corr": dcorr,
            "decode_argmax_ok": bool(dgot.argmax() == dexp.argmax())}


def test_parity_moe():
    r = run_parity("moe")
    assert r["corr"] > 0.999
    assert r["top1"] > 0.9
    assert r["decode_corr"] > 0.999


def test_parity_dense_31b_shape():
    r = run_parity("dense")
    assert r["corr"] > 0.999
    assert r["top1"] > 0.9
    assert r["decode_corr"] > 0.999
    assert r["decode_argmax_ok"]


def main() -> int:
    ok = True
    for arch in ARCHS:
        r = run_parity(arch)
        good = r["corr"] > 0.999 and r["top1"] > 0.9 and r["decode_corr"] > 0.999
        print(f"[{arch}] PARITY", "PASS" if good else "FAIL")
        ok = ok and good
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
