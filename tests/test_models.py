"""Model registry, dense-vs-MoE config loading, sharding and HBM-budget tests.

Runs on CPU: nothing here touches the TPU or the real weights (except the two tests that
opportunistically read a downloaded ``config.json``).
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from gemma4_tpu import model as M  # noqa: E402
from gemma4_tpu.config import load_text_config  # noqa: E402
from gemma4_tpu.models import MODELS, DEFAULT_MODEL_KEY, choices, describe_table, resolve, try_resolve  # noqa: E402

GIB = 2**30

# text_config of google/gemma-4-31B-it, trimmed to the keys the loader reads.
# The MoE keys are present but null, exactly as the real dense checkpoint ships them.
GEMMA4_31B_TEXT = {
    "attention_k_eq_v": True,
    "enable_moe_block": False,
    "expert_intermediate_size": None,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    "moe_intermediate_size": None,
    "num_attention_heads": 32,
    "num_experts": None,
    "num_global_key_value_heads": 4,
    "num_hidden_layers": 60,
    "num_key_value_heads": 16,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    "sliding_window": 1024,
    "top_k_experts": None,
    "vocab_size": 262144,
}

GEMMA4_26B_A4B_TEXT = {
    **GEMMA4_31B_TEXT,
    "enable_moe_block": True,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_size": 2816,
    "intermediate_size": 2112,
    "moe_intermediate_size": 704,
    "num_attention_heads": 16,
    "num_experts": 128,
    "num_global_key_value_heads": 2,
    "num_hidden_layers": 30,
    "num_key_value_heads": 8,
    "top_k_experts": 8,
}


def write_config(text_config, layers=None):
    td = tempfile.mkdtemp()
    tc = dict(text_config)
    if layers is not None:
        tc["layer_types"] = layers
        tc["num_hidden_layers"] = len(layers)
    else:
        tc["layer_types"] = (
            tc["layer_types"] * (tc["num_hidden_layers"] // len(tc["layer_types"]))
        )[: tc["num_hidden_layers"]]
    with open(os.path.join(td, "config.json"), "w") as f:
        json.dump({"text_config": tc, "eos_token_id": [1, 106]}, f)
    return td


# ---------------------------------------------------------------- registry
def test_registry_contains_the_31b():
    spec = resolve("31b")
    assert spec.repo_id == "google/gemma-4-31B-it"
    assert spec.kind == "dense"
    assert spec.default_context == 16384
    assert spec.max_context == 30720


def test_aliases_and_repo_ids_resolve():
    for name in ("31b", "31B", "31b-it", "google/gemma-4-31B-it", " 31b "):
        assert resolve(name).key == "31b"
    assert resolve("26b-a4b").key == "26b-a4b"
    assert resolve("moe").key == "26b-a4b"
    assert resolve(None).key == DEFAULT_MODEL_KEY


def test_unknown_model_lists_the_alternatives():
    try:
        resolve("gpt-2")
    except ValueError as e:
        assert "31b" in str(e) and "26b-a4b" in str(e)
    else:
        raise AssertionError("expected ValueError")
    assert try_resolve("gpt-2") is None


def test_every_spec_is_self_consistent():
    for spec in MODELS:
        assert spec.default_context <= spec.max_context
        assert spec.key in choices()
        assert spec.repo_id.startswith("google/gemma-4-")
    assert "31b" in describe_table()


# ---------------------------------------------------------------- dense config loading
def test_dense_config_loads_with_null_moe_fields():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    assert cfg.enable_moe_block is False
    assert cfg.num_experts == 0 and cfg.top_k_experts == 0 and cfg.moe_intermediate_size == 0
    assert cfg.hidden_size == 5376 and cfg.num_hidden_layers == 60
    assert cfg.layer_types.count("full_attention") == 10
    assert cfg.eos_token_ids == (1, 106)


def test_dense_param_shapes_have_no_expert_tensors():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    shapes = M.param_shapes(cfg)
    assert not [k for k in shapes if "experts" in k or "router" in k]
    assert shapes["layers.0.mlp.gate_proj"] == (21504, 5376)
    # sliding layer: 16 KV heads x 2 query groups x head_dim 256
    assert shapes["layers.0.q_proj"] == (16, 2, 256, 5376)
    # full layer: 4 KV heads x 8 query groups x global head_dim 512, and k == v
    assert shapes["layers.5.q_proj"] == (4, 8, 512, 5376)
    assert "layers.5.v_proj" not in shapes
    assert "layers.0.v_proj" in shapes


def test_moe_config_still_loads():
    cfg = load_text_config(write_config(GEMMA4_26B_A4B_TEXT))
    assert cfg.enable_moe_block is True
    assert (cfg.num_experts, cfg.top_k_experts, cfg.moe_intermediate_size) == (128, 8, 704)
    assert "layers.0.experts.gate_up_proj" in M.param_shapes(cfg)


def test_moe_flag_without_expert_fields_is_rejected():
    broken = {**GEMMA4_31B_TEXT, "enable_moe_block": True}
    try:
        load_text_config(write_config(broken))
    except ValueError as e:
        assert "enable_moe_block" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------- sharding
def test_31b_shards_cleanly_over_eight_chips():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    M.validate_sharding(cfg, 8)  # must not raise
    specs = M.param_specs(cfg)
    assert specs["layers.0.q_proj"] == M.P("tp", None, None, None)   # sliding: split KV heads
    assert specs["layers.5.q_proj"] == M.P(None, "tp", None, None)   # full: split query groups
    assert specs["embed"] == M.P(None, "tp")


def test_26b_a4b_shards_cleanly_over_eight_chips():
    cfg = load_text_config(write_config(GEMMA4_26B_A4B_TEXT))
    M.validate_sharding(cfg, 8)
    assert M.param_specs(cfg)["layers.0.experts.gate_up_proj"] == M.P(None, None, "tp", None)


def test_sharding_validation_reports_the_offending_axis():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    try:
        M.validate_sharding(cfg, 5)
    except ValueError as e:
        assert "hidden_size=5376" in str(e) and "5" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_cache_specs_shard_sliding_and_replicate_full():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    specs = M.cache_specs(cfg)
    assert specs[0] == M.P(None, None, "tp", None)
    assert specs[5] == M.P(None, None, None, None)


# ---------------------------------------------------------------- HBM budget
def test_31b_weights_and_16k_cache_fit_a_v5e8():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    est = M.hbm_estimate(cfg, 8, 1, 16384)
    assert 6.5 < est["weights_bytes_per_chip"] / GIB < 8.0
    assert est["total_bytes_per_chip"] / GIB < 0.80 * 15.75


def test_31b_at_its_ceiling_stays_under_the_hbm_safety_line():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    est = M.hbm_estimate(cfg, 8, 1, resolve("31b").max_context)
    per_chip = est["total_bytes_per_chip"] / GIB
    assert per_chip < 0.80 * 15.75, "the advertised ceiling must actually fit"
    assert per_chip > 0.70 * 15.75, "the ceiling should not be needlessly conservative"


def test_prefill_attention_is_quadratic_and_bounds_prompt_length():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    a = M.prefill_attention_bytes_per_chip(cfg, 8, 1024)
    b = M.prefill_attention_bytes_per_chip(cfg, 8, 2048)
    assert b == 4 * a
    # full-attention layers replicate KV, so they dominate: 32 heads x T^2 x 6 bytes
    assert a == 32 * 1024 * 1024 * 6


def test_safe_prompt_tokens_shrinks_as_hbm_free_space_shrinks():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    roomy = M.safe_prompt_tokens(cfg, 8, 5.0 * GIB)
    tight = M.safe_prompt_tokens(cfg, 8, 1.0 * GIB)
    assert roomy > tight > 0
    assert M.prefill_attention_bytes_per_chip(cfg, 8, roomy) <= 5.0 * GIB
    assert M.safe_prompt_tokens(cfg, 8, 0) == 0


def test_kv_cost_per_token_scales_linearly_with_context():
    cfg = load_text_config(write_config(GEMMA4_31B_TEXT))
    a = M.hbm_estimate(cfg, 8, 1, 8192)["cache_bytes_per_chip"]
    b = M.hbm_estimate(cfg, 8, 1, 16384)["cache_bytes_per_chip"]
    assert abs(b - 2 * a) < 1


def test_default_context_of_every_model_fits_its_own_budget():
    for spec, text in (("31b", GEMMA4_31B_TEXT), ("26b-a4b", GEMMA4_26B_A4B_TEXT)):
        s = resolve(spec)
        cfg = load_text_config(write_config(text))
        est = M.hbm_estimate(cfg, 8, 1, s.default_context)
        assert est["total_bytes_per_chip"] / GIB < 0.80 * 15.75, spec


# ---------------------------------------------------------------- real checkpoint (optional)
def _downloaded_config(repo_key: str):
    spec = resolve(repo_key)
    root = os.environ.get("HF_HOME", "/root/hf_cache")
    base = os.path.join(root, "hub", "models--" + spec.repo_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(base):
        return None
    for snap in os.listdir(base):
        if os.path.exists(os.path.join(base, snap, "config.json")):
            return os.path.join(base, snap)
    return None


def test_real_31b_config_if_downloaded():
    d = _downloaded_config("31b")
    if d is None:
        print("  (skipped: google/gemma-4-31B-it not downloaded)")
        return
    cfg = load_text_config(d)
    assert cfg.hidden_size == 5376
    assert cfg.num_hidden_layers == 60
    assert cfg.enable_moe_block is False
    assert cfg.attention_k_eq_v is True
    M.validate_sharding(cfg, 8)
