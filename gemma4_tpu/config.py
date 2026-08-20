"""Config loading for Gemma-4 MoE text stack (google/gemma-4-26B-A4B*)."""

from __future__ import annotations

import dataclasses
import json
import math
import os


@dataclasses.dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    global_head_dim: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    top_k_experts: int
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    final_logit_softcapping: float | None
    attention_k_eq_v: bool
    enable_moe_block: bool
    layer_types: tuple[str, ...]
    rope_theta_sliding: float
    rope_theta_full: float
    rope_partial_rotary_factor_full: float
    eos_token_ids: tuple[int, ...]

    @property
    def embed_scale(self) -> float:
        return math.sqrt(self.hidden_size)

    def head_dim_for(self, layer_type: str) -> int:
        return self.global_head_dim if layer_type == "full_attention" else self.head_dim

    def kv_heads_for(self, layer_type: str) -> int:
        if layer_type == "full_attention" and self.attention_k_eq_v:
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    def k_eq_v_for(self, layer_type: str) -> bool:
        return self.attention_k_eq_v and layer_type == "full_attention"


def load_text_config(model_dir: str) -> TextConfig:
    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    tc = raw["text_config"]

    gen_eos: list[int] = []
    gen_path = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            gen = json.load(f)
        eos = gen.get("eos_token_id", [])
        gen_eos = [eos] if isinstance(eos, int) else list(eos)
    if not gen_eos:
        eos = raw.get("eos_token_id", tc.get("eos_token_id", 1))
        gen_eos = [eos] if isinstance(eos, int) else list(eos)

    rope = tc["rope_parameters"]
    return TextConfig(
        hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        head_dim=tc["head_dim"],
        global_head_dim=tc.get("global_head_dim") or tc["head_dim"],
        num_key_value_heads=tc["num_key_value_heads"],
        num_global_key_value_heads=tc.get("num_global_key_value_heads", tc["num_key_value_heads"]),
        intermediate_size=tc["intermediate_size"],
        moe_intermediate_size=tc["moe_intermediate_size"],
        num_experts=tc["num_experts"],
        top_k_experts=tc["top_k_experts"],
        sliding_window=tc["sliding_window"],
        rms_norm_eps=tc["rms_norm_eps"],
        vocab_size=tc["vocab_size"],
        final_logit_softcapping=tc.get("final_logit_softcapping"),
        attention_k_eq_v=tc.get("attention_k_eq_v", False),
        enable_moe_block=tc.get("enable_moe_block", False),
        layer_types=tuple(tc["layer_types"]),
        rope_theta_sliding=rope["sliding_attention"]["rope_theta"],
        rope_theta_full=rope["full_attention"]["rope_theta"],
        rope_partial_rotary_factor_full=rope["full_attention"].get("partial_rotary_factor", 1.0),
        eos_token_ids=tuple(gen_eos),
    )
