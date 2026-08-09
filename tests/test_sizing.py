from __future__ import annotations

import unittest

from ktpu.constants import GIB
from ktpu.errors import SizingError
from ktpu.model import ModelProfile, profile_from_config_dict
from ktpu.sizing import calculate_limits


def profile() -> ModelProfile:
    return ModelProfile(
        model_id="test/model",
        weights_bytes=40 * GIB,
        model_limit=131_072,
        num_hidden_layers=32,
        num_key_value_heads=8,
        head_dim=128,
        dtype_bytes=2,
        kv_bytes_per_token=131_072,
        dtype_name="bfloat16",
    )


class ModelProfileTests(unittest.TestCase):
    def test_nested_config_and_conservative_kv(self) -> None:
        value = profile_from_config_dict(
            "google/example",
            {
                "dtype": "bfloat16",
                "text_config": {
                    "max_position_embeddings": 262_144,
                    "num_hidden_layers": 60,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 16,
                    "head_dim": 256,
                },
            },
            weights_bytes=62_000_000_000,
        )
        self.assertEqual(value.model_limit, 262_144)
        self.assertEqual(value.kv_bytes_per_token, 2 * 60 * 16 * 256 * 2)


class SizingTests(unittest.TestCase):
    def test_limits_apply_model_and_user_caps(self) -> None:
        result = calculate_limits(
            profile(),
            hbm_total_bytes=128 * GIB,
            hbm_in_use_bytes=0,
            input_tokens=1_000,
            context_cap=32_000,
            output_cap=4_096,
        )
        self.assertGreaterEqual(result.calculated_safe_context, 32_000)
        self.assertEqual(result.applied_context, 32_000)
        self.assertEqual(result.safe_output_tokens, 31_000)
        self.assertEqual(result.applied_output_tokens, 4_096)
        self.assertEqual(result.runtime_headroom_bytes, int(128 * GIB * 0.15))

    def test_without_output_cap_uses_remaining_context(self) -> None:
        result = calculate_limits(
            profile(),
            hbm_total_bytes=128 * GIB,
            hbm_in_use_bytes=0,
            input_tokens=100,
            context_cap=4_096,
            output_cap=None,
        )
        self.assertEqual(result.applied_output_tokens, 3_996)

    def test_prompt_must_fit(self) -> None:
        with self.assertRaises(SizingError):
            calculate_limits(
                profile(),
                hbm_total_bytes=128 * GIB,
                hbm_in_use_bytes=0,
                input_tokens=4_096,
                context_cap=4_096,
            )

    def test_weights_and_headroom_gate(self) -> None:
        too_large = profile()
        with self.assertRaisesRegex(SizingError, "exceed"):
            calculate_limits(
                too_large,
                hbm_total_bytes=48 * GIB,
                hbm_in_use_bytes=0,
                input_tokens=1,
            )

    def test_model_limit_is_hard_cap(self) -> None:
        small = ModelProfile(
            **{**profile().__dict__, "model_limit": 2_048, "weights_bytes": GIB}
        )
        result = calculate_limits(
            small,
            hbm_total_bytes=128 * GIB,
            hbm_in_use_bytes=0,
            input_tokens=10,
        )
        self.assertEqual(result.calculated_safe_context, 2_048)


if __name__ == "__main__":
    unittest.main()

