import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma4_tpu.limits import allowed_output_tokens


def test_request_below_limit_is_unchanged():
    assert allowed_output_tokens(5000, 54, 32768) == 5000


def test_model_limit_is_reduced_to_fit_prompt():
    assert allowed_output_tokens(32768, 54, 32768) == 32714


def test_oversized_request_is_clamped():
    assert allowed_output_tokens(128000, 54, 32768) == 32714


def test_full_prompt_has_no_output_capacity():
    assert allowed_output_tokens(5000, 32768, 32768) == 0