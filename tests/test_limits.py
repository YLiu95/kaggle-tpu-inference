import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma4_tpu.limits import (
    MAX_OUTPUT_TOKENS,
    allowed_output_tokens,
    check_completion,
    check_prefill_memory,
    check_request,
    env_max_len,
)


def codes(warnings):
    return [w.code for w in warnings]


# ---------------------------------------------------------------- allowed_output_tokens
def test_request_below_limit_is_unchanged():
    assert allowed_output_tokens(5000, 54, 32768) == 5000


def test_model_limit_is_reduced_to_fit_prompt():
    assert allowed_output_tokens(32768, 54, 32768) == 32714


def test_oversized_request_is_clamped():
    assert allowed_output_tokens(128000, 54, 32768) == 32714


def test_full_prompt_has_no_output_capacity():
    assert allowed_output_tokens(5000, 32768, 32768) == 0


def test_per_model_output_cap_is_applied():
    # a 31B daemon started with --max-len 16384 caps output at 16384, not 32768
    assert allowed_output_tokens(30000, 100, 16384, 16384) == 16284


# ---------------------------------------------------------------- check_request
def test_no_warning_when_everything_fits():
    allowed, warnings = check_request(768, 54, 16384, 16384)
    assert allowed == 768
    assert warnings == []


def test_warns_and_truncates_when_prompt_plus_output_exceeds_context():
    allowed, warnings = check_request(16384, 500, 16384, 16384)
    assert allowed == 15884
    assert codes(warnings) == ["output_truncated_by_context"]
    w = warnings[0]
    assert "16,384-token context window" in w.message
    assert "--max-len" in w.remedy


def test_warns_when_request_exceeds_the_output_cap():
    allowed, warnings = check_request(50000, 100, 16384, 16384)
    assert allowed == 16284
    assert "request_exceeds_max_new_tokens" in codes(warnings)
    assert any("--max-len" in w.remedy for w in warnings)


def test_prompt_longer_than_context_blocks_generation_and_names_the_ceiling():
    allowed, warnings = check_request(256, 20000, 16384, 16384, model_max_context=32768)
    assert allowed == 0
    assert codes(warnings) == ["prompt_exceeds_context"]
    w = warnings[0]
    assert "20,000 tokens" in w.message
    assert "--max-len 32768" in w.remedy


def test_prompt_exactly_at_context_still_blocks():
    allowed, warnings = check_request(10, 16384, 16384, 16384)
    assert allowed == 0
    assert codes(warnings) == ["prompt_exceeds_context"]


def test_remedy_mentions_headroom_only_when_a_bigger_context_is_possible():
    _, at_ceiling = check_request(40000, 100, 32768, 32768, model_max_context=32768)
    assert all("would grant the full request" not in w.remedy for w in at_ceiling)
    _, below_ceiling = check_request(40000, 100, 16384, 16384, model_max_context=32768)
    assert any("would grant the full request" in w.remedy for w in below_ceiling)


def test_warnings_serialise_to_events():
    _, warnings = check_request(99999, 10, 4096, 4096)
    events = [w.to_event() for w in warnings]
    assert events
    assert all(e["kind"] == "warning" for e in events)
    assert all(e["message"] and e["remedy"] for e in events)


# ---------------------------------------------------------------- check_completion
def test_no_completion_warning_when_a_stop_token_ended_it():
    assert check_completion(768, 768, stopped_naturally=True) == []


def test_no_completion_warning_when_budget_was_not_reached():
    assert check_completion(120, 768, stopped_naturally=False) == []


def test_warns_when_generation_hits_the_budget():
    warnings = check_completion(768, 768, stopped_naturally=False)
    assert codes(warnings) == ["hit_max_new_tokens"]
    assert "--max-new-tokens" in warnings[0].remedy


# ---------------------------------------------------------------- check_prefill_memory
def test_short_prompt_has_no_prefill_warning():
    assert check_prefill_memory(2000, 4096) == []


def test_long_prompt_warns_about_quadratic_prefill():
    warnings = check_prefill_memory(9000, 4096)
    assert codes(warnings) == ["prompt_prefill_memory_risk"]
    assert "9,000 tokens" in warnings[0].message
    assert "--max-len" in warnings[0].remedy


def test_prefill_check_is_skipped_when_the_budget_is_unknown():
    assert check_prefill_memory(9000, 0) == []


# ---------------------------------------------------------------- env override
def test_env_max_len_override():
    old = os.environ.pop("GEMMA4_MAX_LEN", None)
    try:
        assert env_max_len(16384) == 16384
        os.environ["GEMMA4_MAX_LEN"] = "8192"
        assert env_max_len(16384) == 8192
        os.environ["GEMMA4_MAX_LEN"] = "not-a-number"
        assert env_max_len(16384) == 16384
    finally:
        os.environ.pop("GEMMA4_MAX_LEN", None)
        if old is not None:
            os.environ["GEMMA4_MAX_LEN"] = old


def test_module_default_cap_is_the_v5e8_ceiling():
    assert MAX_OUTPUT_TOKENS == 32768
