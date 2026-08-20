"""Context limits for the resident v5e-8 inference configuration."""

MAX_OUTPUT_TOKENS = 32768
DEFAULT_CONTEXT_TOKENS = 32768


def allowed_output_tokens(requested: int, prompt_tokens: int, context_tokens: int) -> int:
    available = max(0, context_tokens - prompt_tokens)
    return min(max(1, requested), MAX_OUTPUT_TOKENS, available)