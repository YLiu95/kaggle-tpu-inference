"""Registry of the Gemma-4 checkpoints this stack can serve on a Kaggle TPU v5e-8.

Everything that differs per model (HF repo, default resident context, download size,
sharding sanity constraints) lives here so ``setup.sh`` / ``serve.sh`` / ``run.sh`` can
offer a single ``--model`` switch.

The v5e-8 has 8 chips x 15.75 GiB = 126 GiB HBM. ``default_context`` is chosen so that
``weights/chip + KV/chip`` stays under ~65% of a chip, leaving room for prefill
activations; ``max_context`` is the largest value that still fits with a thin margin.
Both are only defaults - ``--max-len`` overrides them (see :mod:`gemma4_tpu.limits`).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    label: str
    kind: str  # "moe" | "dense"
    params_b: float
    weights_gib: float
    download_gb: int
    default_context: int
    max_context: int
    default_max_new_tokens: int
    aliases: tuple[str, ...] = ()
    notes: str = ""

    @property
    def weights_gib_per_chip(self) -> float:
        return self.weights_gib / 8.0


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="26b-a4b",
        repo_id="google/gemma-4-26B-A4B-it",
        label="Gemma-4 26B-A4B (MoE, ~4B active)",
        kind="moe",
        params_b=25.9,
        weights_gib=47.0,
        download_gb=52,
        default_context=32768,
        max_context=32768,
        default_max_new_tokens=768,
        aliases=("26b", "a4b", "moe", "26b-a4b-it", "google/gemma-4-26B-A4B-it"),
        notes="128 experts, top-8; ~6.1 GiB/chip of weights, so a full 32K cache fits.",
    ),
    ModelSpec(
        key="31b",
        repo_id="google/gemma-4-31B-it",
        label="Gemma-4 31B (dense)",
        kind="dense",
        params_b=30.7,
        weights_gib=57.2,
        download_gb=62,
        default_context=16384,
        max_context=32768,
        default_max_new_tokens=768,
        aliases=("31b-it", "dense", "google/gemma-4-31B-it"),
        notes=(
            "60 layers, hidden 5376, no MoE. ~7.2 GiB/chip of weights and 180 KiB/token "
            "of KV per chip, so 16K is the safe default and 32K is the ceiling "
            "(~12.8/15.75 GiB per chip)."
        ),
    ),
    ModelSpec(
        key="12b",
        repo_id="google/gemma-4-12B-it",
        label="Gemma-4 12B (dense)",
        kind="dense",
        params_b=12.0,
        weights_gib=22.4,
        download_gb=25,
        default_context=32768,
        max_context=65536,
        default_max_new_tokens=768,
        aliases=("12b-it", "google/gemma-4-12B-it"),
        notes="Small enough that the context window, not the weights, is the limit.",
    ),
)

DEFAULT_MODEL_KEY = "26b-a4b"

_BY_NAME: dict[str, ModelSpec] = {}
for _spec in MODELS:
    for _name in (_spec.key, _spec.repo_id, *_spec.aliases):
        _BY_NAME[_name.lower()] = _spec


def resolve(name: str | None) -> ModelSpec:
    """Map a user-supplied model key/alias/repo id to a :class:`ModelSpec`."""
    if not name:
        return _BY_NAME[DEFAULT_MODEL_KEY]
    spec = _BY_NAME.get(name.strip().lower())
    if spec is not None:
        return spec
    raise ValueError(
        f"unknown model {name!r}. Choose one of: "
        + ", ".join(f"{s.key} ({s.repo_id})" for s in MODELS)
    )


def try_resolve(name: str | None) -> ModelSpec | None:
    """Like :func:`resolve` but returns ``None`` for unknown names.

    Used on paths that must still work with a checkpoint that is not in the registry
    (for example a local fine-tune passed via ``--model-dir``).
    """
    try:
        return resolve(name)
    except ValueError:
        return None


def choices() -> list[str]:
    return [s.key for s in MODELS]


def describe_table() -> str:
    rows = [("key", "repo", "kind", "params", "weights", "default ctx", "max ctx")]
    for s in MODELS:
        rows.append(
            (
                s.key,
                s.repo_id,
                s.kind,
                f"{s.params_b:.1f}B",
                f"{s.weights_gib:.0f} GiB",
                f"{s.default_context:,}",
                f"{s.max_context:,}",
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def _main() -> int:
    """`python3 -m gemma4_tpu.models [--shell KEY]` - used by setup.sh / serve.sh / run.sh."""
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--shell":
        spec = resolve(args[1] if len(args) > 1 else None)
        print(f"GEMMA4_MODEL_KEY={spec.key}")
        print(f"GEMMA4_MODEL_ID={spec.repo_id}")
        print(f"GEMMA4_MODEL_KIND={spec.kind}")
        print(f"GEMMA4_DEFAULT_MAX_LEN={spec.default_context}")
        print(f"GEMMA4_MODEL_MAX_LEN={spec.max_context}")
        print(f"GEMMA4_DOWNLOAD_GB={spec.download_gb}")
        return 0
    print(describe_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
