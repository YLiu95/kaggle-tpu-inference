"""Compatibility bootstrap executed by the isolated vLLM engine Python."""

from __future__ import annotations

import os


def strip_unsupported_libtpu_flags(value: str) -> str:
    unsupported = {
        "--xla_tpu_use_dynamic_smem_negotiation",
        "--xla_tpu_use_dynamic_smem_negotiation=true",
        "--xla_tpu_use_dynamic_smem_negotiation=false",
    }
    return " ".join(part for part in value.split() if part not in unsupported)


def add_unknown_flag_allowlist(value: str) -> str:
    allow = "--undefok=xla_tpu_use_dynamic_smem_negotiation"
    parts = value.split()
    if allow not in parts:
        parts.append(allow)
    return " ".join(parts)


def main() -> int:
    # Importing the vLLM CLI loads the TPU plugin, whose env_override currently
    # injects a flag absent from Kaggle's host TPU runtime. Strip it only in the
    # known Kaggle v5litepod environment, after plugin loading but before the
    # engine subprocess initializes libtpu.
    from vllm.entrypoints.cli.main import main as vllm_main

    if os.environ.get("KTPU_STRIP_DYNAMIC_SMEM_FLAG") == "1":
        os.environ["LIBTPU_INIT_ARGS"] = add_unknown_flag_allowlist(
            strip_unsupported_libtpu_flags(
                os.environ.get("LIBTPU_INIT_ARGS", "")
            )
        )
    return int(vllm_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
