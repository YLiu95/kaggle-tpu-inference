#!/usr/bin/env python3
"""Predict whether ``weights + KV cache`` fit in HBM before paying for a load.

Loading and compiling the 31B takes minutes; an out-of-memory abort at the end of that
is the most expensive failure mode on a Kaggle TPU. ``setup.sh`` runs this first.

    python3 -m gemma4_tpu.fit_check --model 31b --max-len 32768
    python3 -m gemma4_tpu.fit_check --model-dir /path/to/snapshot --max-len 16384
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma4_tpu import model as M  # noqa: E402
from gemma4_tpu.config import load_text_config  # noqa: E402
from gemma4_tpu.models import resolve  # noqa: E402

GIB = 2**30
V5E_HBM_BYTES = int(15.75 * GIB)
SAFE_FRACTION = 0.80  # leave ~20% for prefill activations and XLA scratch


def device_hbm_bytes(default: int = V5E_HBM_BYTES) -> int:
    try:
        import jax

        return int(jax.devices()[0].memory_stats()["bytes_limit"])
    except Exception:
        return default


def report(cfg, n_devices: int, max_len: int, hbm_bytes: int, batch: int = 1) -> tuple[bool, str]:
    est = M.hbm_estimate(cfg, n_devices, batch, max_len)
    total = est["total_bytes_per_chip"]
    frac = total / hbm_bytes
    per_token = est["kv_bytes_per_token_per_chip"]
    fits_len = int((SAFE_FRACTION * hbm_bytes - est["weights_bytes_per_chip"]) // max(1.0, per_token))
    lines = [
        f"  chips              {n_devices} x {hbm_bytes / GIB:.2f} GiB HBM",
        f"  weights / chip     {est['weights_bytes_per_chip'] / GIB:6.2f} GiB",
        f"  KV cache / chip    {est['cache_bytes_per_chip'] / GIB:6.2f} GiB "
        f"at max-len {max_len:,} ({per_token / 1024:.0f} KiB/token)",
        f"  total / chip       {total / GIB:6.2f} GiB  ({100 * frac:.0f}% of HBM)",
        f"  largest safe max-len for this model: {max(0, fits_len):,} tokens",
    ]
    ok = frac <= SAFE_FRACTION
    if not ok:
        lines.append(
            f"  WARNING: over the {100 * SAFE_FRACTION:.0f}% safety line. Prefill "
            f"activations may not fit -- rerun with --max-len {max(256, fits_len)}."
        )
    return ok, "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="registry key, e.g. 31b / 26b-a4b / 12b")
    ap.add_argument("--model-dir", default=None, help="local snapshot dir (skips the registry)")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--strict", action="store_true", help="exit non-zero if it does not fit")
    args = ap.parse_args()

    model_dir = args.model_dir
    max_len = args.max_len
    if model_dir is None:
        spec = resolve(args.model)
        max_len = max_len or spec.default_context
        from gemma4_tpu.session import resolve_model_dir

        model_dir = resolve_model_dir(spec.repo_id)
    max_len = max_len or 16384

    cfg = load_text_config(model_dir)
    M.validate_sharding(cfg, args.devices)
    ok, text = report(cfg, args.devices, max_len, device_hbm_bytes())
    print(text)
    return 0 if (ok or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
