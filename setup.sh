#!/usr/bin/env bash
# One-time setup: verify the TPU, fetch weights, warm the XLA compilation cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export GEMMA4_XLA_CACHE="${GEMMA4_XLA_CACHE:-/root/.cache/gemma4_jax}"
export MODEL_ID="${MODEL_ID:-google/gemma-4-26B-A4B-it}"
mkdir -p "$HF_HOME" "$GEMMA4_XLA_CACHE"

echo "==> Kaggle secrets -> environment"
eval "$(python3 - <<'PY'
try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    for name in ("HF_TOKEN", "GITHUB_TOKEN"):
        try:
            print(f'export {name}={s.get_secret(name)}')
        except Exception:
            pass
except Exception:
    pass
PY
)"

echo "==> TPU check"
python3 - <<'PY'
import jax
d = jax.devices()
print(f"  {len(d)} x {d[0].device_kind}, "
      f"{d[0].memory_stats()['bytes_limit'] / 2**30:.1f} GiB HBM per chip")
assert len(d) == 8, "expected 8 TPU chips"
PY

echo "==> Downloading ${MODEL_ID} (~52 GB, Xet is flaky on Kaggle -> plain HTTP)"
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    os.environ["MODEL_ID"],
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
    max_workers=8,
)
print("  weights:", p)
open("/root/.gemma4_model_dir", "w").write(p)
PY

export GEMMA4_MODEL_DIR="$(cat /root/.gemma4_model_dir)"
echo "==> Warming the XLA compilation cache (one-off, ~7 min)"
cd "$ROOT"
python3 -u bench.py --steps 8

echo
echo "Setup complete. Weights: $GEMMA4_MODEL_DIR"
echo "Run:  bash run.sh \"your prompt here\""
echo "      (~3.5 min from launch to first token: 15 s weight load + JAX tracing;"
echo "       set GEMMA4_MOE_DECODE=loop to halve that at ~24% lower tok/s)"
