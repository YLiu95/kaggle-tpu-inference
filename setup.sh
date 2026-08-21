#!/usr/bin/env bash
# One-time setup: pick a model, verify the TPU, fetch weights, warm the XLA cache,
# start the persistent daemon.
#
#   bash setup.sh                      # default: gemma-4 26B-A4B (MoE)
#   bash setup.sh --model 31b          # gemma-4 31B instruction-tuned (dense)
#   bash setup.sh --model 31b --max-len 32768
#   bash setup.sh --list               # show every supported model
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/_common.sh"

if [[ "${1:-}" == "--list" || "${1:-}" == "-l" ]]; then
  python3 -m gemma4_tpu.models
  exit 0
fi

# --max-len here becomes the resident context window of the daemon.
prev=""
for a in "$@"; do
  [[ "$prev" == "--max-len" ]] && export GEMMA4_MAX_LEN="$a"
  prev="$a"
done

gemma4_pick_model "$@"
mkdir -p "$HF_HOME" "$GEMMA4_XLA_CACHE"

echo "==> Model: $GEMMA4_MODEL_ID  [$GEMMA4_MODEL_KEY, $GEMMA4_MODEL_KIND]"
echo "    context window: ${GEMMA4_MAX_LEN} tokens (ceiling ${GEMMA4_MODEL_MAX_LEN})"
echo "    change it with: bash setup.sh --model $GEMMA4_MODEL_KEY --max-len N"

echo "==> Kaggle secrets -> environment"
gemma4_load_secrets

echo "==> TPU check"
python3 - <<'PY'
import jax
d = jax.devices()
print(f"  {len(d)} x {d[0].device_kind}, "
      f"{d[0].memory_stats()['bytes_limit'] / 2**30:.1f} GiB HBM per chip")
assert len(d) == 8, "expected 8 TPU chips"
PY

echo "==> Downloading ${GEMMA4_MODEL_ID} (~${GEMMA4_DOWNLOAD_GB} GB, plain HTTP: Xet hangs on Kaggle)"
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    os.environ["GEMMA4_MODEL_ID"],
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
    max_workers=8,
)
print("  weights:", p)
open(os.environ["GEMMA4_MODEL_DIR_FILE"], "w").write(p)
PY

export GEMMA4_MODEL_DIR="$(cat "$GEMMA4_MODEL_DIR_FILE")"
gemma4_remember_model

echo "==> HBM fit check for a ${GEMMA4_MAX_LEN}-token context"
python3 -m gemma4_tpu.fit_check --model-dir "$GEMMA4_MODEL_DIR" --max-len "$GEMMA4_MAX_LEN"

echo "==> Warming the XLA compilation cache (one-off, several minutes)"
cd "$ROOT"
python3 -u bench.py --model "$GEMMA4_MODEL_KEY" --max-len "$GEMMA4_MAX_LEN" --steps 8

echo "==> Starting the persistent TPU daemon"
bash "$ROOT/serve.sh" start --model "$GEMMA4_MODEL_KEY" --max-len "$GEMMA4_MAX_LEN"

echo
echo "Setup complete. Model $GEMMA4_MODEL_ID, weights: $GEMMA4_MODEL_DIR"
echo "Run:  bash run.sh \"your prompt here\""
echo "      Weights stay sharded on the TPU, so every later run starts instantly."
echo "      bash serve.sh status | stop | restart | logs"
echo "      Switch model: bash setup.sh --model 26b-a4b   (see: bash setup.sh --list)"
