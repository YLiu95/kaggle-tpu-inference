#!/usr/bin/env bash
# Streaming inference with a live TPU/throughput dashboard.
#   bash run.sh "prompt"                          -> dashboard
#   bash run.sh "prompt" --plain                  -> plain streaming
#   bash run.sh "prompt" --max-new-tokens 512     -> any run_inference.py flag
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export GEMMA4_XLA_CACHE="${GEMMA4_XLA_CACHE:-/root/.cache/gemma4_jax}"
if [[ -z "${GEMMA4_MODEL_DIR:-}" && -f /root/.gemma4_model_dir ]]; then
  export GEMMA4_MODEL_DIR="$(cat /root/.gemma4_model_dir)"
fi

# The TPU stays claimed for a few seconds after a previous process exits.
for _ in $(seq 1 30); do
  if python3 -c "import jax; assert len(jax.devices())==8" >/dev/null 2>&1; then break; fi
  echo "waiting for TPU to be released..."; sleep 5
done

args=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  args+=(--prompt "$1"); shift
fi
args+=("$@")

cd "$ROOT"
exec python3 -u run_inference.py "${args[@]}"
