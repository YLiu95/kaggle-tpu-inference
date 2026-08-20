#!/usr/bin/env bash
# Streaming inference with a live TPU/throughput dashboard.
#   bash run.sh "prompt"                          -> dashboard
#   bash run.sh "prompt" --plain                  -> plain streaming
#   bash run.sh "prompt" --max-new-tokens 512     -> any run_inference.py flag
#
# Uses the persistent daemon if it is running (instant start); otherwise it starts one
# automatically, unless --local is passed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export GEMMA4_XLA_CACHE="${GEMMA4_XLA_CACHE:-/root/.cache/gemma4_jax}"
export GEMMA4_SOCKET="${GEMMA4_SOCKET:-/tmp/gemma4-tpu.sock}"
if [[ -z "${GEMMA4_MODEL_DIR:-}" && -f /root/.gemma4_model_dir ]]; then
  export GEMMA4_MODEL_DIR="$(cat /root/.gemma4_model_dir)"
fi

args=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  args+=(--prompt "$1"); shift
fi
args+=("$@")

want_local=0
for a in ${args[@]+"${args[@]}"}; do [[ "$a" == "--local" ]] && want_local=1; done

if [[ $want_local -eq 0 ]]; then
  if ! python3 -c "
import sys; sys.path.insert(0, '$ROOT')
from gemma4_tpu.session import daemon_alive
sys.exit(0 if daemon_alive('$GEMMA4_SOCKET') else 1)" 2>/dev/null; then
    bash "$ROOT/serve.sh" start
  fi
fi

cd "$ROOT"
exec python3 -u run_inference.py ${args[@]+"${args[@]}"}
