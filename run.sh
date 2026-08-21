#!/usr/bin/env bash
# Streaming inference with a live TPU/throughput dashboard.
#   bash run.sh "prompt"                              -> dashboard
#   bash run.sh "prompt" --model 31b                  -> pick the model
#   bash run.sh "prompt" --max-new-tokens 4096        -> raise the output budget
#   bash run.sh "prompt" --max-len 32768 --local      -> raise the context window in-process
#   bash run.sh "prompt" --plain                      -> plain streaming
#
# Uses the persistent daemon if it is running (instant start); otherwise it starts one
# automatically, unless --local is passed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/_common.sh"

args=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  args+=(--prompt "$1"); shift
fi
args+=("$@")

prev=""
for a in ${args[@]+"${args[@]}"}; do
  [[ "$prev" == "--max-len" ]] && export GEMMA4_MAX_LEN="$a"
  prev="$a"
done
gemma4_pick_model ${args[@]+"${args[@]}"} || exit 2

want_local=0
for a in ${args[@]+"${args[@]}"}; do [[ "$a" == "--local" ]] && want_local=1; done

if [[ $want_local -eq 0 ]]; then
  if ! python3 -c "
import sys; sys.path.insert(0, '$ROOT')
from gemma4_tpu.session import daemon_alive
sys.exit(0 if daemon_alive('$GEMMA4_SOCKET') else 1)" 2>/dev/null; then
    bash "$ROOT/serve.sh" start --model "$GEMMA4_MODEL_KEY" --max-len "$GEMMA4_MAX_LEN"
  fi
fi

cd "$ROOT"
exec python3 -u run_inference.py --model "$GEMMA4_MODEL_KEY" ${args[@]+"${args[@]}"}
