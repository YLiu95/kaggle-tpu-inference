#!/usr/bin/env bash
# Persistent TPU inference daemon: keeps the sharded weights and compiled XLA programs
# resident so `run.sh` starts generating in milliseconds.
#
#   bash serve.sh start [--model 31b] [--max-len 16384]
#   bash serve.sh status | stop | restart | logs
#
# The 8 chips fit exactly one model at a time, so `restart --model X` is how you switch
# models or change the resident context window.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/_common.sh"

CMD="${1:-start}"
shift || true

prev=""
for a in "$@"; do
  [[ "$prev" == "--max-len" ]] && export GEMMA4_MAX_LEN="$a"
  prev="$a"
done
gemma4_pick_model "$@" || exit 2

LOG="${GEMMA4_SERVER_LOG:-/root/.cache/gemma4_jax/server.log}"
PIDFILE=/root/.cache/gemma4_jax/server.pid
mkdir -p "$(dirname "$LOG")"

alive() { python3 -c "
import sys; sys.path.insert(0, '$ROOT')
from gemma4_tpu.session import daemon_alive
sys.exit(0 if daemon_alive('$GEMMA4_SOCKET') else 1)" 2>/dev/null; }

case "$CMD" in
  start)
    if alive; then
      running="$(python3 -c "
import sys; sys.path.insert(0, '$ROOT')
from gemma4_tpu.session import send_command
print((send_command('status', '$GEMMA4_SOCKET') or {}).get('model_key', '?'))")"
      if [[ "$running" != "$GEMMA4_MODEL_KEY" ]]; then
        echo "daemon already running with model '$running', not '$GEMMA4_MODEL_KEY'."
        echo "The v5e-8 holds one model at a time; switch with:"
        echo "  bash serve.sh restart --model $GEMMA4_MODEL_KEY"
        exit 1
      fi
      echo "daemon already running ($GEMMA4_SOCKET, model $running)"
      exit 0
    fi
    rm -f "$GEMMA4_SOCKET"
    echo "waiting for all 8 TPU devices to become available"
    tpu_ready=0
    for _ in $(seq 1 60); do
      if JAX_PLATFORMS=tpu python3 -c "import jax; assert len(jax.devices()) == 8" \
          >/dev/null 2>&1; then
        tpu_ready=1
        break
      fi
      sleep 5
    done
    if [[ $tpu_ready -ne 1 ]]; then
      echo "TPU devices remained busy for 5 minutes; check for another JAX process"
      exit 1
    fi
    echo "starting $GEMMA4_MODEL_ID with a ${GEMMA4_MAX_LEN}-token context"
    echo "(cold start may take several minutes; later prompts are instant)"
    cd "$ROOT"
    nohup python3 -u -m gemma4_tpu.server \
      --socket "$GEMMA4_SOCKET" --model "$GEMMA4_MODEL_KEY" --max-len "$GEMMA4_MAX_LEN" \
      >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 180); do
      if alive; then
        gemma4_remember_model
        bash "$0" status
        exit 0
      fi
      if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "daemon died, last log lines:"; tail -20 "$LOG"; exit 1
      fi
      sleep 5
    done
    echo "timed out waiting 15 minutes for daemon; see $LOG"; exit 1
    ;;
  status)
    python3 -c "
import json, sys; sys.path.insert(0, '$ROOT')
from gemma4_tpu.session import send_command
st = send_command('status', '$GEMMA4_SOCKET')
print(json.dumps(st, indent=2) if st else 'daemon not running')"
    ;;
  stop)
    python3 -c "
import sys; sys.path.insert(0, '$ROOT')
from gemma4_tpu.session import send_command
print('stopping' if send_command('shutdown', '$GEMMA4_SOCKET') else 'daemon not running')"
    for _ in $(seq 1 12); do alive || break; sleep 2; done
    [[ -f "$PIDFILE" ]] && kill -9 "$(cat "$PIDFILE")" 2>/dev/null; rm -f "$PIDFILE"
    # the TPU stays claimed for a few seconds after the process exits
    sleep 5
    echo stopped
    ;;
  restart) bash "$0" stop; bash "$0" start "$@" ;;
  logs)    tail -n "${1:-40}" -f "$LOG" ;;
  *) echo "usage: bash serve.sh {start|status|stop|restart|logs} [--model KEY] [--max-len N]"
     exit 2 ;;
esac
