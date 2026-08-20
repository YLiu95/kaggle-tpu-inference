#!/usr/bin/env bash
# Persistent TPU inference daemon: keeps the sharded weights and compiled XLA programs
# resident so `run.sh` starts generating in milliseconds.
#
#   bash serve.sh start | status | stop | restart | logs
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export GEMMA4_XLA_CACHE="${GEMMA4_XLA_CACHE:-/root/.cache/gemma4_jax}"
export GEMMA4_SOCKET="${GEMMA4_SOCKET:-/tmp/gemma4-tpu.sock}"
LOG="${GEMMA4_SERVER_LOG:-/root/.cache/gemma4_jax/server.log}"
PIDFILE=/root/.cache/gemma4_jax/server.pid
if [[ -z "${GEMMA4_MODEL_DIR:-}" && -f /root/.gemma4_model_dir ]]; then
  export GEMMA4_MODEL_DIR="$(cat /root/.gemma4_model_dir)"
fi
mkdir -p "$(dirname "$LOG")"

alive() { python3 -c "
import sys; sys.path.insert(0, '$ROOT')
from gemma4_tpu.session import daemon_alive
sys.exit(0 if daemon_alive('$GEMMA4_SOCKET') else 1)" 2>/dev/null; }

case "${1:-start}" in
  start)
    if alive; then echo "daemon already running ($GEMMA4_SOCKET)"; exit 0; fi
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
    echo "starting 32K daemon (cold start may take several minutes; later prompts are instant)"
    cd "$ROOT"
    nohup python3 -u -m gemma4_tpu.server --socket "$GEMMA4_SOCKET" "${@:2}" >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 180); do
      if alive; then
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
  restart) bash "$0" stop; bash "$0" start "${@:2}" ;;
  logs)    tail -n "${2:-40}" -f "$LOG" ;;
  *) echo "usage: bash serve.sh {start|status|stop|restart|logs}"; exit 2 ;;
esac
