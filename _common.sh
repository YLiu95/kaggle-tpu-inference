#!/usr/bin/env bash
# Shared model/context resolution for setup.sh, serve.sh and run.sh.
#
# Sourced, never executed. Reads the model from (highest priority first):
#   1. a `--model <key>` flag anywhere in "$@"
#   2. $MODEL
#   3. the model recorded by the last successful setup.sh (/root/.gemma4_model)
#   4. the registry default (26b-a4b)
#
# Exports: GEMMA4_MODEL_KEY GEMMA4_MODEL_ID GEMMA4_MODEL_KIND GEMMA4_DEFAULT_MAX_LEN
#          GEMMA4_MODEL_MAX_LEN GEMMA4_DOWNLOAD_GB GEMMA4_MODEL_DIR GEMMA4_SOCKET
#          HF_HOME GEMMA4_XLA_CACHE

GEMMA4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export GEMMA4_XLA_CACHE="${GEMMA4_XLA_CACHE:-/root/.cache/gemma4_jax}"
export GEMMA4_SOCKET="${GEMMA4_SOCKET:-/tmp/gemma4-tpu.sock}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"   # Xet transfers hang on Kaggle
STATE_FILE=/root/.gemma4_model

gemma4_pick_model() {
  local want="" prev=""
  for a in "$@"; do
    if [[ "$prev" == "--model" ]]; then want="$a"; fi
    prev="$a"
  done
  if [[ -z "$want" ]]; then want="${MODEL:-}"; fi
  if [[ -z "$want" && -f "$STATE_FILE" ]]; then want="$(cat "$STATE_FILE")"; fi

  local out
  if ! out="$(cd "$GEMMA4_ROOT" && python3 -m gemma4_tpu.models --shell "$want" 2>&1)"; then
    echo "$out" >&2
    echo "" >&2
    (cd "$GEMMA4_ROOT" && python3 -m gemma4_tpu.models) >&2
    return 2
  fi
  eval "export $(echo "$out" | tr '\n' ' ')"

  # Weights are cached per model so switching back and forth does not re-download.
  local dirfile="/root/.gemma4_model_dir.${GEMMA4_MODEL_KEY}"
  export GEMMA4_MODEL_DIR_FILE="$dirfile"
  if [[ -z "${GEMMA4_MODEL_DIR:-}" && -f "$dirfile" ]]; then
    export GEMMA4_MODEL_DIR="$(cat "$dirfile")"
  fi
  export GEMMA4_MAX_LEN="${GEMMA4_MAX_LEN:-$GEMMA4_DEFAULT_MAX_LEN}"
}

gemma4_remember_model() {
  echo "$GEMMA4_MODEL_KEY" > "$STATE_FILE"
}

gemma4_load_secrets() {
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
}
