#!/usr/bin/env bash
set -euo pipefail

repo_url="${KTPU_REPO_URL:-https://github.com/YLiu95/kaggle-tpu-inference.git}"
repo_dir="${KTPU_REPO_DIR:-$HOME/kaggle-tpu-inference}"
run_setup=0
clear_state=0

usage() {
  cat <<'EOF'
Usage: scripts/resume.sh [--setup] [--clear-tpu]

Re-clone or fast-forward the repository, rebuild the lightweight CLI
environment with binary dependencies, verify the pushed checkpoint, run tests,
show reusable Hugging Face cache state, and print the next inference command.

  --setup      Run the checkpoint-gated `ktpu setup --engine vllm`.
  --clear-tpu  Clear stale managed/vLLM TPU owners before status checks.
EOF
}

while (($#)); do
  case "$1" in
    --setup) run_setup=1 ;;
    --clear-tpu) clear_state=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -d "$repo_dir/.git" ]]; then
  git clone "$repo_url" "$repo_dir"
fi

cd "$repo_dir"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to overwrite an unclean recovery checkout: $repo_dir" >&2
  exit 2
fi

git fetch origin main
git checkout main
git merge --ff-only origin/main

echo "Local checkpoint:  $(git rev-parse HEAD)"
echo "Remote checkpoint: $(git ls-remote --exit-code origin refs/heads/main | awk '{print $1}')"

python -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=:all: \
  "hatchling>=1.26" \
  "huggingface-hub>=0.25" \
  "httpx>=0.27" \
  "psutil>=5.9" \
  "rich>=13" \
  "transformers>=4.45" \
  "typer>=0.12"
python -m pip install --no-deps -e .

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -q
scripts/checkpoint.sh

hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ -d "$hf_home" ]]; then
  echo "Reusing Hugging Face cache: $hf_home"
  du -sh "$hf_home" 2>/dev/null || true
else
  echo "No surviving Hugging Face cache found at: $hf_home"
fi

if ((clear_state)); then
  ktpu clear-tpu --force
fi
ktpu status

if ((run_setup)); then
  ktpu setup --engine vllm
fi

cat <<'EOF'

Resume inference with, for example:
  ktpu run --model google/gemma-4-31B-it \
    --prompt "Explain why checkpoint-first TPU workflows are safer." \
    --max-output 1024

HF_TOKEN is read from the environment when needed. Do not put it in files,
Git remotes, shell commands, or logs.
EOF

