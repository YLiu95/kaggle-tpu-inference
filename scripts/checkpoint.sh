#!/usr/bin/env bash
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
cd "$repo"

export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -q

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing checkpoint: the working tree is not clean." >&2
  echo "Review, commit, and then re-run this script." >&2
  exit 2
fi

branch="$(git branch --show-current)"
if [[ "$branch" != "main" ]]; then
  echo "Refusing checkpoint: current branch is '$branch', not 'main'." >&2
  exit 2
fi

remote="$(git remote get-url origin)"
if [[ "$remote" =~ ^https?://[^/]*@ ]]; then
  echo "Refusing checkpoint: origin URL contains credentials." >&2
  exit 2
fi

if [[ "${1:-}" == "--push" ]]; then
  git push origin main
fi

local_sha="$(git rev-parse HEAD)"
remote_sha="$(git ls-remote --exit-code origin refs/heads/main | awk '{print $1}')"
if [[ "$local_sha" != "$remote_sha" ]]; then
  echo "Checkpoint mismatch: local=$local_sha remote=$remote_sha" >&2
  echo "Run: scripts/checkpoint.sh --push" >&2
  exit 3
fi

echo "Checkpoint confirmed: $local_sha"

