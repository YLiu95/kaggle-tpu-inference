# Resume after a Kaggle environment crash

The last safe source checkpoint is always `origin/main`. Runtime state,
downloaded model weights, engine environments, and logs are intentionally not
part of the checkpoint.

## Automated recovery

```bash
export KTPU_REPO_DIR="$HOME/kaggle-tpu-inference"
bash "$KTPU_REPO_DIR/scripts/resume.sh" --clear-tpu
```

If the repository itself did not survive:

```bash
git clone https://github.com/YLiu95/kaggle-tpu-inference.git \
  "$HOME/kaggle-tpu-inference"
bash "$HOME/kaggle-tpu-inference/scripts/resume.sh" --clear-tpu
```

Add `--setup` if the separately installed vLLM TPU engine is missing.

## Manual checklist

1. **Recover the pushed source**

   ```bash
   git clone https://github.com/YLiu95/kaggle-tpu-inference.git
   cd kaggle-tpu-inference
   git fetch origin main
   git checkout main
   git merge --ff-only origin/main
   git rev-parse HEAD
   git ls-remote origin refs/heads/main
   ```

   The two SHA values must match.

2. **Reinstall only binary dependencies and the local package**

   Follow the lightweight installation commands in `README.md`. The engine
   itself is restored with:

   ```bash
   ktpu setup --engine vllm
   ```

   Setup fails instead of building a dependency from source.

3. **Reuse surviving model cache**

   `ktpu` defaults to `$HF_HOME` or `~/.cache/huggingface`. Do not delete this
   directory if it survived. `--local-files-only` prevents network access and
   verifies that the required tokenizer/config/weights are already cached.

4. **Detect stale state**

   ```bash
   ktpu status
   ktpu clear-tpu
   ```

   If a stale non-managed vLLM process owns TPU devices:

   ```bash
   ktpu clear-tpu --force
   ```

   `--force` still refuses to kill owners that do not look like vLLM or
   `tpu-inference`.

5. **Re-run tests and checkpoint verification**

   ```bash
   python -m unittest discover -s tests -v
   scripts/checkpoint.sh
   ```

6. **Resume at setup or inference**

   ```bash
   ktpu setup --engine vllm  # only if needed
   ktpu run --model google/gemma-4-31B-it \
     --prompt "Your prompt" --max-output 4096
   ```

## Credential safety

Provide `HF_TOKEN` through Kaggle Secrets/the ephemeral environment only. Never
write GitHub or Hugging Face tokens into files, remotes, logs, command
arguments, notebooks, or Git history.

