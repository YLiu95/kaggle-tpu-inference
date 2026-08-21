# RESTART NOTES — read this first after a Kaggle crash

Kaggle TPU sessions die: the notebook times out, the VM is reclaimed, the tunnel drops,
or a JAX process wedges the chips. **Nothing outside this git repo survives.** These are
the exact steps to get back to a working, generating daemon, and the state that has to be
rebuilt each time.

## 0. What survives a restart

| thing | survives? | where |
|---|---|---|
| this source tree | only via GitHub | `github.com/YLiu95/kaggle-tpu-inference` |
| downloaded weights (52-62 GB) | no | `$HF_HOME=/root/hf_cache` |
| XLA compilation cache | no | `/root/.cache/gemma4_jax` |
| the resident daemon (sharded weights) | no | `/tmp/gemma4-tpu.sock` |
| chosen model | no | `/root/.gemma4_model` |

So: **push early, push often.** Everything else is reproducible with one command.

## 1. Cold restart (fresh Kaggle TPU v5e-8 VM)

```bash
cd /root && rm -rf kaggle-tpu-inference && \
git clone https://github.com/YLiu95/kaggle-tpu-inference.git && \
cd kaggle-tpu-inference && bash setup.sh --model 31b
```

`setup.sh` is idempotent: it re-pulls the Kaggle secrets, re-checks that JAX sees 8 chips,
skips the download if `$HF_HOME` still has the snapshot, runs the HBM fit check, warms the
XLA cache, and starts the daemon. Budget roughly:

| step | 31B | 26B-A4B |
|---|---|---|
| download | ~8 min at 130 MB/s | ~7 min |
| load + shard | 23 s | 14 s |
| cold XLA compile | ~7 min | ~6 min |

## 2. Warm restart (VM alive, daemon gone)

```bash
cd /root/kaggle-tpu-inference && bash serve.sh start --model 31b
```

## 3. Restoring git access after a crash

The token lives in Kaggle secrets, not on disk:

```bash
cd /root/kaggle-tpu-inference
export GITHUB_TOKEN=$(python3 -c "
from kaggle_secrets import UserSecretsClient
print(UserSecretsClient().get_secret('GITHUB_TOKEN'))")
git remote set-url origin \
  https://x-access-token:${GITHUB_TOKEN}@github.com/YLiu95/kaggle-tpu-inference.git
git config user.email "agent@kaggle.local" && git config user.name "kaggle-tpu-agent"
```

`HF_TOKEN` comes from the same place; `setup.sh` exports both automatically.

## 4. Symptoms and fixes

**`open(/dev/vfio/N): Device or resource busy`**
A dead process still holds the chips for 10-30 s. `serve.sh start` already polls
`jax.devices()` for up to 5 minutes. If it never clears:
```bash
pkill -9 -f gemma4_tpu.server; pkill -9 -f run_inference; sleep 20
bash serve.sh start --model 31b
```

**The daemon is serving the wrong model.**
The 8 chips hold exactly one model. `run.sh` refuses rather than silently using the wrong
one:
```bash
bash serve.sh restart --model 31b        # or --model 26b-a4b / 12b
```

**Out of memory during load or first prefill.**
Check before paying for the load:
```bash
python3 -m gemma4_tpu.fit_check --model 31b --max-len 16384
```
Lower `--max-len` until it reports under 80% of HBM.

**`run.sh` hangs at "waiting for all 8 TPU devices".**
Another JAX process is alive. `ps aux | grep -E "jax|gemma4"` and kill it.

**Tests need the TPU but the daemon owns it.**
They do not — the suite runs on 8 simulated CPU devices:
```bash
python3 -m pytest tests/ -q          # safe to run while the daemon is up
GEMMA4_TEST_PLATFORM=tpu python3 -m pytest tests/ -q   # only after serve.sh stop
```

## 5. Verifying a restart actually worked

```bash
bash serve.sh status                                   # pid, model_key, max_len, HBM
python3 -m pytest tests/ -q                            # 50 tests, no TPU needed
bash run.sh "Say hello in one sentence." --max-new-tokens 64 --no-think --plain
```

## 6. Work-in-progress hygiene

- Commit and push after every self-contained change; the VM can vanish mid-edit.
- Never leave a long download or compile as the only copy of progress — the snapshot path
  is recorded in `/root/.gemma4_model_dir.<key>` and re-derived by `setup.sh` anyway.
- `bash serve.sh logs` tails `/root/.cache/gemma4_jax/server.log`, which is the only
  record of what the daemon did before it died.
