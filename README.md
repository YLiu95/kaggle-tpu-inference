# Crash-Safe Kaggle TPU Inference CLI

`ktpu` is a checkpoint-first command-line workflow for running one streamed
Gemma 4 request on a local Kaggle TPU VM. It detects hardware, sizes context
from model and HBM metadata, applies conservative CPU/process limits, monitors
TPU/host utilization, and writes mirrored CSV telemetry.

## Safety model

Engine installation, TPU initialization, model loading, XLA compilation, and
inference are treated as risky. Before `setup` or `run`, `ktpu`:

1. runs the lightweight unit test suite;
2. requires a clean `main` branch;
3. requires local `HEAD` to equal `origin/main`;
4. rejects credential-bearing HTTP Git remotes;
5. checks RAM, disk, normalized CPU load, TPU ownership/utilization, and
   conflicting vLLM processes.

The vLLM process binds only to `127.0.0.1`, runs at reduced priority with a
bounded CPU affinity and thread environment, and is configured for one
sequence/request (`--max-num-seqs 1`). Engine setup permits binary wheels only
and stops rather than compiling a missing dependency from source.

## Install the lightweight CLI

```bash
cd ~/kaggle-tpu-inference
python -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: \
  "hatchling>=1.26" "huggingface-hub>=0.25" "httpx>=0.27" \
  "psutil>=5.9" "rich>=13" "transformers>=4.45" "typer>=0.12"
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

Do not store GitHub or Hugging Face tokens in the repository, Git remote,
commands, or logs. If model authentication is needed, expose `HF_TOKEN` only
through the ephemeral environment.

## Commands

```bash
# Hardware, engine, server, and telemetry state (does not initialize JAX)
ktpu status

# Binary-only, pinned vLLM TPU installation; checkpoint gate runs first
ktpu setup --engine vllm

# One streamed reasoning + response request
ktpu run \
  --model google/gemma-4-31B-it \
  --prompt "A snail climbs a 20-foot well: +3 feet by day, -2 at night. How many days?" \
  --max-output 4096

# Stop a managed server and discard stale state
ktpu clear-tpu

# Also clear non-managed owners only when their command looks like vLLM
ktpu clear-tpu --force
```

Prompts can also be supplied by `--prompt-file` or stdin. Useful run options
include `--context`, `--max-output`, `--cpu-limit`, `--no-thinking`,
`--local-files-only`, and `--startup-timeout`.

## Hardware-aware sizing

The CLI reads:

- TPU generation, chip count, per-chip/total HBM, and current runtime metrics;
- model weight file sizes and declared context limit;
- text-layer count, KV-head count, head dimension, and KV dtype;
- the exact token count of the rendered chat template.

For safety, KV bytes/token use the full-attention upper bound:

```text
KV bytes/token = 2 × layers × KV heads × head dimension × dtype bytes
```

The HBM budget subtracts current use, weights, 5% weight overhead, and runtime
headroom (`max(15% of HBM, 8 GiB)`). KV sizing adds 10% overhead and rounds
context down to a 256-token bucket. Then:

```text
context = min(calculated safe context, optional user context cap)
max output = min(context - rendered input tokens, optional output cap)
```

The default CLI output cap is 4096 tokens; passing a different
`--max-output` changes it. Every calculation input and applied limit is printed
before server launch.

## Monitoring and logs

During startup and streaming, the terminal shows TPU chip count, HBM,
TensorCore duty cycle, bounded-process CPU, host RAM, time to first token
(TTFT), generated tokens, and token speed.

CSV rows are flushed continuously to:

```text
~/.local/state/ktpu/logs/inference-*.csv
```

When `/kaggle/working` is writable, every row is also written to:

```text
/kaggle/working/ktpu/logs/inference-*.csv
```

Server logs are copied there on shutdown. Runtime state, logs, downloaded
weights, caches, and secrets are excluded from Git.

## Checkpoint helper

After reviewing and committing changes:

```bash
scripts/checkpoint.sh --push
```

This runs tests, requires a clean `main`, pushes, and verifies the remote SHA.
It deliberately does not create commits or stage files for you.

## Recovery

See [`RESUME.md`](RESUME.md), or run:

```bash
scripts/resume.sh --clear-tpu
```

Use `--setup` when the engine installation did not survive.

## Tested versions

The implementation pins the risky engine environment:

| Component | Version |
|---|---:|
| Python | 3.12 |
| `vllm-tpu` | 0.26.0 |
| `tpu-inference` | 0.26.0 (engine dependency) |
| `uv` | 0.12.3 |
| `jax` / `jaxlib` | 0.10.2 |
| `torch` | 2.10.0 |
| `transformers` (engine) | 5.14.1 |

Exact validation results and the last successful end-to-end command are
tracked in [`docs/VALIDATION.md`](docs/VALIDATION.md).
