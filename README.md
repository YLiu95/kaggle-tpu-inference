# Gemma-4 on a Kaggle TPU v5e-8

A from-scratch **JAX / GSPMD** inference stack for Google's Gemma-4 text models that runs
on all **8 TPU v5e chips** of a Kaggle TPU VM and streams tokens to the terminal with a
live throughput + TPU-utilisation dashboard. No vLLM, no torch-xla — `jax.jit` with
explicit `NamedSharding` all the way down.

**Crashed? Start at [RESTART_NOTES.md](RESTART_NOTES.md).**

## Supported models

Pick the checkpoint in the very first command — the model registry
([`gemma4_tpu/models.py`](gemma4_tpu/models.py)) carries the per-model context window,
HBM budget and download size.

| key | checkpoint | arch | params | bf16 weights | default context | ceiling on a v5e-8 |
|---|---|---|---|---|---|---|
| `26b-a4b` *(default)* | `google/gemma-4-26B-A4B-it` | MoE, 128 experts top-8 | 25.9B (~4B active) | 47 GiB | 32,768 | 32,768 |
| `31b` | `google/gemma-4-31B-it` | dense | 30.7B | 57 GiB | 16,384 | 30,720 |
| `12b` | `google/gemma-4-12B-it` | dense | 12.0B | 22 GiB | 32,768 | 65,536 |

```bash
bash setup.sh --list      # the same table, from the registry
```

The 8 chips hold exactly one model at a time. `run.sh` refuses to talk to a daemon that is
serving a different one rather than silently answering with the wrong model.

---

## 1. Setup (run once) — choose your model here

```bash
# --- SETUP ------------------------------------------------------------------
cd /root && rm -rf kaggle-tpu-inference && \
git clone https://github.com/YLiu95/kaggle-tpu-inference.git && \
cd kaggle-tpu-inference && bash setup.sh --model 31b
```

Swap `--model 31b` for `--model 26b-a4b` or `--model 12b`. Add `--max-len N` to change the
resident context window (see [limits](#context-window-and-output-limits)):

```bash
bash setup.sh --model 31b --max-len 30720     # largest context the 31B fits on a v5e-8
bash setup.sh --model 26b-a4b                 # the MoE model, 32K context
```

`setup.sh` pulls `HF_TOKEN` / `GITHUB_TOKEN` from Kaggle secrets, verifies that JAX sees
8 chips, downloads the checkpoint over plain HTTP (Xet hangs on Kaggle, see notes), runs an
**HBM fit check** before paying for the load, warms the persistent XLA compilation cache,
and starts the **persistent TPU daemon** so the weights stay sharded across the 8 chips and
the XLA programs stay compiled between runs.

## 2. Run inference (re-runnable, instant)

```bash
# --- RUN --------------------------------------------------------------------
cd /root/kaggle-tpu-inference && \
bash run.sh "Explain why a systolic array suits matrix multiplication, then work through a 2x2 example." \
     --max-new-tokens 768 --temperature 1.0
```

Re-run with any prompt as often as you like: it talks to the daemon over a Unix socket, so
there is **no reload and no recompile** — first token in tens of milliseconds.

```
$ bash run.sh "..."                    # instant (daemon already up)
$ bash run.sh "..." --model 31b        # errors out if the daemon holds another model
$ bash serve.sh restart --model 31b    # switch model / context window
$ bash serve.sh status                 # pid / model / uptime / requests / decode step time
$ bash serve.sh stop                   # release the 8 chips
$ bash serve.sh logs
```

Useful flags: `--plain` (no dashboard, raw ANSI streaming), `--no-think` (skip the
reasoning channel), `--top-p`, `--top-k`, `--seed`, `--system`, `--model-dir`,
`--list-models`, `--local` (skip the daemon and load in-process).

---

## Context window and output limits

Two separate budgets, and the stack **warns instead of failing** whenever a request bumps
into either one. Every warning names the limit that was hit and the exact command that
raises it.

| budget | what it is | how to change it | cost |
|---|---|---|---|
| **context window** (`--max-len`) | the resident KV cache the daemon allocated at start-up | `bash serve.sh restart --max-len N`, or `GEMMA4_MAX_LEN=N`, or `bash run.sh ... --max-len N --local` | reallocates the cache and forces one recompile |
| **max new tokens** (`--max-new-tokens`) | how many tokens *this* request may generate | `bash run.sh "..." --max-new-tokens N` | free |

Four warnings, all emitted as structured events so the daemon, the dashboard and
`--plain` mode print the same thing ([`gemma4_tpu/limits.py`](gemma4_tpu/limits.py)):

```
$ bash run.sh "..." --max-new-tokens 50000
warning (request_exceeds_max_new_tokens): you asked for 50,000 new tokens but the
  per-request cap is 16,384
  how to fix: the cap is MAX_OUTPUT_TOKENS in gemma4_tpu/limits.py and can never exceed
  the resident context window, so raise the resident context with
  `bash serve.sh restart --max-len N` ...
warning (output_truncated_by_context): prompt 54 + requested 50,000 = 50,054 tokens
  exceeds the 16,384-token context window, so max-new-tokens was reduced to 16,330
  how to fix: ... Serving this model with `--max-len 30720` (ceiling 30,720) would grant
  the full request.
```

| code | fires when | effect |
|---|---|---|
| `output_truncated_by_context` | `prompt + max_new_tokens > context` | output silently clamped to what fits, with the adjustment printed |
| `request_exceeds_max_new_tokens` | `max_new_tokens` above the per-request cap | clamped |
| `prompt_exceeds_context` | the **input alone** does not fit | request refused before any TPU work |
| `prompt_prefill_memory_risk` | the prompt is long enough that prefill attention may exhaust HBM | proceeds, but warns first |
| `hit_max_new_tokens` | generation stopped at the budget rather than at an end-of-turn token | printed after the answer, so you know it is truncated |

The last one matters in practice: Gemma-4 emits a reasoning channel first, so a truncated
run often produces zero answer tokens. The warning tells you to raise
`--max-new-tokens` rather than leaving you guessing.

### Why the 31B defaults to 16K and not 32K

`python3 -m gemma4_tpu.fit_check --model 31b --max-len N` predicts the HBM split before
anything is loaded — an OOM after a 6-minute load is the most expensive failure mode on a
Kaggle TPU.

```
$ python3 -m gemma4_tpu.fit_check --model 31b --max-len 16384
  chips              8 x 15.75 GiB HBM
  weights / chip       7.15 GiB
  KV cache / chip      2.81 GiB at max-len 16,384 (180 KiB/token)
  total / chip         9.96 GiB  (63% of HBM)
  largest safe max-len for this model: 31,755 tokens
```

At 32,768 the same check reports 12.77 GiB/chip (81%) and warns. 30,720 is the registry
ceiling: 12.4 GiB/chip, just under the 80% line, leaving room for prefill activations.

---

## How it works

| piece | choice |
|---|---|
| framework | JAX 0.10.2 + GSPMD (`jax.jit` with `NamedSharding`), no vLLM/torch-xla |
| mesh | 1-D `tp` mesh over all 8 chips, activations replicated |
| sliding layers | q/k/v/o split over the KV heads (8 on the 26B-A4B, 16 on the 31B); q heads follow their KV head, so grouped-query attention needs **zero** cross-chip traffic |
| full layers | too few KV heads to split (2 / 4), so KV is replicated and the 8 query *groups* are split instead |
| dense MLP | Megatron column/row split over `intermediate_size` |
| MoE (26B-A4B only) | split over `moe_intermediate_size`: every chip keeps all 128 experts but 1/8 of each expert's hidden dim. The top-8 lookup stays a **local** HBM read and the combine is one small all-reduce — no expert all-to-all |
| embeddings / lm_head | split over `hidden_size`; lookup is local, lm_head is an all-reduce over the contracted dim |
| prefill | bucketed to multiples of 256; MoE evaluated densely |
| decode | one `dynamic_slice` per selected expert (see notes — a 2.3x win on the MoE model) |
| streaming | decode steps are dispatched 3 steps ahead and tokens are fed back as device arrays, so the host loop never stalls the TPU |

Dense and MoE share one code path: `enable_moe_block=False` in the checkpoint config drops
the router/expert branch and leaves the dense MLP. The dense checkpoints ship the MoE keys
as JSON `null`, which the config loader normalises to 0.

### Measured on Kaggle TPU v5e-8

| metric | `31b` (dense) | `26b-a4b` (MoE) |
|---|---|---|
| weights | 57.2 GiB bf16, **7.15 GiB/chip** | 47.0 GiB bf16, 6.1 GiB/chip |
| weight load + shard | **23.4 s** | ~14 s |
| cold XLA compile | **291 s prefill + 221 s decode** (60 layers) | ~115 s + ~255 s (30 layers) |
| repeat run via the daemon | **~0 s startup, TTFT 56-81 ms** | ~0 s, TTFT 44-69 ms |
| resident context (default / ceiling) | **16,384 / 30,720** | 32,768 / 32,768 |
| HBM at the default context | **9.96 / 15.75 GiB per chip (63%)** | ~7.7 GiB (49%) |
| KV cache | **180 KiB/token/chip** (880 KiB/token total) | ~45 KiB/token/chip |
| prefill (66-token prompt, 256 bucket) | **818 tok/s, TTFT 81 ms** | ~6.9k tok/s padded |
| decode, batch 1 | **19.88 ms/token = 51.5 tok/s** | ~9.4 ms/token = ~111 tok/s |
| implied HBM bandwidth while decoding | **~386 GB/s per chip (47% of the 819 GB/s peak)** | ~150 GB/s per chip |
| longest prompt that fits alongside the cache | **5,120 tokens** (see the prefill note) | ~7,168 tokens |

Verified end to end on the 31B: a 3,168-token prompt answered correctly; a 30,022-token
prompt refused with `prompt_exceeds_context` before any TPU work; `--max-new-tokens 50000`
clamped to 16,356 with both warnings printed; `--max-new-tokens 30` truncated with
`hit_max_new_tokens`.

The dense 31B is genuinely memory-bandwidth-bound — it reads all 30.7B parameters per
token and reaches 47% of peak HBM bandwidth, against ~18% for the MoE model, whose batch-1
decode is dominated by per-layer launch latency instead.

---

## Tests

```bash
python3 -m pytest tests/ -q       # 50 tests, no TPU required
```

The suite runs on the **CPU backend forced to 8 devices**
(`--xla_force_host_platform_device_count=8`), so it exercises the real 8-way `tp` mesh,
every `NamedSharding` spec and every collective — and it is safe to run while the daemon
owns the actual chips. `GEMMA4_TEST_PLATFORM=tpu` re-runs the same tests on silicon (stop
the daemon first).

| file | covers |
|---|---|
| `tests/test_parity.py` | numerical parity against HuggingFace `transformers` for **both** architectures: a 26B-A4B-shaped MoE model and a 31B-shaped dense model (16 sliding KV heads / 4 global, so the full-attention layers shard query groups). `corr > 0.9999`, identical argmax on every position, prefill and decode paths both checked |
| `tests/test_limits.py` | the pure limit/warning logic: clamping, every warning code, the remedy text, `GEMMA4_MAX_LEN` |
| `tests/test_models.py` | the registry, dense-vs-MoE config loading (null MoE fields), sharding divisibility, the HBM and prefill-memory estimators |
| `tests/test_warnings.py` | the full event stream through a stubbed engine: which warnings fire, in what order, and that a too-long prompt never reaches the TPU |

---

## Repo layout

```
setup.sh              one-time environment + weights + XLA cache + daemon start (--model)
serve.sh              start/status/stop/restart/logs for the persistent TPU daemon
run.sh                streaming inference entry point (auto-starts the daemon)
_common.sh            shared model/context resolution for the three scripts
run_inference.py      CLI: daemon client, or in-process fallback
bench.py              decode microbenchmark + ablation harness
gemma4_tpu/
  models.py           model registry: repo ids, context defaults, HBM ceilings
  config.py           Gemma-4 text config loader (dense + MoE)
  limits.py           context/output budgets and the warnings that explain them
  fit_check.py        predicts weights+KV HBM per chip before loading anything
  model.py            the JAX model: attention, MoE, RoPE, masks, sampling, sharding
  weights.py          streaming safetensors -> sharded device arrays
  engine.py           jitted prefill/decode + async-dispatch streaming loop
  server.py           persistent daemon: holds the TPU, serves generations over a socket
  session.py          unified event stream (local or remote) + prompt building
  ui.py               rich dashboard rendering, driven purely by events
  stream.py           incremental detokeniser + reasoning-channel splitter
  tpu_monitor.py      per-chip HBM / duty-cycle sampling
tests/                see above
```

---

## Kaggle TPU field notes

Problems hit while building this, and what fixed them. Anything marked *unresolved* is a
live hazard for other agents doing AI work on Kaggle TPUs.

- **The dense 31B needs a different sharding decision from the MoE 26B.** Its
  full-attention layers have 4 KV heads, its sliding layers 16, and 32 query heads. Splitting
  KV heads works for the sliding layers (16/8 = 2 per chip) but not for the full layers, so
  those replicate KV and split the 8 query groups instead. `validate_sharding()` now checks
  every axis up front — a non-divisible axis otherwise shows up as silent XLA padding or as
  an error minutes into a load.
- **Prefill, not the KV cache, is what limits prompt length.** Attention materialises
  `[heads, T, T]` float32 scores, so cost grows with the *square* of the prompt. On the 31B
  the full-attention layers replicate KV, so every chip carries all 32 heads:
  `32 x T^2 x 6` bytes. At the default 16K context that leaves room for about **5,120
  prompt tokens** (the daemon reports the number as `safe_prompt_tokens`). The stack
  estimates this at start-up and warns (`prompt_prefill_memory_risk`) instead of OOM-ing.
  *Unresolved:* the real fix is chunked or flash-style prefill, which costs another set of
  XLA compiles.
- **Every new prompt-length bucket costs a fresh compile.** Prefill is bucketed to
  multiples of 256 and each bucket is a separate `jit` specialisation. The first
  3,168-token prompt on the 31B took **178 s to first token** — almost all of it compiling
  the 3,328 bucket — and then microseconds on every later prompt of similar length. Warm
  the buckets you care about, or expect one slow request per new prompt-size band.
- **A too-large `--max-len` only fails after the load.** Hence `fit_check.py` and the
  start-up HBM warning. 80% of HBM is the practical line: above that, prefill activations
  stop fitting.
- **Dense Gemma-4 checkpoints set the MoE config keys to `null`, not absent.**
  `tc["num_experts"]` returns `None`, not a `KeyError`, so `int(None)` blows up in a way
  that looks like a corrupt config. The loader normalises `None -> 0` and cross-checks
  against `enable_moe_block`.
- **`huggingface_hub` Xet transfer hangs.** The 50 GB shard downloaded to full size, then
  the process sat in `futex_wait` forever with the blob still named `*.incomplete` and an
  unwritten safetensors header, so `safe_open()` also hung. Fix: set `HF_HUB_DISABLE_XET=1`
  and use plain HTTP. `HF_HUB_ENABLE_HF_TRANSFER` is a no-op (deprecated) in
  `huggingface_hub` 1.21.
- **TPU chips stay claimed after a process dies.** For ~10-30 s after SIGKILL you get
  `open(/dev/vfio/N): Device or resource busy`. Never chain TPU runs back to back — poll
  `jax.devices()` until it succeeds (`serve.sh` does this).
- **Tests must not need the TPU.** The daemon owns all 8 chips for the whole session, so a
  TPU-only test suite can never be run without tearing down the thing you are testing.
  Forcing the CPU backend to 8 devices keeps the sharding coverage and removes the conflict.
- **libtpu's metrics gRPC server never starts on Kaggle.** libtpu logs `Could not find
  SliceBuilder port 8471 in any of the 0 ports provided in tpu_process_addresses="local"`,
  so nothing listens on `TPU_RUNTIME_METRICS_PORTS` (8431-8438) and `tpu-info` reports
  `HBM N/A / duty N/A`. Setting `TPU_PROCESS_ADDRESSES=localhost:8471` does not help.
  *Workaround:* read per-chip HBM from `jax.Device.memory_stats()` inside the owning process
  and report a **measured** TensorCore-busy % (blocking device time per step x tokens/s).
  *Unresolved:* no real hardware duty cycle, and `tpu-info` also warns that libtpu 0.0.17 is
  partly incompatible with Python 3.12.
- **`jnp.take` over the expert axis is a trap.** Gathering the top-8 of 128 experts with
  `jnp.take` cost 15.2 ms/token — the same as evaluating *all 128* experts (14.2 ms). XLA
  materialises the gather instead of reading only the selected slices. Replacing it with an
  unrolled `lax.dynamic_index_in_dim` per selected expert dropped decode to 6.7 ms/token
  (2.3x). Worth checking on any MoE you port to TPU.
- **Batch-1 decode is overhead-bound, not bandwidth-bound** on the MoE model. Ablations:
  MoE ~1.5 ms, attention ~3.7 ms, rest ~1.5 ms — attention's cost is dominated by per-layer
  op/collective launch latency, not arithmetic. Fixing that needs kernel fusion (Pallas) or
  larger batches, not more sharding. The dense 31B is the opposite: it is genuinely
  weight-bandwidth-bound, because all 30.7B parameters are read every token.
- **XLA compile time dominates cold start, and it scales with layer count.** 30 unrolled
  layers (26B-A4B) take ~6 min to trace/lower/compile; the 31B's 60 layers take about twice
  that. Set `jax_compilation_cache_dir` (plus
  `jax_persistent_cache_min_entry_size_bytes=-1` and
  `jax_persistent_cache_min_compile_time_secs=1.0`) — but note the cache only skips the XLA
  backend compile, **not** JAX tracing/lowering. Any change to input *shardings* (e.g. an
  uncommitted `jnp.zeros` vs a `device_put` with `NamedSharding`) silently invalidates the
  cache and forces a full recompile. Because none of this can be cached away, the only real
  fix is to **not exit the process**: keep a daemon resident that owns the TPU, the sharded
  weights and the compiled executables (`serve.sh` / `gemma4_tpu/server.py`), and drive it
  from a thin socket client.
- **Warm-up must use the exact same shardings as the real call**, otherwise `jit`
  respecialises and your first "real" request pays the whole compile again (this showed up
  as a 148 s TTFT).
- **`aten::_grouped_mm` is not implemented on CPU**, so `transformers`' default MoE kernel
  cannot run a Gemma-4 reference on the host. Set `config._experts_implementation = "eager"`
  to get a CPU-runnable reference for parity testing.
- **Gemma-4 architecture gotchas** (easy to get silently wrong): attention scaling is `1.0`
  (not `1/sqrt(d)`) because q/k are RMS-normed first; full-attention layers use
  `head_dim=512` with `k == v` (no `v_proj` in the checkpoint) while sliding layers use 256;
  full-attention RoPE is `proportional` with `partial_rotary_factor=0.25`, i.e. the upper
  3/4 of each head is NoPE (zero inverse frequency); `final_logit_softcapping=30`; the MoE
  branch consumes the residual *before* the dense MLP's pre-norm; and `embed_scale` is
  computed in bf16, so `sqrt(2816)` rounds to 53.0, not 53.066.
- **Reasoning tokens** are delimited by the single-token markers `<|channel>` (id 100) and
  `<channel|>` (id 101) and only appear when the chat template is rendered with
  `enable_thinking=True`. Split on **token ids**, not decoded text — a text-based splitter
  silently classified an entire 768-token generation as "reasoning". Also suppress the stop
  token (`<turn|>`, id 106) or it gets printed as literal text.
- *Unresolved:* transparent hugepages are disabled in the Kaggle image, which JAX warns
  slows TPU runtime start/stop; enabling it needs host root access.
- *Unresolved:* an earlier attempt in this repo used `vllm-tpu` 0.26.0, which injects
  `--xla_tpu_use_dynamic_smem_negotiation=true` — a flag the Kaggle `v5litepod` runtime
  rejects, killing the engine before any weights load. That is why this implementation talks
  to JAX directly.
