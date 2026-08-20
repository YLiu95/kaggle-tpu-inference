# Gemma-4 26B-A4B on a Kaggle TPU v5e-8

A from-scratch **JAX / GSPMD** inference stack for `google/gemma-4-26B-A4B-it`
(26B total / ~4B active MoE) that runs on all **8 TPU v5e chips** of a Kaggle TPU VM and
streams tokens to the terminal with a live throughput + TPU-utilisation dashboard.

```
TTFT 44-69 ms  •  ~111 tok/s decode at 32K  •  47 GiB bf16 weights split 8 ways
```

Correctness is pinned to HuggingFace `transformers` by a parity test
(`tests/test_parity.py`) that builds a small random Gemma-4 with the same structure
(MoE + sliding/full attention + `k==v` global attention + logit softcap) and compares
logits: `corr = 0.99997`, identical argmax on every position.

---

## 1. Setup (run once)

```bash
# --- SETUP ------------------------------------------------------------------
cd /root && rm -rf kaggle-tpu-inference && \
git clone https://github.com/YLiu95/kaggle-tpu-inference.git && \
cd kaggle-tpu-inference && bash setup.sh
```

`setup.sh` pulls `HF_TOKEN` from Kaggle secrets, verifies that JAX sees 8 chips,
downloads the ~52 GB checkpoint over plain HTTP (Xet hangs on Kaggle, see notes),
warms the persistent XLA compilation cache (~7 min, once), and starts the **persistent
TPU daemon** so the weights stay sharded across the 8 chips and the XLA programs stay
compiled between runs.

## 2. Run inference (re-runnable, instant)

```bash
# --- RUN --------------------------------------------------------------------
cd /root/kaggle-tpu-inference && \
bash run.sh "Explain why a systolic array suits matrix multiplication, then work through a 2x2 example." \
     --max-new-tokens 768 --temperature 1.0
```

Re-run that command with any prompt as often as you like: it talks to the daemon over a
Unix socket, so there is **no reload and no recompile** — first token in tens of ms.

The daemon reserves a **32,768-token context cache** on the v5e-8. `--max-new-tokens`
is adjusted automatically instead of failing: a request for 32,768 new tokens with a
54-token rendered prompt becomes 32,714, while a request for 5,000 remains 5,000. The
terminal prints the adjustment before generation.

```
$ bash run.sh "..."          # 1st time after setup: instant (daemon already up)
using persistent TPU daemon (pid 78944, up 658s, 4 prior requests)
TTFT 44 ms  •  110.7 tok/s (32K resident cache)

$ bash serve.sh status       # pid / uptime / requests / decode step time
$ bash serve.sh stop         # release the 8 chips
$ bash serve.sh restart
$ bash serve.sh logs
```

If the daemon is not running, `run.sh` starts it automatically (a cold 32K compile can
take several minutes) and then streams. Pass `--local` to skip the daemon entirely and
load everything in-process instead.

Live dashboard (updates ~10x/s while tokens stream):

```
╭──────────────────────────────── Throughput ─────────────────────────────────╮
│      TTFT       44.0 ms    prompt   32 tok      prefill 726 tok/s   gen  352 │
│       now   110.73 tok/s      avg  110.73 tok/s    ITL     9.0 ms            │
│ reasoning   347 tok        answer    5 tok       phase  done                 │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────── TPU utilisation ───────────────────────────────╮
│ chip  kind          HBM used   HBM total  used %      peak   duty            │
│    0  TPU v5 lite   6.12 GiB   15.75 GiB    38.9  6.30 GiB    n/a            │
│  ...  (all 8 chips, identical -> weights really are sharded 8 ways)          │
│  aggregate   HBM 48.96 / 125.98 GiB over 8 chips   libtpu duty n/a           │
│ tensorcore   100.0% busy (measured: 9.41 ms device time per decode step)     │
│   achieved     0.85 TFLOP/s      940.0 GB/s (14.3% of HBM BW)                │
│        host   RSS 13.2 GB   CPU 190%                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── stream (reasoning dimmed) ───────────────────────────╮
│ ### 3. Calculate where they meet:                                            │
│ $$\text{Distance from A} = 60\text{ km/h} \times 3\text{ hours} = 180$$      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Useful flags: `--plain` (no dashboard, raw ANSI streaming), `--no-think`
(skip the reasoning channel), `--top-p`, `--top-k`, `--seed`,
`--system`, `--model-dir`.

---

## How it works

| piece | choice |
|---|---|
| framework | JAX 0.10.2 + GSPMD (`jax.jit` with `NamedSharding`), no vLLM/torch-xla |
| mesh | 1-D `tp` mesh over all 8 chips, activations replicated |
| sliding layers (25) | q/k/v/o split over the 8 KV heads; q heads follow their KV head, so grouped-query attention needs **zero** cross-chip traffic |
| full layers (5) | only 2 KV heads, so KV is replicated and the 8 query *groups* are split instead |
| dense MLP | Megatron column/row split over `intermediate_size` |
| MoE (128 experts, top-8) | split over `moe_intermediate_size`: every chip keeps all 128 experts but 1/8 of each expert's hidden dim. The top-8 lookup stays a **local** HBM read and the combine is one small all-reduce — no expert all-to-all |
| embeddings / lm_head | split over `hidden_size`; lookup is local, lm_head is an all-reduce over the contracted dim |
| prefill | dense MoE (all 128 experts in one matmul), bucketed to multiples of 256 |
| decode | one `dynamic_slice` per selected expert (see notes — this is a 2.3x win) |
| streaming | decode steps are dispatched 3 steps ahead and tokens are fed back as device arrays, so the host loop never stalls the TPU (measured 98%+ TensorCore busy) |

### Measured on Kaggle TPU v5e-8

| metric | value |
|---|---|
| weights | 47.0 GiB bf16, 6.1-6.3 GiB per chip (39% HBM), identical on all 8 |
| weight load + shard | ~14 s |
| cold start | ~115 s prefill + ~255 s decode compile |
| warm start (cached XLA) | ~14 s load + ~95 s + ~115 s ≈ 3.5 min — the residue is JAX tracing/lowering, which the persistent cache does *not* skip |
| **repeat run via the daemon** | **~0 s startup, TTFT 44-69 ms** |
| resident context / max output | 32,768 tokens / prompt-adjusted 32,768 tokens |
| 32K KV-cache footprint | ~1.41 GiB/chip; projected total ~7.7/15.75 GiB HBM/chip |
| prefill | 256-token bucket in 37 ms (~6.9k tok/s padded) |
| TTFT (67-token prompt) | **58 ms** |
| decode, 32K resident cache | **~9.4 ms/token = ~111 tok/s** at batch 1 |
| TensorCore busy while streaming | ~100% (async dispatch keeps the queue full) |
| achieved HBM bandwidth | ~1.2 TB/s aggregate (~19% of the 6.5 TB/s peak) |
| MoE decode variants | unrolled `dynamic_slice` **6.72 ms** · `fori_loop` 8.79 ms (half the compile time) · `jnp.take` gather 15.21 ms · dense-128 14.18 ms |

Live limit validation: `--max-new-tokens 5000` was admitted unchanged; 32,768 became
32,736 for a 32-token prompt. Both requests generated 352 tokens (347 reasoning + 5
answer), stopped naturally, and ran at 110.7 tok/s.

Set `GEMMA4_MOE_DECODE=loop` to halve decode compile time at ~24% lower throughput,
or `=take` / use `--ablate densemoe` in `bench.py` to reproduce the comparison.

---

## Repo layout

```
setup.sh              one-time environment + weights + XLA cache + daemon start
serve.sh              start/status/stop/restart/logs for the persistent TPU daemon
run.sh                streaming inference entry point (auto-starts the daemon)
run_inference.py      CLI: daemon client, or in-process fallback
bench.py              decode microbenchmark + ablation harness
gemma4_tpu/
  config.py           Gemma-4 text config loader
  model.py            the JAX model: attention, MoE, RoPE, masks, sampling, sharding
  weights.py          streaming safetensors -> sharded device arrays
  engine.py           jitted prefill/decode + async-dispatch streaming loop
  server.py           persistent daemon: holds the TPU, serves generations over a socket
  session.py          unified event stream (local or remote) + prompt building
  ui.py               rich dashboard rendering, driven purely by events
  stream.py           incremental detokeniser + reasoning-channel splitter
  tpu_monitor.py      per-chip HBM / duty-cycle sampling
tests/test_parity.py  HF transformers numerical parity check
```

---

## Kaggle TPU field notes

Problems hit while building this, and what fixed them. Anything marked *unresolved*
is a live hazard for other agents doing AI work on Kaggle TPUs.

- **Large outputs originally failed before inference.** The daemon had a hard-coded
  4,096-token KV cache, so even `--max-new-tokens 5000` raised
  `prompt + max_new exceeds max_len`. The v5e-8 can comfortably hold the model plus a
  32,768-token cache (~7.7/15.75 GiB projected HBM per chip). The daemon now reserves
  32K and automatically clamps output to `min(requested, 32768 - prompt_tokens)`, with
  the adjustment printed in the terminal instead of failing.
- **`huggingface_hub` Xet transfer hangs.** The 50 GB shard downloaded to full size,
  then the process sat in `futex_wait` forever with the blob still named `*.incomplete`
  and an unwritten safetensors header, so `safe_open()` also hung. Fix: set
  `HF_HUB_DISABLE_XET=1` and use plain HTTP. `HF_HUB_ENABLE_HF_TRANSFER` is a no-op
  (deprecated) in `huggingface_hub` 1.21.
- **TPU chips stay claimed after a process dies.** For ~10-30 s after SIGKILL you get
  `open(/dev/vfio/N): Device or resource busy`. Never chain TPU runs back to back —
  poll `jax.devices()` until it succeeds (`run.sh` does this).
- **libtpu's metrics gRPC server never starts on Kaggle.** libtpu logs
  `Could not find SliceBuilder port 8471 in any of the 0 ports provided in
  tpu_process_addresses="local"`, so nothing listens on `TPU_RUNTIME_METRICS_PORTS`
  (8431-8438) and `tpu-info` reports `HBM N/A / duty N/A`. Setting
  `TPU_PROCESS_ADDRESSES=localhost:8471` does not help. *Workaround:* read per-chip HBM
  from `jax.Device.memory_stats()` inside the owning process and report a **measured**
  TensorCore-busy % (blocking device time per step x tokens/s). *Unresolved:* no real
  hardware duty cycle, and `tpu-info` also warns that libtpu 0.0.17 is partly
  incompatible with Python 3.12.
- **`jnp.take` over the expert axis is a trap.** Gathering the top-8 of 128 experts with
  `jnp.take` cost 15.2 ms/token — the same as evaluating *all 128* experts (14.2 ms).
  XLA materialises the gather instead of reading only the selected slices. Replacing it
  with an unrolled `lax.dynamic_index_in_dim` per selected expert dropped decode to
  6.7 ms/token (2.3x). Worth checking on any MoE you port to TPU.
- **Batch-1 decode is overhead-bound, not bandwidth-bound.** At 6.7 ms/token we move
  ~1.2 TB/s aggregate = ~18% of HBM peak. Ablations: MoE ~1.5 ms, attention ~3.7 ms,
  rest ~1.5 ms — attention's cost is dominated by ~30 x ~100 us of per-layer op/collective
  launch latency, not by arithmetic. Fixing that needs kernel fusion (Pallas) or larger
  batches, not more sharding.
- **XLA compile time dominates cold start.** 30 unrolled layers take ~2 min to
  trace/lower plus ~2-4 min to compile. Set `jax_compilation_cache_dir` (plus
  `jax_persistent_cache_min_entry_size_bytes=-1` and
  `jax_persistent_cache_min_compile_time_secs=1.0`) — but note the cache only skips the
  XLA backend compile, **not** JAX tracing/lowering (~95 s here). Any change to input
  *shardings* (e.g. an uncommitted `jnp.zeros` vs a `device_put` with `NamedSharding`)
  silently invalidates the cache and forces a full recompile. Because none of this can be
  cached away, the only real fix is to **not exit the process**: keep a daemon resident
  that owns the TPU, the sharded weights and the compiled executables
  (`serve.sh` / `gemma4_tpu/server.py`), and drive it from a thin socket client.
- **Warm-up must use the exact same shardings as the real call**, otherwise `jit`
  respecialises and your first "real" request pays the whole compile again (this showed
  up as a 148 s TTFT).
- **`aten::_grouped_mm` is not implemented on CPU**, so `transformers`' default MoE
  kernel cannot run a Gemma-4 reference on the host. Set
  `config._experts_implementation = "eager"` to get a CPU-runnable reference for parity
  testing.
- **Gemma-4 architecture gotchas** (easy to get silently wrong): attention scaling is
  `1.0` (not `1/sqrt(d)`) because q/k are RMS-normed first; full-attention layers use
  `head_dim=512` with `k == v` (no `v_proj` in the checkpoint) while sliding layers use
  256; full-attention RoPE is `proportional` with `partial_rotary_factor=0.25`, i.e. the
  upper 3/4 of each head is NoPE (zero inverse frequency); `final_logit_softcapping=30`;
  the MoE branch consumes the residual *before* the dense MLP's pre-norm; and
  `embed_scale = bf16(sqrt(2816)) = 53.0`, not 53.066.
- **Reasoning tokens** are delimited by the single-token markers `<|channel>` (id 100) and
  `<channel|>` (id 101) and only appear when the chat template is rendered with
  `enable_thinking=True`. Split on **token ids**, not decoded text — a text-based splitter
  silently classified an entire 768-token generation as "reasoning". Also suppress the
  stop token (`<turn|>`, id 106) or it gets printed as literal text.
- *Unresolved:* transparent hugepages are disabled in the Kaggle image, which JAX warns
  slows TPU runtime start/stop; enabling it needs host root access.
- *Unresolved:* an earlier attempt in this repo used `vllm-tpu` 0.26.0, which injects
  `--xla_tpu_use_dynamic_smem_negotiation=true` — a flag the Kaggle `v5litepod` runtime
  rejects, killing the engine before any weights load. That is why this implementation
  talks to JAX directly.
