# Gemma-4 26B-A4B on a Kaggle TPU v5e-8

A from-scratch **JAX / GSPMD** inference stack for `google/gemma-4-26B-A4B-it`
(26B total / ~4B active MoE) that runs on all **8 TPU v5e chips** of a Kaggle TPU VM and
streams tokens to the terminal with a live throughput + TPU-utilisation dashboard.

```
TTFT ~64 ms  •  ~150 tok/s decode  •  6.7 ms/token  •  47 GiB bf16 weights split 8 ways
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
and warms the persistent XLA compilation cache so later runs start in ~40 s
instead of ~6 min.

## 2. Run inference

```bash
# --- RUN --------------------------------------------------------------------
cd /root/kaggle-tpu-inference && \
bash run.sh "Explain why a systolic array suits matrix multiplication, then work through a 2x2 example." \
     --max-new-tokens 768 --temperature 1.0
```

Live dashboard (updates ~10x/s while tokens stream):

```
╭──────────────────────────────── Throughput ─────────────────────────────────╮
│      TTFT       63.6 ms    prompt   61 tok     prefill  959 tok/s   gen  300 │
│       now   149.83 tok/s      avg  149.55 tok/s    ITL     6.7 ms            │
│ reasoning   251 tok        answer   49 tok       phase  answering            │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────── TPU utilisation ───────────────────────────────╮
│ chip  kind          HBM used   HBM total  used %      peak   duty            │
│    0  TPU v5 lite   6.29 GiB   15.75 GiB    39.9  6.29 GiB    n/a            │
│  ...  (all 8 chips, identical -> weights really are sharded 8 ways)          │
│  tensorcore   98.4% busy (measured: 6.72 ms device time per decode step)     │
│    achieved     1.15 TFLOP/s   1194.0 GB/s (18.2% of HBM BW)                 │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── stream (reasoning dimmed) ───────────────────────────╮
│ *   Vehicle 1 (Train A): Speed = 60 km/h, starting point = City A.           │
│ ...                                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Useful flags: `--plain` (no dashboard, raw ANSI streaming), `--no-think`
(skip the reasoning channel), `--max-len`, `--top-p`, `--top-k`, `--seed`,
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
| weights | 47.0 GiB bf16, 5.9 GiB per chip (39.9% HBM) |
| weight load + shard | ~14 s |
| first compile (cold) | ~115 s prefill + ~255 s decode; cached afterwards |
| prefill | 256-token bucket in 36.9 ms (~6.9k tok/s padded) |
| TTFT (61-token prompt) | **64 ms** |
| decode | **6.72 ms/token = 149 tok/s** at batch 1 |
| MoE decode variants | `dynamic_slice` **6.72 ms** · `jnp.take` gather 15.21 ms · dense-128 14.18 ms |

---

## Repo layout

```
setup.sh              one-time environment + weights + XLA cache warm-up
run.sh                streaming inference entry point
run_inference.py      CLI, rich dashboard, reasoning/answer channel split
bench.py              decode microbenchmark + ablation harness
gemma4_tpu/
  config.py           Gemma-4 text config loader
  model.py            the JAX model: attention, MoE, RoPE, masks, sampling, sharding
  weights.py          streaming safetensors -> sharded device arrays
  engine.py           jitted prefill/decode + async-dispatch streaming loop
  stream.py           incremental detokeniser + <|channel>thought splitter
  tpu_monitor.py      per-chip HBM / duty-cycle sampling
tests/test_parity.py  HF transformers numerical parity check
```

---

## Kaggle TPU field notes

Problems hit while building this, and what fixed them. Anything marked *unresolved*
is a live hazard for other agents doing AI work on Kaggle TPUs.

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
  silently invalidates the cache and forces a full recompile.
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
- **Reasoning tokens** are delimited by `<|channel>thought` ... `<channel|>` and only
  appear when the chat template is rendered with `enable_thinking=True`. The markers can
  straddle a token boundary, so the splitter buffers partial suffixes.
- *Unresolved:* transparent hugepages are disabled in the Kaggle image, which JAX warns
  slows TPU runtime start/stop; enabling it needs host root access.
- *Unresolved:* an earlier attempt in this repo used `vllm-tpu` 0.26.0, which injects
  `--xla_tpu_use_dynamic_smem_negotiation=true` — a flag the Kaggle `v5litepod` runtime
  rejects, killing the engine before any weights load. That is why this implementation
  talks to JAX directly.
