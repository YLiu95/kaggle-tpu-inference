#!/usr/bin/env python3
"""Streaming Gemma-4 26B-A4B inference on a Kaggle TPU v5e-8 with live metrics."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ.setdefault("HF_HOME", "/root/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jax

# Persist XLA binaries: the 30-layer unrolled graphs take minutes to compile once.
CACHE_DIR = os.environ.get("GEMMA4_XLA_CACHE", "/root/.cache/gemma4_jax")
os.makedirs(CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemma4_tpu.engine import Engine  # noqa: E402
from gemma4_tpu.stream import ChannelSplitter, IncrementalDetokenizer  # noqa: E402
from gemma4_tpu.tpu_monitor import TpuMonitor, peak_bandwidth, peak_flops  # noqa: E402

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"
console = Console()


# ---------------------------------------------------------------------------- helpers
def resolve_model_dir(model_id: str, local_dir: str | None) -> str:
    if local_dir:
        return local_dir
    env = os.environ.get("GEMMA4_MODEL_DIR")
    if env and os.path.isdir(env):
        return env
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        max_workers=16,
    )


def kv_bytes_per_token(cfg) -> int:
    return sum(2 * 2 * cfg.kv_heads_for(lt) * cfg.head_dim_for(lt) for lt in cfg.layer_types)


def fmt_gb(x: float) -> str:
    return f"{x / 2**30:6.2f}"


def header_panel(engine: Engine, model_id: str) -> Panel:
    cfg = engine.cfg
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan", justify="right")
    t.add_column(style="white")
    t.add_row("model", f"{model_id}  [dim](bf16, MoE {cfg.num_experts}E top-{cfg.top_k_experts})[/]")
    t.add_row(
        "params",
        f"{engine.param_bytes / 2 / 1e9:.1f}B total  |  "
        f"{engine.active_params / 1e9:.2f}B active/token  |  "
        f"{engine.param_bytes / 2**30:.1f} GiB weights",
    )
    t.add_row(
        "layers",
        f"{cfg.num_hidden_layers} "
        f"({sum(1 for x in cfg.layer_types if x == 'sliding_attention')} sliding /"
        f"{sum(1 for x in cfg.layer_types if x == 'full_attention')} full), "
        f"hidden {cfg.hidden_size}, vocab {cfg.vocab_size}",
    )
    t.add_row(
        "sharding",
        f"1-D 'tp' mesh over {engine.n_devices} chips  |  "
        f"attn heads + MLP + expert-intermediate split",
    )
    t.add_row(
        "kv cache",
        f"{engine.cache_bytes / 2**30:.2f} GiB @ max_len={engine.max_len} "
        f"({kv_bytes_per_token(cfg) / 1024:.0f} KiB/token)",
    )
    return Panel(t, title="[bold]Gemma-4 on TPU v5e-8[/]", border_style="cyan")


def tpu_panel(snap, tokens_per_s: float, engine: Engine, pos: int) -> Panel:
    t = Table(box=None, pad_edge=False, expand=True)
    t.add_column("chip", justify="right", style="bold")
    t.add_column("kind")
    t.add_column("HBM used", justify="right")
    t.add_column("HBM total", justify="right")
    t.add_column("%", justify="right")
    t.add_column("peak", justify="right")
    t.add_column("duty", justify="right")
    for c in snap.chips:
        pct = 100.0 * c.hbm_used / c.hbm_limit if c.hbm_limit else 0.0
        colour = "green" if pct < 75 else ("yellow" if pct < 90 else "red")
        duty = f"{c.duty_cycle:5.1f}%" if c.duty_cycle is not None else "  n/a"
        t.add_row(
            str(c.index),
            c.kind,
            f"{fmt_gb(c.hbm_used)} GiB",
            f"{fmt_gb(c.hbm_limit)} GiB",
            f"[{colour}]{pct:5.1f}[/]",
            f"{fmt_gb(c.hbm_peak)} GiB",
            duty,
        )
    n = max(1, len(snap.chips))
    flops = 2 * engine.active_params * tokens_per_s
    bw = (2 * engine.active_params + kv_bytes_per_token(engine.cfg) * pos) * tokens_per_s
    foot = Table.grid(padding=(0, 2))
    foot.add_column(style="bold cyan", justify="right")
    foot.add_column()
    foot.add_row(
        "aggregate",
        f"HBM {fmt_gb(snap.hbm_used)} / {fmt_gb(snap.hbm_limit)} GiB   "
        f"duty {('%.1f%%' % snap.mean_duty) if snap.mean_duty is not None else 'n/a'}   "
        f"[dim]src={snap.duty_source}[/]",
    )
    foot.add_row(
        "achieved",
        f"{flops / 1e12:6.2f} TFLOP/s ({100 * flops / peak_flops(n):.2f}% of peak)   "
        f"{bw / 1e9:7.1f} GB/s ({100 * bw / peak_bandwidth(n):.1f}% of HBM BW)",
    )
    foot.add_row("host", f"RSS {snap.host_rss_gb:.1f} GB   CPU {snap.host_cpu_pct:.0f}%")
    return Panel(Group(t, foot), title="[bold]TPU utilisation[/]", border_style="magenta")


def speed_panel(m: dict) -> Panel:
    t = Table.grid(padding=(0, 3))
    for _ in range(4):
        t.add_column(style="bold cyan", justify="right")
        t.add_column(justify="left")
    t.add_row(
        "TTFT", f"[bold yellow]{m['ttft'] * 1000:8.1f} ms[/]",
        "prompt", f"{m['prompt_tokens']} tok",
        "prefill", f"{m['prefill_tps']:.0f} tok/s",
        "gen", f"{m['tokens']} tok",
    )
    t.add_row(
        "now", f"[bold green]{m['instant_tps']:6.2f} tok/s[/]",
        "avg", f"[bold]{m['avg_tps']:6.2f} tok/s[/]",
        "ITL", f"{m['itl_ms']:6.1f} ms",
        "elapsed", f"{m['elapsed']:6.2f} s",
    )
    t.add_row(
        "reasoning", f"{m['reasoning_tokens']} tok",
        "answer", f"{m['answer_tokens']} tok",
        "phase", f"[bold]{m['phase']}[/]",
        "ctx", f"{m['pos']}/{m['max_len']}",
    )
    return Panel(t, title="[bold]Throughput[/]", border_style="green")


def output_panel(segments: list[tuple[str, str]], max_lines: int = 22) -> Panel:
    text = Text()
    for mode, chunk in segments:
        text.append(chunk, style="italic dim yellow" if mode == "reasoning" else "white")
    lines = text.split("\n")
    if len(lines) > max_lines:
        text = Text("\n").join(lines[-max_lines:])
    return Panel(text, title="[bold]stream (reasoning dimmed)[/]", border_style="blue")


# ---------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="Explain why TPUs use a systolic array, then give one concrete example of an operation it accelerates.")
    ap.add_argument("--system", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", dest="think", action="store_true", default=True)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--plain", action="store_true", help="plain stdout streaming, no dashboard")
    ap.add_argument("--refresh", type=float, default=10.0)
    args = ap.parse_args()

    console.rule("[bold cyan]Gemma-4 26B-A4B / TPU v5e-8 / JAX SPMD")
    model_dir = resolve_model_dir(args.model, args.model_dir)
    console.print(f"[dim]weights:[/] {model_dir}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)

    monitor = TpuMonitor()
    monitor.start()

    with console.status("[bold]loading + sharding weights across 8 TPU chips...") as status:
        def progress(done, total, name):
            if done % 25 == 0 or done == total:
                status.update(f"[bold]sharding weights {done}/{total}[/]  [dim]{name}[/]")

        engine = Engine(model_dir, max_len=args.max_len, top_k=args.top_k, progress=progress)
    console.print(
        f"[green]loaded[/] {engine.param_bytes / 2**30:.1f} GiB in {engine.load_seconds:.1f}s "
        f"({engine.param_bytes / 2**30 / engine.load_seconds:.2f} GiB/s)"
    )

    with console.status("[bold]compiling XLA programs (prefill + decode)..."):
        ct = engine.compile_all()
    console.print(
        f"[green]compiled[/] prefill {ct['prefill_compile_s']:.1f}s, "
        f"decode {ct['decode_compile_s']:.1f}s"
    )

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    encoded = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=args.think,
        tokenize=True,
        return_dict=True,
    )
    prompt_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    while isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_ids = [int(x) for x in prompt_ids]

    console.print(header_panel(engine, args.model))
    console.print(f"[bold]prompt[/] ({len(prompt_ids)} tokens): {args.prompt}\n")

    detok = IncrementalDetokenizer(tok)
    splitter = ChannelSplitter()
    segments: list[tuple[str, str]] = []
    stamps: deque[float] = deque(maxlen=24)
    counts = {"reasoning": 0, "answer": 0}

    metrics = {
        "ttft": 0.0, "prompt_tokens": len(prompt_ids), "prefill_tps": 0.0,
        "tokens": 0, "instant_tps": 0.0, "avg_tps": 0.0, "itl_ms": 0.0,
        "elapsed": 0.0, "reasoning_tokens": 0, "answer_tokens": 0,
        "phase": "prefill", "pos": len(prompt_ids), "max_len": engine.max_len,
    }
    t_first = None
    last_render = 0.0

    def render():
        snap = monitor.latest
        return Group(
            speed_panel(metrics),
            tpu_panel(snap, metrics["instant_tps"], engine, metrics["pos"]),
            output_panel(segments),
        )

    gen = engine.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    if args.plain:
        for ev in gen:
            if ev["kind"] == "prefill":
                metrics["ttft"] = ev["ttft"]
                console.print(
                    f"[yellow]TTFT {ev['ttft'] * 1000:.0f} ms[/] "
                    f"({ev['prompt_tokens']} prompt tokens, "
                    f"{ev['prompt_tokens'] / ev['ttft']:.0f} tok/s prefill)"
                )
                t_first = ev["t"]
            for mode, chunk in splitter.feed(detok.add(ev["token"])):
                sys.stdout.write(
                    f"\033[2;33m{chunk}\033[0m" if mode == "reasoning" else chunk
                )
                sys.stdout.flush()
            if ev["kind"] == "decode":
                stamps.append(ev["t"])
        elapsed = time.perf_counter() - (t_first or time.perf_counter())
        n = detok.ids.__len__()
        print(f"\n\n[{n} tokens, {(n - 1) / max(elapsed, 1e-9):.2f} tok/s]")
        monitor.stop()
        return 0

    with Live(render(), console=console, refresh_per_second=args.refresh, transient=False) as live:
        for ev in gen:
            now = ev["t"]
            if ev["kind"] == "prefill":
                metrics["ttft"] = ev["ttft"]
                metrics["prefill_tps"] = ev["prompt_tokens"] / max(ev["ttft"], 1e-9)
                metrics["phase"] = "decode"
                t_first = now
            else:
                stamps.append(now)
            metrics["tokens"] += 1
            metrics["pos"] = len(prompt_ids) + metrics["tokens"]
            if t_first is not None:
                metrics["elapsed"] = now - t_first
                if metrics["tokens"] > 1:
                    metrics["avg_tps"] = (metrics["tokens"] - 1) / max(metrics["elapsed"], 1e-9)
            if len(stamps) > 1:
                span = stamps[-1] - stamps[0]
                metrics["instant_tps"] = (len(stamps) - 1) / max(span, 1e-9)
                metrics["itl_ms"] = 1000.0 * span / (len(stamps) - 1)

            for mode, chunk in splitter.feed(detok.add(ev["token"])):
                segments.append((mode, chunk))
                counts[mode] += 0
            metrics["phase"] = "reasoning" if splitter.mode == "reasoning" else "answering"
            if splitter.mode == "reasoning":
                metrics["reasoning_tokens"] += 1
            else:
                metrics["answer_tokens"] += 1

            if now - last_render > 1.0 / args.refresh:
                live.update(render())
                last_render = now
        for mode, chunk in splitter.flush():
            segments.append((mode, chunk))
        metrics["phase"] = "done"
        live.update(render())

    monitor.stop()
    full = "".join(c for m, c in segments if m == "answer")
    console.rule("[bold green]final answer")
    console.print(full.strip())
    console.rule()
    console.print(
        f"[bold]TTFT[/] {metrics['ttft'] * 1000:.0f} ms   "
        f"[bold]tokens[/] {metrics['tokens']}   "
        f"[bold]avg[/] {metrics['avg_tps']:.2f} tok/s   "
        f"[bold]reasoning[/] {metrics['reasoning_tokens']} tok"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
