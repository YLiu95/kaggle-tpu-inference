"""Terminal rendering for the streaming dashboard.

Driven purely by a stream of JSON-able events, so the same code renders a local
in-process run and a run served by the persistent daemon.
"""

from __future__ import annotations

import sys
import time
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

V5E_BF16_TFLOPS = 197.0  # per chip, dense bf16
V5E_HBM_GBPS = 819.0  # per chip


def fmt_gb(x: float) -> str:
    return f"{x / 2**30:6.2f}"


def header_panel(info: dict) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan", justify="right")
    t.add_column(style="white")
    if info.get("num_experts"):
        arch = f"bf16, MoE {info['num_experts']}E top-{info['top_k_experts']}"
    else:
        arch = "bf16, dense MLP"
    t.add_row("model", f"{info['model']}  [dim]({arch})[/]")
    t.add_row(
        "params",
        f"{info['param_bytes'] / 2 / 1e9:.1f}B total  |  "
        f"{info['active_params'] / 1e9:.2f}B active/token  |  "
        f"{info['param_bytes'] / 2**30:.1f} GiB weights",
    )
    t.add_row(
        "layers",
        f"{info['num_layers']} ({info['sliding_layers']} sliding /{info['full_layers']} full), "
        f"hidden {info['hidden_size']}, vocab {info['vocab_size']}",
    )
    t.add_row(
        "sharding",
        f"1-D 'tp' mesh over {info['n_devices']} chips  |  "
        + (
            "attn heads + MLP + expert-intermediate split"
            if info.get("num_experts")
            else "attn heads + MLP intermediate split"
        ),
    )
    ceiling = info.get("model_max_context") or info["max_len"]
    room = "" if ceiling <= info["max_len"] else f"  [dim](ceiling {ceiling:,})[/]"
    t.add_row(
        "context",
        f"{info['max_len']:,} tokens resident{room}  |  KV "
        f"{info['cache_bytes'] / 2**30:.2f} GiB "
        f"({info['kv_bytes_per_token'] / 1024:.0f} KiB/token)  |  "
        f"max-new-tokens cap {info.get('max_output_tokens', info['max_len']):,}",
    )
    if info.get("served_by"):
        t.add_row("session", f"[green]{info['served_by']}[/]")
    return Panel(t, title="[bold]Gemma-4 on TPU v5e-8[/]", border_style="cyan")


def print_warning(console: Console, ev: dict) -> None:
    console.print(f"[bold yellow]warning[/] ({ev.get('code')}): {ev.get('message')}")
    console.print(f"[dim]  how to fix: {ev.get('remedy')}[/]")


def tpu_panel(snap: dict, tokens_per_s: float, info: dict, pos: int) -> Panel:
    chips = snap.get("chips", [])
    t = Table(box=None, pad_edge=False)
    for name, just in (
        ("chip", "right"), ("kind", "left"), ("HBM used", "right"),
        ("HBM total", "right"), ("used %", "right"), ("peak", "right"), ("duty", "right"),
    ):
        t.add_column(name, justify=just)
    for c in chips:
        limit = c["hbm_limit"] or 1
        pct = 100.0 * c["hbm_used"] / limit
        colour = "green" if pct < 75 else ("yellow" if pct < 90 else "red")
        duty = f"{c['duty_cycle']:5.1f}%" if c.get("duty_cycle") is not None else "  n/a"
        t.add_row(
            str(c["index"]), c["kind"],
            f"{fmt_gb(c['hbm_used'])} GiB", f"{fmt_gb(c['hbm_limit'])} GiB",
            f"[{colour}]{pct:5.1f}[/]", f"{fmt_gb(c['hbm_peak'])} GiB", duty,
        )

    n = max(1, len(chips))
    used = sum(c["hbm_used"] for c in chips)
    limit = sum(c["hbm_limit"] for c in chips)
    flops = 2 * info["active_params"] * tokens_per_s
    bw = (2 * info["active_params"] + info["kv_bytes_per_token"] * pos) * tokens_per_s
    step_s = info.get("decode_step_s") or 0.0
    busy = min(100.0, 100.0 * tokens_per_s * step_s) if step_s else None

    foot = Table.grid(padding=(0, 2))
    foot.add_column(style="bold cyan", justify="right")
    foot.add_column()
    foot.add_row(
        "aggregate",
        f"HBM {fmt_gb(used)} / {fmt_gb(limit)} GiB over {n} chips   "
        f"[dim]libtpu duty {snap.get('duty_source', 'unavailable')}[/]",
    )
    if busy is not None:
        foot.add_row(
            "tensorcore",
            f"[bold green]{busy:5.1f}%[/] busy "
            f"[dim](measured: {1000 * step_s:.2f} ms device time per decode step)[/]",
        )
    foot.add_row(
        "achieved",
        f"{flops / 1e12:6.2f} TFLOP/s ({100 * flops / (V5E_BF16_TFLOPS * 1e12 * n):.2f}% of "
        f"bf16 peak)   {bw / 1e9:7.1f} GB/s "
        f"({100 * bw / (V5E_HBM_GBPS * 1e9 * n):.1f}% of HBM BW)",
    )
    foot.add_row(
        "host",
        f"RSS {snap.get('host_rss_gb', 0):.1f} GB   CPU {snap.get('host_cpu_pct', 0):.0f}%",
    )
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


def run_ui(events, console: Console, plain: bool = False, refresh: float = 10.0) -> dict:
    """Consume the unified event stream and draw it. Returns the final metrics."""
    info: dict = {}
    segments: list[tuple[str, str]] = []
    warnings: list[dict] = []
    deferred_warnings: list[dict] = []
    stamps: deque[float] = deque(maxlen=24)
    snap: dict = {"chips": []}
    metrics = {
        "ttft": 0.0, "prompt_tokens": 0, "prefill_tps": 0.0, "tokens": 0,
        "instant_tps": 0.0, "avg_tps": 0.0, "itl_ms": 0.0, "elapsed": 0.0,
        "reasoning_tokens": 0, "answer_tokens": 0, "phase": "starting",
        "pos": 0, "max_len": 0,
    }
    t_first = None
    live = None
    last_render = 0.0

    def render():
        return Group(speed_panel(metrics), tpu_panel(snap, metrics["instant_tps"], info,
                                                     metrics["pos"]), output_panel(segments))

    try:
        for ev in events:
            kind = ev["kind"]
            if kind == "info":
                info = ev
                metrics["max_len"] = info["max_len"]
                if not plain:
                    console.print(header_panel(info))
                continue
            if kind == "prompt":
                metrics["prompt_tokens"] = ev["tokens"]
                metrics["pos"] = ev["tokens"]
                console.print(f"[bold]prompt[/] ({ev['tokens']} tokens): {ev['text']}\n")
                continue
            if kind == "tpu":
                snap = ev
            elif kind == "warning":
                warnings.append(ev)
                if live is None:
                    print_warning(console, ev)
                else:
                    deferred_warnings.append(ev)
            elif kind in ("prefill", "token"):
                if not plain and live is None:
                    live = Live(render(), console=console, refresh_per_second=refresh)
                    live.start()
                now = ev["t"]
                if kind == "prefill":
                    metrics["ttft"] = ev["ttft"]
                    metrics["prefill_tps"] = ev["prompt_tokens"] / max(ev["ttft"], 1e-9)
                    metrics["phase"] = "decode"
                    t_first = now
                    if plain:
                        console.print(
                            f"[yellow]TTFT {ev['ttft'] * 1000:.0f} ms[/] "
                            f"({ev['prompt_tokens']} prompt tokens, "
                            f"{metrics['prefill_tps']:.0f} tok/s prefill)"
                        )
                else:
                    stamps.append(now)
                metrics["tokens"] += 1
                metrics["pos"] = metrics["prompt_tokens"] + metrics["tokens"]
                if t_first is not None:
                    metrics["elapsed"] = now - t_first
                    if metrics["tokens"] > 1:
                        metrics["avg_tps"] = (metrics["tokens"] - 1) / max(metrics["elapsed"], 1e-9)
                if len(stamps) > 1:
                    span = stamps[-1] - stamps[0]
                    metrics["instant_tps"] = (len(stamps) - 1) / max(span, 1e-9)
                    metrics["itl_ms"] = 1000.0 * span / (len(stamps) - 1)

                mode = ev.get("mode", "answer")
                metrics["reasoning_tokens" if mode == "reasoning" else "answer_tokens"] += 1
                metrics["phase"] = "reasoning" if mode == "reasoning" else "answering"
                text = ev.get("text", "")
                if text:
                    if plain:
                        sys.stdout.write(
                            f"\033[2;33m{text}\033[0m" if mode == "reasoning" else text
                        )
                        sys.stdout.flush()
                    else:
                        segments.append((mode, text))
            elif kind == "done":
                metrics["phase"] = "done"
            elif kind == "error":
                metrics["phase"] = "error"
                console.print(f"[red]server error:[/] {ev.get('message')}")

            if live is not None and time.perf_counter() - last_render > 1.0 / refresh:
                live.update(render())
                last_render = time.perf_counter()
    finally:
        if live is not None:
            live.update(render())
            live.stop()

    metrics["answer_text"] = "".join(c for m, c in segments if m == "answer").strip()
    metrics["warnings"] = warnings
    # Warnings raised while the dashboard owned the terminal could not be printed then;
    # the caller shows them once the screen is free.
    metrics["pending_warnings"] = deferred_warnings
    return metrics
