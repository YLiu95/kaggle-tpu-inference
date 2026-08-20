#!/usr/bin/env python3
"""Streaming Gemma-4 26B-A4B inference on a Kaggle TPU v5e-8 with live metrics.

Uses the persistent daemon (``serve.sh``) when it is running, so repeat prompts start
instantly; otherwise falls back to loading and compiling in-process.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console  # noqa: E402

from gemma4_tpu.session import (  # noqa: E402
    SOCKET_PATH,
    daemon_alive,
    remote_events,
    send_command,
)
from gemma4_tpu.limits import DEFAULT_CONTEXT_TOKENS  # noqa: E402
from gemma4_tpu.ui import run_ui  # noqa: E402

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"
console = Console()


def local_events(args, request):
    """Load, shard and compile in this process, then stream (slow path)."""
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import jax

    cache_dir = os.environ.get("GEMMA4_XLA_CACHE", "/root/.cache/gemma4_jax")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    from transformers import AutoTokenizer

    from gemma4_tpu.engine import Engine
    from gemma4_tpu.session import generate_events, model_info, resolve_model_dir
    from gemma4_tpu.tpu_monitor import TpuMonitor

    model_dir = resolve_model_dir(args.model, args.model_dir)
    console.print(f"[dim]weights:[/] {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)

    monitor = TpuMonitor()
    monitor.start()
    with console.status("[bold]loading + sharding weights across 8 TPU chips...") as status:
        def progress(done, total, name):
            if done % 25 == 0 or done == total:
                status.update(f"[bold]sharding weights {done}/{total}[/]  [dim]{name}[/]")

        engine = Engine(model_dir, max_len=args.max_len, top_k=args.top_k, progress=progress)
    console.print(
        f"[green]loaded[/] {engine.param_bytes / 2**30:.1f} GiB in {engine.load_seconds:.1f}s"
    )
    with console.status("[bold]compiling XLA programs (prefill + decode)..."):
        ct = engine.compile_all()
    console.print(
        f"[green]compiled[/] prefill {ct['prefill_compile_s']:.1f}s, "
        f"decode {ct['decode_compile_s']:.1f}s, "
        f"steady-state decode step {1000 * ct['decode_step_s']:.2f} ms"
    )
    console.print(
        "[dim]tip: run `bash serve.sh start` once to keep this state resident and make "
        "later prompts start instantly.[/]"
    )
    info = model_info(engine, args.model, served_by=None)
    try:
        yield from generate_events(engine, tok, monitor, request, info)
    finally:
        monitor.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default="Explain why TPUs use a systolic array, then give one concrete example of an operation it accelerates.")
    ap.add_argument("--system", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-len", type=int, default=DEFAULT_CONTEXT_TOKENS)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", dest="think", action="store_true", default=True)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--plain", action="store_true", help="plain stdout streaming, no dashboard")
    ap.add_argument("--refresh", type=float, default=10.0)
    ap.add_argument("--local", action="store_true", help="ignore the daemon, run in-process")
    ap.add_argument("--socket", default=SOCKET_PATH)
    args = ap.parse_args()

    request = {
        "prompt": args.prompt,
        "system": args.system,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "think": args.think,
    }

    console.rule("[bold cyan]Gemma-4 26B-A4B / TPU v5e-8 / JAX SPMD")
    if args.local and daemon_alive(args.socket):
        console.print(
            "[red]--local cannot claim the TPU while the daemon holds it.[/] "
            "Run [bold]bash serve.sh stop[/] first."
        )
        return 1
    if not args.local and daemon_alive(args.socket):
        st = send_command("status", args.socket) or {}
        console.print(
            f"[green]using persistent TPU daemon[/] "
            f"[dim](pid {st.get('pid')}, up {st.get('uptime_s', 0):.0f}s, "
            f"{st.get('requests', 0)} prior requests)[/]"
        )
        events = remote_events(request, args.socket)
    else:
        events = local_events(args, request)

    metrics = run_ui(events, console, plain=args.plain, refresh=args.refresh)

    if args.plain:
        print(
            f"\n\n[{metrics['tokens']} tokens, {metrics['avg_tps']:.2f} tok/s, "
            f"TTFT {metrics['ttft'] * 1000:.0f} ms]"
        )
    else:
        console.rule("[bold green]final answer")
        console.print(
            metrics["answer_text"]
            or "[yellow](token budget consumed by the reasoning channel; "
               "raise --max-new-tokens)[/]"
        )
        console.rule()
    console.print(
        f"[bold]TTFT[/] {metrics['ttft'] * 1000:.0f} ms   "
        f"[bold]tokens[/] {metrics['tokens']}   "
        f"[bold]avg[/] {metrics['avg_tps']:.2f} tok/s   "
        f"[bold]reasoning[/] {metrics['reasoning_tokens']}   "
        f"[bold]answer[/] {metrics['answer_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
