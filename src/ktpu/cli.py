from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.table import Table
from rich.text import Text

from ktpu.checkpoint import verify_checkpoint
from ktpu.constants import (
    DEFAULT_CPU_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_MONITOR_INTERVAL,
    DEFAULT_OUTPUT_CAP,
    DEFAULT_STARTUP_TIMEOUT,
    GIB,
    TESTED_VLLM_TPU_VERSION,
    cache_dir,
)
from ktpu.engine import load_engine_installation, require_engine, setup_vllm
from ktpu.errors import KtpuError, SafetyError
from ktpu.hardware import detect_hardware, read_tpu_telemetry
from ktpu.model import ModelProfile, load_model_profile
from ktpu.prompting import PromptInfo, load_and_tokenize_prompt
from ktpu.safety import SafetyReport, evaluate_preflight
from ktpu.server import launch_server
from ktpu.sizing import SizingResult, calculate_limits
from ktpu.state import clear_tpu_state, load_server_state, state_process
from ktpu.streaming import StreamChunk, stream_chat
from ktpu.telemetry import (
    CsvTelemetryLogger,
    InferenceStats,
    RuntimeSnapshot,
    TelemetryMonitor,
)
from ktpu.util import find_free_loopback_port, human_bytes, tail

app = typer.Typer(
    name="ktpu",
    help="Crash-safe, hardware-aware Gemma inference on Kaggle TPUs.",
    no_args_is_help=True,
)
console = Console(stderr=True)


def _fail(exc: Exception) -> None:
    console.print(f"[bold red]Error:[/bold red] {exc}")
    raise typer.Exit(1)


def _hardware_table(hardware: object) -> Table:
    table = Table(title="Detected hardware", show_header=False)
    table.add_column("Item", style="cyan")
    table.add_column("Value")
    table.add_row("TPU", f"{hardware.tpu_type} × {hardware.tpu_chip_count}")
    table.add_row("HBM per chip", human_bytes(hardware.hbm_per_chip_bytes))
    table.add_row("Total HBM", human_bytes(hardware.hbm_total_bytes))
    table.add_row(
        "CPU",
        f"{hardware.cpu_count} logical ({hardware.allowed_cpu_count} allowed)",
    )
    table.add_row(
        "RAM",
        f"{human_bytes(hardware.ram_available_bytes)} available / "
        f"{human_bytes(hardware.ram_total_bytes)}",
    )
    table.add_row("1m load", f"{hardware.load_1m:.2f}")
    table.add_row(
        "Disk",
        f"{human_bytes(hardware.disk_free_bytes)} free / "
        f"{human_bytes(hardware.disk_total_bytes)}",
    )
    telemetry = hardware.telemetry
    table.add_row(
        "TPU telemetry",
        (
            f"HBM {human_bytes(telemetry.hbm_used_bytes)} used, "
            f"duty {telemetry.duty_cycle_percent:.1f}%"
            if telemetry.duty_cycle_percent is not None
            else telemetry.source
        ),
    )
    return table


def _sizing_table(profile: ModelProfile, result: SizingResult) -> Table:
    table = Table(title="Safe request sizing", show_header=False)
    table.add_column("Input", style="cyan")
    table.add_column("Value", overflow="fold")
    table.add_row("Model revision", profile.revision or "default")
    table.add_row("Model weights", human_bytes(result.weights_bytes))
    table.add_row("Model dtype", profile.dtype_name)
    table.add_row("Model limit", f"{result.model_limit:,} tokens")
    table.add_row("TPU HBM", human_bytes(result.hbm_total_bytes))
    table.add_row("Current HBM use", human_bytes(result.hbm_in_use_bytes))
    table.add_row("Weight overhead", human_bytes(result.weight_overhead_bytes))
    table.add_row("Runtime headroom", human_bytes(result.runtime_headroom_bytes))
    table.add_row("KV budget", human_bytes(result.kv_budget_bytes))
    table.add_row(
        "KV per token",
        f"{human_bytes(result.kv_bytes_per_token)} raw / "
        f"{human_bytes(result.effective_kv_bytes_per_token)} conservative",
    )
    table.add_row(
        "Safe context", f"{result.calculated_safe_context:,} tokens"
    )
    table.add_row(
        "Applied context",
        f"{result.applied_context:,} tokens"
        + (
            f" (user cap {result.user_context_cap:,})"
            if result.user_context_cap is not None
            else ""
        ),
    )
    table.add_row("Rendered input", f"{result.input_tokens:,} tokens")
    table.add_row("Safe output", f"{result.safe_output_tokens:,} tokens")
    table.add_row(
        "Applied output",
        f"{result.applied_output_tokens:,} tokens"
        + (
            f" (user cap {result.user_output_cap:,})"
            if result.user_output_cap is not None
            else ""
        ),
    )
    return table


def _report_warnings(report: SafetyReport) -> None:
    for warning in report.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _format_snapshot(snapshot: RuntimeSnapshot | None) -> Text:
    if snapshot is None:
        return Text("Collecting telemetry…", style="dim")
    hbm = (
        f"{snapshot.tpu_hbm_used_gib:.1f}/{snapshot.tpu_hbm_total_gib:.1f} GiB"
        if snapshot.tpu_hbm_used_gib is not None
        and snapshot.tpu_hbm_total_gib is not None
        else "n/a"
    )
    duty = (
        f"{snapshot.tpu_duty_cycle_percent:.1f}%"
        if snapshot.tpu_duty_cycle_percent is not None
        else "n/a"
    )
    cpu = (
        f"{snapshot.cpu_percent_of_affinity:.1f}%"
        if snapshot.cpu_percent_of_affinity is not None
        else "n/a"
    )
    ttft = f"{snapshot.ttft_s:.2f}s" if snapshot.ttft_s is not None else "pending"
    speed = (
        f"{snapshot.tokens_per_s:.2f} tok/s"
        if snapshot.tokens_per_s is not None
        else "pending"
    )
    return Text(
        f"{snapshot.phase} | TPU {snapshot.tpu_chips} chips, HBM {hbm}, "
        f"duty {duty} | CPU {cpu} of affinity | RAM {snapshot.ram_percent:.1f}% | "
        f"TTFT {ttft} | {snapshot.output_tokens} tokens, {speed}"
    )


class DynamicTelemetry:
    def __init__(self, monitor: TelemetryMonitor) -> None:
        self.monitor = monitor

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        yield _format_snapshot(self.monitor.latest)


def _confirm_runtime_tpus(expected: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = read_tpu_telemetry()
        if (
            last.source == "libtpu"
            and len(last.per_chip_hbm_used_bytes) == expected
        ):
            return
        time.sleep(1.0)
    count = len(last.per_chip_hbm_used_bytes) if last is not None else 0
    raise SafetyError(
        f"Could not confirm runtime telemetry for all {expected} TPUs "
        f"(reported {count})."
    )


def _read_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt is not None and prompt_file is not None:
        raise KtpuError("Use either --prompt or --prompt-file, not both.")
    if prompt_file is not None:
        try:
            return prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise KtpuError(f"Could not read prompt file: {exc}") from exc
    if prompt is not None:
        return prompt
    if not sys.stdin.isatty():
        value = sys.stdin.read()
        if value:
            return value
    return typer.prompt("Prompt")


@app.command()
def setup(
    engine: Annotated[
        str, typer.Option("--engine", help="Inference engine (only vllm is supported).")
    ] = "vllm",
    version: Annotated[
        str, typer.Option("--version", help="Pinned binary vllm-tpu version.")
    ] = TESTED_VLLM_TPU_VERSION,
    cpu_limit: Annotated[
        int, typer.Option("--cpu-limit", min=1, help="Maximum CPUs for setup.")
    ] = DEFAULT_CPU_LIMIT,
) -> None:
    """Install a binary-only vLLM TPU engine after checkpoint validation."""
    try:
        if engine != "vllm":
            raise KtpuError("Only --engine vllm is supported.")
        installation = setup_vllm(version=version, cpu_limit=cpu_limit)
        console.print(
            f"[green]Installed[/green] vllm-tpu {installation.version} at "
            f"{installation.root}"
        )
        for name, installed in sorted(installation.installed_versions.items()):
            console.print(f"  {name}: {installed}")
    except KtpuError as exc:
        _fail(exc)


@app.command()
def status() -> None:
    """Show hardware, engine, managed server, and telemetry status."""
    try:
        hardware = detect_hardware(cache_dir())
        console.print(_hardware_table(hardware))
        installation = load_engine_installation()
        if installation is None:
            console.print("[yellow]Engine:[/yellow] not installed")
        else:
            console.print(
                f"[green]Engine:[/green] vllm-tpu {installation.version} "
                f"({installation.root})"
            )
        state = load_server_state()
        process = state_process(state) if state is not None else None
        if state is None:
            console.print("Managed server: stopped")
        elif process is None:
            console.print(
                f"[yellow]Managed server:[/yellow] stale state for PID {state.pid}"
            )
        else:
            healthy = False
            try:
                healthy = (
                    httpx.get(f"{state.base_url}/health", timeout=1.0).status_code
                    == 200
                )
            except httpx.HTTPError:
                pass
            console.print(
                f"[green]Managed server:[/green] PID {state.pid}, "
                f"{'healthy' if healthy else 'starting/unhealthy'}, model {state.model}"
            )
            console.print(f"Server log: {state.log_path}")
            if not healthy:
                recent = tail(state.log_path, 10)
                if recent:
                    console.print(recent)
    except KtpuError as exc:
        _fail(exc)


@app.command("clear-tpu")
def clear_tpu(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Also terminate non-managed TPU owners only when they look like vLLM.",
        ),
    ] = False,
) -> None:
    """Stop managed vLLM processes and clear stale server state."""
    killed, owners = clear_tpu_state(force=force)
    if killed:
        console.print(f"Stopped PIDs: {', '.join(str(pid) for pid in killed)}")
    if owners:
        console.print("[yellow]TPU devices remain owned:[/yellow]")
        for path, pid in sorted(owners.items()):
            console.print(f"  {path}: PID {pid}")
        if not force:
            console.print("Re-run with --force to clear vLLM-like owners.")
        raise typer.Exit(2)
    console.print("[green]TPU state is clear.[/green]")


@app.command()
def run(
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="User prompt.")
    ] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option("--prompt-file", exists=True, dir_okay=False, readable=True),
    ] = None,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    system: Annotated[str | None, typer.Option("--system")] = None,
    context: Annotated[
        int | None,
        typer.Option("--context", min=1, help="Optional context cap in tokens."),
    ] = None,
    max_output: Annotated[
        int | None,
        typer.Option(
            "--max-output",
            min=1,
            help="Optional output cap; defaults to a conservative 4096.",
        ),
    ] = DEFAULT_OUTPUT_CAP,
    temperature: Annotated[
        float, typer.Option("--temperature", min=0.0, max=2.0)
    ] = 0.2,
    thinking: Annotated[
        bool, typer.Option("--thinking/--no-thinking")
    ] = True,
    cpu_limit: Annotated[
        int, typer.Option("--cpu-limit", min=1)
    ] = DEFAULT_CPU_LIMIT,
    startup_timeout: Annotated[
        int, typer.Option("--startup-timeout", min=60)
    ] = DEFAULT_STARTUP_TIMEOUT,
    monitor_interval: Annotated[
        float, typer.Option("--monitor-interval", min=0.1)
    ] = DEFAULT_MONITOR_INTERVAL,
    local_files_only: Annotated[
        bool,
        typer.Option(
            "--local-files-only",
            help="Use only an existing Hugging Face cache/local model.",
        ),
    ] = False,
) -> None:
    """Size, launch, monitor, and stream one Gemma chat request."""
    server = None
    monitor = None
    csv_logger = None
    try:
        user_prompt = _read_prompt(prompt, prompt_file)
        if not user_prompt.strip():
            raise KtpuError("Prompt is empty.")
        installation = require_engine()
        checkpoint = verify_checkpoint(run_tests=True)

        profile = load_model_profile(
            model, local_files_only=local_files_only
        )
        tokenizer, prompt_info = load_and_tokenize_prompt(
            model,
            user_prompt,
            system=system,
            revision=profile.revision,
            local_files_only=local_files_only,
            enable_thinking=thinking,
        )
        hardware = detect_hardware(cache_dir())
        required_ram = max(32 * GIB, int(profile.weights_bytes * 1.20))
        required_disk = max(10 * GIB, int(profile.weights_bytes * 1.10))
        report = evaluate_preflight(
            hardware,
            operation="model startup and inference",
            required_ram_bytes=required_ram,
            required_disk_bytes=required_disk,
            require_tpu=True,
        )
        report.require_safe()
        _report_warnings(report)
        sizing = calculate_limits(
            profile,
            hbm_total_bytes=hardware.hbm_total_bytes,
            hbm_in_use_bytes=hardware.telemetry.hbm_used_bytes or 0,
            input_tokens=prompt_info.input_tokens,
            context_cap=context,
            output_cap=max_output,
        )
        console.print(_hardware_table(hardware))
        console.print(_sizing_table(profile, sizing))
        console.print(
            f"Checkpoint: [green]{checkpoint.commit}[/green] (matches origin/main)"
        )

        port = find_free_loopback_port()
        server, cpus = launch_server(
            installation,
            checkpoint,
            model=model,
            port=port,
            tensor_parallel_size=hardware.tpu_chip_count,
            max_model_len=sizing.applied_context,
            enable_thinking=thinking,
            cpu_limit=min(cpu_limit, hardware.allowed_cpu_count),
        )
        csv_logger = CsvTelemetryLogger()
        stats = InferenceStats()
        stats.set_phase("server_startup")
        monitor = TelemetryMonitor(
            server_pid=server.state.pid,
            affinity_cpu_count=len(cpus),
            logger=csv_logger,
            stats=stats,
            interval=monitor_interval,
        )
        monitor.start()
        dynamic = DynamicTelemetry(monitor)
        with Live(dynamic, console=console, refresh_per_second=4, transient=False):
            server.wait_ready(startup_timeout)
            stats.set_phase("runtime_validation")
            _confirm_runtime_tpus(hardware.tpu_chip_count)
            stats.start_request()
            current_channel: str | None = None

            def display_chunk(chunk: StreamChunk) -> None:
                nonlocal current_channel
                if chunk.channel != current_channel:
                    current_channel = chunk.channel
                    heading = (
                        "\n=== Reasoning ===\n"
                        if chunk.channel == "reasoning"
                        else "\n=== Response ===\n"
                    )
                    sys.stdout.write(heading)
                sys.stdout.write(chunk.text)
                sys.stdout.flush()

            def count_tokens(text: str) -> int:
                return len(tokenizer.encode(text, add_special_tokens=False))

            result = stream_chat(
                base_url=server.state.base_url,
                model=model,
                messages=prompt_info.messages,
                max_tokens=sizing.applied_output_tokens,
                temperature=temperature,
                enable_thinking=thinking,
                on_chunk=display_chunk,
                on_first_token=lambda _value: stats.first_token(),
                on_token_count=stats.update_tokens,
                count_tokens=count_tokens,
            )
            stats.update_tokens(result.completion_tokens)
            stats.set_phase("complete")
        sys.stdout.write("\n")
        sys.stdout.flush()
        monitor.stop(
            note=(
                f"complete; ttft={result.ttft_s}; "
                f"tokens={result.completion_tokens}; tps={result.tokens_per_second}"
            )
        )
        monitor = None
        console.print(f"CSV log: {csv_logger.primary}")
        if csv_logger.mirror is not None:
            console.print(f"Kaggle mirror: {csv_logger.mirror}")
    except (KtpuError, OSError) as exc:
        _fail(exc)
    finally:
        if monitor is not None:
            monitor.stop(note="aborted")
        if csv_logger is not None:
            csv_logger.close()
        if server is not None:
            server.stop()

