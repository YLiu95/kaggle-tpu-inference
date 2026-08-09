from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import psutil

from ktpu.constants import kaggle_mirror_dir, state_dir
from ktpu.hardware import TpuTelemetry, read_tpu_telemetry
from ktpu.util import ensure_private_directory


CSV_FIELDS = [
    "timestamp_utc",
    "phase",
    "elapsed_s",
    "server_pid",
    "cpu_percent_of_affinity",
    "cpu_cores_used",
    "ram_used_gib",
    "ram_percent",
    "tpu_chips",
    "tpu_hbm_used_gib",
    "tpu_hbm_total_gib",
    "tpu_hbm_percent",
    "tpu_duty_cycle_percent",
    "ttft_s",
    "output_tokens",
    "tokens_per_s",
    "note",
]


@dataclass
class RuntimeSnapshot:
    timestamp_utc: str
    phase: str
    elapsed_s: float
    server_pid: int
    cpu_percent_of_affinity: float | None
    cpu_cores_used: float | None
    ram_used_gib: float
    ram_percent: float
    tpu_chips: int
    tpu_hbm_used_gib: float | None
    tpu_hbm_total_gib: float | None
    tpu_hbm_percent: float | None
    tpu_duty_cycle_percent: float | None
    ttft_s: float | None
    output_tokens: int
    tokens_per_s: float | None
    note: str = ""


class CsvTelemetryLogger:
    def __init__(
        self,
        primary: Path | None = None,
        mirror: Path | bool | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        primary_dir = ensure_private_directory(state_dir() / "logs")
        self.primary = primary or primary_dir / f"inference-{timestamp}.csv"
        mirror_root = kaggle_mirror_dir()
        if mirror is False:
            self.mirror = None
        elif isinstance(mirror, Path):
            self.mirror = mirror
        elif mirror_root is not None:
            self.mirror = mirror_root / "logs" / self.primary.name
        else:
            self.mirror = None
        self._lock = threading.Lock()
        self._handles: list[object] = []
        self._writers: list[csv.DictWriter] = []
        for path in [self.primary, self.mirror]:
            if path is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", encoding="utf-8", newline="")
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            handle.flush()
            self._handles.append(handle)
            self._writers.append(writer)

    def write(self, snapshot: RuntimeSnapshot) -> None:
        row = asdict(snapshot)
        with self._lock:
            for writer, handle in zip(self._writers, self._handles):
                writer.writerow(row)
                handle.flush()  # type: ignore[attr-defined]

    def close(self) -> None:
        with self._lock:
            for handle in self._handles:
                try:
                    handle.close()  # type: ignore[attr-defined]
                except OSError:
                    pass
            self._handles.clear()
            self._writers.clear()

    def __enter__(self) -> "CsvTelemetryLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class InferenceStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.request_started: float | None = None
        self.first_token_at: float | None = None
        self.output_tokens = 0
        self.phase = "preflight"

    def start_request(self) -> None:
        with self._lock:
            self.request_started = time.monotonic()
            self.phase = "inference"

    def first_token(self) -> None:
        with self._lock:
            if self.first_token_at is None:
                self.first_token_at = time.monotonic()

    def update_tokens(self, tokens: int) -> None:
        with self._lock:
            self.output_tokens = max(0, int(tokens))

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def snapshot(self, now: float) -> tuple[str, float | None, int, float | None]:
        with self._lock:
            phase = self.phase
            started = self.request_started
            first = self.first_token_at
            tokens = self.output_tokens
        ttft = first - started if first is not None and started is not None else None
        speed = None
        if first is not None and tokens > 0:
            duration = max(now - first, 1e-9)
            speed = tokens / duration
        return phase, ttft, tokens, speed


class TelemetryMonitor:
    def __init__(
        self,
        *,
        server_pid: int,
        affinity_cpu_count: int,
        logger: CsvTelemetryLogger,
        stats: InferenceStats,
        interval: float = 1.0,
        telemetry_reader: Callable[[], TpuTelemetry] = read_tpu_telemetry,
    ) -> None:
        self.server_pid = server_pid
        self.affinity_cpu_count = max(1, affinity_cpu_count)
        self.logger = logger
        self.stats = stats
        self.interval = max(0.1, interval)
        self.telemetry_reader = telemetry_reader
        self.started = time.monotonic()
        self.latest: RuntimeSnapshot | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_processes: dict[int, psutil.Process] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="ktpu-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self, note: str = "") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 3))
        if note:
            snapshot = self.sample(note=note)
            self.latest = snapshot
            self.logger.write(snapshot)

    def _processes(self) -> list[psutil.Process]:
        try:
            root = psutil.Process(self.server_pid)
            values = [root, *root.children(recursive=True)]
        except psutil.Error:
            values = []
        result = []
        for process in values:
            try:
                if process.pid not in self._cpu_processes:
                    process.cpu_percent(None)
                    self._cpu_processes[process.pid] = process
                result.append(self._cpu_processes[process.pid])
            except psutil.Error:
                continue
        return result

    def sample(self, *, note: str = "") -> RuntimeSnapshot:
        now = time.monotonic()
        cpu_percent = 0.0
        seen_cpu = False
        for process in self._processes():
            try:
                cpu_percent += process.cpu_percent(None)
                seen_cpu = True
            except psutil.Error:
                continue
        cpu_cores = cpu_percent / 100 if seen_cpu else None
        normalized_cpu = (
            cpu_percent / self.affinity_cpu_count if seen_cpu else None
        )
        memory = psutil.virtual_memory()
        try:
            telemetry = self.telemetry_reader()
        except Exception as exc:
            telemetry = TpuTelemetry(source=f"fallback:{type(exc).__name__}")
        used = telemetry.hbm_used_bytes
        total = telemetry.hbm_total_bytes
        hbm_percent = (
            (100.0 * used / total) if used is not None and total else None
        )
        phase, ttft, tokens, speed = self.stats.snapshot(now)
        return RuntimeSnapshot(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            elapsed_s=round(now - self.started, 3),
            server_pid=self.server_pid,
            cpu_percent_of_affinity=(
                round(normalized_cpu, 3) if normalized_cpu is not None else None
            ),
            cpu_cores_used=round(cpu_cores, 3) if cpu_cores is not None else None,
            ram_used_gib=round(memory.used / 1024**3, 3),
            ram_percent=round(float(memory.percent), 3),
            tpu_chips=telemetry.chip_count,
            tpu_hbm_used_gib=round(used / 1024**3, 3) if used is not None else None,
            tpu_hbm_total_gib=round(total / 1024**3, 3) if total is not None else None,
            tpu_hbm_percent=round(hbm_percent, 3) if hbm_percent is not None else None,
            tpu_duty_cycle_percent=(
                round(telemetry.duty_cycle_percent, 3)
                if telemetry.duty_cycle_percent is not None
                else None
            ),
            ttft_s=round(ttft, 3) if ttft is not None else None,
            output_tokens=tokens,
            tokens_per_s=round(speed, 3) if speed is not None else None,
            note=note,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            snapshot = self.sample()
            self.latest = snapshot
            self.logger.write(snapshot)
            self._stop.wait(self.interval)
