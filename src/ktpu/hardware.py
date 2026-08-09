from __future__ import annotations

import glob
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import psutil

from ktpu.constants import GIB


@dataclass(frozen=True)
class TpuSpec:
    name: str
    hbm_gib: int
    devices_per_chip: int = 1


TPU_BY_PCI_ID: dict[str, TpuSpec] = {
    "0x0027": TpuSpec("v2/v3", 8, 2),
    "0x005e": TpuSpec("v4", 32),
    "0x0063": TpuSpec("v5e", 16),
    "0x0062": TpuSpec("v5p", 95),
    "0x006f": TpuSpec("v6e", 32),
    "0x0076": TpuSpec("v7x", 192, 2),
}

TPU_BY_ENV_PREFIX: tuple[tuple[str, TpuSpec], ...] = (
    ("v5litepod", TpuSpec("v5e", 16)),
    ("v5e", TpuSpec("v5e", 16)),
    ("v5p", TpuSpec("v5p", 95)),
    ("v6e", TpuSpec("v6e", 32)),
    ("v7", TpuSpec("v7x", 192, 2)),
    ("v4", TpuSpec("v4", 32)),
    ("v3", TpuSpec("v3", 16, 2)),
    ("v2", TpuSpec("v2", 8, 2)),
)


@dataclass
class TpuTelemetry:
    chip_count: int = 0
    hbm_used_bytes: int | None = None
    hbm_total_bytes: int | None = None
    duty_cycle_percent: float | None = None
    per_chip_hbm_used_bytes: list[int] = field(default_factory=list)
    per_chip_duty_cycle_percent: list[float] = field(default_factory=list)
    source: str = "unavailable"


@dataclass
class HardwareInfo:
    tpu_type: str
    tpu_chip_count: int
    hbm_per_chip_bytes: int
    hbm_total_bytes: int
    cpu_count: int
    allowed_cpu_count: int
    ram_total_bytes: int
    ram_available_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    load_1m: float
    tpu_owners: dict[str, int]
    telemetry: TpuTelemetry

    @property
    def normalized_load(self) -> float:
        return self.load_1m / max(1, self.allowed_cpu_count)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def detect_tpu_from_sysfs(
    pci_root: Path = Path("/sys/bus/pci/devices"),
) -> tuple[TpuSpec | None, int]:
    matches: list[TpuSpec] = []
    try:
        device_paths = list(pci_root.iterdir())
    except OSError:
        return None, 0
    for device_path in device_paths:
        if _read(device_path / "vendor") != "0x1ae0":
            continue
        device_id = _read(device_path / "device")
        if not device_id:
            continue
        spec = TPU_BY_PCI_ID.get(device_id)
        if spec is not None:
            if device_id == "0x0027" and _read(device_path / "subsystem_device") == "0x004f":
                spec = TpuSpec("v3", 16, 2)
            matches.append(spec)
    if not matches:
        return None, 0
    names = {item.name for item in matches}
    if len(names) != 1:
        return TpuSpec("mixed", min(item.hbm_gib for item in matches)), len(matches)
    return matches[0], len(matches)


def _env_chip_count() -> int:
    bounds = os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS", "")
    try:
        values = [int(value) for value in bounds.split(",") if value]
    except ValueError:
        return 0
    product = 1
    for value in values:
        product *= value
    return product if values else 0


def detect_tpu_static() -> tuple[TpuSpec | None, int]:
    spec, count = detect_tpu_from_sysfs()
    if spec is not None:
        return spec, count
    accelerator = os.environ.get("TPU_ACCELERATOR_TYPE", "").lower()
    for prefix, candidate in TPU_BY_ENV_PREFIX:
        if accelerator.startswith(prefix):
            return candidate, _env_chip_count()
    return None, 0


def find_tpu_owners() -> dict[str, int]:
    owners: dict[str, int] = {}
    for fd_path in glob.glob("/proc/[0-9]*/fd/[0-9]*"):
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if not re.fullmatch(r"/dev/(?:accel\d+|vfio/\d+)", target):
            continue
        match = re.match(r"/proc/(\d+)/", fd_path)
        if match:
            owners[target] = int(match.group(1))
    return owners


def process_command(pid: int) -> str:
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except (psutil.Error, OSError):
        return ""


def conflicting_server_processes(
    processes: Iterable[psutil.Process] | None = None,
) -> dict[int, str]:
    conflicts: dict[int, str] = {}
    iterator = processes if processes is not None else psutil.process_iter(["pid", "cmdline"])
    for process in iterator:
        try:
            pid = int(process.pid)
            cmdline_value = process.info.get("cmdline") if hasattr(process, "info") else process.cmdline()
            command = " ".join(cmdline_value or [])
        except (psutil.Error, OSError, ValueError):
            continue
        lowered = command.lower()
        if "vllm serve" in lowered or (
            ("uvicorn" in lowered or "api_server" in lowered) and "vllm" in lowered
        ):
            conflicts[pid] = command
    return conflicts


def _metric_values(response: object) -> list[object]:
    try:
        values = list(response.metric.metrics)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return []
    return sorted(
        values,
        key=lambda item: getattr(getattr(getattr(item, "attribute", None), "value", None), "int_attr", 0),
    )


def read_tpu_telemetry(timeout: float = 0.75) -> TpuTelemetry:
    """Read libtpu metrics without creating a JAX client or initializing TPU."""
    spec, static_count = detect_tpu_static()
    try:
        import grpc
        from tpu_info.metrics import MetricName
        from tpu_info.proto import tpu_metric_service_pb2 as pb2
        from tpu_info.proto import tpu_metric_service_pb2_grpc as pb2_grpc

        channel = grpc.secure_channel("localhost:8431", grpc.local_channel_credentials())
        client = pb2_grpc.RuntimeMetricServiceStub(channel)

        def fetch(metric: object) -> list[object]:
            response = client.GetRuntimeMetric(
                pb2.MetricRequest(metric_name=metric.value), timeout=timeout
            )
            return _metric_values(response)

        totals = fetch(MetricName.TOTAL_MEMORY)
        usages = fetch(MetricName.MEMORY_USAGE)
        duties = fetch(MetricName.DUTY_CYCLE_PCT)
        total_values = [int(item.gauge.as_int) for item in totals]  # type: ignore[attr-defined]
        usage_values = [int(item.gauge.as_int) for item in usages]  # type: ignore[attr-defined]
        duty_values = [float(item.gauge.as_double) for item in duties]  # type: ignore[attr-defined]
        if duties and spec and len(duty_values) != len(usage_values):
            expanded: list[float] = []
            for value in duty_values:
                expanded.extend([value] * spec.devices_per_chip)
            duty_values = expanded
        count = static_count or len(usage_values)
        return TpuTelemetry(
            chip_count=count,
            hbm_used_bytes=sum(usage_values) if usage_values else None,
            hbm_total_bytes=sum(total_values) if total_values else None,
            duty_cycle_percent=(
                sum(duty_values) / len(duty_values) if duty_values else None
            ),
            per_chip_hbm_used_bytes=usage_values,
            per_chip_duty_cycle_percent=duty_values,
            source="libtpu",
        )
    except Exception:
        return TpuTelemetry(
            chip_count=static_count,
            hbm_total_bytes=(static_count * spec.hbm_gib * GIB if spec else None),
            source="static",
        )


def detect_hardware(disk_path: Path | None = None) -> HardwareInfo:
    spec, chip_count = detect_tpu_static()
    memory = psutil.virtual_memory()
    target = disk_path or Path.home()
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    try:
        allowed = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = psutil.cpu_count(logical=True) or 1
    cpu_count = psutil.cpu_count(logical=True) or allowed
    load_1m = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    telemetry = read_tpu_telemetry()
    hbm_per_chip = (spec.hbm_gib * GIB) if spec else 0
    static_total = hbm_per_chip * chip_count
    return HardwareInfo(
        tpu_type=spec.name if spec else "none",
        tpu_chip_count=chip_count,
        hbm_per_chip_bytes=hbm_per_chip,
        hbm_total_bytes=telemetry.hbm_total_bytes or static_total,
        cpu_count=cpu_count,
        allowed_cpu_count=allowed,
        ram_total_bytes=int(memory.total),
        ram_available_bytes=int(memory.available),
        disk_total_bytes=int(usage.total),
        disk_free_bytes=int(usage.free),
        load_1m=float(load_1m),
        tpu_owners=find_tpu_owners(),
        telemetry=telemetry,
    )
