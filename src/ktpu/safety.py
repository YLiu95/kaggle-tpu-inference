from __future__ import annotations

import os
from dataclasses import dataclass, field

from ktpu.constants import GIB
from ktpu.errors import SafetyError
from ktpu.hardware import HardwareInfo, conflicting_server_processes, process_command


@dataclass
class SafetyReport:
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return not self.issues

    def require_safe(self) -> None:
        if self.issues:
            details = "\n".join(f"- {item}" for item in self.issues)
            raise SafetyError(f"Safety preflight failed:\n{details}")


def evaluate_preflight(
    hardware: HardwareInfo,
    *,
    operation: str,
    required_ram_bytes: int = 16 * GIB,
    required_disk_bytes: int = 10 * GIB,
    max_normalized_load: float = 0.80,
    require_tpu: bool = True,
    ignore_pids: set[int] | None = None,
    server_processes: dict[int, str] | None = None,
) -> SafetyReport:
    ignore = set(ignore_pids or ())
    ignore.add(os.getpid())
    report = SafetyReport()
    if require_tpu and hardware.tpu_chip_count <= 0:
        report.issues.append("No local TPU chips were detected.")
    if hardware.ram_available_bytes < required_ram_bytes:
        report.issues.append(
            f"Available RAM is below the {required_ram_bytes / GIB:.1f} GiB "
            f"requirement for {operation}."
        )
    if hardware.disk_free_bytes < required_disk_bytes:
        report.issues.append(
            f"Free disk is below the {required_disk_bytes / GIB:.1f} GiB "
            f"requirement for {operation}."
        )
    if hardware.normalized_load > max_normalized_load:
        report.issues.append(
            f"Normalized 1-minute CPU load is {hardware.normalized_load:.2f}, "
            f"above the {max_normalized_load:.2f} safety threshold."
        )
    owners = {
        path: pid
        for path, pid in hardware.tpu_owners.items()
        if pid not in ignore
    }
    if owners:
        descriptions = ", ".join(
            f"{path}=PID {pid} ({process_command(pid) or 'unknown'})"
            for path, pid in sorted(owners.items())
        )
        report.issues.append(f"TPU devices are already owned: {descriptions}")
    conflicts = (
        conflicting_server_processes()
        if server_processes is None
        else server_processes
    )
    conflicts = {pid: cmd for pid, cmd in conflicts.items() if pid not in ignore}
    if conflicts:
        descriptions = ", ".join(
            f"PID {pid} ({command})" for pid, command in sorted(conflicts.items())
        )
        report.issues.append(f"Conflicting inference server processes found: {descriptions}")
    telemetry = hardware.telemetry
    if telemetry.hbm_used_bytes and hardware.hbm_total_bytes:
        fraction = telemetry.hbm_used_bytes / hardware.hbm_total_bytes
        if fraction > 0.05 and not owners:
            report.issues.append(
                f"TPU HBM is already {fraction:.1%} utilized without an identified owner."
            )
    if (
        telemetry.duty_cycle_percent is not None
        and telemetry.duty_cycle_percent > 5.0
        and not owners
    ):
        report.issues.append(
            f"TPU duty cycle is already {telemetry.duty_cycle_percent:.1f}% "
            "without an identified owner."
        )
    if hardware.allowed_cpu_count < 4:
        report.warnings.append(
            f"Only {hardware.allowed_cpu_count} CPUs are available to this process."
        )
    return report

