"""Live TPU utilisation sampling.

Two independent sources are combined:

* ``jax.Device.memory_stats()`` - always available inside the process that owns the TPU.
* ``tpu_info`` gRPC to the libtpu runtime metric server (``localhost:8431``) - gives the
  hardware **duty cycle**, but is not always reachable on Kaggle (libtpu/py3.12 mismatch),
  so every call is guarded.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import jax


@dataclass
class ChipStat:
    index: int
    kind: str
    hbm_used: int = 0
    hbm_limit: int = 0
    hbm_peak: int = 0
    duty_cycle: float | None = None


@dataclass
class Snapshot:
    chips: list[ChipStat] = field(default_factory=list)
    duty_source: str = "unavailable"
    host_rss_gb: float = 0.0
    host_cpu_pct: float = 0.0
    timestamp: float = 0.0

    @property
    def hbm_used(self) -> int:
        return sum(c.hbm_used for c in self.chips)

    @property
    def hbm_limit(self) -> int:
        return sum(c.hbm_limit for c in self.chips)

    @property
    def mean_duty(self) -> float | None:
        vals = [c.duty_cycle for c in self.chips if c.duty_cycle is not None]
        return sum(vals) / len(vals) if vals else None


class TpuMonitor:
    def __init__(self, poll_interval: float = 0.4):
        self.poll_interval = poll_interval
        self.devices = jax.devices()
        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._duty_fn = self._init_duty()
        self._proc = None
        try:
            import psutil

            self._proc = psutil.Process()
            self._proc.cpu_percent(None)
        except Exception:
            pass

    # ------------------------------------------------------------------ duty cycle
    def _init_duty(self):
        try:
            from tpu_info import device as tdev, metrics as tmetrics

            chip_type, _ = tdev.get_local_chips()
            if chip_type is None:
                return None
            probe = tmetrics.get_chip_usage(chip_type)
            if not probe:
                return None
            self.duty_source = "libtpu gRPC :8431"
            return lambda: tmetrics.get_chip_usage(chip_type)
        except Exception:
            return None

    # ------------------------------------------------------------------ sampling
    def sample(self) -> Snapshot:
        chips = []
        for i, d in enumerate(self.devices):
            try:
                st = d.memory_stats() or {}
            except Exception:
                st = {}
            chips.append(
                ChipStat(
                    index=i,
                    kind=d.device_kind,
                    hbm_used=int(st.get("bytes_in_use", 0)),
                    hbm_limit=int(st.get("bytes_limit", 0)),
                    hbm_peak=int(st.get("peak_bytes_in_use", 0)),
                )
            )
        source = "unavailable"
        if self._duty_fn is not None:
            try:
                for u in self._duty_fn():
                    if u.device_id < len(chips):
                        chips[u.device_id].duty_cycle = float(u.duty_cycle_pct)
                        if not chips[u.device_id].hbm_limit:
                            chips[u.device_id].hbm_limit = int(u.total_memory)
                source = "libtpu gRPC :8431"
            except Exception:
                self._duty_fn = None

        snap = Snapshot(chips=chips, duty_source=source, timestamp=time.time())
        if self._proc is not None:
            try:
                snap.host_rss_gb = self._proc.memory_info().rss / 1e9
                snap.host_cpu_pct = self._proc.cpu_percent(None)
            except Exception:
                pass
        return snap

    def _loop(self):
        while not self._stop.is_set():
            snap = self.sample()
            with self._lock:
                self._snapshot = snap
            self._stop.wait(self.poll_interval)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def latest(self) -> Snapshot:
        with self._lock:
            return self._snapshot


V5E_BF16_TFLOPS = 197.0  # per chip, dense bf16
V5E_HBM_GBPS = 819.0  # per chip


def peak_flops(n_chips: int) -> float:
    return V5E_BF16_TFLOPS * 1e12 * n_chips


def peak_bandwidth(n_chips: int) -> float:
    return V5E_HBM_GBPS * 1e9 * n_chips
