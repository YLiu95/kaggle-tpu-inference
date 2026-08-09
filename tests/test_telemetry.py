from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ktpu.hardware import TpuTelemetry
from ktpu.telemetry import (
    CSV_FIELDS,
    CsvTelemetryLogger,
    InferenceStats,
    RuntimeSnapshot,
    TelemetryMonitor,
)


def snapshot() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        timestamp_utc="2026-01-01T00:00:00+00:00",
        phase="inference",
        elapsed_s=1.0,
        server_pid=1,
        cpu_percent_of_affinity=10.0,
        cpu_cores_used=1.0,
        ram_used_gib=2.0,
        ram_percent=3.0,
        tpu_chips=8,
        tpu_hbm_used_gib=4.0,
        tpu_hbm_total_gib=128.0,
        tpu_hbm_percent=3.125,
        tpu_duty_cycle_percent=50.0,
        ttft_s=0.5,
        output_tokens=10,
        tokens_per_s=20.0,
    )


class CsvTests(unittest.TestCase):
    def test_csv_is_written_and_mirrored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary.csv"
            mirror = root / "mirror.csv"
            with CsvTelemetryLogger(primary, mirror) as logger:
                logger.write(snapshot())
            for path in (primary, mirror):
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["tpu_chips"], "8")
                self.assertEqual(list(rows[0].keys()), CSV_FIELDS)

    def test_monitor_falls_back_when_tpu_metrics_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = CsvTelemetryLogger(Path(temporary) / "log.csv", False)

            def fail() -> TpuTelemetry:
                raise RuntimeError("no metrics")

            monitor = TelemetryMonitor(
                server_pid=1,
                affinity_cpu_count=2,
                logger=logger,
                stats=InferenceStats(),
                telemetry_reader=fail,
            )
            value = monitor.sample()
            logger.close()
        self.assertEqual(value.tpu_chips, 0)
        self.assertIsNone(value.tpu_hbm_used_gib)


if __name__ == "__main__":
    unittest.main()
