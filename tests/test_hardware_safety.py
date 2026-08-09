from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ktpu.constants import GIB
from ktpu.hardware import (
    HardwareInfo,
    TpuTelemetry,
    detect_tpu_from_sysfs,
)
from ktpu.restrictions import bounded_environment, make_preexec_fn, select_cpu_affinity
from ktpu.safety import evaluate_preflight


def hardware(**overrides) -> HardwareInfo:
    values = {
        "tpu_type": "v5e",
        "tpu_chip_count": 8,
        "hbm_per_chip_bytes": 16 * GIB,
        "hbm_total_bytes": 128 * GIB,
        "cpu_count": 96,
        "allowed_cpu_count": 96,
        "ram_total_bytes": 300 * GIB,
        "ram_available_bytes": 250 * GIB,
        "disk_total_bytes": 1000 * GIB,
        "disk_free_bytes": 500 * GIB,
        "load_1m": 2.0,
        "tpu_owners": {},
        "telemetry": TpuTelemetry(chip_count=8, hbm_total_bytes=128 * GIB),
    }
    values.update(overrides)
    return HardwareInfo(**values)


class HardwareTests(unittest.TestCase):
    def test_detects_v5e_from_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(8):
                path = root / f"0000:00:0{index}.0"
                path.mkdir()
                (path / "vendor").write_text("0x1ae0\n")
                (path / "device").write_text("0x0063\n")
            spec, count = detect_tpu_from_sysfs(root)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "v5e")
        self.assertEqual(spec.hbm_gib, 16)
        self.assertEqual(count, 8)


class SafetyTests(unittest.TestCase):
    def test_safe_configuration_passes(self) -> None:
        report = evaluate_preflight(
            hardware(),
            operation="test",
            server_processes={},
        )
        self.assertTrue(report.safe)

    def test_low_ram_disk_load_and_ownership_are_gated(self) -> None:
        report = evaluate_preflight(
            hardware(
                ram_available_bytes=GIB,
                disk_free_bytes=GIB,
                load_1m=100.0,
                tpu_owners={"/dev/vfio/0": 999999},
            ),
            operation="test",
            required_ram_bytes=2 * GIB,
            required_disk_bytes=2 * GIB,
            server_processes={},
        )
        text = "\n".join(report.issues)
        self.assertIn("RAM", text)
        self.assertIn("disk", text)
        self.assertIn("CPU load", text)
        self.assertIn("owned", text)

    def test_conflicting_server_is_gated(self) -> None:
        report = evaluate_preflight(
            hardware(),
            operation="test",
            server_processes={1234: "vllm serve model"},
        )
        self.assertFalse(report.safe)
        self.assertIn("Conflicting", report.issues[0])


class RestrictionTests(unittest.TestCase):
    def test_affinity_and_thread_limits_are_bounded(self) -> None:
        cpus = select_cpu_affinity(set(range(32)), limit=8)
        self.assertEqual(cpus, tuple(range(8)))
        env, selected = bounded_environment({"PATH": "/bin"}, cpu_limit=4)
        self.assertLessEqual(len(selected), 4)
        self.assertEqual(env["MAX_JOBS"], "1")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(env["OMP_NUM_THREADS"], str(len(selected)))

    def test_preexec_applies_affinity_and_nice(self) -> None:
        import unittest.mock as mock

        callback = make_preexec_fn((1, 2), nice=7)
        with (
            mock.patch("ktpu.restrictions.os.setsid") as setsid,
            mock.patch("ktpu.restrictions.os.sched_setaffinity") as affinity,
            mock.patch("ktpu.restrictions.os.nice") as nice,
        ):
            callback()
        setsid.assert_called_once()
        affinity.assert_called_once_with(0, {1, 2})
        nice.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()

