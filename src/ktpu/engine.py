from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ktpu.checkpoint import Checkpoint, verify_checkpoint
from ktpu.constants import (
    ENGINE_MANIFEST,
    TESTED_UV_VERSION,
    TESTED_VLLM_TPU_VERSION,
    data_dir,
)
from ktpu.errors import EngineError
from ktpu.hardware import detect_hardware
from ktpu.restrictions import bounded_environment, make_preexec_fn
from ktpu.safety import evaluate_preflight
from ktpu.util import atomic_write_json, ensure_private_directory, read_json


@dataclass(frozen=True)
class EngineInstallation:
    engine: str
    version: str
    root: Path
    python: Path
    executable: Path
    installed_versions: dict[str, str]
    checkpoint_commit: str | None


def manifest_path() -> Path:
    return data_dir() / ENGINE_MANIFEST


def load_engine_installation() -> EngineInstallation | None:
    value = read_json(manifest_path())
    if not value:
        return None
    try:
        installation = EngineInstallation(
            engine=str(value["engine"]),
            version=str(value["version"]),
            root=Path(value["root"]),
            python=Path(value["python"]),
            executable=Path(value["executable"]),
            installed_versions={
                str(key): str(item)
                for key, item in dict(value.get("installed_versions", {})).items()
            },
            checkpoint_commit=(
                str(value["checkpoint_commit"])
                if value.get("checkpoint_commit")
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not installation.python.is_file() or not installation.executable.is_file():
        return None
    return installation


def _run_restricted(command: list[str], *, cpu_limit: int) -> None:
    env, cpus = bounded_environment(cpu_limit=cpu_limit)
    try:
        subprocess.run(
            command,
            check=True,
            env=env,
            preexec_fn=make_preexec_fn(cpus),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineError(f"Engine setup command failed: {command[0]}") from exc


def _installed_versions(python: Path) -> dict[str, str]:
    script = """
import importlib.metadata as m
import json
names = ["vllm-tpu", "tpu-inference", "jax", "jaxlib", "torch", "torch-xla", "transformers"]
result = {}
for name in names:
    try:
        result[name] = m.version(name)
    except m.PackageNotFoundError:
        pass
print(json.dumps(result, sort_keys=True))
"""
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        return dict(json.loads(result.stdout))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise EngineError("Installed engine could not be verified.") from exc


def setup_vllm(
    *,
    version: str = TESTED_VLLM_TPU_VERSION,
    cpu_limit: int = 16,
    checkpoint: bool = True,
) -> EngineInstallation:
    checkpoint_result: Checkpoint | None = (
        verify_checkpoint(run_tests=True) if checkpoint else None
    )
    hardware = detect_hardware(data_dir())
    evaluate_preflight(
        hardware,
        operation="engine setup",
        required_ram_bytes=8 * 1024**3,
        required_disk_bytes=20 * 1024**3,
        require_tpu=True,
    ).require_safe()
    root = ensure_private_directory(data_dir() / "engines")
    target = root / f"vllm-{version}-{int(time.time())}"
    if target.exists():
        raise EngineError(f"Engine target unexpectedly exists: {target}")
    try:
        _run_restricted(
            [sys.executable, "-m", "venv", str(target)], cpu_limit=cpu_limit
        )
        python = target / "bin/python"
        pip = target / "bin/pip"
        _run_restricted(
            [
                str(pip),
                "install",
                "--only-binary=:all:",
                f"uv=={TESTED_UV_VERSION}",
            ],
            cpu_limit=cpu_limit,
        )
        uv = target / "bin/uv"
        _run_restricted(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(python),
                "--only-binary",
                ":all:",
                f"vllm-tpu=={version}",
            ],
            cpu_limit=cpu_limit,
        )
        executable = target / "bin/vllm"
        if not executable.is_file():
            raise EngineError("vLLM executable was not installed.")
        versions = _installed_versions(python)
        if versions.get("vllm-tpu") != version:
            raise EngineError(
                f"Expected vllm-tpu {version}, found {versions.get('vllm-tpu')!r}."
            )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    installation = EngineInstallation(
        engine="vllm",
        version=version,
        root=target,
        python=python,
        executable=executable,
        installed_versions=versions,
        checkpoint_commit=checkpoint_result.commit if checkpoint_result else None,
    )
    atomic_write_json(
        manifest_path(),
        {
            "engine": installation.engine,
            "version": installation.version,
            "root": str(installation.root),
            "python": str(installation.python),
            "executable": str(installation.executable),
            "installed_versions": installation.installed_versions,
            "checkpoint_commit": installation.checkpoint_commit,
        },
    )
    return installation


def require_engine() -> EngineInstallation:
    installation = load_engine_installation()
    if installation is None:
        raise EngineError("vLLM TPU is not set up. Run: ktpu setup --engine vllm")
    return installation

