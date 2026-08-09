from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
import psutil

from ktpu.checkpoint import Checkpoint
from ktpu.constants import kaggle_mirror_dir, state_dir
from ktpu.engine import EngineInstallation
from ktpu.errors import EngineError
from ktpu.restrictions import bounded_environment, make_preexec_fn
from ktpu.state import ServerState, remove_server_state, save_server_state
from ktpu.util import ensure_private_directory, tail


def build_vllm_command(
    installation: EngineInstallation,
    *,
    model: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    max_model_len: int,
    enable_thinking: bool,
) -> list[str]:
    bootstrap = Path(__file__).with_name("engine_bootstrap.py")
    command = [
        str(installation.python),
        str(bootstrap),
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        "1",
        "--generation-config",
        "vllm",
        "--no-enable-log-requests",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "audio": 0}, separators=(",", ":")),
    ]
    if "gemma-4" in model.lower():
        command.extend(
            [
                "--reasoning-parser",
                "gemma4",
                "--default-chat-template-kwargs",
                json.dumps(
                    {"enable_thinking": bool(enable_thinking)}, separators=(",", ":")
                ),
                "--disable-chunked-mm-input",
            ]
        )
    return command


@dataclass
class ServerProcess:
    process: subprocess.Popen[bytes]
    state: ServerState
    log_handle: object
    mirror_log: Path | None

    def wait_ready(
        self,
        timeout: float,
        callback: Callable[[], None] | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout
        health_url = f"{self.state.base_url}/health"
        while time.monotonic() < deadline:
            returncode = self.process.poll()
            if returncode is not None:
                raise EngineError(
                    f"vLLM exited with status {returncode} before becoming ready.\n"
                    f"{tail(self.state.log_path, 120)}"
                )
            try:
                response = httpx.get(health_url, timeout=2.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if callback is not None:
                callback()
            time.sleep(1.0)
        raise EngineError(
            f"vLLM did not become ready within {timeout:.0f} seconds.\n"
            f"{tail(self.state.log_path, 120)}"
        )

    def stop(self, timeout: float = 20.0) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.state.process_group, 15)
                self.process.wait(timeout=timeout)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.state.process_group, 9)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        try:
            self.log_handle.close()  # type: ignore[attr-defined]
        except OSError:
            pass
        if self.mirror_log is not None:
            try:
                self.mirror_log.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.state.log_path, self.mirror_log)
            except OSError:
                pass
        remove_server_state()


def launch_server(
    installation: EngineInstallation,
    checkpoint: Checkpoint,
    *,
    model: str,
    port: int,
    tensor_parallel_size: int,
    max_model_len: int,
    enable_thinking: bool,
    cpu_limit: int,
) -> tuple[ServerProcess, tuple[int, ...]]:
    host = "127.0.0.1"
    command = build_vllm_command(
        installation,
        model=model,
        host=host,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        enable_thinking=enable_thinking,
    )
    log_dir = ensure_private_directory(state_dir() / "logs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"server-{stamp}.log"
    log_handle = log_path.open("ab", buffering=0)
    try:
        os.chmod(log_path, 0o600)
    except OSError:
        pass
    env, cpus = bounded_environment(cpu_limit=cpu_limit)
    env.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    env.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/ktpu/xla"))
    if os.environ.get("TPU_ACCELERATOR_TYPE", "").lower().startswith("v5litepod"):
        env["KTPU_STRIP_DYNAMIC_SMEM_FLAG"] = "1"
    try:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=make_preexec_fn(cpus),
        )
        ps_process = psutil.Process(process.pid)
        process_group = os.getpgid(process.pid)
    except Exception as exc:
        log_handle.close()
        raise EngineError(f"Could not launch vLLM: {exc}") from exc
    state = ServerState(
        pid=process.pid,
        create_time=ps_process.create_time(),
        process_group=process_group,
        model=model,
        host=host,
        port=port,
        log_path=log_path,
        checkpoint_commit=checkpoint.commit,
        command=command,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    save_server_state(state)
    mirror_root = kaggle_mirror_dir()
    mirror_log = mirror_root / "logs" / log_path.name if mirror_root else None
    return (
        ServerProcess(
            process=process,
            state=state,
            log_handle=log_handle,
            mirror_log=mirror_log,
        ),
        cpus,
    )
