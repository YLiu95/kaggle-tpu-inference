from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from ktpu.constants import SERVER_STATE, state_dir
from ktpu.hardware import find_tpu_owners, process_command
from ktpu.util import atomic_write_json, read_json


@dataclass(frozen=True)
class ServerState:
    pid: int
    create_time: float
    process_group: int
    model: str
    host: str
    port: int
    log_path: Path
    checkpoint_commit: str
    command: list[str]
    started_at_utc: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def server_state_path() -> Path:
    return state_dir() / SERVER_STATE


def save_server_state(state: ServerState) -> None:
    atomic_write_json(
        server_state_path(),
        {
            "pid": state.pid,
            "create_time": state.create_time,
            "process_group": state.process_group,
            "model": state.model,
            "host": state.host,
            "port": state.port,
            "log_path": str(state.log_path),
            "checkpoint_commit": state.checkpoint_commit,
            "command": state.command,
            "started_at_utc": state.started_at_utc,
        },
    )


def load_server_state() -> ServerState | None:
    value = read_json(server_state_path())
    if not value:
        return None
    try:
        return ServerState(
            pid=int(value["pid"]),
            create_time=float(value["create_time"]),
            process_group=int(value["process_group"]),
            model=str(value["model"]),
            host=str(value["host"]),
            port=int(value["port"]),
            log_path=Path(value["log_path"]),
            checkpoint_commit=str(value["checkpoint_commit"]),
            command=[str(item) for item in value.get("command", [])],
            started_at_utc=str(value["started_at_utc"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def state_process(state: ServerState) -> psutil.Process | None:
    try:
        process = psutil.Process(state.pid)
        if abs(process.create_time() - state.create_time) > 1.0:
            return None
        return process
    except psutil.Error:
        return None


def remove_server_state() -> None:
    try:
        server_state_path().unlink()
    except FileNotFoundError:
        pass


def _stop_group(process_group: int, timeout: float) -> None:
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop_managed_server(timeout: float = 20.0) -> bool:
    state = load_server_state()
    if state is None:
        remove_server_state()
        return False
    process = state_process(state)
    if process is None:
        remove_server_state()
        return False
    _stop_group(state.process_group, timeout)
    remove_server_state()
    return True


def clear_tpu_state(*, force: bool = False) -> tuple[list[int], dict[str, int]]:
    killed: list[int] = []
    state = load_server_state()
    if state is not None and state_process(state) is not None:
        pid = state.pid
        if stop_managed_server():
            killed.append(pid)
    else:
        remove_server_state()
    owners = find_tpu_owners()
    if force:
        candidate_pids = sorted(set(owners.values()))
        for pid in candidate_pids:
            command = process_command(pid).lower()
            if not any(
                marker in command
                for marker in ("vllm", "tpu_inference", "tpu-inference")
            ):
                continue
            try:
                process = psutil.Process(pid)
                group = os.getpgid(pid)
                _stop_group(group, 10.0)
                process.wait(timeout=2)
                killed.append(pid)
            except (psutil.Error, OSError):
                continue
        owners = find_tpu_owners()
    return sorted(set(killed)), owners

