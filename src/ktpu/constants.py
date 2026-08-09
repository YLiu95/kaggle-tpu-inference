from __future__ import annotations

import os
from pathlib import Path

GIB = 1024**3
MIB = 1024**2

DEFAULT_MODEL = "google/gemma-4-31B-it"
TESTED_VLLM_TPU_VERSION = "0.26.0"
TESTED_UV_VERSION = "0.12.3"

DEFAULT_HEADROOM_FRACTION = 0.15
DEFAULT_WEIGHT_OVERHEAD_FRACTION = 0.05
DEFAULT_KV_OVERHEAD_FRACTION = 0.10
DEFAULT_CONTEXT_BUCKET = 256
DEFAULT_OUTPUT_CAP = 4096
DEFAULT_CPU_LIMIT = 16
DEFAULT_NICE = 10
DEFAULT_STARTUP_TIMEOUT = 30 * 60
DEFAULT_MONITOR_INTERVAL = 1.0


def data_dir() -> Path:
    override = os.environ.get("KTPU_DATA_DIR")
    return Path(override).expanduser() if override else Path.home() / ".local/share/ktpu"


def state_dir() -> Path:
    override = os.environ.get("KTPU_STATE_DIR")
    return Path(override).expanduser() if override else Path.home() / ".local/state/ktpu"


def cache_dir() -> Path:
    override = os.environ.get("KTPU_CACHE_DIR")
    return Path(override).expanduser() if override else Path.home() / ".cache/ktpu"


def kaggle_mirror_dir() -> Path | None:
    override = os.environ.get("KTPU_KAGGLE_MIRROR")
    path = Path(override).expanduser() if override else Path("/kaggle/working/ktpu")
    if path.parent.exists() and os.access(path.parent, os.W_OK):
        return path
    return None


ENGINE_MANIFEST = "engine-vllm.json"
SERVER_STATE = "server.json"

