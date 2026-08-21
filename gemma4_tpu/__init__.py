"""Gemma-4 TPU inference. Heavy deps are imported lazily so the thin client stays fast."""

from .models import MODELS, ModelSpec, resolve as resolve_model  # noqa: F401

__all__ = [
    "Engine",
    "MODELS",
    "ModelSpec",
    "TextConfig",
    "TpuMonitor",
    "load_text_config",
    "resolve_model",
]


def __getattr__(name):
    if name == "Engine":
        from .engine import Engine

        return Engine
    if name == "TpuMonitor":
        from .tpu_monitor import TpuMonitor

        return TpuMonitor
    if name in ("TextConfig", "load_text_config"):
        from . import config

        return getattr(config, name)
    raise AttributeError(name)
