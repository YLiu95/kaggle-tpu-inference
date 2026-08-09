"""Crash-safe Kaggle TPU inference utilities."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kaggle-tpu-inference")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]

