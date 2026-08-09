from __future__ import annotations

import os
from collections.abc import Callable

from ktpu.constants import DEFAULT_CPU_LIMIT, DEFAULT_NICE


def select_cpu_affinity(
    available: set[int] | None = None, limit: int = DEFAULT_CPU_LIMIT
) -> tuple[int, ...]:
    if limit <= 0:
        raise ValueError("CPU affinity limit must be positive.")
    if available is None:
        try:
            available = set(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = set(range(os.cpu_count() or 1))
    ordered = sorted(available)
    return tuple(ordered[: min(limit, len(ordered))])


def bounded_environment(
    base: dict[str, str] | None = None,
    *,
    cpu_limit: int = DEFAULT_CPU_LIMIT,
) -> tuple[dict[str, str], tuple[int, ...]]:
    env = dict(os.environ if base is None else base)
    cpus = select_cpu_affinity(limit=cpu_limit)
    thread_count = max(1, len(cpus))
    values = {
        "OMP_NUM_THREADS": str(thread_count),
        "MKL_NUM_THREADS": str(thread_count),
        "OPENBLAS_NUM_THREADS": str(thread_count),
        "NUMEXPR_NUM_THREADS": str(thread_count),
        "RAYON_NUM_THREADS": str(thread_count),
        "TF_NUM_INTRAOP_THREADS": str(thread_count),
        "TF_NUM_INTEROP_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MAX_JOBS": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "NINJAFLAGS": "-j1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    env.update(values)
    return env, cpus


def make_preexec_fn(
    cpus: tuple[int, ...], nice: int = DEFAULT_NICE
) -> Callable[[], None]:
    def restrict() -> None:
        os.setsid()
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpus))
        if nice:
            try:
                os.nice(nice)
            except OSError:
                pass

    return restrict

