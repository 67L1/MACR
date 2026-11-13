from __future__ import annotations

import importlib
import pkgutil

from .base import Benchmark

REGISTRY: dict[str, Benchmark] = {}

for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name.startswith("_") or _module_info.name == "base":
        continue
    _module = importlib.import_module(f"{__name__}.{_module_info.name}")
    _benchmark = getattr(_module, "BENCHMARK", None)
    if not isinstance(_benchmark, Benchmark):
        continue
    if not _benchmark.name:
        raise ValueError(f"benchmarks/{_module_info.name}.py: BENCHMARK.name is empty")
    if _benchmark.name in REGISTRY:
        raise ValueError(f"Duplicate benchmark name: {_benchmark.name}")
    REGISTRY[_benchmark.name] = _benchmark


def get_benchmark(name: str) -> Benchmark:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}; available: {sorted(REGISTRY)}") from None
