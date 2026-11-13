from __future__ import annotations

import argparse
import contextlib
import functools
from typing import Any, Iterator

import core.inference as base_inference
from benchmarks.base import Benchmark
from macr_ext.config import ExtConfig
from macr_ext.relay_ext import MACRRelayExt

_REPLACED_NAMES = ("MACRRelay", "build_output_filename", "build_experiment")


@contextlib.contextmanager
def installed(ext: ExtConfig, benchmark_name: str) -> Iterator[None]:
    originals = {name: getattr(base_inference, name) for name in _REPLACED_NAMES}
    original_filename = originals["build_output_filename"]
    original_experiment = originals["build_experiment"]

    def build_output_filename(benchmark: Benchmark, args: argparse.Namespace) -> str:
        name = original_filename(benchmark, args)
        return f"{name.removesuffix('.jsonl')}_{ext.tag()}.jsonl"

    def build_experiment(*args: Any, **kwargs: Any) -> dict[str, Any]:
        experiment = original_experiment(*args, **kwargs)
        experiment["macr_ext"] = ext.metadata()
        return experiment

    base_inference.MACRRelay = functools.partial(
        MACRRelayExt, ext_config=ext, benchmark_name=benchmark_name
    )
    base_inference.build_output_filename = build_output_filename
    base_inference.build_experiment = build_experiment
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(base_inference, name, value)


def run_ext(
    benchmark: Benchmark,
    args: argparse.Namespace,
    config: base_inference.RuntimeConfig,
    ext: ExtConfig,
) -> None:
    with installed(ext, benchmark.name):
        base_inference.run(benchmark, args, config)
