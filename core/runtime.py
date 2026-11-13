from __future__ import annotations

import atexit
import importlib.metadata
import io
import os
import random
import re
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from transformers import set_seed

warnings.filterwarnings("ignore")


class Profiler:
    total_prep_time = 0.0
    total_gen_time = 0.0
    total_bidding_time = 0.0


_PROFILER_LOCK = threading.Lock()


@contextmanager
def timer(name: str, silent: bool = False):
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    if not silent:
        print(f" [timer] {name}: {elapsed:.4f}s")
    with _PROFILER_LOCK:
        if "preprocess" in name:
            Profiler.total_prep_time += elapsed
        elif "generation" in name:
            Profiler.total_gen_time += elapsed
        elif "bidding total" in name:
            Profiler.total_bidding_time += elapsed


def add_profiler_time(prep: float = 0.0, gen: float = 0.0, bidding: float = 0.0) -> None:
    with _PROFILER_LOCK:
        Profiler.total_prep_time += prep
        Profiler.total_gen_time += gen
        Profiler.total_bidding_time += bidding


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    print(f"[seed] Global seed set to {seed}")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        if image_path:
            with Image.open(image_path) as image:
                return image.convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            return image.convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value).__name__}")


_FLASH_ATTN_VALIDATED = False
_FLASH_ATTN_VERSION: str | None = None


def installed_flash_attn_version() -> str | None:
    try:
        return importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        return None


def require_flash_attn() -> str:
    global _FLASH_ATTN_VALIDATED, _FLASH_ATTN_VERSION
    if _FLASH_ATTN_VALIDATED:
        return _FLASH_ATTN_VERSION or "unknown"
    package_version = installed_flash_attn_version()
    try:
        import flash_attn
    except Exception as error:
        raise RuntimeError(
            "All-expert FlashAttention-2 mode requires a working flash-attn "
            "installation compatible with the active PyTorch and CUDA build."
        ) from error
    _FLASH_ATTN_VERSION = package_version or str(
        getattr(sys.modules.get("flash_attn"), "__version__", "unknown")
    )
    _FLASH_ATTN_VALIDATED = True
    print(f"[flash-attn] validated package version={_FLASH_ATTN_VERSION}")
    return _FLASH_ATTN_VERSION


class TeeStream:
    def __init__(self, terminal: Any, log_handle: Any) -> None:
        self.terminal = terminal
        self.log_handle = log_handle

    def write(self, data: str) -> int:
        terminal_result = self.terminal.write(data)
        self.log_handle.write(data)
        return terminal_result if isinstance(terminal_result, int) else len(data)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.terminal, name)


def install_log_tee(log_path: Path) -> None:
    log_handle = log_path.open("x", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    def close_log() -> None:
        try:
            log_handle.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()

    atexit.register(close_log)


def cuda_index(device_value: str) -> int:
    match = re.fullmatch(r"cuda:(\d+)", device_value.strip())
    if match is None:
        raise ValueError(f"Device must have the form cuda:N, got {device_value!r}")
    return int(match.group(1))


def reset_peak_memory(devices: dict[str, str]) -> None:
    for device_value in devices.values():
        torch.cuda.reset_peak_memory_stats(torch.device(device_value))


def collect_peak_memory(devices: dict[str, str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model_name, device_value in devices.items():
        device = torch.device(device_value)
        try:
            result[model_name] = {
                "device": device_value,
                "name": torch.cuda.get_device_name(device),
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device)
                / (1024**2),
            }
        except Exception as error:
            result[model_name] = {
                "device": device_value,
                "error": f"{type(error).__name__}: {error}",
            }
    return result
