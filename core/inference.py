from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import tempfile
import time
import traceback
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from benchmarks.base import Benchmark
from core.adapters import (
    FLASH_ATTN_IMPLEMENTATION,
    INTERNVL_INPUT_SIZE,
    INTERNVL_MAX_NUM,
    INTERNVL_USE_THUMBNAIL,
    MINICPM_ENABLE_THINKING,
    MINICPM_ITERATION,
    MINICPM_MAX_SLICE_NUMS,
    QWEN_MAX_PIXELS,
)
from core.prompts import relay_instruction_sha256, sha256_text
from core.relay import OPTIMIZATION_PROFILE, MACRRelay
from core.runtime import (
    cuda_index,
    decode_image,
    install_log_tee,
    installed_flash_attn_version,
    json_default,
    set_global_seed,
    utc_now,
)
from core.worker_main import WorkerSpec
from core.workers import WorkerDiedError, shutdown_workers, start_model_workers

MODEL_NAMES = ("Qwen3-VL", "InternVL", "MiniCPM-V")
ADAPTER_FACTORIES = {
    "InternVL": "core.adapters:InternVLAdapter",
    "Qwen3-VL": "core.adapters:Qwen3VLAdapter",
    "MiniCPM-V": "core.adapters:MiniCPMVAdapter",
}
VARIANT_TAG = "flashattn2_kvledger"
SCHEMA_VERSION = 3
ORI_INPUT_TOKEN_DEFINITION = (
    "per model prefill length of the initial prompt+image input (no assistant "
    "prefix); image = visual-embedding positions (Qwen <|image_pad|>, InternVL "
    "<IMG_CONTEXT>, MiniCPM image_bound spans); text = total - image"
)


@dataclass
class RuntimeConfig:
    model_paths: dict[str, str]
    dataset_paths: dict[str, str]
    devices: dict[str, str]
    stats_dir: Path
    results_dir: Path
    logs_dir: Path


def resolve_dataset_path(benchmark: Benchmark, args: argparse.Namespace, config: RuntimeConfig) -> Path:
    value = args.dataset_path or config.dataset_paths.get(benchmark.name, "")
    if not str(value).strip():
        raise ValueError(
            f'Dataset path is empty: fill DATASET_PATHS["{benchmark.name}"] at the top '
            "of main.py (or pass --dataset-path)"
        )
    return Path(value).expanduser().resolve()


def resolve_model_paths(config: RuntimeConfig) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for model_name in MODEL_NAMES:
        value = str(config.model_paths.get(model_name, "")).strip()
        if not value:
            raise ValueError(f'Model path is empty: fill MODEL_PATHS["{model_name}"] at the top of main.py')
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f'MODEL_PATHS["{model_name}"] does not exist: {path}')
        resolved[model_name] = str(path)
    return resolved


def validate_args(args: argparse.Namespace, config: RuntimeConfig) -> None:
    if not math.isfinite(args.z_threshold):
        raise ValueError("--z-threshold must be finite")
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.initial_trial_windows <= 0:
        raise ValueError("--initial-trial-windows must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.cleanup_every < 0:
        raise ValueError("--cleanup-every cannot be negative")
    if args.sample_ids and args.max_samples is not None:
        raise ValueError("--sample-ids and --max-samples cannot be used together")
    if args.extend_from and not args.extend:
        raise ValueError("--extend-from requires --extend")
    if args.extend and args.sample_ids:
        raise ValueError("--extend cannot be used with --sample-ids")
    disabled_models = args.disable_candidate_cache_for or []
    if len(set(disabled_models)) != len(disabled_models):
        raise ValueError("--disable-candidate-cache-for contains duplicate models")
    missing_devices = [name for name in MODEL_NAMES if name not in config.devices]
    if missing_devices:
        raise ValueError(f"DEVICES in main.py is missing: {missing_devices}")
    for name in MODEL_NAMES:
        cuda_index(config.devices[name])


def normalized_sample_ids(benchmark: Benchmark, values: list[str] | None) -> list[str]:
    sample_ids = [benchmark.normalize_sample_id(value) for value in values or []]
    duplicates = [key for key, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"--sample-ids contains duplicate ID(s): {duplicates[:10]}")
    return sample_ids


def select_dataset_rows(dataset: Any, benchmark: Benchmark, args: argparse.Namespace) -> Any:
    if args.sample_ids:
        requested = normalized_sample_ids(benchmark, args.sample_ids)
        index_by_id = {
            str(value): index for index, value in enumerate(dataset[benchmark.id_field])
        }
        missing = [sample_id for sample_id in requested if sample_id not in index_by_id]
        if missing:
            raise ValueError(f"Requested sample ID(s) not found in dataset: {missing}")
        return dataset.select([index_by_id[sample_id] for sample_id in requested])
    if args.max_samples is not None:
        return dataset.select(range(min(args.max_samples, len(dataset))))
    return dataset


def inference_only_view(dataset: Any, benchmark: Benchmark) -> Any:
    removable = [
        column for column in dataset.column_names if column not in benchmark.inference_fields
    ]
    inference_dataset = dataset.remove_columns(removable)
    if set(inference_dataset.column_names) != set(benchmark.inference_fields):
        raise RuntimeError(
            f"Inference dataset isolation failed; columns are {inference_dataset.column_names}"
        )
    return inference_dataset


def parameter_tag(value: float) -> str:
    text = format(value, ".12g")
    return text.replace("-", "neg").replace("+", "pos").replace(".", "p")


def build_output_filename(benchmark: Benchmark, args: argparse.Namespace) -> str:
    if args.sample_ids:
        selected = normalized_sample_ids(benchmark, args.sample_ids)
        digest = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()[:10]
        sample_tag = f"ids{len(selected)}_{digest}"
    else:
        sample_tag = "full" if args.max_samples is None else f"n{args.max_samples}"
    disabled_models = sorted(args.disable_candidate_cache_for or [])
    cache_policy_tag = "" if not disabled_models else "_cacheoff" + "-".join(disabled_models)
    return (
        f"responses_{benchmark.name}_{benchmark.split}_{VARIANT_TAG}_"
        f"z{parameter_tag(args.z_threshold)}_w{args.window_size}_"
        f"init{args.initial_trial_windows}_max{args.max_new_tokens}_"
        f"seed{args.seed}_{sample_tag}{cache_policy_tag}.jsonl"
    )


def build_log_filename(benchmark: Benchmark, args: argparse.Namespace) -> str:
    experiment_name = build_output_filename(benchmark, args).removesuffix(".jsonl")
    if experiment_name.startswith("responses_"):
        experiment_name = experiment_name[len("responses_") :]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"loginfo_{experiment_name}_{timestamp}.log"


def resolve_output_path(benchmark: Benchmark, args: argparse.Namespace, config: RuntimeConfig) -> Path:
    if args.output_file:
        path = Path(args.output_file).expanduser().resolve()
    else:
        path = (config.results_dir / benchmark.name / build_output_filename(benchmark, args)).resolve()
    if path.suffix.lower() != ".jsonl":
        raise ValueError("Output file must use the .jsonl suffix")
    return path


def read_latest_records(
    path: Path, id_field: str
) -> tuple[OrderedDict[str, dict[str, Any]], dict[str, Any]]:
    latest: OrderedDict[str, dict[str, Any]] = OrderedDict()
    diagnostics: dict[str, Any] = {
        "malformed_lines": [],
        "ignored_partial_tail": False,
        "duplicates": 0,
    }
    if not path.exists():
        return latest, diagnostics
    if not path.is_file():
        raise FileNotFoundError(f"Output path is not a regular file: {path}")

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        diagnostics["ignored_partial_tail"] = True
        lines = lines[:-1]
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
            if not isinstance(record, dict):
                raise TypeError("record is not an object")
            sample_id = str(record[id_field])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            diagnostics["malformed_lines"].append(line_number)
            continue
        if sample_id in latest:
            diagnostics["duplicates"] += 1
        latest[sample_id] = record
    diagnostics["file_size_bytes"] = len(raw)
    return latest, diagnostics


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )


def build_experiment(
    benchmark: Benchmark,
    args: argparse.Namespace,
    config: RuntimeConfig,
    dataset_path: Path,
    model_paths: dict[str, str],
    dataset_details: dict[str, Any],
) -> dict[str, Any]:
    experiment: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": benchmark.name,
        "split": benchmark.split,
        "expected_total": benchmark.expected_total,
        "dataset_path": str(dataset_path),
        "dataset_fingerprint": dataset_details.get("dataset_fingerprint"),
        "dataset_rows": dataset_details.get("rows"),
        "stats_dir": str(config.stats_dir.resolve()),
        "z_threshold": args.z_threshold,
        "window_size": args.window_size,
        "initial_trial_windows": args.initial_trial_windows,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "runtime_variant": "macr_all_models_flash_attention_2_optcache_kv_ledger",
        "optimization": {
            "profile": OPTIMIZATION_PROFILE,
            "sample_static_inputs": True,
            "static_tensor_storage": "cpu_pinned_memory",
            "async_device_transfer_when_pinned": True,
            "winner_continuation_state_promotion": True,
            "pending_token_contract": "past_excludes_exactly_one_pending_token",
            "strict_owner_sample_history_and_cache_length_validation": True,
            "invalid_state_policy": "ledger_resync_then_full_prefill_fallback",
            "candidate_generation_policy": "per_model_kv_ledger_lcp_crop_then_incremental_prefill",
            "cross_model_kv_transfer": False,
            "kv_ledger": {
                "scope": "one resident cache per model per sample, released at sample end",
                "cold_prefill": "full multimodal prefill on the first call of the sample "
                "(initial trial) or when no cached prefix covers every visual position",
                "reuse": "longest common token prefix with the retokenized committed text; "
                "the rest is cropped and only the new tokens are prefilled",
                "min_uncached_tokens": 1,
                "requires_neutral_generation_config": ["InternVL", "MiniCPM-V"],
            },
            "operator_disabled_candidate_cache_models": sorted(
                args.disable_candidate_cache_for or []
            ),
        },
        "prompt_protocol": "core_relay_instruction_plus_benchmark_task_prompt",
        "relay_instruction_sha256": relay_instruction_sha256(),
        "task_prompt_sha256": benchmark.task_prompt_sha256(),
        "ground_truth_in_response_jsonl": False,
        "inference_columns": list(benchmark.inference_fields),
        "ori_input_token_definition": ORI_INPUT_TOKEN_DEFINITION,
        "model_paths": dict(model_paths),
        "device_mapping": {name: config.devices[name] for name in MODEL_NAMES},
        "flash_attention": {
            "enabled_for_all_models": True,
            "flash_attn_package_version": installed_flash_attn_version(),
            "Qwen3-VL": FLASH_ATTN_IMPLEMENTATION,
            "InternVL": "use_flash_attn=True",
            "MiniCPM-V": FLASH_ATTN_IMPLEMENTATION,
        },
        "model_settings": {
            "Qwen3-VL": {
                "torch_dtype": "torch.bfloat16",
                "attention_implementation": FLASH_ATTN_IMPLEMENTATION,
                "do_sample": False,
                "max_pixels": QWEN_MAX_PIXELS,
                "chat_template_tokenize": True,
                "add_generation_prompt": True,
            },
            "InternVL": {
                "torch_dtype": "torch.bfloat16",
                "low_cpu_mem_usage": False,
                "use_flash_attn": True,
                "tokenizer_use_fast": False,
                "input_size": INTERNVL_INPUT_SIZE,
                "max_num": INTERNVL_MAX_NUM,
                "use_thumbnail": INTERNVL_USE_THUMBNAIL,
                "do_sample": False,
            },
            "MiniCPM-V": {
                "torch_dtype": "torch.bfloat16",
                "attention_implementation": FLASH_ATTN_IMPLEMENTATION,
                "sampling": False,
                "enable_thinking": MINICPM_ENABLE_THINKING,
                "max_slice_nums": MINICPM_MAX_SLICE_NUMS,
                "iteration": MINICPM_ITERATION,
            },
        },
        "token_accounting_scope": "generated_tokens_only",
    }
    if args.sample_ids:
        selected = normalized_sample_ids(benchmark, args.sample_ids)
        experiment["expected_selected_total"] = len(selected)
        experiment["sample_selection"] = {"strategy": f"explicit_{benchmark.id_field}", "ids": selected}
    return experiment


def validate_existing_experiments(
    records: OrderedDict[str, dict[str, Any]], experiment: dict[str, Any]
) -> None:
    for sha_key in ("relay_instruction_sha256", "task_prompt_sha256"):
        mismatched = [
            sample_id
            for sample_id, record in records.items()
            if isinstance(record.get("experiment"), dict)
            and record["experiment"].get(sha_key) != experiment[sha_key]
        ]
        if mismatched:
            raise ValueError(
                f"Refusing to append: {sha_key} differs from existing records "
                f"(e.g. {mismatched[:5]}); the prompt changed since this file was written"
            )

    expected = stable_json(experiment)
    incompatible: list[str] = []
    missing: list[str] = []
    for sample_id, record in records.items():
        existing = record.get("experiment")
        if not isinstance(existing, dict):
            missing.append(sample_id)
        elif stable_json(existing) != expected:
            incompatible.append(sample_id)
    if missing or incompatible:
        parts = []
        if missing:
            parts.append(f"missing experiment metadata: {missing[:5]}")
        if incompatible:
            parts.append(f"incompatible experiment metadata: {incompatible[:5]}")
        raise ValueError(
            "Refusing to append records from a different experiment (" + "; ".join(parts) + ")"
        )


_EXTEND_PROBE_SAMPLES = 987654321


def find_extend_source(
    benchmark: Benchmark, args: argparse.Namespace, output_path: Path, target_count: int
) -> Path:
    probe = copy.copy(args)
    probe.max_samples = _EXTEND_PROBE_SAMPLES
    probe_name = build_output_filename(benchmark, probe)
    marker = f"n{_EXTEND_PROBE_SAMPLES}"
    if probe_name.count(marker) != 1:
        raise RuntimeError(f"Cannot locate the sample tag in {probe_name!r}")
    prefix, suffix = probe_name.split(marker)
    pattern = re.compile(re.escape(prefix) + r"n([1-9]\d*)" + re.escape(suffix))
    directory = output_path.parent
    candidates: list[Path] = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            match = pattern.fullmatch(path.name)
            if match and path.is_file() and int(match.group(1)) < target_count:
                candidates.append(path)
    if len(candidates) != 1:
        found = ", ".join(path.name for path in candidates) if candidates else "none"
        raise ValueError(
            f"--extend needs exactly one {prefix}n<M>{suffix} with M < {target_count} "
            f"in {directory} (found: {found}); pass the source file with --extend-from"
        )
    return candidates[0]


EXTEND_IGNORED_KEYS = ("max_samples", "device_mapping", "stats_dir")


def extend_comparable(experiment: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in experiment.items() if key not in EXTEND_IGNORED_KEYS}


def validate_extend_source(
    records: OrderedDict[str, dict[str, Any]],
    experiment: dict[str, Any],
    source_path: Path,
    target_count: int,
) -> int:
    if not records:
        raise ValueError(f"--extend source has no records: {source_path}")
    current = extend_comparable(experiment)
    expected = stable_json(current)
    source_max_samples = set()
    for sample_id, record in records.items():
        existing = record.get("experiment")
        if not isinstance(existing, dict):
            raise ValueError(
                f"--extend source record {sample_id} has no experiment metadata: {source_path}"
            )
        if "sample_selection" in existing:
            raise ValueError(f"--extend cannot import a --sample-ids output file: {source_path}")
        existing = extend_comparable(existing)
        if stable_json(existing) != expected:
            differing = sorted(
                key
                for key in set(existing) | set(current)
                if key not in existing
                or key not in current
                or stable_json(existing[key]) != stable_json(current[key])
            )
            raise ValueError(
                f"Refusing to extend from {source_path}: experiment metadata differs beyond "
                f"{'/'.join(EXTEND_IGNORED_KEYS)} (record {sample_id}; keys {differing})"
            )
        source_max_samples.add(record["experiment"].get("max_samples"))
    if len(source_max_samples) != 1:
        raise ValueError(f"--extend source mixes max_samples values {source_max_samples}: {source_path}")
    max_samples = source_max_samples.pop()
    if not isinstance(max_samples, int) or isinstance(max_samples, bool):
        raise ValueError(f"--extend source was not produced with --max-samples: {source_path}")
    if max_samples >= target_count:
        raise ValueError(
            f"--extend source covers {max_samples} samples, but only {target_count} are selected now"
        )
    return max_samples


def import_extend_source(
    benchmark: Benchmark,
    args: argparse.Namespace,
    output_path: Path,
    experiment: dict[str, Any],
    selected_ids: list[str],
) -> None:
    if output_path.exists():
        print(f"[extend] {output_path} already exists; skipping import and resuming")
        return
    target_count = len(selected_ids)
    if args.extend_from:
        source_path = Path(args.extend_from).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"--extend-from is not a file: {source_path}")
    else:
        source_path = find_extend_source(benchmark, args, output_path, target_count)
    records, diagnostics = read_latest_records(source_path, benchmark.id_field)
    source_max_samples = validate_extend_source(records, experiment, source_path, target_count)
    source_ids = selected_ids[:source_max_samples]
    outside = sorted(set(records).difference(source_ids))
    if outside:
        raise ValueError(
            f"--extend source contains ID(s) outside the first {source_max_samples} "
            f"selected samples: {outside[:10]}"
        )

    imported_from = {
        "path": str(source_path),
        "max_samples": source_max_samples,
        "device_mapping": next(iter(records.values()))["experiment"].get("device_mapping"),
        "imported_at": utc_now(),
    }
    temp_file = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".importing",
        delete=False,
    )
    try:
        with temp_file as handle:
            for sample_id in source_ids:
                if sample_id not in records:
                    continue
                record = dict(records[sample_id])
                record["experiment"] = experiment
                record["imported_from"] = imported_from
                handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_file.name, output_path)
    except BaseException:
        Path(temp_file.name).unlink(missing_ok=True)
        raise
    statuses = Counter(str(record.get("status")) for record in records.values())
    print(
        f"[extend] Imported {len(records)} record(s) {dict(statuses)} from {source_path} "
        f"(n{source_max_samples}) diagnostics={diagnostics}"
    )


def report_shared_gpus(devices: dict[str, str]) -> None:
    for device_value in sorted(set(devices.values())):
        sharing = [name for name in MODEL_NAMES if devices[name] == device_value]
        if len(sharing) > 1:
            print(
                f"[device] {', '.join(sharing)} share {device_value}; their worker processes "
                "are time-sliced on that GPU"
            )


def build_worker_specs(
    args: argparse.Namespace, model_paths: dict[str, str], devices: dict[str, str]
) -> list[WorkerSpec]:
    disabled = set(args.disable_candidate_cache_for or [])
    return [
        WorkerSpec(
            name=name,
            factory=factory,
            model_path=model_paths[name],
            device=devices[name],
            init_kwargs=(
                {"max_new_tokens": args.max_new_tokens} if name == "MiniCPM-V" else {}
            ),
            disable_candidate_cache=name in disabled,
            seed=args.seed,
        )
        for name, factory in ADAPTER_FACTORIES.items()
    ]


def reset_worker_peak_memory(models: dict[str, Any]) -> None:
    for adapter in models.values():
        adapter.reset_peak_memory()


def collect_worker_peak_memory(models: dict[str, Any]) -> dict[str, dict[str, Any]]:
    peaks: dict[str, dict[str, Any]] = {}
    for adapter in models.values():
        peaks.update(adapter.collect_peak_memory())
    return peaks


def run(benchmark: Benchmark, args: argparse.Namespace, config: RuntimeConfig) -> None:
    if args.max_new_tokens is None:
        args.max_new_tokens = benchmark.max_new_tokens
    validate_args(args, config)
    dataset_path = resolve_dataset_path(benchmark, args, config)
    model_paths = resolve_model_paths(config)
    stats_dir = config.stats_dir.resolve()

    output_path = resolve_output_path(benchmark, args, config)
    log_dir = (config.logs_dir / benchmark.name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / build_log_filename(benchmark, args)
    install_log_tee(log_path)
    print(f"[log] Terminal output is mirrored to {log_path}")

    torch.cuda.empty_cache()
    gc.collect()
    set_global_seed(args.seed)
    print(f"[data] Loading {benchmark.name} ({benchmark.split}): {dataset_path}")
    complete_dataset, dataset_details = benchmark.load_dataset(dataset_path)
    all_columns = set(complete_dataset.column_names)
    selected_dataset = select_dataset_rows(complete_dataset, benchmark, args)
    dataset = inference_only_view(selected_dataset, benchmark)
    del selected_dataset
    del complete_dataset

    first_image = decode_image(dataset[0][benchmark.image_field])
    first_image.close()
    print(
        f"[data] Loaded {len(dataset)} selected samples ({dataset_details['rows']} total); "
        f"strategy={dataset_details['strategy']} inference_columns={dataset.column_names}"
    )

    experiment = build_experiment(
        benchmark, args, config, dataset_path, model_paths, dataset_details
    )
    selected_ids = [str(value) for value in dataset[benchmark.id_field]]
    if args.extend:
        import_extend_source(benchmark, args, output_path, experiment, selected_ids)
    existing, existing_diagnostics = read_latest_records(output_path, benchmark.id_field)
    validate_existing_experiments(existing, experiment)
    print(
        f"[resume] output={output_path} existing_unique={len(existing)} "
        f"diagnostics={existing_diagnostics}"
    )

    unexpected_existing = sorted(set(existing).difference(selected_ids))
    if unexpected_existing:
        raise ValueError(
            f"Output contains ID(s) outside the selected dataset: {unexpected_existing[:10]}"
        )
    skip_ids = {
        sample_id
        for sample_id, record in existing.items()
        if record.get("status") == "success" or not args.retry_errors
    }
    pending_indices = [
        index for index, sample_id in enumerate(selected_ids) if sample_id not in skip_ids
    ]
    print(
        f"[run] selected={len(dataset)} skipped={len(dataset) - len(pending_indices)} "
        f"pending={len(pending_indices)} output={output_path}"
    )
    if not pending_indices:
        print(f"[done] No pending {benchmark.name} samples")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the three-model relay")
    devices = experiment["device_mapping"]
    requested_indices = [cuda_index(devices[name]) for name in MODEL_NAMES]
    device_count = torch.cuda.device_count()
    if max(requested_indices) >= device_count:
        raise RuntimeError(
            f"Requested logical CUDA devices {requested_indices}, but only "
            f"{device_count} device(s) are visible"
        )

    print(
        "[model] Starting one worker process per expert (FlashAttention-2); "
        "the three models load in parallel"
    )
    report_shared_gpus(devices)
    models = start_model_workers(build_worker_specs(args, model_paths, devices))
    try:
        for model_name in args.disable_candidate_cache_for or []:
            print(
                f"[optcache] {model_name} candidate-KV promotion disabled by CLI; "
                "static input caching remains active"
            )
        relay = MACRRelay(
            models_dict=models,
            window_size=args.window_size,
            initial_trial_windows=args.initial_trial_windows,
            z_threshold=args.z_threshold,
            stats_dir=str(stats_dir),
        )
        run_pending_samples(
            benchmark,
            args,
            relay,
            models,
            dataset,
            selected_ids,
            pending_indices,
            experiment,
            output_path,
            existing_diagnostics,
            forbidden_keys=all_columns.difference(benchmark.inference_fields),
        )
    finally:
        shutdown_workers(models)

    print(f"[done] Raw responses saved to {output_path}")
    print(f"[log] Run log saved to {log_path}")
    print(f"[evaluate] Add the file to INPUT_FILES in eva_{benchmark.name}.py and run it")


def run_pending_samples(
    benchmark: Benchmark,
    args: argparse.Namespace,
    relay: MACRRelay,
    models: dict[str, Any],
    dataset: Any,
    selected_ids: list[str],
    pending_indices: list[int],
    experiment: dict[str, Any],
    output_path: Path,
    existing_diagnostics: dict[str, Any],
    *,
    forbidden_keys: set[str],
) -> None:
    from tqdm import tqdm

    with output_path.open("a", encoding="utf-8") as output_handle:
        if output_path.stat().st_size and existing_diagnostics.get("ignored_partial_tail"):
            output_handle.write("\n")
            output_handle.flush()

        with tempfile.TemporaryDirectory(prefix=f"{benchmark.name}_relay_") as temp_dir:
            for completed_index, dataset_index in enumerate(
                tqdm(pending_indices, desc=f"{benchmark.name} bidding relay"), start=1
            ):
                sample_id = selected_ids[dataset_index]
                sample_started = time.perf_counter()
                relay.reset_sample_state()
                image: Image.Image | None = None
                problem = dataset[dataset_index]
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    benchmark.id_field: problem[benchmark.id_field],
                    **{field: problem.get(field) for field in benchmark.record_fields},
                    "status": "error",
                    "eval_prompt": None,
                    "eval_prompt_sha256": None,
                    "response": "",
                    "error": None,
                    "experiment": experiment,
                    "started_at": utc_now(),
                }
                reset_worker_peak_memory(models)
                worker_died = False
                try:
                    eval_prompt = benchmark.build_prompt(problem)
                    record["eval_prompt"] = eval_prompt
                    record["eval_prompt_sha256"] = sha256_text(eval_prompt)
                    image = decode_image(problem[benchmark.image_field])
                    record["image_meta"] = {
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                        "original_pixels": image.width * image.height,
                        "temporary_format": "PNG",
                    }
                    image_path = Path(temp_dir) / f"sample_{dataset_index:06d}.png"
                    image.save(image_path, format="PNG")
                    response = relay.generate_response(
                        image_path=str(image_path),
                        user_prompt=eval_prompt,
                        max_tokens=args.max_new_tokens,
                    )
                    record["response"] = response
                    record["status"] = "success"
                except WorkerDiedError:
                    worker_died = True
                    raise
                except Exception as error:
                    record["error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    print(f"\n[error] {benchmark.id_field}={sample_id}: {type(error).__name__}: {error}")
                finally:
                    if image is not None:
                        image.close()
                    if not worker_died:
                        record["relay_stats"] = relay.last_relay_stats
                        record["usage"] = relay.last_usage
                        record["ori_input_token"] = relay.last_ori_input_token
                        record["cache_stats"] = relay.last_cache_stats
                        timing = dict(relay.last_timing)
                        timing["sample_total_seconds"] = time.perf_counter() - sample_started
                        timing["total_seconds"] = timing["sample_total_seconds"]
                        record["timing"] = timing
                        record["gpu"] = collect_worker_peak_memory(models)
                        record["recorded_at"] = utc_now()

                        leaked_keys = sorted(forbidden_keys.intersection(record))
                        if leaked_keys:
                            raise RuntimeError(f"Evaluation-only fields leaked into record: {leaked_keys}")
                        output_handle.write(
                            json.dumps(record, ensure_ascii=False, default=json_default) + "\n"
                        )
                        output_handle.flush()

                        if args.cleanup_every and completed_index % args.cleanup_every == 0:
                            gc.collect()
