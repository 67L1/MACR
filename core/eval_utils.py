from __future__ import annotations

import json
import math
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MODEL_ORDER = ("Qwen3-VL", "InternVL", "MiniCPM-V")
OUTPUT_ALIASES = ("response", "pred", "model_output", "output")
ORI_INPUT_FIELDS = ("total", "image", "text")


def dataset_path(name: str) -> Path:
    from main import DATASET_PATHS

    value = str(DATASET_PATHS.get(name, "")).strip()
    if not value:
        raise ValueError(f'Dataset path is empty: fill DATASET_PATHS["{name}"] at the top of main.py')
    return Path(value).expanduser().resolve()


def resolve_input_files(input_files: Sequence[str], dataset_name: str) -> list[Path]:
    if input_files:
        from main import PROJECT_ROOT

        paths = [(Path(PROJECT_ROOT) / Path(value).expanduser()).resolve() for value in input_files]
    else:
        from main import RESULTS_DIR

        result_dir = Path(RESULTS_DIR) / dataset_name
        paths = sorted(result_dir.glob("*.jsonl"))
        if not paths:
            raise ValueError(
                f"INPUT_FILES is empty and no *.jsonl found in {result_dir}; "
                "add response files to INPUT_FILES"
            )
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def record_id(record: Mapping[str, Any], id_aliases: Sequence[str]) -> str:
    for key in id_aliases:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise KeyError(f"record has none of the ID fields: {tuple(id_aliases)}")


def read_jsonl_snapshot(
    path: Path, id_aliases: Sequence[str]
) -> tuple[OrderedDict[str, dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Response JSONL not found: {path}")

    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    ignored_partial_tail = False
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        ignored_partial_tail = True
        lines = lines[:-1]

    latest: OrderedDict[str, dict[str, Any]] = OrderedDict()
    malformed_lines: list[int] = []
    duplicate_records = 0
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
            if not isinstance(record, dict):
                raise TypeError("JSON value is not an object")
            sample_id = record_id(record, id_aliases)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            malformed_lines.append(line_number)
            continue
        if sample_id in latest:
            duplicate_records += 1
        latest[sample_id] = record

    diagnostics = {
        "file_size_bytes": len(raw),
        "complete_physical_lines": len(lines),
        "ignored_partial_tail": ignored_partial_tail,
        "malformed_complete_lines": malformed_lines,
        "duplicate_records": duplicate_records,
    }
    return latest, diagnostics


def first_n(records: "OrderedDict[Any, dict[str, Any]]", max_eval: int | None) -> "OrderedDict[Any, dict[str, Any]]":
    if max_eval is not None and max_eval <= 0:
        raise ValueError("MAX_EVAL must be a positive integer or None")
    kept: OrderedDict[Any, dict[str, Any]] = OrderedDict()
    for n, (key, record) in enumerate(records.items()):
        if max_eval is not None and n >= max_eval:
            break
        kept[key] = record
    return kept


def response_text(record: Mapping[str, Any]) -> tuple[str, str | None]:
    for key in OUTPUT_ALIASES:
        value = record.get(key)
        if isinstance(value, str):
            return value, key
    return "", None


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _first_int(mapping: Mapping[str, Any], aliases: Sequence[str]) -> int | None:
    for key in aliases:
        value = _safe_int(mapping.get(key))
        if value is not None:
            return value
    return None


def generated_token_counts(record: Mapping[str, Any]) -> tuple[int | None, int | None]:
    usage = record.get("usage")
    combined = dict(usage) if isinstance(usage, dict) else {}
    for key in ("generated_tokens_raw", "generated_tokens", "output_tokens"):
        if key not in combined and key in record:
            combined[key] = record[key]
    generated = _first_int(combined, ("generated_tokens_raw", "generated_tokens", "output_tokens"))
    attempted = _first_int(
        combined, ("attempted_generated_tokens_raw", "attempted_tokens", "attempted")
    )
    committed = _first_int(
        combined, ("committed_generated_tokens_raw", "committed_tokens", "committed")
    )
    if attempted is None and generated is not None:
        attempted = generated
    if committed is None and generated is not None:
        committed = generated
    if committed is None:
        relay = record.get("relay_stats")
        if isinstance(relay, dict):
            committed = _safe_int(relay.get("generated_tokens"))
    return attempted, committed


def sample_latency(record: Mapping[str, Any]) -> float | None:
    timing = record.get("timing")
    if not isinstance(timing, dict):
        return None
    for key in ("sample_total_seconds", "total_seconds"):
        value = _safe_float(timing.get(key))
        if value is not None:
            return value
    return None


def _stat(total: float, count: int) -> dict[str, Any]:
    return {"total": total, "count": count, "average": total / count if count else None}


def summarize_basic(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    ori_sums: dict[str, dict[str, list[int]]] = {}
    attempted_total = attempted_count = 0
    committed_total = committed_count = 0
    latency_total = 0.0
    latency_count = 0

    for record in records:
        by_model = record.get("ori_input_token")
        if isinstance(by_model, dict):
            for model_name, counts in by_model.items():
                if not isinstance(counts, dict):
                    continue
                slot = ori_sums.setdefault(
                    str(model_name), {field: [0, 0] for field in ORI_INPUT_FIELDS}
                )
                for field in ORI_INPUT_FIELDS:
                    value = _safe_int(counts.get(field))
                    if value is not None:
                        slot[field][0] += value
                        slot[field][1] += 1

        attempted, committed = generated_token_counts(record)
        if attempted is not None:
            attempted_total += attempted
            attempted_count += 1
        if committed is not None:
            committed_total += committed
            committed_count += 1

        latency = sample_latency(record)
        if latency is not None:
            latency_total += latency
            latency_count += 1

    ordered_models = [name for name in MODEL_ORDER if name in ori_sums] + sorted(
        name for name in ori_sums if name not in MODEL_ORDER
    )
    return {
        "records": len(records),
        "ori_input_token": {
            name: {field: _stat(*ori_sums[name][field]) for field in ORI_INPUT_FIELDS}
            for name in ordered_models
        },
        "attempted": _stat(attempted_total, attempted_count),
        "committed": _stat(committed_total, committed_count),
        "latency": _stat(latency_total, latency_count),
    }


def _fmt(value: float | int | None, decimals: int = 1) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def _count(stat: Mapping[str, Any]) -> str:
    return f"{int(stat['total']):,}" if stat["count"] else "unavailable"


def format_accuracy(correct: int, total: int) -> str:
    accuracy = correct / total if total else 0.0
    return f"{accuracy * 100:.2f}% ({correct}/{total})"


def print_basic_report(
    path: Path,
    evaluated: int,
    expected: int | None,
    acc_line: str,
    basic: Mapping[str, Any],
    note: str | None = None,
) -> None:
    expected_text = "?" if expected is None else str(expected)
    print(f"\n--- {path}  (n={evaluated}/{expected_text}) ---")
    if note:
        print(f"  [{note}]")
    print(f"ACC: {acc_line}")

    ori = basic["ori_input_token"]
    if ori:
        print("ori_input_token (input, total | avg per sample):")
        width = max(len(name) for name in [*ori, "ALL"])
        all_totals = {field: 0 for field in ORI_INPUT_FIELDS}
        for name, fields in ori.items():
            totals = "  ".join(
                f"{field}={_count(fields[field])}" for field in ORI_INPUT_FIELDS
            )
            averages = "  ".join(
                f"{field}={_fmt(fields[field]['average'])}" for field in ORI_INPUT_FIELDS
            )
            print(f"  {name:<{width}} : {totals}  | avg {averages}")
            for field in ORI_INPUT_FIELDS:
                all_totals[field] += fields[field]["total"]
        totals = "  ".join(f"{field}={all_totals[field]:,}" for field in ORI_INPUT_FIELDS)
        print(f"  {'ALL':<{width}} : {totals}")
    else:
        print("ori_input_token (input): unavailable")

    attempted = basic["attempted"]
    committed = basic["committed"]
    if attempted["count"] or committed["count"]:
        discarded = (
            f"{attempted['total'] - committed['total']:,}"
            if attempted["count"] == committed["count"] and attempted["count"]
            else "unavailable"
        )
        print(
            f"Generated tokens (output, total): attempted={_count(attempted)}  "
            f"committed={_count(committed)}  discarded={discarded}  "
            f"| avg attempted={_fmt(attempted['average'])} committed={_fmt(committed['average'])}"
        )
    else:
        print("Generated tokens (output): unavailable")

    latency = basic["latency"]
    if latency["count"]:
        print(f"Latency (s/sample): avg={_fmt(latency['average'], 2)} total={_fmt(latency['total'], 2)}")
    else:
        print("Latency (s/sample): unavailable")


def basic_overall_metrics(basic: Mapping[str, Any]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for name, fields in basic["ori_input_token"].items():
        metrics[f"ori_input_token {name} avg total"] = fields["total"]["average"]
    metrics["Attempted tokens avg/sample"] = basic["attempted"]["average"]
    metrics["Committed tokens avg/sample"] = basic["committed"]["average"]
    metrics["Latency avg (s/sample)"] = basic["latency"]["average"]
    return metrics


def print_overall(results: Sequence[tuple[Path, Mapping[str, float | None]]]) -> None:
    if len(results) <= 1:
        return
    labels: list[str] = []
    for _, metrics in results:
        for label in metrics:
            if label not in labels:
                labels.append(label)
    print("\n==============================")
    print(f"--- Overall ({len(results)} files) ---")
    print("==============================")
    for label in labels:
        values = [
            float(metrics[label])
            for _, metrics in results
            if metrics.get(label) is not None
        ]
        if not values:
            print(f"{label}: unavailable")
            continue
        print(
            f"{label}: mean={statistics.fmean(values):.4f} "
            f"std={statistics.pstdev(values):.4f}"
        )
    print("==============================")
