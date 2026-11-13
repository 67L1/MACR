from __future__ import annotations

INPUT_FILES: list = []
MAX_EVAL = None
VERBOSE = False

import hashlib
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

from core import eval_utils as eu

try:
    import pyarrow.parquet as pq
except ImportError as error:
    raise ImportError(
        "eva_hallusionbench.py requires pyarrow; install it with "
        "`pip install pyarrow`"
    ) from error


SPLIT = "image"
EXPECTED_IMAGE_ROWS = 951

IDENTITY_FIELDS = (
    "category",
    "subcategory",
    "set_id",
    "figure_id",
    "question_id",
)
GROUND_TRUTH_COLUMNS = (
    "category",
    "subcategory",
    "visual_input",
    "set_id",
    "figure_id",
    "sample_note",
    "question_id",
    "question",
    "gt_answer_details",
    "gt_answer",
    "filename",
)
QUESTION_GROUP_FIELDS = (
    "category",
    "subcategory",
    "set_id",
    "question_id",
)
FIGURE_GROUP_FIELDS = (
    "category",
    "subcategory",
    "set_id",
    "figure_id",
)

SampleKey = tuple[str, str, str, str, str]


def _as_identity_text(value: Any) -> str:
    if value is None:
        raise ValueError("identity value is null")
    return str(value).strip()


def _sample_key(record: dict[str, Any]) -> SampleKey:
    return tuple(_as_identity_text(record[field]) for field in IDENTITY_FIELDS)


def load_ground_truth(
    path: Path,
) -> tuple[OrderedDict[SampleKey, dict[str, Any]], list[SampleKey], str]:
    if not path.is_file():
        raise FileNotFoundError(f"HallusionBench parquet not found: {path}")

    schema_names = set(pq.read_schema(path).names)
    missing_columns = [name for name in GROUND_TRUTH_COLUMNS if name not in schema_names]
    if missing_columns:
        raise ValueError(f"Ground truth is missing columns: {missing_columns}")

    rows = pq.read_table(path, columns=list(GROUND_TRUTH_COLUMNS)).to_pylist()
    ground_truth: OrderedDict[SampleKey, dict[str, Any]] = OrderedDict()
    row_keys: list[SampleKey] = []
    fingerprint = hashlib.sha256()

    for row_index, raw_row in enumerate(rows):
        row = {
            field: "" if raw_row[field] is None else str(raw_row[field]).strip()
            for field in GROUND_TRUTH_COLUMNS
        }
        try:
            key = _sample_key(row)
        except ValueError as error:
            raise ValueError(f"Invalid identity at parquet row {row_index}: {error}") from error
        if any(not value for value in key):
            raise ValueError(f"Empty identity field at parquet row {row_index}: {key}")
        if key in ground_truth:
            raise ValueError(f"Duplicate ground-truth identity at row {row_index}: {key}")
        if row["gt_answer"] not in {"0", "1"}:
            raise ValueError(
                f"Invalid gt_answer at parquet row {row_index}: {row['gt_answer']!r}"
            )
        if row["visual_input"] not in {"1", "2"}:
            raise ValueError(
                "Expected the HallusionBench image split, but parquet row "
                f"{row_index} has visual_input={row['visual_input']!r}"
            )

        row["row_index"] = row_index
        ground_truth[key] = row
        row_keys.append(key)
        canonical = json.dumps(
            {field: row[field] for field in GROUND_TRUTH_COLUMNS},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint.update(canonical.encode("utf-8"))
        fingerprint.update(b"\n")

    if not ground_truth:
        raise ValueError("Ground-truth parquet contains no rows")
    return ground_truth, row_keys, fingerprint.hexdigest()


def _parse_pid(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean pid is invalid")
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"pid is not an integer: {value!r}")
    return int(text)


def resolve_record_key(
    record: dict[str, Any],
    ground_truth: OrderedDict[SampleKey, dict[str, Any]],
    row_keys: Sequence[SampleKey],
) -> tuple[SampleKey | None, str | None]:
    pid_key: SampleKey | None = None
    if "pid" in record:
        try:
            pid = _parse_pid(record["pid"])
        except ValueError:
            return None, "invalid_pid"
        if pid < 0 or pid >= len(row_keys):
            return None, "unknown_pid"
        pid_key = row_keys[pid]

    present_fields = [field for field in IDENTITY_FIELDS if field in record]
    composite_key: SampleKey | None = None
    if len(present_fields) == len(IDENTITY_FIELDS):
        try:
            composite_key = _sample_key(record)
        except ValueError:
            return None, "invalid_composite_identity"
        if composite_key not in ground_truth:
            return None, "unknown_composite_identity"
    elif present_fields and pid_key is None:
        return None, "incomplete_composite_identity"

    if pid_key is None and composite_key is None:
        return None, "missing_identity"
    if pid_key is not None and composite_key is not None and pid_key != composite_key:
        return None, "identity_conflict"

    key = composite_key if composite_key is not None else pid_key
    assert key is not None

    if composite_key is None and present_fields:
        expected = ground_truth[key]
        for field in present_fields:
            try:
                actual_value = _as_identity_text(record[field])
            except ValueError:
                return None, "invalid_partial_identity"
            if actual_value != expected[field]:
                return None, "identity_conflict"
    return key, None


def read_jsonl_snapshot(
    path: Path,
    ground_truth: OrderedDict[SampleKey, dict[str, Any]],
    row_keys: Sequence[SampleKey],
) -> tuple[OrderedDict[SampleKey, dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Response JSONL not found: {path}")
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    ignored_partial_tail = False
    if lines and not lines[-1].endswith(b"\n"):
        ignored_partial_tail = True
        lines = lines[:-1]

    latest: OrderedDict[SampleKey, dict[str, Any]] = OrderedDict()
    malformed_lines: list[int] = []
    identity_errors: Counter[str] = Counter()
    identity_error_lines: dict[str, list[int]] = {}
    duplicate_records = 0

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_lines.append(line_number)
            continue
        if not isinstance(record, dict):
            malformed_lines.append(line_number)
            continue

        key, error_kind = resolve_record_key(record, ground_truth, row_keys)
        if error_kind is not None:
            identity_errors[error_kind] += 1
            identity_error_lines.setdefault(error_kind, []).append(line_number)
            continue
        assert key is not None
        if key in latest:
            duplicate_records += 1
            del latest[key]
        latest[key] = record

    diagnostics = {
        "file_size_bytes": len(raw),
        "complete_physical_lines": len(lines),
        "ignored_partial_tail": ignored_partial_tail,
        "malformed_complete_lines": malformed_lines,
        "identity_errors": dict(identity_errors),
        "identity_error_lines": identity_error_lines,
        "duplicate_records": duplicate_records,
    }
    return latest, diagnostics


ANSWER_TOKEN = r"(?:yes|no|0|1)"
BOXED_ANSWER_RE = re.compile(
    rf"\\boxed\s*\{{\s*(?P<answer>{ANSWER_TOKEN})\s*\}}", re.IGNORECASE
)
LABELED_ANSWER_RE = re.compile(
    rf"[*_`]*\s*(?:final\s+answer|answer|答案)\s*[*_`]*"
    rf"\s*(?:is|为)?\s*[*_`:：=\-]*\s*[*_`]*"
    rf"(?P<answer>{ANSWER_TOKEN})(?![\w])[*_`]*",
    re.IGNORECASE,
)
DIRECT_ANSWER_RE = re.compile(
    rf"^\s*(?P<answer>{ANSWER_TOKEN})\s*[.!。！]?\s*$", re.IGNORECASE
)
LEADING_YES_NO_RE = re.compile(
    r"^\s*(?P<answer>yes|no)(?=\s|[,，.:：;；!！。]|$)", re.IGNORECASE
)
REASONING_CLOSE_RE = re.compile(r"</(?:think|analysis|reasoning)\s*>", re.IGNORECASE)


def _normalize_answer_token(token: str) -> str:
    return "1" if token.strip().lower() in {"1", "yes"} else "0"


def extract_binary_answer(response: Any) -> dict[str, Any]:
    if not isinstance(response, str) or not response.strip():
        return {
            "prediction": None,
            "status": "failed",
            "method": "empty_response",
            "candidate": None,
            "warnings": ["response is empty or is not a string"],
        }

    text = response.strip()
    close_matches = list(REASONING_CLOSE_RE.finditer(text))
    candidate = text[close_matches[-1].end() :].strip() if close_matches else text
    if not candidate:
        return {
            "prediction": None,
            "status": "failed",
            "method": "empty_final_segment",
            "candidate": candidate,
            "warnings": ["no text follows the final reasoning tag"],
        }

    explicit_matches: list[tuple[int, str, str]] = []
    for method, pattern in (
        ("boxed", BOXED_ANSWER_RE),
        ("labeled", LABELED_ANSWER_RE),
    ):
        for match in pattern.finditer(candidate):
            explicit_matches.append(
                (match.start(), _normalize_answer_token(match.group("answer")), method)
            )

    if explicit_matches:
        values = {value for _, value, _ in explicit_matches}
        if len(values) != 1:
            return {
                "prediction": None,
                "status": "failed",
                "method": "ambiguous_explicit_answers",
                "candidate": candidate,
                "warnings": ["conflicting explicit final answers were found"],
            }
        explicit_matches.sort(key=lambda item: item[0])
        _, prediction, method = explicit_matches[-1]
        return {
            "prediction": prediction,
            "status": "success",
            "method": method,
            "candidate": candidate,
            "warnings": [],
        }

    unwrapped_candidate = candidate.strip().strip("*_`").strip()
    direct_match = DIRECT_ANSWER_RE.fullmatch(unwrapped_candidate)
    if direct_match:
        return {
            "prediction": _normalize_answer_token(direct_match.group("answer")),
            "status": "success",
            "method": "direct",
            "candidate": candidate,
            "warnings": [],
        }

    leading_match = LEADING_YES_NO_RE.match(candidate)
    if leading_match:
        return {
            "prediction": _normalize_answer_token(leading_match.group("answer")),
            "status": "success",
            "method": "leading_yes_no",
            "candidate": candidate,
            "warnings": [],
        }

    method = "uncertain" if re.fullmatch(r"\s*uncertain\s*[.!]?\s*", candidate, re.I) else "no_explicit_answer"
    return {
        "prediction": None,
        "status": "failed",
        "method": method,
        "candidate": candidate,
        "warnings": ["no unambiguous explicit 0/1 or yes/no answer was found"],
    }


def failed_extraction(reason: str, method: str) -> dict[str, Any]:
    return {
        "prediction": None,
        "status": "failed",
        "method": method,
        "candidate": None,
        "warnings": [reason],
    }


def _metric(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def aggregate_accuracy(
    rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    if group_fields is None:
        return _metric(sum(bool(row["correct"]) for row in rows), len(rows))

    groups: OrderedDict[tuple[str, ...], list[bool]] = OrderedDict()
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(bool(row["correct"]))
    correct = sum(all(values) for values in groups.values())
    return _metric(correct, len(groups))


def build_official_scores(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "aAcc": aggregate_accuracy(rows),
        "qAcc": aggregate_accuracy(rows, QUESTION_GROUP_FIELDS),
        "fAcc": aggregate_accuracy(rows, FIGURE_GROUP_FIELDS),
    }


def build_grouped_scores(
    rows: Sequence[dict[str, Any]], field: str
) -> OrderedDict[str, dict[str, dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return OrderedDict((name, build_official_scores(items)) for name, items in groups.items())


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _accumulate_token_group(
    destination: dict[str, dict[str, int]], source: Any
) -> None:
    if not isinstance(source, dict):
        return
    for group_name, values in source.items():
        if not isinstance(values, dict):
            continue
        group = destination.setdefault(
            str(group_name), {"attempted": 0, "committed": 0, "discarded": 0}
        )
        for field in ("attempted", "committed", "discarded"):
            group[field] += _integer(values.get(field, 0))


def summarize_usage(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    attempted = 0
    committed = 0
    discarded = 0
    invariant_violations = 0
    available = 0
    by_model: dict[str, dict[str, int]] = {}
    by_phase: dict[str, dict[str, int]] = {}

    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        available += 1
        record_attempted = _integer(usage.get("attempted_generated_tokens_raw", 0))
        record_committed = _integer(usage.get("committed_generated_tokens_raw", 0))
        record_discarded = _integer(usage.get("discarded_generated_tokens_raw", 0))
        attempted += record_attempted
        committed += record_committed
        discarded += record_discarded
        if record_attempted != record_committed + record_discarded:
            invariant_violations += 1
        _accumulate_token_group(by_model, usage.get("by_model"))
        _accumulate_token_group(by_phase, usage.get("by_phase"))

    count = len(records)
    return {
        "available": available,
        "attempted": attempted,
        "committed": committed,
        "discarded": discarded,
        "overhead_ratio": attempted / committed if committed else 0.0,
        "average_attempted": attempted / count if count else 0.0,
        "average_committed": committed / count if count else 0.0,
        "average_discarded": discarded / count if count else 0.0,
        "invariant_violations": invariant_violations,
        "by_model": by_model,
        "by_phase": by_phase,
    }


def summarize_relay(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    available = 0
    questions_with_handoff = 0
    handoffs = 0
    triggers = 0
    initial_models: Counter[str] = Counter()
    final_models: Counter[str] = Counter()
    for record in records:
        relay = record.get("relay_stats")
        if not isinstance(relay, dict):
            continue
        available += 1
        handoff_count = _integer(relay.get("handoff_count", 0))
        handoffs += handoff_count
        triggers += _integer(relay.get("zscore_trigger_count", 0))
        if handoff_count:
            questions_with_handoff += 1
        if relay.get("initial_model"):
            initial_models[str(relay["initial_model"])] += 1
        if relay.get("final_model"):
            final_models[str(relay["final_model"])] += 1
    return {
        "available": available,
        "questions_with_handoff": questions_with_handoff,
        "handoffs": handoffs,
        "zscore_triggers": triggers,
        "average_handoffs": handoffs / available if available else 0.0,
        "initial_models": dict(initial_models),
        "final_models": dict(final_models),
    }


def summarize_timing(
    records: Iterable[dict[str, Any]], attempted: int, committed: int
) -> dict[str, Any]:
    fields = (
        "preprocess_seconds",
        "model_generation_seconds",
        "initial_selection_wall_seconds",
        "bidding_wall_seconds",
        "response_wall_seconds",
        "sample_total_seconds",
    )
    totals = {field: 0.0 for field in fields}
    available = 0
    for record in records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        available += 1
        for field in fields:
            totals[field] += _number(timing.get(field, 0.0))
    totals["available"] = available
    totals["average_sample_total_seconds"] = (
        totals["sample_total_seconds"] / available if available else 0.0
    )
    totals["attempted_tokens_per_model_generation_second"] = (
        attempted / totals["model_generation_seconds"]
        if totals["model_generation_seconds"]
        else 0.0
    )
    totals["committed_tokens_per_response_second"] = (
        committed / totals["response_wall_seconds"]
        if totals["response_wall_seconds"]
        else 0.0
    )
    return totals


def evaluate_records(
    raw_records: OrderedDict[SampleKey, dict[str, Any]],
    ground_truth: OrderedDict[SampleKey, dict[str, Any]],
    partial: bool = False,
) -> dict[str, Any]:
    scored_rows: list[dict[str, Any]] = []
    extraction_statuses: Counter[str] = Counter()
    extraction_methods: Counter[str] = Counter()
    generation_statuses: Counter[str] = Counter()

    scored_items = (
        [(key, ground_truth[key]) for key in raw_records]
        if partial
        else list(ground_truth.items())
    )
    for key, problem in scored_items:
        raw_record = raw_records.get(key)
        if raw_record is None:
            extraction = failed_extraction("response record is missing", "missing_record")
            generation_status = "missing"
            response = ""
        else:
            generation_status = str(raw_record.get("status", "missing"))
            response = raw_record.get("response", "")
            if generation_status.lower() == "success":
                extraction = extract_binary_answer(response)
            else:
                extraction = failed_extraction(
                    "generation record is not successful", "generation_error"
                )

        prediction = extraction["prediction"]
        scored = dict(problem)
        scored.update(
            {
                "response": response,
                "prediction": prediction,
                "correct": prediction == problem["gt_answer"],
                "generation_status": generation_status,
                "extraction_status": extraction["status"],
                "extraction_method": extraction["method"],
            }
        )
        scored_rows.append(scored)
        generation_statuses[generation_status] += 1
        extraction_statuses[extraction["status"]] += 1
        extraction_methods[extraction["method"]] += 1

    records = list(raw_records.values())
    usage = summarize_usage(records)
    return {
        "dataset_total": len(ground_truth),
        "known_records": len(raw_records),
        "coverage": len(raw_records) / len(ground_truth) if ground_truth else 0.0,
        "provisional": not partial and len(raw_records) != len(ground_truth),
        "generation_statuses": dict(generation_statuses),
        "extraction_statuses": dict(extraction_statuses),
        "extraction_methods": dict(extraction_methods),
        "official_scores": build_official_scores(scored_rows),
        "by_category": build_grouped_scores(scored_rows, "category"),
        "by_subcategory": build_grouped_scores(scored_rows, "subcategory"),
        "by_visual_input": build_grouped_scores(scored_rows, "visual_input"),
        "relay": summarize_relay(records),
        "usage": usage,
        "timing": summarize_timing(records, usage["attempted"], usage["committed"]),
    }


def _format_metric(metric: dict[str, Any]) -> str:
    return (
        f"{metric['correct']}/{metric['total']} "
        f"({float(metric['accuracy']) * 100:.2f}%)"
    )


def print_score_bundle(scores: dict[str, dict[str, Any]], indent: str = "  ") -> None:
    for metric_name in ("aAcc", "qAcc", "fAcc"):
        print(f"{indent}{metric_name}: {_format_metric(scores[metric_name])}")


def print_grouped_scores(
    label: str, groups: OrderedDict[str, dict[str, dict[str, Any]]]
) -> None:
    print(f"{label}:")
    for group_name, scores in groups.items():
        values = "  ".join(
            f"{metric}={scores[metric]['accuracy'] * 100:.2f}%"
            for metric in ("aAcc", "qAcc", "fAcc")
        )
        print(f"  {group_name}: {values}")


def print_token_groups(label: str, groups: dict[str, dict[str, int]]) -> None:
    print(f"  {label}:")
    if not groups:
        print("    unavailable")
        return
    for name, values in groups.items():
        print(
            f"    {name}: attempted={values['attempted']} "
            f"committed={values['committed']} discarded={values['discarded']}"
        )


def print_verbose(
    path: Path, diagnostics: dict[str, Any], result: dict[str, Any]
) -> None:
    print("[VERBOSE]")
    print(
        f"Snapshot: bytes={diagnostics['file_size_bytes']} "
        f"complete_lines={diagnostics['complete_physical_lines']} "
        f"partial_tail_ignored={diagnostics['ignored_partial_tail']}"
    )
    print(
        f"Malformed complete lines: {len(diagnostics['malformed_complete_lines'])} "
        f"{diagnostics['malformed_complete_lines'][:20]}"
    )
    print(f"Identity errors: {diagnostics['identity_errors']}")
    for error_kind, line_numbers in diagnostics["identity_error_lines"].items():
        print(f"  {error_kind} lines: {line_numbers[:20]}")
    print(f"Duplicate records (latest retained): {diagnostics['duplicate_records']}")
    print(
        f"Coverage: {result['known_records']}/{result['dataset_total']} "
        f"({result['coverage'] * 100:.2f}%)"
    )
    print(f"Generation status: {result['generation_statuses']}")
    print(f"Extraction status: {result['extraction_statuses']}")
    print(f"Extraction methods: {result['extraction_methods']}")

    qualifier = "PROVISIONAL; missing records count as wrong" if result["provisional"] else "complete"
    print(f"Official HallusionBench image metrics ({qualifier}):")
    print_score_bundle(result["official_scores"])
    print_grouped_scores("By category", result["by_category"])
    print_grouped_scores("By subcategory", result["by_subcategory"])
    print_grouped_scores("By visual_input", result["by_visual_input"])

    relay = result["relay"]
    print("Relay stats:")
    print(f"  Records with stats: {relay['available']}")
    print(f"  Questions with handoff: {relay['questions_with_handoff']}")
    print(f"  Total handoffs: {relay['handoffs']}")
    print(f"  Z-score triggers: {relay['zscore_triggers']}")
    print(f"  Average handoffs/record with stats: {relay['average_handoffs']:.4f}")
    print(f"  Initial models: {relay['initial_models']}")
    print(f"  Final models: {relay['final_models']}")

    usage = result["usage"]
    print("Output token usage:")
    print(f"  Records with usage: {usage['available']}")
    print(f"  Attempted: {usage['attempted']}")
    print(f"  Committed: {usage['committed']}")
    print(f"  Discarded: {usage['discarded']}")
    print(f"  Compute overhead ratio: {usage['overhead_ratio']:.4f}x")
    print(f"  Average attempted/matched record: {usage['average_attempted']:.2f}")
    print(f"  Average committed/matched record: {usage['average_committed']:.2f}")
    print(f"  Average discarded/matched record: {usage['average_discarded']:.2f}")
    print(f"  Accounting invariant violations: {usage['invariant_violations']}")
    print_token_groups("By model", usage["by_model"])
    print_token_groups("By phase", usage["by_phase"])

    timing = result["timing"]
    print("Timing:")
    print(f"  Records with timing: {timing['available']}")
    print(f"  Preprocess seconds (summed): {timing['preprocess_seconds']:.4f}")
    print(f"  Model generation seconds (summed): {timing['model_generation_seconds']:.4f}")
    print(
        "  Initial selection wall seconds (summed): "
        f"{timing['initial_selection_wall_seconds']:.4f}"
    )
    print(f"  Bidding wall seconds (summed): {timing['bidding_wall_seconds']:.4f}")
    print(f"  Response wall seconds (summed): {timing['response_wall_seconds']:.4f}")
    print(f"  Sample total seconds (summed): {timing['sample_total_seconds']:.4f}")
    print(f"  Average sample total seconds: {timing['average_sample_total_seconds']:.4f}")
    print(
        "  Attempted tokens/model-generation-second: "
        f"{timing['attempted_tokens_per_model_generation_second']:.4f}"
    )
    print(
        "  Committed tokens/response-second: "
        f"{timing['committed_tokens_per_response_second']:.4f}"
    )


DATASET = "hallusionbench"


def resolve_parquet(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.rglob(f"{SPLIT}-*.parquet"))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(
            f"Expected exactly one {SPLIT}-*.parquet under {path}, found {len(matches)}"
        )
    raise FileNotFoundError(f"HallusionBench parquet not found: {path}")


def main() -> None:
    parquet_path = resolve_parquet(eu.dataset_path(DATASET))
    ground_truth, row_keys, fingerprint = load_ground_truth(parquet_path)
    if VERBOSE:
        print(f"[ground truth] split={SPLIT} rows={len(ground_truth)} fingerprint={fingerprint}")
    if len(ground_truth) != EXPECTED_IMAGE_ROWS:
        print(
            f"Warning: expected {EXPECTED_IMAGE_ROWS} rows in the current public image "
            f"split, found {len(ground_truth)}; continuing with the loaded snapshot.",
            file=sys.stderr,
        )

    partial = MAX_EVAL is not None
    processed = []
    for path in eu.resolve_input_files(INPUT_FILES, DATASET):
        try:
            raw_records, diagnostics = read_jsonl_snapshot(path, ground_truth, row_keys)
            raw_records = eu.first_n(raw_records, MAX_EVAL)
            result = evaluate_records(raw_records, ground_truth, partial=partial)
            basic = eu.summarize_basic(raw_records.values())
            scores = result["official_scores"]
            acc_line = "  ".join(
                f"{name}={eu.format_accuracy(scores[name]['correct'], scores[name]['total'])}"
                for name in ("aAcc", "qAcc", "fAcc")
            )
            eu.print_basic_report(
                path,
                evaluated=result["known_records"],
                expected=len(ground_truth),
                acc_line=acc_line,
                basic=basic,
                note=(
                    "PROVISIONAL: missing records counted as wrong"
                    if result["provisional"]
                    else None
                ),
            )
            if VERBOSE:
                print_verbose(path, diagnostics, result)
            processed.append(
                (
                    path,
                    {
                        **{
                            f"{name} (%)": scores[name]["accuracy"] * 100
                            for name in ("aAcc", "qAcc", "fAcc")
                        },
                        **eu.basic_overall_metrics(basic),
                    },
                )
            )
        except FileNotFoundError as error:
            print(f"\nError: {error}. Skipping.")
        except Exception as error:
            print(f"\nError while processing {path}: {type(error).__name__}: {error}")
    if not processed:
        raise RuntimeError("No input files were processed successfully")
    eu.print_overall(processed)


if __name__ == "__main__":
    main()
