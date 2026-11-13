from __future__ import annotations

INPUT_FILES: list = []
MAX_EVAL = None
VERBOSE = False

import ast
import json
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core import eval_utils as eu

DATASET = "logicvista"
EXPECTED_TOTAL = 448
OFFICIAL_UNSCORABLE_IDS = frozenset({"v1_382"})
DATASET_SPLIT = "test"
ID_ALIASES = ("id", "pid", "index")

Answer = frozenset[str]

GROUND_TRUTH_REQUIRED_COLUMNS = {
    "id",
    "question",
    "answer",
    "skill",
    "broad_capability",
    "specific_capability",
}


def parse_options(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(value)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    return []


_ATOM = r"(?:[A-Za-z]|\d+)"
_WRAPPED_ATOM = rf"(?:[\(\[\{{]?\s*({_ATOM})\s*[\)\]\}}]?)"
_SEPARATOR = r"(?:\s*(?:,|，|、|/|;|；|\||&|\band\b|\bor\b|\b和\b)\s*)"
_CHOICE_SEQUENCE = re.compile(
    rf"^\s*{_WRAPPED_ATOM}(?:{_SEPARATOR}{_WRAPPED_ATOM})*",
    flags=re.IGNORECASE,
)
_STANDALONE_ATOM = re.compile(rf"(?<![A-Za-z0-9])({_ATOM})(?![A-Za-z0-9])")


def _normalize_atom(value: Any) -> str | None:
    text = str(value).strip().strip("*`_'\".!,;:：")
    while len(text) >= 2 and (text[0], text[-1]) in {
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
    }:
        text = text[1:-1].strip()
    if re.fullmatch(r"[A-Za-z]", text):
        return text.upper()
    if re.fullmatch(r"\d+", text):
        return text
    return None


def _answer_from_values(values: Iterable[Any]) -> Answer | None:
    normalized: list[str] = []
    for value in values:
        atom = _normalize_atom(value)
        if atom is None:
            return None
        normalized.append(atom)
    return frozenset(normalized) if normalized else None


def _sequence_tokens(text: str, *, require_full: bool) -> Answer | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^(?:\*{1,2}|`)+", "", cleaned).lstrip()
    cleaned = re.sub(
        r"^(?:(?:the\s+)?(?:option|options|choice|choices)\s*)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    match = _CHOICE_SEQUENCE.match(cleaned)
    if not match:
        return None

    remainder = cleaned[match.end() :]
    if require_full:
        remainder = remainder.strip().strip("*`_'\".!,;:：)]}")
        if remainder:
            return None

    tokens = [item.upper() if item.isalpha() else item for item in _STANDALONE_ATOM.findall(match.group(0))]
    return frozenset(tokens) if tokens else None


def normalize_answer(value: Any) -> Answer | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return _answer_from_values(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return None
        return frozenset({str(int(value))})
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return _answer_from_values(parsed)

    boxed = re.fullmatch(r"\\boxed\s*\{(.*)\}", text, flags=re.IGNORECASE | re.DOTALL)
    if boxed:
        text = boxed.group(1)
    return _sequence_tokens(text, require_full=True)


def _select_split(dataset: Any, split: str) -> Any:
    from datasets import DatasetDict

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found; available splits: {list(dataset)}")
        return dataset[split]
    return dataset


def load_ground_truth(
    path: Path,
    split: str,
    expected_total: int = EXPECTED_TOTAL,
) -> tuple[OrderedDict[str, dict[str, Any]], dict[str, Any]]:
    from datasets import DownloadConfig, load_dataset, load_from_disk

    offline_download_config = DownloadConfig(local_files_only=True)

    if not path.exists():
        raise FileNotFoundError(f"LogicVista path not found: {path}")

    source_files: list[Path] = []
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Unsupported dataset file: {path}")
        source_files = [path]
        strategy = "parquet_file"
        dataset = load_dataset(
            "parquet",
            data_files={split: str(path)},
            split=split,
            download_config=offline_download_config,
        )
    else:
        split_parquets = sorted(path.rglob(f"{split}-*.parquet"))
        source_files = split_parquets or sorted(path.rglob("*.parquet"))
        if source_files:
            strategy = "parquet_directory"
            dataset = load_dataset(
                "parquet",
                data_files={split: [str(item) for item in source_files]},
                split=split,
                download_config=offline_download_config,
            )
        else:
            try:
                dataset = _select_split(load_from_disk(str(path)), split)
                strategy = "datasets_load_from_disk"
            except Exception as load_from_disk_error:
                try:
                    dataset = load_dataset(
                        str(path),
                        split=split,
                        download_config=offline_download_config,
                    )
                    strategy = "local_dataset_script"
                except Exception as local_dataset_error:
                    raise RuntimeError(
                        f"Could not load LogicVista from {path}. load_from_disk "
                        f"failed with {type(load_from_disk_error).__name__}: "
                        f"{load_from_disk_error}; local dataset loading failed with "
                        f"{type(local_dataset_error).__name__}: {local_dataset_error}"
                    ) from local_dataset_error

    missing_columns = sorted(
        GROUND_TRUTH_REQUIRED_COLUMNS.difference(dataset.column_names)
    )
    if missing_columns:
        raise ValueError(
            f"LogicVista is missing evaluation columns: {', '.join(missing_columns)}"
        )
    if "image" in dataset.column_names:
        dataset = dataset.remove_columns(["image"])
    if len(dataset) != expected_total:
        raise ValueError(
            f"Expected exactly {expected_total} LogicVista {split} rows, "
            f"found {len(dataset)}"
        )

    ground_truth: OrderedDict[str, dict[str, Any]] = OrderedDict()
    invalid_answers: list[str] = []
    for index in range(len(dataset)):
        row = dataset[index]
        sample_id = str(row["id"])
        if not sample_id.strip():
            raise ValueError(f"LogicVista row {index} has an empty id")
        if sample_id in ground_truth:
            raise ValueError(f"LogicVista contains duplicate id: {sample_id}")
        if normalize_answer(row.get("answer")) is None:
            invalid_answers.append(sample_id)
        ground_truth[sample_id] = {
            "id": sample_id,
            "question": row.get("question", ""),
            "answer": row.get("answer"),
            "skill": row.get("skill", []),
            "broad_capability": row.get("broad_capability", []),
            "specific_capability": row.get("specific_capability", []),
        }

    unexpected_invalid_answers = sorted(
        set(invalid_answers).difference(OFFICIAL_UNSCORABLE_IDS)
    )
    if unexpected_invalid_answers:
        raise ValueError(
            "LogicVista contains unrecognized unscorable answer(s); refusing "
            f"to guess ground truth for ID(s): {unexpected_invalid_answers[:20]}"
        )
    details = {
        "path": str(path),
        "split": split,
        "strategy": strategy,
        "source_files": [str(item.resolve()) for item in source_files],
        "rows": len(ground_truth),
        "scorable_rows": len(ground_truth) - len(invalid_answers),
        "unscorable_ids": invalid_answers,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "columns": list(dataset.column_names),
    }
    return ground_truth, details


def _allowed_choices(record: Mapping[str, Any]) -> set[str] | None:
    options = parse_options(record.get("options"))
    if not 2 <= len(options) <= 26:
        return None
    letters = {chr(ord("A") + index) for index in range(len(options))}
    numbers = {str(index + 1) for index in range(len(options))}
    return letters | numbers


def _validate_prediction(answer: Answer | None, record: Mapping[str, Any]) -> Answer | None:
    if not answer:
        return None
    allowed = _allowed_choices(record)
    if allowed is not None and not answer.issubset(allowed):
        return None
    return answer


_ANSWER_MARKER = re.compile(
    r"(?ix)"
    r"(?:"
    r"final\s+answer\s*(?:is|:|=)?|"
    r"answer\s*(?:is|:|=)|"
    r"correct\s+(?:answer|option|choice)\s*(?:is|:|=)?|"
    r"最终答案\s*(?:是|为|[:：])?|"
    r"答案\s*(?:是|为|[:：])|"
    r"故选|所以(?:选择|选)"
    r")"
)
_LABELED_CHOICE = re.compile(
    r"(?ix)(?:option|options|choice|choices)\s*"
    r"(?:is|are|:|=)?\s*"
)
_BOXED = re.compile(r"\\boxed\s*\{([^{}\r\n]+)\}", flags=re.IGNORECASE)
def _answer_after_position(
    response: str,
    position: int,
    record: Mapping[str, Any],
) -> Answer | None:
    tail = response[position:].lstrip(" \t:：=")
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    first_content = lines[0] if lines else tail.strip()
    first_content = first_content.lstrip(" :：=")
    return _validate_prediction(
        _sequence_tokens(first_content, require_full=False), record
    )


def parse_compatible_answer(
    response: str,
    record: Mapping[str, Any],
) -> tuple[Answer | None, str]:
    if not isinstance(response, str) or not response.strip():
        return None, "missing_output"

    explicit_candidates: list[tuple[int, Answer, str]] = []
    for match in _ANSWER_MARKER.finditer(response):
        answer = _answer_after_position(response, match.end(), record)
        if answer:
            explicit_candidates.append((match.start(), answer, "answer_marker"))

    for match in _BOXED.finditer(response):
        answer = _validate_prediction(
            _sequence_tokens(match.group(1), require_full=True), record
        )
        if answer:
            explicit_candidates.append((match.start(), answer, "boxed_choice"))

    if explicit_candidates:
        _, answer, method = max(explicit_candidates, key=lambda item: item[0])
        return answer, method

    labeled_candidates: list[tuple[int, Answer, str]] = []
    for match in _LABELED_CHOICE.finditer(response):
        answer = _answer_after_position(response, match.end(), record)
        if answer:
            labeled_candidates.append((match.start(), answer, "labeled_choice"))

    if labeled_candidates:
        _, answer, method = max(labeled_candidates, key=lambda item: item[0])
        return answer, method

    stripped = response.strip()
    whole = _validate_prediction(_sequence_tokens(stripped, require_full=True), record)
    if whole:
        return whole, "bare_choice"

    nonempty_lines = [line.strip() for line in response.splitlines() if line.strip()]
    if nonempty_lines:
        final_line = _validate_prediction(
            _sequence_tokens(nonempty_lines[-1], require_full=True), record
        )
        if final_line:
            return final_line, "final_choice_line"

    options = parse_options(record.get("options"))
    if options:
        lowered_response = response.lower()
        option_positions: list[tuple[int, str]] = []
        for index, option in enumerate(options):
            option_text = option.strip().lower()
            if not option_text:
                continue
            position = lowered_response.rfind(option_text)
            if position >= 0:
                option_positions.append((position, chr(ord("A") + index)))
        if option_positions:
            last_position = max(position for position, _ in option_positions)
            last_choices = {
                choice for position, choice in option_positions if position == last_position
            }
            if len(last_choices) == 1:
                return frozenset(last_choices), "option_text"

    return None, "failed"


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


_USAGE_TOP_LEVEL_FIELDS = (
    "input_tokens",
    "prompt_tokens",
    "text_prompt_tokens",
    "generated_tokens_raw",
    "generated_tokens",
    "output_tokens",
    "response_tokens",
    "total_tokens",
    "attempted_generated_tokens_raw",
    "attempted_tokens",
    "committed_generated_tokens_raw",
    "committed_tokens",
    "discarded_generated_tokens_raw",
    "discarded_tokens",
)


def _usage_mapping(record: Mapping[str, Any]) -> dict[str, Any] | None:
    usage_value = record.get("usage")
    combined = dict(usage_value) if isinstance(usage_value, dict) else {}
    for field in _USAGE_TOP_LEVEL_FIELDS:
        if field not in combined and field in record:
            combined[field] = record[field]
    return combined or None


def _normalized_token_counts(usage: Mapping[str, Any]) -> dict[str, int | None]:
    generated = _first_int(
        usage, ("generated_tokens_raw", "generated_tokens", "output_tokens")
    )
    attempted = _first_int(
        usage,
        ("attempted_generated_tokens_raw", "attempted_tokens", "attempted"),
    )
    committed = _first_int(
        usage,
        ("committed_generated_tokens_raw", "committed_tokens", "committed"),
    )
    discarded = _first_int(
        usage,
        ("discarded_generated_tokens_raw", "discarded_tokens", "discarded"),
    )
    if attempted is None and generated is not None:
        attempted = generated
    if committed is None and generated is not None:
        committed = generated
    if discarded is None and attempted is not None and committed is not None:
        discarded = attempted - committed

    input_tokens = _first_int(usage, ("input_tokens", "prompt_tokens"))
    total_tokens = _first_int(usage, ("total_tokens",))
    if total_tokens is None and input_tokens is not None and generated is not None:
        total_tokens = input_tokens + generated
    return {
        "input_tokens": input_tokens,
        "text_prompt_tokens": _first_int(usage, ("text_prompt_tokens",)),
        "generated_tokens_raw": generated,
        "response_tokens": _first_int(usage, ("response_tokens",)),
        "total_tokens": total_tokens,
        "attempted": attempted,
        "committed": committed,
        "discarded": discarded,
    }


def _group_counts(values: Mapping[str, Any]) -> dict[str, int | None]:
    return {
        "attempted": _first_int(
            values,
            ("attempted", "attempted_tokens", "attempted_generated_tokens_raw"),
        ),
        "committed": _first_int(
            values,
            ("committed", "committed_tokens", "committed_generated_tokens_raw"),
        ),
        "discarded": _first_int(
            values,
            ("discarded", "discarded_tokens", "discarded_generated_tokens_raw"),
        ),
    }


def _accumulate_token_groups(
    destination: dict[str, dict[str, int]], source: Any
) -> None:
    if not isinstance(source, dict):
        return
    for name, values in source.items():
        if not isinstance(values, dict):
            continue
        normalized = _group_counts(values)
        group = destination.setdefault(
            str(name), {"attempted": 0, "committed": 0, "discarded": 0}
        )
        for field, value in normalized.items():
            if value is not None:
                group[field] += value


def summarize_usage(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    fields = (
        "input_tokens",
        "text_prompt_tokens",
        "generated_tokens_raw",
        "response_tokens",
        "total_tokens",
        "attempted",
        "committed",
        "discarded",
    )
    totals = {field: 0 for field in fields}
    available = {field: 0 for field in fields}
    records_with_usage = 0
    invariant_violations = 0
    by_model: dict[str, dict[str, int]] = {}
    by_phase: dict[str, dict[str, int]] = {}

    for record in records:
        usage = _usage_mapping(record)
        if usage is None:
            continue
        records_with_usage += 1
        counts = _normalized_token_counts(usage)
        for field, value in counts.items():
            if value is not None:
                totals[field] += value
                available[field] += 1
        attempted = counts["attempted"]
        committed = counts["committed"]
        discarded = counts["discarded"]
        if (
            attempted is not None
            and committed is not None
            and discarded is not None
            and attempted != committed + discarded
        ):
            invariant_violations += 1
        _accumulate_token_groups(by_model, usage.get("by_model"))
        _accumulate_token_groups(by_phase, usage.get("by_phase"))

    averages = {
        field: totals[field] / available[field] if available[field] else None
        for field in fields
    }
    attempted_total = totals["attempted"] if available["attempted"] else None
    committed_total = totals["committed"] if available["committed"] else None
    overhead_ratio = (
        attempted_total / committed_total
        if attempted_total is not None and committed_total
        else None
    )
    return {
        "records_total": len(records),
        "records_with_usage": records_with_usage,
        "totals": totals,
        "available": available,
        "averages": averages,
        "overhead_ratio": overhead_ratio,
        "invariant_violations": invariant_violations,
        "by_model": by_model,
        "by_phase": by_phase,
    }


TIMING_ALIASES = {
    "preprocess_seconds": ("preprocess_seconds",),
    "model_generation_seconds": ("model_generation_seconds", "generation_seconds"),
    "initial_selection_wall_seconds": ("initial_selection_wall_seconds",),
    "bidding_wall_seconds": ("bidding_wall_seconds",),
    "response_wall_seconds": ("response_wall_seconds", "generation_seconds"),
    "sample_total_seconds": ("sample_total_seconds", "total_seconds"),
}


def _first_float(mapping: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    for key in aliases:
        value = _safe_float(mapping.get(key))
        if value is not None:
            return value
    return None


def summarize_timing(
    records: Iterable[Mapping[str, Any]], usage: Mapping[str, Any]
) -> dict[str, Any]:
    totals = {field: 0.0 for field in TIMING_ALIASES}
    available = {field: 0 for field in TIMING_ALIASES}
    records_with_timing = 0
    for record in records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        records_with_timing += 1
        for field, aliases in TIMING_ALIASES.items():
            value = _first_float(timing, aliases)
            if value is not None:
                totals[field] += value
                available[field] += 1

    averages = {
        field: totals[field] / available[field] if available[field] else None
        for field in TIMING_ALIASES
    }
    attempted = usage["totals"]["attempted"] if usage["available"]["attempted"] else None
    committed = usage["totals"]["committed"] if usage["available"]["committed"] else None
    model_seconds = (
        totals["model_generation_seconds"]
        if available["model_generation_seconds"]
        else None
    )
    response_seconds = (
        totals["response_wall_seconds"] if available["response_wall_seconds"] else None
    )
    return {
        "records_with_timing": records_with_timing,
        "totals": totals,
        "available": available,
        "averages": averages,
        "attempted_tokens_per_model_generation_second": (
            attempted / model_seconds if attempted is not None and model_seconds else None
        ),
        "committed_tokens_per_response_second": (
            committed / response_seconds if committed is not None and response_seconds else None
        ),
    }


def summarize_relay(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
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
        handoff_count = _safe_int(relay.get("handoff_count")) or 0
        trigger_count = _safe_int(relay.get("zscore_trigger_count"))
        if trigger_count is None:
            trigger_count = _safe_int(relay.get("trigger_count")) or 0
        handoffs += handoff_count
        triggers += trigger_count
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
        "triggers": triggers,
        "average_handoffs": handoffs / available if available else None,
        "initial_models": dict(initial_models),
        "final_models": dict(final_models),
    }


def _score_result(correct: int, total: int) -> dict[str, int | float]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def _metadata_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(normalized))
    if value is None or not str(value).strip():
        return []
    return [str(value).strip()]


def _add_group_result(
    groups: dict[str, dict[str, list[int]]],
    field: str,
    value: Any,
    is_correct: bool,
) -> None:
    field_groups = groups.setdefault(field, {})
    for label in _metadata_values(value):
        counts = field_groups.setdefault(label, [0, 0])
        counts[0] += int(is_correct)
        counts[1] += 1


def _finalize_group_results(
    groups: Mapping[str, Mapping[str, Sequence[int]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for field, labels in groups.items():
        result[field] = {}
        for label, counts in labels.items():
            correct, total = counts
            result[field][label] = _score_result(correct, total)
    return result


def evaluate_records(
    records: OrderedDict[str, dict[str, Any]],
    ground_truth: Mapping[str, Mapping[str, Any]],
    expected_total: int = EXPECTED_TOTAL,
) -> dict[str, Any]:
    correct = 0
    total = 0
    extraction_methods: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    output_fields: Counter[str] = Counter()
    parse_failures = 0
    unknown_ids: list[str] = []
    matched_ids: set[str] = set()
    matched_scorable_ids: set[str] = set()
    matched_unscorable_ids: list[str] = []
    group_counts: dict[str, dict[str, list[int]]] = {}

    unscorable_ground_truth_ids = [
        sample_id
        for sample_id, record in ground_truth.items()
        if normalize_answer(record.get("answer")) is None
    ]
    expected_scorable_total = len(ground_truth) - len(unscorable_ground_truth_ids)

    for sample_id, record in records.items():
        status_counts[str(record.get("status", "unspecified"))] += 1
        response, output_field = eu.response_text(record)
        output_fields[output_field or "missing"] += 1
        gold_record = ground_truth.get(sample_id)
        if gold_record is None:
            unknown_ids.append(sample_id)
            continue
        matched_ids.add(sample_id)
        gold = normalize_answer(gold_record.get("answer"))
        if gold is None:
            matched_unscorable_ids.append(sample_id)
            continue
        matched_scorable_ids.add(sample_id)

        prediction, method = parse_compatible_answer(response, gold_record)
        extraction_methods[method] += 1
        parse_failures += int(prediction is None)
        total += 1
        is_correct = prediction == gold
        correct += int(is_correct)
        for field in ("skill", "broad_capability", "specific_capability"):
            _add_group_result(
                group_counts,
                field,
                gold_record.get(field),
                is_correct,
            )

    accuracy = _score_result(correct, total)
    records_list = list(records.values())
    usage = summarize_usage(records_list)
    timing = summarize_timing(records_list, usage)
    missing_ids = [sample_id for sample_id in ground_truth if sample_id not in matched_ids]
    missing_scorable_ids = [
        sample_id
        for sample_id, record in ground_truth.items()
        if normalize_answer(record.get("answer")) is not None
        and sample_id not in matched_scorable_ids
    ]
    matched_records = len(matched_ids)
    return {
        "expected_total": expected_total,
        "expected_scorable_total": expected_scorable_total,
        "unique_records": len(records),
        "matched_records": matched_records,
        "matched_scorable_records": len(matched_scorable_ids),
        "coverage": matched_records / expected_total if expected_total else None,
        "scorable_coverage": (
            len(matched_scorable_ids) / expected_scorable_total
            if expected_scorable_total
            else None
        ),
        "unknown_ids": unknown_ids,
        "missing_ids": missing_ids,
        "missing_scorable_ids": missing_scorable_ids,
        "unscorable_ground_truth_ids": unscorable_ground_truth_ids,
        "matched_unscorable_ids": matched_unscorable_ids,
        "unscored_missing_answer": len(unscorable_ground_truth_ids),
        "parse_failures": parse_failures,
        "accuracy": accuracy,
        "full_progress_lower_bound": (
            correct / expected_scorable_total if expected_scorable_total else None
        ),
        "all_rows_progress_lower_bound": (
            correct / expected_total if expected_total else None
        ),
        "extraction_methods": dict(extraction_methods),
        "group_accuracy": _finalize_group_results(group_counts),
        "status_counts": dict(status_counts),
        "output_fields": dict(output_fields),
        "usage": usage,
        "timing": timing,
        "relay": summarize_relay(records_list),
    }


def _format_score(score: Mapping[str, Any]) -> str:
    return f"{score['correct']}/{score['total']} ({float(score['accuracy']) * 100:.2f}%)"


def _format_optional(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def print_token_groups(label: str, groups: Mapping[str, Mapping[str, int]]) -> None:
    print(f"  {label}:")
    if not groups:
        print("    unavailable")
        return
    for name in sorted(groups):
        values = groups[name]
        print(
            f"    {name}: attempted={values['attempted']} "
            f"committed={values['committed']} discarded={values['discarded']}"
        )


def print_accuracy_groups(
    label: str,
    groups: Mapping[str, Mapping[str, Any]],
) -> None:
    print(f"  {label}:")
    if not groups:
        print("    unavailable")
        return
    for name in sorted(groups):
        print(f"    {name}: {_format_score(groups[name])}")


def print_verbose(
    path: Path, diagnostics: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    print("[VERBOSE]")
    print(
        f"Snapshot: bytes={diagnostics['file_size_bytes']} "
        f"complete_lines={diagnostics['complete_physical_lines']} "
        f"partial_tail_ignored={diagnostics['ignored_partial_tail']}"
    )
    malformed = diagnostics["malformed_complete_lines"]
    print(f"Malformed complete lines: {len(malformed)} {malformed[:20]}")
    print(f"Duplicate ID records: {diagnostics['duplicate_records']}")
    coverage = result["coverage"]
    print(
        f"Coverage: {result['matched_records']}/{result['expected_total']} "
        f"({_format_optional(coverage * 100 if coverage is not None else None)}%)"
    )
    scorable_coverage = result["scorable_coverage"]
    print(
        f"Scorable coverage: {result['matched_scorable_records']}/"
        f"{result['expected_scorable_total']} "
        f"({_format_optional(scorable_coverage * 100 if scorable_coverage is not None else None)}%)"
    )
    print(
        "Official unscorable ground truth: "
        f"{result['unscorable_ground_truth_ids']} "
        f"(present in responses: {result['matched_unscorable_ids']})"
    )
    print(
        f"Response IDs: unique={result['unique_records']} "
        f"unknown={len(result['unknown_ids'])} missing_rows={len(result['missing_ids'])} "
        f"missing_scorable={len(result['missing_scorable_ids'])}"
    )
    if result["unknown_ids"]:
        print(f"  Unknown IDs (first 20): {result['unknown_ids'][:20]}")
    if result["missing_ids"]:
        print(f"  Missing IDs (first 20): {result['missing_ids'][:20]}")
    if result["missing_scorable_ids"]:
        print(
            "  Missing scorable IDs (first 20): "
            f"{result['missing_scorable_ids'][:20]}"
        )
    print(f"Generation status: {result['status_counts']}")
    print(f"Output fields: {result['output_fields']}")

    print("Accuracy:")
    print(f"  LogicVista: {_format_score(result['accuracy'])}")
    lower_bound = result["full_progress_lower_bound"]
    print(
        "  Full-scorable progress lower bound: "
        f"{result['accuracy']['correct']}/{result['expected_scorable_total']} "
        f"({_format_optional(lower_bound * 100 if lower_bound is not None else None)}%)"
    )
    all_rows_lower_bound = result["all_rows_progress_lower_bound"]
    print(
        "  All-448-row progress (not ACC): "
        f"{result['accuracy']['correct']}/{result['expected_total']} "
        f"({_format_optional(all_rows_lower_bound * 100 if all_rows_lower_bound is not None else None)}%)"
    )
    print(f"  Parse failures: {result['parse_failures']}")
    print(f"  Extraction methods: {result['extraction_methods']}")

    print("Accuracy breakdown:")
    group_accuracy = result["group_accuracy"]
    print_accuracy_groups("skill", group_accuracy.get("skill", {}))
    print_accuracy_groups(
        "broad_capability", group_accuracy.get("broad_capability", {})
    )
    print_accuracy_groups(
        "specific_capability", group_accuracy.get("specific_capability", {})
    )

    usage = result["usage"]
    print("Token usage:")
    print(f"  Records with usage: {usage['records_with_usage']}/{usage['records_total']}")
    labels = {
        "input_tokens": "Input",
        "text_prompt_tokens": "Text prompt",
        "generated_tokens_raw": "Generated raw",
        "response_tokens": "Response",
        "total_tokens": "Total",
        "attempted": "Attempted",
        "committed": "Committed",
        "discarded": "Discarded",
    }
    for field, label in labels.items():
        count = usage["available"][field]
        total = usage["totals"][field] if count else None
        average = usage["averages"][field]
        print(
            f"  {label}: total={_format_optional(total, 0)} "
            f"average={_format_optional(average)} available={count}"
        )
    overhead = _format_optional(usage["overhead_ratio"], 4)
    print(f"  Compute overhead ratio: {overhead + 'x' if overhead != 'unavailable' else overhead}")
    print(f"  Accounting invariant violations: {usage['invariant_violations']}")
    print_token_groups("By model", usage["by_model"])
    print_token_groups("By phase", usage["by_phase"])

    timing = result["timing"]
    print("Timing:")
    print(f"  Records with timing: {timing['records_with_timing']}/{result['unique_records']}")
    for field in TIMING_ALIASES:
        count = timing["available"][field]
        total = timing["totals"][field] if count else None
        average = timing["averages"][field]
        print(
            f"  {field}: total={_format_optional(total, 4)} "
            f"average={_format_optional(average, 4)} available={count}"
        )
    print(
        "  Attempted tokens/model-generation-second: "
        f"{_format_optional(timing['attempted_tokens_per_model_generation_second'], 4)}"
    )
    print(
        "  Committed tokens/response-second: "
        f"{_format_optional(timing['committed_tokens_per_response_second'], 4)}"
    )

    relay = result["relay"]
    print("Relay stats:")
    print(f"  Records with relay stats: {relay['available']}")
    if relay["available"]:
        print(f"  Questions with handoff: {relay['questions_with_handoff']}")
        print(f"  Total handoffs: {relay['handoffs']}")
        print(f"  Total triggers: {relay['triggers']}")
        print(f"  Average handoffs/record: {_format_optional(relay['average_handoffs'], 4)}")
        print(f"  Initial models: {relay['initial_models']}")
        print(f"  Final models: {relay['final_models']}")


def main() -> None:
    if not DATASET_SPLIT.strip():
        raise ValueError("DATASET_SPLIT cannot be empty")
    dataset_path = eu.dataset_path(DATASET)
    ground_truth, ground_truth_details = load_ground_truth(
        dataset_path,
        DATASET_SPLIT,
        EXPECTED_TOTAL,
    )
    if VERBOSE:
        print(
            f"[ground-truth] path={dataset_path} split={DATASET_SPLIT} "
            f"rows={len(ground_truth)} scorable={ground_truth_details['scorable_rows']} "
            f"strategy={ground_truth_details['strategy']}"
        )

    processed = []
    for path in eu.resolve_input_files(INPUT_FILES, DATASET):
        try:
            records, diagnostics = eu.read_jsonl_snapshot(path, ID_ALIASES)
            records = eu.first_n(records, MAX_EVAL)
            result = evaluate_records(records, ground_truth, EXPECTED_TOTAL)
            known = [record for key, record in records.items() if key in ground_truth]
            basic = eu.summarize_basic(known)
            correct = result["accuracy"]["correct"]
            evaluated = result["matched_records"]
            eu.print_basic_report(
                path,
                evaluated=evaluated,
                expected=EXPECTED_TOTAL,
                acc_line=eu.format_accuracy(correct, evaluated),
                basic=basic,
                note=(
                    f"{len(result['unknown_ids'])} record(s) with unknown id skipped"
                    if result["unknown_ids"]
                    else None
                ),
            )
            if VERBOSE:
                print_verbose(path, diagnostics, result)
            processed.append(
                (
                    path,
                    {
                        "ACC (%)": correct / evaluated * 100 if evaluated else 0.0,
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
