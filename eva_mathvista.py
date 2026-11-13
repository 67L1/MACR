from __future__ import annotations

INPUT_FILES: list = []
MAX_EVAL = None
VERBOSE = False


import ast
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

from core import eval_utils as eu

DATASET = "mathvista"
SPLIT = "testmini"
EXPECTED_TOTAL = 1000
TAIL_CHARS = 300
ID_ALIASES = ("pid",)

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?(?![\w.])"
)
INTEGER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?![\w.]|,\d)"
)


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def strip_candidate(text: str) -> str:
    candidate = text.strip()
    candidate = candidate.strip("`*_ \t\r\n")
    if candidate.startswith("$") and candidate.endswith("$") and len(candidate) >= 2:
        candidate = candidate[1:-1].strip()
    candidate = re.sub(r"[。；;]+$", "", candidate).strip()
    return candidate


def extract_balanced_after(text: str, marker: str, opening: str, closing: str) -> list[str]:
    values: list[str] = []
    start = 0
    while True:
        marker_index = text.find(marker, start)
        if marker_index < 0:
            break
        content_start = marker_index + len(marker)
        depth = 1
        index = content_start
        while index < len(text):
            character = text[index]
            if character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if depth == 0:
                    values.append(text[content_start:index])
                    start = index + 1
                    break
            index += 1
        else:
            start = content_start
    return values


def extract_balanced_lists(text: str) -> list[str]:
    values: list[str] = []
    depth = 0
    start: int | None = None
    for index, character in enumerate(text):
        if character == "[":
            if depth == 0:
                start = index
            depth += 1
        elif character == "]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                values.append(text[start : index + 1])
                start = None
    return values


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def parse_choice_candidate(candidate: str, choices: list[Any]) -> tuple[str, str | None, str | None]:
    candidate = strip_candidate(candidate)
    valid_letters = [chr(ord("A") + index) for index in range(len(choices))]
    valid_set = set(valid_letters)

    compact = candidate.strip()
    exact_match = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?[.)]?", compact)
    if exact_match and exact_match.group(1).upper() in valid_set:
        return "ok", exact_match.group(1).upper(), None

    leading_match = re.match(
        r"^\s*(?:option\s*)?\(?([A-Za-z])\)?(?:\s*[-:：.)]|\s*$)",
        compact,
        flags=re.IGNORECASE,
    )
    if leading_match and leading_match.group(1).upper() in valid_set:
        return "ok", leading_match.group(1).upper(), None

    normalized_candidate = normalize_space(compact).casefold().strip(" .")
    matching_choice_letters = [
        valid_letters[index]
        for index, choice in enumerate(choices)
        if normalize_space(choice).casefold().strip(" .") == normalized_candidate
    ]
    if len(matching_choice_letters) == 1:
        return "ok", matching_choice_letters[0], None
    if len(matching_choice_letters) > 1:
        return "ambiguous", None, "candidate matches multiple identical choice texts"

    option_letters = [
        match.upper()
        for match in re.findall(r"(?<![A-Za-z])\(?([A-Za-z])\)?(?![A-Za-z])", compact)
        if match.upper() in valid_set
    ]
    option_letters = unique_preserving_order(option_letters)
    if len(option_letters) == 1:
        return "ok", option_letters[0], None
    if len(option_letters) > 1:
        return "ambiguous", None, f"multiple option letters: {option_letters}"
    return "invalid", None, "no valid option letter or exact choice text"


def parse_integer_candidate(candidate: str) -> tuple[str, str | None, str | None]:
    candidate = strip_candidate(candidate)
    if NUMBER_PATTERN.search(candidate):
        decimal_like = [
            value
            for value in NUMBER_PATTERN.findall(candidate)
            if "." in value or "e" in value.lower()
        ]
        if decimal_like:
            return "invalid", None, f"integer candidate contains non-integer number(s): {decimal_like}"
    values = unique_preserving_order(value.replace(",", "") for value in INTEGER_PATTERN.findall(candidate))
    if len(values) == 1:
        return "ok", str(int(values[0])), None
    if len(values) > 1:
        return "ambiguous", None, f"multiple integers: {values}"
    return "invalid", None, "no integer found"


def parse_float_candidate(candidate: str) -> tuple[str, str | None, str | None]:
    candidate = strip_candidate(candidate)
    values = unique_preserving_order(value.replace(",", "") for value in NUMBER_PATTERN.findall(candidate))
    if len(values) == 1:
        try:
            value = float(values[0])
        except ValueError:
            return "invalid", None, f"invalid float: {values[0]}"
        if not math.isfinite(value):
            return "invalid", None, "non-finite float"
        return "ok", values[0], None
    if len(values) > 1:
        return "ambiguous", None, f"multiple numbers: {values}"
    return "invalid", None, "no numeric value found"


def parse_list_candidate(candidate: str) -> tuple[str, str | None, str | None]:
    candidate = strip_candidate(candidate)
    list_texts = extract_balanced_lists(candidate)
    if not list_texts and candidate.startswith("[") and candidate.endswith("]"):
        list_texts = [candidate]

    parsed_values: list[str] = []
    for list_text in list_texts:
        try:
            value = ast.literal_eval(list_text)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, list):
            continue
        if not all(isinstance(element, (int, float)) and not isinstance(element, bool) for element in value):
            continue
        parsed_values.append(str(value))
    parsed_values = unique_preserving_order(parsed_values)
    if len(parsed_values) == 1:
        return "ok", parsed_values[0], None
    if len(parsed_values) > 1:
        return "ambiguous", None, f"multiple lists: {parsed_values}"
    return "invalid", None, "no valid numeric Python list found"


def parse_candidate(candidate: str, problem: dict[str, Any]) -> tuple[str, str | None, str | None]:
    if problem.get("question_type") == "multi_choice":
        choices = problem.get("choices") or []
        return parse_choice_candidate(candidate, list(choices))
    answer_type = problem.get("answer_type")
    if answer_type == "integer":
        return parse_integer_candidate(candidate)
    if answer_type == "float":
        return parse_float_candidate(candidate)
    if answer_type == "list":
        return parse_list_candidate(candidate)
    return "invalid", None, f"unsupported answer type: {answer_type}"


def candidate_sources(response: str, tail_chars: int) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []

    bold_answers = re.findall(
        r"\*\*\s*Answer\s*[:：]\s*(.*?)\s*\*\*",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if bold_answers:
        sources.append(("bold_answer", bold_answers[-1]))

    labeled_answers = re.findall(
        r"(?im)^\s*(?:final\s+answer|answer)\s*[:：]\s*(.+?)\s*$",
        response,
    )
    if labeled_answers:
        sources.append(("labeled_answer", labeled_answers[-1]))

    boxed_answers = extract_balanced_after(response, r"\boxed{", "{", "}")
    if boxed_answers:
        sources.append(("boxed", boxed_answers[-1]))

    tail = response[-tail_chars:] if tail_chars > 0 else response
    sources.append(("tail_fallback", tail))
    return sources


def extract_answer(response: str, problem: dict[str, Any], tail_chars: int = 300) -> dict[str, Any]:
    if not isinstance(response, str) or not response.strip():
        return {
            "extraction": "",
            "extraction_status": "failed",
            "extraction_method": "none",
            "candidate": None,
            "warnings": ["empty model response"],
        }

    warnings: list[str] = []
    for method, candidate in candidate_sources(response, tail_chars):
        status, extraction, warning = parse_candidate(candidate, problem)
        if status == "ok":
            return {
                "extraction": extraction,
                "extraction_status": "ok",
                "extraction_method": method,
                "candidate": candidate,
                "warnings": warnings,
            }
        if warning:
            warnings.append(f"{method}: {warning}")
        if status == "ambiguous":
            return {
                "extraction": "",
                "extraction_status": "ambiguous",
                "extraction_method": method,
                "candidate": candidate,
                "warnings": warnings,
            }

    return {
        "extraction": "",
        "extraction_status": "failed",
        "extraction_method": "none",
        "candidate": None,
        "warnings": warnings,
    }


def levenshtein_distance(left: str, right: str) -> int:
    try:
        from Levenshtein import distance

        return int(distance(left, right))
    except ImportError:
        previous = list(range(len(right) + 1))
        for left_index, left_character in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_character in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1] + (left_character != right_character),
                    )
                )
            previous = current
        return previous[-1]


def official_normalize(problem: dict[str, Any], extraction: Any) -> str | None:
    choices = problem.get("choices") or []
    question_type = problem.get("question_type")
    answer_type = problem.get("answer_type")
    precision = problem.get("precision")

    if question_type == "multi_choice":
        try:
            extraction_text = str(extraction).strip()
        except Exception:
            extraction_text = ""
        parenthesized = re.findall(r"\(([a-zA-Z])\)", extraction_text)
        if parenthesized:
            extraction_text = parenthesized[0].upper()
        letters = [chr(ord("A") + index) for index in range(len(choices))]
        if extraction_text in letters:
            return str(choices[letters.index(extraction_text)])
        if not choices:
            return None
        distances = [levenshtein_distance(extraction_text, str(choice)) for choice in choices]
        return str(choices[distances.index(min(distances))])

    if answer_type == "integer":
        try:
            return str(int(float(extraction)))
        except (TypeError, ValueError, OverflowError):
            return None
    if answer_type == "float":
        try:
            return str(round(float(extraction), int(precision)))
        except (TypeError, ValueError, OverflowError):
            return None
    if answer_type == "list":
        try:
            return str(extraction)
        except Exception:
            return None
    return None


def strict_normalize(problem: dict[str, Any], extraction: Any) -> str | None:
    choices = problem.get("choices") or []
    question_type = problem.get("question_type")
    answer_type = problem.get("answer_type")
    precision = problem.get("precision")
    text = str(extraction).strip() if extraction is not None else ""
    if not text:
        return None

    if question_type == "multi_choice":
        letters = [chr(ord("A") + index) for index in range(len(choices))]
        parenthesized = re.fullmatch(r"\(([A-Za-z])\)", text)
        if parenthesized:
            text = parenthesized.group(1).upper()
        if text.upper() in letters and len(text) == 1:
            return str(choices[letters.index(text.upper())])
        exact_matches = [
            str(choice)
            for choice in choices
            if normalize_space(choice).casefold() == normalize_space(text).casefold()
        ]
        return exact_matches[0] if len(exact_matches) == 1 else None

    if answer_type == "integer":
        if not re.fullmatch(r"[-+]?\d+", text):
            return None
        return str(int(text))
    if answer_type == "float":
        if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
            return None
        try:
            return str(round(float(text), int(precision)))
        except (TypeError, ValueError, OverflowError):
            return None
    if answer_type == "list":
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        return str(value) if isinstance(value, list) else None
    return None


def get_problem_value(problem: dict[str, Any], key: str) -> Any:
    if key in {"question_type", "answer_type"}:
        return problem.get(key)
    return (problem.get("metadata") or {}).get(key)


def build_scores(problems: list[dict[str, Any]], truth_field: str) -> dict[str, Any]:
    total = len(problems)
    correct = sum(bool(problem.get(truth_field)) for problem in problems)
    scores: dict[str, Any] = {
        "average": {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        }
    }
    target_keys = [
        "question_type",
        "answer_type",
        "language",
        "source",
        "category",
        "task",
        "context",
        "grade",
        "skills",
    ]
    for key in target_keys:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for problem in problems:
            value = get_problem_value(problem, key)
            values = value if key == "skills" and isinstance(value, list) else [value]
            for group_value in values:
                group_name = str(group_value)
                grouped.setdefault(group_name, []).append(problem)
        key_scores: dict[str, Any] = {}
        for group_name, group_problems in grouped.items():
            group_correct = sum(bool(problem.get(truth_field)) for problem in group_problems)
            group_total = len(group_problems)
            key_scores[group_name] = {
                "accuracy": group_correct / group_total if group_total else 0.0,
                "correct": group_correct,
                "total": group_total,
            }
        scores[key] = dict(
            sorted(
                key_scores.items(),
                key=lambda item: float(item[1]["accuracy"]),
                reverse=True,
            )
        )
    return scores


def load_ground_truth(parquet_path: Path, split: str) -> tuple[OrderedDict[str, dict[str, Any]], str | None]:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={split: str(parquet_path)},
        split=split,
    )
    if "decoded_image" in dataset.column_names:
        dataset = dataset.remove_columns("decoded_image")
    required = {
        "pid",
        "question",
        "choices",
        "answer",
        "question_type",
        "answer_type",
        "precision",
        "metadata",
    }
    missing = sorted(required.difference(dataset.column_names))
    if missing:
        raise ValueError(f"MathVista parquet is missing scoring columns: {', '.join(missing)}")

    problems: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for problem in dataset:
        pid = str(problem["pid"])
        if pid in problems:
            raise ValueError(f"Duplicate pid in MathVista parquet: {pid}")
        problem["pid"] = pid
        problems[pid] = problem
    return problems, getattr(dataset, "_fingerprint", None)


def failed_extraction(reason: str) -> dict[str, Any]:
    return {
        "extraction": "",
        "extraction_status": "failed",
        "extraction_method": "generation_error",
        "candidate": None,
        "warnings": [reason],
    }


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
    destination: dict[str, dict[str, int]],
    source: Any,
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
    by_model: dict[str, dict[str, int]] = {}
    by_phase: dict[str, dict[str, int]] = {}

    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
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
    records = list(records)
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


def summarize_timing(records: Iterable[dict[str, Any]], attempted: int, committed: int) -> dict[str, Any]:
    records = list(records)
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
    raw_records: OrderedDict[str, dict[str, Any]],
    ground_truth: OrderedDict[str, dict[str, Any]],
    tail_chars: int,
) -> dict[str, Any]:
    unknown_pids = [pid for pid in raw_records if pid not in ground_truth]
    known_records = OrderedDict(
        (pid, record) for pid, record in raw_records.items() if pid in ground_truth
    )
    scored_problems: list[dict[str, Any]] = []
    successful_scored: list[dict[str, Any]] = []
    extraction_statuses: Counter[str] = Counter()
    extraction_methods: Counter[str] = Counter()

    for pid, raw_record in known_records.items():
        problem = dict(ground_truth[pid])
        response = raw_record.get("response", "")
        if raw_record.get("status") == "success":
            extraction_result = extract_answer(response, problem, tail_chars=tail_chars)
        else:
            extraction_result = failed_extraction("generation record is not successful")

        extraction = extraction_result["extraction"]
        official_prediction = official_normalize(problem, extraction)
        strict_prediction = strict_normalize(problem, extraction)
        answer = str(problem["answer"])
        scored = dict(problem)
        scored.update(
            {
                "response": response,
                "extraction": extraction,
                "prediction": official_prediction,
                "strict_prediction": strict_prediction,
                "true_false": official_prediction == answer,
                "strict_true_false": strict_prediction == answer,
                "generation_status": raw_record.get("status"),
            }
        )
        scored_problems.append(scored)
        if raw_record.get("status") == "success":
            successful_scored.append(scored)
        extraction_statuses[extraction_result["extraction_status"]] += 1
        extraction_methods[extraction_result["extraction_method"]] += 1

    official_scores = build_scores(scored_problems, "true_false")
    strict_scores = build_scores(scored_problems, "strict_true_false")
    success_official = build_scores(successful_scored, "true_false")["average"]
    success_strict = build_scores(successful_scored, "strict_true_false")["average"]
    records = list(known_records.values())
    usage = summarize_usage(records)
    timing = summarize_timing(records, usage["attempted"], usage["committed"])
    status_counts = Counter(str(record.get("status", "missing")) for record in records)

    return {
        "dataset_total": len(ground_truth),
        "known_records": len(known_records),
        "unknown_pids": unknown_pids,
        "coverage": len(known_records) / len(ground_truth) if ground_truth else 0.0,
        "status_counts": dict(status_counts),
        "extraction_statuses": dict(extraction_statuses),
        "extraction_methods": dict(extraction_methods),
        "official_scores": official_scores,
        "strict_scores": strict_scores,
        "success_official": success_official,
        "success_strict": success_strict,
        "relay": summarize_relay(records),
        "usage": usage,
        "timing": timing,
    }


def _format_accuracy(values: dict[str, Any]) -> str:
    return (
        f"{values['correct']}/{values['total']} "
        f"({float(values['accuracy']) * 100:.2f}%)"
    )


def print_grouped_scores(scores: dict[str, Any]) -> None:
    for key, groups in scores.items():
        if key == "average" or not isinstance(groups, dict):
            continue
        print(f"  {key}:")
        for group_name, values in groups.items():
            print(f"    {group_name}: {_format_accuracy(values)}")


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
    path: Path,
    diagnostics: dict[str, Any],
    result: dict[str, Any],
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
    print(f"Duplicate pid records: {diagnostics['duplicate_records']}")
    print(
        f"Coverage: {result['known_records']}/{result['dataset_total']} "
        f"({result['coverage'] * 100:.2f}%)"
    )
    print(f"Generation status: {result['status_counts']}")
    if result["unknown_pids"]:
        print(
            f"Unknown pids skipped: {len(result['unknown_pids'])} "
            f"{result['unknown_pids'][:20]}"
        )
    print(f"Extraction status: {result['extraction_statuses']}")
    print(f"Extraction methods: {result['extraction_methods']}")

    official = result["official_scores"]["average"]
    strict = result["strict_scores"]["average"]
    print("Accuracy:")
    print(f"  Official-compatible (all completed): {_format_accuracy(official)}")
    print(f"  Strict (all completed): {_format_accuracy(strict)}")
    print(
        f"  Official-compatible (successful only): "
        f"{_format_accuracy(result['success_official'])}"
    )
    print(f"  Strict (successful only): {_format_accuracy(result['success_strict'])}")
    print("Official-compatible grouped accuracy:")
    print_grouped_scores(result["official_scores"])

    relay = result["relay"]
    print("Relay stats:")
    print(f"  Records with stats: {relay['available']}")
    print(f"  Questions with handoff: {relay['questions_with_handoff']}")
    print(f"  Total handoffs: {relay['handoffs']}")
    print(f"  Z-score triggers: {relay['zscore_triggers']}")
    print(f"  Average handoffs/record: {relay['average_handoffs']:.4f}")
    print(f"  Initial models: {relay['initial_models']}")
    print(f"  Final models: {relay['final_models']}")

    usage = result["usage"]
    print("Token usage:")
    print(f"  Attempted: {usage['attempted']}")
    print(f"  Committed: {usage['committed']}")
    print(f"  Discarded: {usage['discarded']}")
    print(f"  Compute overhead ratio: {usage['overhead_ratio']:.4f}x")
    print(f"  Average attempted/record: {usage['average_attempted']:.2f}")
    print(f"  Average committed/record: {usage['average_committed']:.2f}")
    print(f"  Average discarded/record: {usage['average_discarded']:.2f}")
    print(f"  Accounting invariant violations: {usage['invariant_violations']}")
    print_token_groups("By model", usage["by_model"])
    print_token_groups("By phase", usage["by_phase"])

    timing = result["timing"]
    print("Timing:")
    print(f"  Preprocess seconds (summed): {timing['preprocess_seconds']:.4f}")
    print(
        f"  Model generation seconds (summed): "
        f"{timing['model_generation_seconds']:.4f}"
    )
    print(
        f"  Initial selection wall seconds (summed): "
        f"{timing['initial_selection_wall_seconds']:.4f}"
    )
    print(f"  Bidding wall seconds (summed): {timing['bidding_wall_seconds']:.4f}")
    print(f"  Response wall seconds (summed): {timing['response_wall_seconds']:.4f}")
    print(f"  Sample total seconds (summed): {timing['sample_total_seconds']:.4f}")
    print(
        f"  Average sample total seconds: "
        f"{timing['average_sample_total_seconds']:.4f}"
    )
    print(
        f"  Attempted tokens/model-generation-second: "
        f"{timing['attempted_tokens_per_model_generation_second']:.4f}"
    )
    print(
        f"  Committed tokens/response-second: "
        f"{timing['committed_tokens_per_response_second']:.4f}"
    )


def resolve_parquet(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.rglob(f"{SPLIT}-*.parquet"))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"Expected exactly one {SPLIT}-*.parquet under {path}, found {len(matches)}")
    raise FileNotFoundError(f"MathVista parquet not found: {path}")


def main() -> None:
    if TAIL_CHARS <= 0:
        raise ValueError("TAIL_CHARS must be positive")
    parquet_path = resolve_parquet(eu.dataset_path(DATASET))
    ground_truth, fingerprint = load_ground_truth(parquet_path, SPLIT)
    if VERBOSE:
        print(f"[ground truth] split={SPLIT} rows={len(ground_truth)} fingerprint={fingerprint}")

    processed = []
    for path in eu.resolve_input_files(INPUT_FILES, DATASET):
        try:
            raw_records, diagnostics = eu.read_jsonl_snapshot(path, ID_ALIASES)
            raw_records = eu.first_n(raw_records, MAX_EVAL)
            result = evaluate_records(raw_records, ground_truth, TAIL_CHARS)
            known = [record for pid, record in raw_records.items() if pid in ground_truth]
            basic = eu.summarize_basic(known)
            official = result["official_scores"]["average"]
            eu.print_basic_report(
                path,
                evaluated=result["known_records"],
                expected=EXPECTED_TOTAL,
                acc_line=eu.format_accuracy(official["correct"], official["total"]),
                basic=basic,
                note=(
                    f"{len(result['unknown_pids'])} record(s) with unknown pid skipped"
                    if result["unknown_pids"]
                    else None
                ),
            )
            if VERBOSE:
                print_verbose(path, diagnostics, result)
            processed.append(
                (path, {"ACC (%)": official["accuracy"] * 100, **eu.basic_overall_metrics(basic)})
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
