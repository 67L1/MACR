from __future__ import annotations

INPUT_FILES: list = []
MAX_EVAL = None
VERBOSE = False

import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from core import eval_utils as eu

DATASET = "realworldqa"
SPLIT = "test"
EXPECTED_TOTAL = 765
ID_ALIASES = ("pid",)

ANSWER_INSTRUCTIONS = {
    "option_letter": (
        "Please answer directly with only the letter of the correct option and nothing else."
    ),
    "short_answer": "Please answer directly with a single word or number.",
}

ANSWER_RE = re.compile(r"(?im)^[ \t]*[*_`]*[ \t]*answer[ \t]*[*_`]*[ \t]*[:：][ \t]*(.*)$")


def resolve_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Unsupported dataset file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.rglob(f"{SPLIT}-*.parquet")) or sorted(path.rglob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"RealWorldQA parquet not found: {path}")


def _answer_format(question: Any) -> str:
    text = question if isinstance(question, str) else ""
    for answer_format, instruction in ANSWER_INSTRUCTIONS.items():
        if text.endswith(instruction):
            return answer_format
    return "unknown"


def load_ground_truth(path: Path) -> "OrderedDict[str, dict[str, Any]]":
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for file in resolve_parquet_files(path):
        rows.extend(pq.read_table(file, columns=["question", "answer"]).to_pylist())
    ground_truth: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for pid, row in enumerate(rows):
        ground_truth[str(pid)] = {
            "answer": "" if row["answer"] is None else str(row["answer"]),
            "answer_format": _answer_format(row["question"]),
        }
    if len(ground_truth) != EXPECTED_TOTAL:
        print(
            f"Warning: expected {EXPECTED_TOTAL} RealWorldQA rows, found {len(ground_truth)}",
            file=sys.stderr,
        )
    return ground_truth


def extract_answer(response: str) -> str | None:
    matches = ANSWER_RE.findall(response or "")
    if not matches:
        return None
    text = matches[-1].strip().strip("*_`").strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text or None


def is_correct(prediction: str | None, gold: dict[str, Any]) -> bool:
    gold_text = str(gold["answer"]).strip()
    return prediction is not None and bool(gold_text) and prediction.casefold() == gold_text.casefold()


def evaluate_records(records, ground_truth) -> dict[str, Any]:
    total = 0
    correct = 0
    statuses: Counter = Counter()
    no_answer = 0
    by_format: dict[str, list[int]] = {}
    for key, record in records.items():
        gold = ground_truth[key]
        total += 1
        status = str(record.get("status", "success"))
        statuses[status] += 1
        ok = False
        if status == "success":
            prediction = extract_answer(eu.response_text(record)[0])
            if prediction is None:
                no_answer += 1
            ok = is_correct(prediction, gold)
        correct += int(ok)
        group = by_format.setdefault(str(gold.get("answer_format", "unknown")), [0, 0])
        group[0] += int(ok)
        group[1] += 1
    return {
        "total": total,
        "correct": correct,
        "statuses": dict(statuses),
        "no_answer": no_answer,
        "by_format": by_format,
    }


def main() -> None:
    ground_truth = load_ground_truth(eu.dataset_path(DATASET))
    processed = []
    for path in eu.resolve_input_files(INPUT_FILES, DATASET):
        try:
            records, diagnostics = eu.read_jsonl_snapshot(path, ID_ALIASES)
            records = eu.first_n(records, MAX_EVAL)
            unknown_ids = [key for key in records if key not in ground_truth]
            known = OrderedDict((k, r) for k, r in records.items() if k in ground_truth)
            result = evaluate_records(known, ground_truth)
            basic = eu.summarize_basic(known.values())
            eu.print_basic_report(
                path,
                evaluated=result["total"],
                expected=EXPECTED_TOTAL,
                acc_line=eu.format_accuracy(result["correct"], result["total"]),
                basic=basic,
                note=f"{len(unknown_ids)} record(s) with unknown id skipped" if unknown_ids else None,
            )
            if VERBOSE:
                print(f"[VERBOSE] snapshot diagnostics: {diagnostics}")
                print(f"[VERBOSE] generation status: {result['statuses']}")
                print(f"[VERBOSE] success records without an Answer line: {result['no_answer']}")
                for name, (group_correct, group_total) in result["by_format"].items():
                    print(f"[VERBOSE] {name}: {eu.format_accuracy(group_correct, group_total)}")
                if unknown_ids:
                    print(f"[VERBOSE] unknown ids: {unknown_ids[:20]}")
            accuracy = result["correct"] / result["total"] * 100 if result["total"] else 0.0
            processed.append((path, {"ACC (%)": accuracy, **eu.basic_overall_metrics(basic)}))
        except FileNotFoundError as error:
            print(f"\nError: {error}. Skipping.")
        except Exception as error:
            print(f"\nError while processing {path}: {type(error).__name__}: {error}")
    if not processed:
        raise RuntimeError("No input files were processed successfully")
    eu.print_overall(processed)


if __name__ == "__main__":
    main()
