from __future__ import annotations

INPUT_FILES: list = []
MAX_EVAL = None
VERBOSE = False

import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from core import eval_utils as eu

DATASET = "mmstar"
EXPECTED_TOTAL = 1500
ID_ALIASES = ("index", "id", "pid")


def calculate_f1_macro(y_true, y_pred):
    labels = sorted(set(y_true))
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            stats[true]["tp"] += 1
        else:
            stats[true]["fn"] += 1
            stats[pred]["fp"] += 1

    f1_scores = []
    for label in labels:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]

        if (tp + fp) == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if (tp + fn) == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        f1_scores.append(f1)

    if not f1_scores:
        return 0.0

    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1


letter_switch = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
}


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"^[\s\*\_\-\=\:\|\"'`]+", "", s)
    s = re.sub(r"[\s\*\_\-\=\:\|\"'`\.!,;？。！、]+$", "", s)
    return s


def try_parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def extract_from_pred(pred_raw: str) -> str:
    if pred_raw is None:
        return ""
    pred = pred_raw.lower()
    tail = None
    idx = pred.find("answer:")
    if idx != -1:
        tail = pred[idx + len("answer:"):]
    else:
        idx2 = pred.find("answer")
        if idx2 != -1:
            tail = pred[idx2 + len("answer"):]
    if tail is None:
        tail = pred
    tail = re.sub(r"^\s*[:：=\-–—~>*()\[\]]*\s*(?:is|are|=)?\s*", "", tail)
    tail = tail.splitlines()[0] if tail else ""
    tail = normalize_text(tail)
    tail = re.sub(r"^[\*\_`]+|[\*\_`]+$", "", tail).strip()
    return tail


def parse_options_from_question(question: str) -> Dict[str, str]:
    options = {}
    pattern = re.compile(r"^\s*\(?([A-Z])\)?[.\s]\s*(.*)", re.MULTILINE)
    matches = pattern.findall(question)
    for match in matches:
        option_letter = match[0]
        option_text = match[1].strip()
        if len(option_letter) == 1 and option_text:
            options[option_letter] = option_text
    return options


def compare_pred_answer(pred_text: str, ans_text: str) -> bool:
    pred_norm = normalize_text(pred_text)
    ans_norm = normalize_text(ans_text)
    p_num = try_parse_number(pred_norm)
    a_num = try_parse_number(ans_norm)

    if p_num is not None and a_num is not None:
        return abs(p_num - a_num) < 1e-9

    if pred_norm == ans_norm:
        return True
    if ans_norm in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        correct_option_text = letter_switch[ans_norm]
        if correct_option_text in pred_norm or pred_norm in correct_option_text:
            return True
        elif correct_option_text == 'zero':
            correct_option_text = 'no'
            if correct_option_text in pred_norm or pred_norm in correct_option_text:
                return True

    if len(ans_norm) > 1:
        pattern = r'\b' + re.escape(ans_norm) + r'\b'
        if re.search(pattern, pred_norm):
            return True
    if pred_norm in ['a', 'b'] and ans_norm in ['yes', 'no']:
        if pred_norm == 'a' and ans_norm == 'yes':
            return True
        elif pred_norm == 'b' and ans_norm == 'no':
            return True

    return False


def parquet_files(path: Path) -> list:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("val-*.parquet")) or sorted(path.rglob("*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"MMStar parquet not found: {path}")


def load_ground_truth(path: Path) -> "OrderedDict[str, dict[str, str]]":
    import pyarrow.parquet as pq

    rows = []
    for parquet_path in parquet_files(path):
        rows.extend(pq.read_table(parquet_path, columns=["index", "question", "answer"]).to_pylist())
    ground_truth: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    for row in rows:
        sample_id = str(row["index"])
        if sample_id in ground_truth:
            raise ValueError(f"Duplicate MMStar index: {sample_id}")
        ground_truth[sample_id] = {
            "question": str(row.get("question", "")).strip(),
            "answer": str(row.get("answer", "")).strip(),
        }
    return ground_truth


def evaluate_records(records: "OrderedDict[str, dict[str, Any]]", ground_truth) -> dict:
    y_true = []
    y_pred = []
    total = 0
    correct = 0
    relay_stats_available = 0
    questions_with_handoff = 0
    total_handoffs = 0
    markov_trigger_count = 0
    zscore_trigger_count = 0
    both_trigger_count = 0

    for obj in records.values():
        gold = ground_truth[eu.record_id(obj, ID_ALIASES)]
        total += 1
        ans = gold["answer"]
        pred_raw = eu.response_text(obj)[0].strip()
        question = gold["question"]
        pred_extracted = extract_from_pred(pred_raw)
        relay_stats = obj.get("relay_stats")
        if isinstance(relay_stats, dict):
            relay_stats_available += 1
            handoff_count = int(relay_stats.get("handoff_count", 0) or 0)
            total_handoffs += handoff_count
            markov_trigger_count += int(relay_stats.get("markov_trigger_count", 0) or 0)
            zscore_trigger_count += int(relay_stats.get("zscore_trigger_count", 0) or 0)
            both_trigger_count += int(relay_stats.get("both_trigger_count", 0) or 0)
            if handoff_count > 0:
                questions_with_handoff += 1

        ok = False
        options_map = {}

        if compare_pred_answer(pred_extracted, ans):
            ok = True
        elif pred_extracted not in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
            options_map = parse_options_from_question(question)
            ans_key = ans.upper()
            if ans_key in options_map:
                correct_option_text = options_map[ans_key].lower()
                if correct_option_text in pred_extracted or pred_extracted in correct_option_text:
                    ok = True
                if not ok and correct_option_text in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    correct_option_text = letter_switch[correct_option_text]
                    if correct_option_text in pred_extracted or pred_extracted in correct_option_text:
                        ok = True
                    elif correct_option_text == 'zero':
                        correct_option_text = 'no'
                        if correct_option_text in pred_extracted or pred_extracted in correct_option_text:
                            ok = True

        if ok:
            correct += 1

        y_true.append(ans)
        if ok:
            y_pred.append(ans)
        else:
            if pred_extracted not in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                pred_extracted = 'a'
            y_pred.append(pred_extracted)

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "f1_macro": calculate_f1_macro(y_true, y_pred),
        "relay_stats_available": relay_stats_available,
        "questions_with_handoff": questions_with_handoff,
        "total_handoffs": total_handoffs,
        "markov_trigger_count": markov_trigger_count,
        "zscore_trigger_count": zscore_trigger_count,
        "both_trigger_count": both_trigger_count,
    }


def print_verbose(result: dict, unknown_ids: list, diagnostics: dict) -> None:
    total = result["total"]
    print("[VERBOSE]")
    print(f"Correct: {result['correct']}")
    print(f"F1 (Macro): {result['f1_macro']:.6f}")
    if unknown_ids:
        print(f"Unknown ids skipped: {len(unknown_ids)} {unknown_ids[:20]}")
    print(
        f"Snapshot: malformed_lines={len(diagnostics['malformed_complete_lines'])} "
        f"duplicates={diagnostics['duplicate_records']} "
        f"partial_tail_ignored={diagnostics['ignored_partial_tail']}"
    )
    if result["relay_stats_available"] > 0:
        total_handoffs = result["total_handoffs"]
        questions_with_handoff = result["questions_with_handoff"]
        print("Relay Trigger Stats:")
        print(f"Questions with handoff: {questions_with_handoff}")
        print(f"Total handoffs: {total_handoffs}")
        print(f"Markov-triggered count: {result['markov_trigger_count']}")
        print(f"ZScore-triggered count: {result['zscore_trigger_count']}")
        print(f"Both-triggered count: {result['both_trigger_count']}")
        print(f"Avg handoffs/question: {(total_handoffs / total if total > 0 else 0.0):.4f}")
        print(
            "Avg handoffs/question with handoff: "
            f"{(total_handoffs / questions_with_handoff if questions_with_handoff > 0 else 0.0):.4f}"
        )
    else:
        print("Relay Trigger Stats: unavailable")


def main() -> None:
    ground_truth = load_ground_truth(eu.dataset_path(DATASET))
    processed = []
    for path in eu.resolve_input_files(INPUT_FILES, DATASET):
        try:
            records, diagnostics = eu.read_jsonl_snapshot(path, ID_ALIASES)
            records = eu.first_n(records, MAX_EVAL)
            unknown_ids = [key for key in records if key not in ground_truth]
            known = OrderedDict(
                (key, record) for key, record in records.items() if key in ground_truth
            )
            result = evaluate_records(known, ground_truth)
            basic = eu.summarize_basic(known.values())
            eu.print_basic_report(
                path,
                evaluated=result["total"],
                expected=EXPECTED_TOTAL,
                acc_line=eu.format_accuracy(result["correct"], result["total"]),
                basic=basic,
                note=f"{len(unknown_ids)} record(s) with unknown index skipped" if unknown_ids else None,
            )
            if VERBOSE:
                print_verbose(result, unknown_ids, diagnostics)
            processed.append(
                (path, {"ACC (%)": result["accuracy"] * 100, **eu.basic_overall_metrics(basic)})
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
