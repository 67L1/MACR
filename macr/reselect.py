from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from macr_ext import answer_bid


def reselect_record(record: dict[str, Any], rule: str) -> tuple[dict[str, Any], bool | None]:
    relay_stats = record.get("relay_stats")
    info = relay_stats.get("answer_bid") if isinstance(relay_stats, dict) else None
    if not isinstance(info, dict) or info.get("status") != "done":
        return record, None
    candidates = [answer_bid.AnswerCandidate.from_record(item) for item in info["candidates"]]
    selection = answer_bid.select_answer(candidates, rule, info["incumbent_model"])
    updated = copy.deepcopy(record)
    prefix = str(record.get("response", ""))[: int(info["prefix_chars"])]
    updated["response"] = prefix + selection.winner.answer_text
    updated["relay_stats"]["answer_bid"].update(
        {
            "reselect_rule": rule,
            "winner_model": selection.winner.model,
            "winner_answer": selection.winner.normalized,
            "fallback": selection.fallback,
            "changed": selection.changed,
        }
    )
    return updated, selection.changed


def read_complete_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        lines = lines[:-1]
    records: list[dict[str, Any]] = []
    skipped = 0
    for raw_line in lines:
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            skipped += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            skipped += 1
    return records, skipped


def reselect_file(
    path: Path, rules: Sequence[str], out_dir: Path | None = None
) -> dict[str, Path]:
    records, skipped = read_complete_records(path)
    statuses: Counter[str] = Counter()
    agreements: Counter[str] = Counter()
    for record in records:
        relay_stats = record.get("relay_stats")
        info = relay_stats.get("answer_bid") if isinstance(relay_stats, dict) else None
        statuses[info.get("status", "?") if isinstance(info, dict) else "none"] += 1
        if isinstance(info, dict) and info.get("status") == "done":
            agreements[str(info.get("agreement"))] += 1
    print(f"\n[reselect] {path}")
    print(
        f"  records={len(records)} skipped_lines={skipped} "
        f"answer_bid_status={dict(statuses)}"
    )
    print(f"  agreement={dict(sorted(agreements.items()))}")

    target_dir = out_dir or path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for rule in rules:
        output_path = target_dir / f"{path.stem}_reselect-{rule}.jsonl"
        changed = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                updated, is_changed = reselect_record(record, rule)
                changed += bool(is_changed)
                handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
        print(f"  {rule:<9} -> {output_path}  (answer differs from incumbent: {changed})")
        outputs[rule] = output_path
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline answer re-selection over logged answer-bid candidates."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="answer-bid JSONL file(s)")
    parser.add_argument(
        "--rules",
        nargs="+",
        choices=answer_bid.RESELECT_RULES,
        default=list(answer_bid.RESELECT_RULES),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: next to each input file"
    )
    args = parser.parse_args(argv)
    for path in args.inputs:
        reselect_file(path.expanduser().resolve(), args.rules, args.out_dir)


if __name__ == "__main__":
    main()
