from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .base import Benchmark

IDENTITY_FIELDS = (
    "category",
    "subcategory",
    "set_id",
    "figure_id",
    "question_id",
)

HALLUSIONBENCH_TASK_PROMPT_TEMPLATE = """Carefully inspect the image and answer the yes-or-no question step by step.

Treat the image as the authoritative visual context. Determine whether the statement asked by the question is true or false.

Show your reasoning before giving the final answer.
Your final answer must be Yes or No.
Use "Answer:" exactly once, only on the final line.
Do not give alternative answers or write anything after the final answer line.

Required response format:
Reasoning:
<your step-by-step reasoning>

Answer: <Yes or No>

Question:
{question}"""


def _identity_text(value: Any, *, field: str, row_index: int) -> str:
    if value is None:
        raise ValueError(f"HallusionBench row {row_index} has null identity field {field!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"HallusionBench row {row_index} has empty identity field {field!r}")
    return text


class HallusionBench(Benchmark):
    name = "hallusionbench"
    split = "image"
    id_field = "pid"
    image_field = "image"
    inference_fields = ("pid", *IDENTITY_FIELDS, "question", "image")
    record_fields = (*IDENTITY_FIELDS, "question")
    required_columns = (
        "visual_input",
        "sample_note",
        "gt_answer_details",
        "gt_answer",
        "filename",
    )
    expected_total = 951
    max_new_tokens = 2048
    TASK_PROMPT_TEMPLATE = HALLUSIONBENCH_TASK_PROMPT_TEMPLATE

    def prepare_dataset(self, dataset: Any) -> Any:
        needed = {*IDENTITY_FIELDS, "gt_answer", "visual_input"}
        missing = sorted(needed.difference(dataset.column_names))
        if missing:
            raise ValueError(f"HallusionBench is missing columns: {', '.join(missing)}")

        invalid_answers = [
            (row_index, value)
            for row_index, value in enumerate(dataset["gt_answer"])
            if str(value).strip() not in {"0", "1"}
        ]
        if invalid_answers:
            raise ValueError(
                f"HallusionBench contains invalid gt_answer value(s): {invalid_answers[:10]}"
            )
        invalid_visual_inputs = [
            (row_index, value)
            for row_index, value in enumerate(dataset["visual_input"])
            if str(value).strip() not in {"1", "2"}
        ]
        if invalid_visual_inputs:
            raise ValueError(
                "Expected only the HallusionBench image split, but found invalid "
                f"visual_input value(s): {invalid_visual_inputs[:10]}"
            )

        identity_columns = {field: dataset[field] for field in IDENTITY_FIELDS}
        identity_keys = [
            tuple(
                _identity_text(identity_columns[field][row_index], field=field, row_index=row_index)
                for field in IDENTITY_FIELDS
            )
            for row_index in range(len(dataset))
        ]
        duplicates = [key for key, count in Counter(identity_keys).items() if count > 1]
        if duplicates:
            raise ValueError(
                f"HallusionBench contains duplicate composite identity key(s): {duplicates[:10]}"
            )

        if "pid" in dataset.column_names:
            dataset = dataset.remove_columns("pid")
        return dataset.add_column("pid", list(range(len(dataset))))

    def normalize_sample_id(self, value: Any) -> str:
        text = str(value).strip()
        if not re.fullmatch(r"\d+", text):
            raise ValueError(f"--sample-ids values must be non-negative integer pids, got {value!r}")
        return str(int(text))


BENCHMARK = HallusionBench()
