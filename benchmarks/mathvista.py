from __future__ import annotations

import json
from typing import Any

from .base import Benchmark

FORMAT_INSTRUCTION = (
    "Your final response line must begin with the literal string \"**Answer: \" "
    "and end with the literal string \"**\". "
)

ANSWER_REQUIREMENTS = {
    "multi_choice": (
        "Between the colon and the closing asterisks, write only the single "
        "option letter selected from the given choices."
    ),
    "integer": (
        "Between the colon and the closing asterisks, write only the final "
        "integer, without units or explanatory text."
    ),
    "float": (
        "Between the colon and the closing asterisks, write only the final "
        "number rounded to {precision} decimal {decimal_word}, without units "
        "or explanatory text."
    ),
    "list": (
        "Between the colon and the closing asterisks, write only the final "
        "Python-style list."
    ),
    "other": "Between the colon and the closing asterisks, write only the final answer.",
}


class MathVista(Benchmark):
    name = "mathvista"
    split = "testmini"
    id_field = "pid"
    image_field = "decoded_image"
    inference_fields = (
        "pid",
        "query",
        "question_type",
        "answer_type",
        "precision",
        "decoded_image",
    )
    record_fields = ("query", "question_type", "answer_type", "precision")
    required_columns = ("question", "choices", "metadata", "answer")
    expected_total = 1000
    max_new_tokens = 2048

    TASK_PROMPT_TEMPLATE = (
        "{query}\n"
        "Think step by step.\n\n"
        + FORMAT_INSTRUCTION
        + "{answer_requirement}"
    )

    def build_task_prompt(self, row: dict[str, Any]) -> str:
        if row.get("question_type") == "multi_choice":
            answer_requirement = ANSWER_REQUIREMENTS["multi_choice"]
        elif row.get("answer_type") == "integer":
            answer_requirement = ANSWER_REQUIREMENTS["integer"]
        elif row.get("answer_type") == "float":
            precision_value = row.get("precision")
            precision = int(precision_value) if precision_value is not None else 1
            decimal_word = "place" if precision == 1 else "places"
            answer_requirement = ANSWER_REQUIREMENTS["float"].format(
                precision=precision, decimal_word=decimal_word
            )
        elif row.get("answer_type") == "list":
            answer_requirement = ANSWER_REQUIREMENTS["list"]
        else:
            answer_requirement = ANSWER_REQUIREMENTS["other"]
        return self.TASK_PROMPT_TEMPLATE.format(
            query=str(row["query"]).strip(),
            answer_requirement=answer_requirement,
        )

    def task_prompt_signature(self) -> str:
        return self.TASK_PROMPT_TEMPLATE + "\n" + json.dumps(
            ANSWER_REQUIREMENTS, sort_keys=True, ensure_ascii=False
        )


BENCHMARK = MathVista()
