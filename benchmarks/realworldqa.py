from __future__ import annotations

import json
import re
from typing import Any

from .base import Benchmark

ANSWER_INSTRUCTIONS = {
    "option_letter": (
        "Please answer directly with only the letter of the correct option and nothing else."
    ),
    "short_answer": "Please answer directly with a single word or number.",
}

ANSWER_REQUIREMENTS = {
    "option_letter": "Your final answer must be only the letter of the correct option.",
    "short_answer": (
        "Your final answer must be only a single word or number, without explanation."
    ),
}

ANSWER_PLACEHOLDERS = {
    "option_letter": "<option letter>",
    "short_answer": "<single word or number>",
}

REALWORLDQA_TASK_PROMPT_TEMPLATE = """Carefully inspect the image and answer the question step by step.

Show your reasoning before giving the final answer.
{answer_requirement}
Use "Answer:" exactly once, only on the final line.
Do not write anything after the final answer line.

Required response format:
Reasoning:
<your step-by-step reasoning>

Answer: {answer_placeholder}

Question:
{question}"""


def strip_answer_instruction(question: Any) -> tuple[str, str]:
    if not isinstance(question, str):
        raise ValueError(f"question is not a string: {question!r}")
    for answer_format, instruction in ANSWER_INSTRUCTIONS.items():
        if question.endswith("\n" + instruction):
            if question.count(instruction) != 1:
                raise ValueError("answer instruction appears more than once")
            body = question[: -len(instruction)].rstrip()
            if not body:
                raise ValueError("question body is empty")
            return body, answer_format
    raise ValueError("question does not end with a known answer instruction line")


class RealWorldQA(Benchmark):
    name = "realworldqa"
    split = "test"
    id_field = "pid"
    image_field = "image"
    inference_fields = ("pid", "question", "answer_format", "image")
    record_fields = ("question", "answer_format")
    required_columns = ("answer",)
    expected_total = 765
    max_new_tokens = 2048
    TASK_PROMPT_TEMPLATE = REALWORLDQA_TASK_PROMPT_TEMPLATE

    def prepare_dataset(self, dataset: Any) -> Any:
        missing = sorted({"question", "answer", "image"}.difference(dataset.column_names))
        if missing:
            raise ValueError(f"RealWorldQA is missing columns: {', '.join(missing)}")

        questions: list[str] = []
        answer_formats: list[str] = []
        errors: list[tuple[int, str]] = []
        for row_index, (question, answer) in enumerate(
            zip(dataset["question"], dataset["answer"])
        ):
            try:
                body, answer_format = strip_answer_instruction(question)
            except ValueError as error:
                errors.append((row_index, str(error)))
                continue
            answer_text = "" if answer is None else str(answer).strip()
            if not answer_text:
                errors.append((row_index, "empty answer"))
            elif answer_format == "option_letter" and not re.fullmatch(r"[A-Z]", answer_text):
                errors.append((row_index, f"option answer is not a letter: {answer_text!r}"))
            questions.append(body)
            answer_formats.append(answer_format)
        if errors:
            raise ValueError(f"RealWorldQA contains invalid row(s): {errors[:10]}")

        drop = [column for column in ("question", "answer_format", "pid") if column in dataset.column_names]
        dataset = dataset.remove_columns(drop)
        dataset = dataset.add_column("question", questions)
        dataset = dataset.add_column("answer_format", answer_formats)
        return dataset.add_column("pid", list(range(len(dataset))))

    def normalize_sample_id(self, value: Any) -> str:
        text = str(value).strip()
        if not re.fullmatch(r"\d+", text):
            raise ValueError(f"--sample-ids values must be non-negative integer pids, got {value!r}")
        return str(int(text))

    def build_task_prompt(self, row: dict[str, Any]) -> str:
        answer_format = row.get("answer_format")
        if answer_format not in ANSWER_REQUIREMENTS:
            raise ValueError(f"Unknown RealWorldQA answer_format: {answer_format!r}")
        return self.TASK_PROMPT_TEMPLATE.format(
            answer_requirement=ANSWER_REQUIREMENTS[answer_format],
            answer_placeholder=ANSWER_PLACEHOLDERS[answer_format],
            question=str(row["question"]).strip(),
        )

    def task_prompt_signature(self) -> str:
        return self.TASK_PROMPT_TEMPLATE + "\n" + json.dumps(
            {
                "instructions_removed": ANSWER_INSTRUCTIONS,
                "requirements": ANSWER_REQUIREMENTS,
                "placeholders": ANSWER_PLACEHOLDERS,
            },
            sort_keys=True,
            ensure_ascii=False,
        )


BENCHMARK = RealWorldQA()
