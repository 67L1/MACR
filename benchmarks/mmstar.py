from __future__ import annotations

from .base import Benchmark


class MMStar(Benchmark):
    name = "mmstar"
    split = "val"
    id_field = "index"
    image_field = "image"
    inference_fields = ("index", "question", "image")
    record_fields = ("question",)
    required_columns = ("answer",)
    expected_total = 1500
    max_new_tokens = 2048

    TASK_PROMPT_TEMPLATE = (
        "{question}\n"
        "Compare all options. "
        "Think step by step, but choose exactly one option from A, B, C, and D.\n"
        "You must give your final answer using the **EXACT format** below:\n"
        "**Answer: [Your Final Option]**\n\n"
        "For example:\n"
        "[Here is your reasoning text].\n"
        "**Answer: B**"
    )


BENCHMARK = MMStar()
