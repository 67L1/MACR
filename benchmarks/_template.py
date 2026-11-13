from __future__ import annotations

from .base import Benchmark


class MyDataset(Benchmark):
    name = "my_dataset"
    split = "test"
    id_field = "id"
    image_field = "image"
    inference_fields = ("id", "question", "image")
    record_fields = ("question",)
    required_columns = ("answer",)
    expected_total = None
    max_new_tokens = 2048

    TASK_PROMPT_TEMPLATE = (
        "Carefully analyze the image and question, and solve the problem step by step.\n"
        "End your response with one final line: Answer: <your answer>\n\n"
        "Question:\n"
        "{question}"
    )


BENCHMARK = MyDataset()
