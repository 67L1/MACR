from __future__ import annotations

from .base import Benchmark

LOGICVISTA_TASK_PROMPT_TEMPLATE = """Carefully analyze the image and question, and solve the problem step by step.

Show your reasoning before giving the final answer. 

End your response with exactly one final answer line using the exact option label shown in the image or question.
For multiple answers, use uppercase labels in ascending order separated by ", ".
In the final answer line, do not return option content, zero-based indices, explanations, alternatives, or repeated labels.
Do not write anything after the final answer line.

Required response format:
Reasoning:
<your step-by-step reasoning>

Answer: <label(s)>

Final answer examples:
Answer: C
Answer: A, C
Answer: 3

Question:
{question}"""


class LogicVista(Benchmark):
    name = "logicvista"
    split = "test"
    id_field = "id"
    image_field = "image"
    inference_fields = ("id", "question", "image")
    record_fields = ("question",)
    required_columns = (
        "answer",
        "reasoning",
        "skill",
        "broad_capability",
        "specific_capability",
    )
    expected_total = 448
    max_new_tokens = 2048
    TASK_PROMPT_TEMPLATE = LOGICVISTA_TASK_PROMPT_TEMPLATE


BENCHMARK = LogicVista()
