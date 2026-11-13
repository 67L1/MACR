from __future__ import annotations

import hashlib

RELAY_INSTRUCTION = (
    "You are solving questions in a bidding relay system. "
    "The answer may be continued from another model's earlier draft, and that "
    "draft may be wrong. Treat previous reasoning as tentative, re-inspect the "
    "image and question, and correct unsupported conclusions."
)


def compose_prompt(task_prompt: str) -> str:
    return f"{RELAY_INSTRUCTION}\n\n{task_prompt}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relay_instruction_sha256() -> str:
    return sha256_text(RELAY_INSTRUCTION)
