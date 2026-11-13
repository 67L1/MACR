from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from macr_ext import answer_bid, boundary

EXT_VERSION = 1
DELAY_CHOICES = ("off",) + boundary.PRESETS
ANSWER_CHOICES = ("off",) + answer_bid.RULES
_DELAY_TAGS = {"newline": "nl", "sentence": "sent", "clause": "clause"}
_ANSWER_TAGS = {"max_conf": "maxconf", "majority": "majority"}


def add_ext_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("macr_ext (delayed bidding / answer bidding)")
    group.add_argument(
        "--delay-bid",
        "--delay_bid",
        dest="delay_bid",
        choices=DELAY_CHOICES,
        default="sentence",
        help="Delayed bidding: after z drops below the threshold, keep writing until the next "
        "boundary, then bid; off = bid immediately (original MACR). "
        "newline = line breaks; sentence = sentence-ending punctuation and line breaks "
        "(default); clause = sentence plus commas, semicolons and colons.",
    )
    group.add_argument(
        "--max-bid-delay",
        "--max_bid_delay",
        dest="max_bid_delay",
        type=int,
        default=32,
        help="Delayed bidding: force the bid once this many tokens have passed since the "
        "trigger (default 32).",
    )
    group.add_argument(
        "--answer-bid",
        "--answer_bid",
        dest="answer_bid",
        choices=ANSWER_CHOICES,
        default="majority",
        help="Answer bidding: once an answer marker is written, every model answers after it; "
        "off = the last model in charge answers alone (original MACR); "
        "max_conf = the answer with the highest z wins; "
        "majority = hard majority vote, ties broken by z (default).",
    )
    group.add_argument(
        "--answer-max-tokens",
        "--answer_max_tokens",
        dest="answer_max_tokens",
        type=int,
        default=32,
        help="Answer bidding: maximum tokens per answer candidate (default 32).",
    )
    group.add_argument(
        "--answer-force-marker",
        "--answer_force_marker",
        dest="answer_force_marker",
        action="store_true",
        help="Answer bidding: if the response ends without an answer marker, append the "
        "dataset's standard marker and run answer bidding anyway.",
    )


@dataclass(frozen=True)
class ExtConfig:
    delay_bid: str = "sentence"
    max_bid_delay: int = 32
    answer_bid: str = "majority"
    answer_max_tokens: int = 32
    answer_force_marker: bool = False

    def __post_init__(self) -> None:
        if self.delay_bid not in DELAY_CHOICES:
            raise ValueError(f"--delay-bid must be one of {DELAY_CHOICES}")
        if self.answer_bid not in ANSWER_CHOICES:
            raise ValueError(f"--answer-bid must be one of {ANSWER_CHOICES}")
        if self.max_bid_delay < 1:
            raise ValueError("--max-bid-delay must be at least 1")
        if self.answer_max_tokens < 1:
            raise ValueError("--answer-max-tokens must be at least 1")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExtConfig":
        return cls(
            delay_bid=args.delay_bid,
            max_bid_delay=args.max_bid_delay,
            answer_bid=args.answer_bid,
            answer_max_tokens=args.answer_max_tokens,
            answer_force_marker=bool(args.answer_force_marker),
        )

    @property
    def delay_enabled(self) -> bool:
        return self.delay_bid != "off"

    @property
    def answer_enabled(self) -> bool:
        return self.answer_bid != "off"

    def tag(self) -> str:
        parts = []
        if self.delay_enabled:
            parts.append(f"db{_DELAY_TAGS[self.delay_bid]}-md{self.max_bid_delay}")
        if self.answer_enabled:
            part = f"ab{_ANSWER_TAGS[self.answer_bid]}-t{self.answer_max_tokens}"
            parts.append(part + ("-fm" if self.answer_force_marker else ""))
        return "ext-" + ("_".join(parts) if parts else "off")

    def describe(self) -> str:
        delay = (
            f"delay_bid={self.delay_bid} (max_bid_delay={self.max_bid_delay})"
            if self.delay_enabled
            else "delay_bid=off"
        )
        answer = (
            f"answer_bid={self.answer_bid} (answer_max_tokens={self.answer_max_tokens}, "
            f"force_marker={self.answer_force_marker})"
            if self.answer_enabled
            else "answer_bid=off"
        )
        return f"{delay} | {answer}"

    def metadata(self) -> dict[str, Any]:
        return {
            "version": EXT_VERSION,
            "tag": self.tag(),
            "delay_bid": self.delay_bid,
            "max_bid_delay": self.max_bid_delay if self.delay_enabled else None,
            "boundary_rules_sha256": boundary.rules_sha256() if self.delay_enabled else None,
            "answer_bid": self.answer_bid,
            "answer_max_tokens": self.answer_max_tokens if self.answer_enabled else None,
            "answer_force_marker": self.answer_force_marker if self.answer_enabled else None,
            "answer_rules_sha256": answer_bid.rules_sha256() if self.answer_enabled else None,
        }
