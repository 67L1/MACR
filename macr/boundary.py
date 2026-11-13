from __future__ import annotations

import hashlib
import re

PRESETS = ("newline", "sentence", "clause")

_CLOSERS = r"[\"'”’)\]}*_$`]*"
_ASCII_MARKS = {"sentence": ".!?", "clause": ".!?,;:"}
_CJK_MARKS = {"sentence": "。！？", "clause": "。！？，；："}

_ENUMERATOR_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:[-*•][ \t]*)?(?:\*\*)?"
    r"(?:(?:step|option|choice|case|part)[ \t]*)?"
    r"\(?(?:\d{1,3}|[a-z]|[ivx]{1,4})\)?\.$",
    re.IGNORECASE,
)
_ABBREVIATION_RE = re.compile(
    r"(?:^|[^a-z])(?:e\.g|i\.e|etc|vs|approx|fig|figs|eq|eqs|cf|resp|viz|al|mr|mrs|ms|dr)\.$",
    re.IGNORECASE,
)
_TAIL_CHARS = 256


def _false_terminal(text_through_mark: str) -> bool:
    if not text_through_mark.endswith("."):
        return False
    return bool(
        _ENUMERATOR_RE.search(text_through_mark)
        or _ABBREVIATION_RE.search(text_through_mark)
    )


class BoundaryDetector:
    def __init__(self, preset: str) -> None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown boundary preset {preset!r}; choose from {PRESETS}")
        self.preset = preset
        ascii_marks = re.escape(_ASCII_MARKS.get(preset, ""))
        cjk_marks = _CJK_MARKS.get(preset, "")
        self._terminal_re = (
            re.compile(rf"([{ascii_marks}]){_CLOSERS}$") if ascii_marks else None
        )
        self._terminal_space_re = (
            re.compile(rf"([{ascii_marks}]){_CLOSERS}[ \t]+$") if ascii_marks else None
        )
        self._cjk_re = re.compile(rf"[{cjk_marks}]{_CLOSERS}$") if cjk_marks else None

    def after_commit(self, text: str, token: str) -> str | None:
        if "\n" in token:
            return "newline"
        tail = text[-_TAIL_CHARS:]
        if self._cjk_re is not None and self._cjk_re.search(tail):
            return "cjk"
        if self._terminal_space_re is not None:
            match = self._terminal_space_re.search(tail)
            if match and not _false_terminal(tail[: match.start(1) + 1]):
                return "punct_space"
        return None

    def before_token(self, text: str, token: str) -> str | None:
        if self._terminal_re is None:
            return None
        if not token or token[0] not in " \t" or "\n" in token or not token.strip():
            return None
        tail = text[-_TAIL_CHARS:]
        match = self._terminal_re.search(tail)
        if match and not _false_terminal(tail[: match.start(1) + 1]):
            return "lookahead"
        return None


def rules_sha256() -> str:
    parts = (
        _CLOSERS,
        repr(sorted(_ASCII_MARKS.items())),
        repr(sorted(_CJK_MARKS.items())),
        _ENUMERATOR_RE.pattern,
        _ABBREVIATION_RE.pattern,
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
