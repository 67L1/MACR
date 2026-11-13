from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Sequence

RULES = ("max_conf", "majority")
RESELECT_RULES = RULES + ("incumbent",)

ANSWER_MARKER_RE = re.compile(
    r"(?m)(?:^|(?<=\*\*))[ \t]*(?:\*\*)?[ \t]*"
    r"(?:Final[ \t]+[Aa]nswer|Answer|ANSWER)[ \t]*(?:\*\*)?[ \t]*[:：]"
)
_MARKER_LOOKBACK = 64

FORCED_MARKERS = {"mmstar": "\n\n**Answer:", "mathvista": "\n\n**Answer:"}
DEFAULT_FORCED_MARKER = "\n\nAnswer:"

_FORMAT_ONLY_RE = re.compile(r"[\s*_`$]*")
NORMALIZER_VERSION = "eva-extractors+complete-line-v1"


def find_answer_marker(text: str, since: int = 0) -> int | None:
    for match in ANSWER_MARKER_RE.finditer(text, max(0, since - _MARKER_LOOKBACK)):
        if match.end() > since:
            return match.end()
    return None


def forced_marker(benchmark_name: str) -> str:
    return FORCED_MARKERS.get(benchmark_name, DEFAULT_FORCED_MARKER)


def answer_line(text: str) -> str:
    first_visible = re.search(r"\S", text)
    if first_visible is None:
        return ""
    newline = text.find("\n", first_visible.start())
    return text if newline < 0 else text[:newline]


@dataclass
class AnswerCandidate:
    model: str
    answer_text: str
    normalized: str | None
    z: float | None
    sum_logprob: float | None
    mean_logprob: float | None
    n_tokens: int
    answer_tokens: int
    content_tokens: int
    line_complete: bool | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "AnswerCandidate":
        return cls(**{item.name: record.get(item.name) for item in fields(cls)})


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def build_candidate(
    *,
    model: str,
    text: str,
    token_details: Sequence[tuple[str, int, float]],
    finished: bool,
    z_score: Callable[[list[float]], float],
    normalize: Callable[[str], str | None],
) -> AnswerCandidate:
    answer_text = answer_line(text)
    line_complete = finished or text[len(answer_text) :].startswith("\n")
    pieces = [str(token_text) for token_text, _, _ in token_details]
    span_end = len(answer_line("".join(pieces)))
    answer_tokens = 0
    content_logprobs: list[float] = []
    offset = 0
    for piece, (_, _, log_prob) in zip(pieces, token_details):
        if offset >= span_end:
            break
        answer_tokens += 1
        if not _FORMAT_ONLY_RE.fullmatch(piece):
            content_logprobs.append(float(log_prob))
        offset += len(piece)
    if finished and not "".join(pieces[answer_tokens:]):
        answer_tokens = len(pieces)

    z = sum_logprob = mean_logprob = None
    if content_logprobs:
        z = _finite(float(z_score(content_logprobs)))
        sum_logprob = sum(content_logprobs)
        mean_logprob = sum_logprob / len(content_logprobs)
    return AnswerCandidate(
        model=model,
        answer_text=answer_text,
        normalized=(
            normalize(answer_text) if answer_text.strip() and line_complete else None
        ),
        z=z,
        sum_logprob=sum_logprob,
        mean_logprob=mean_logprob,
        n_tokens=len(token_details),
        answer_tokens=answer_tokens,
        content_tokens=len(content_logprobs),
        line_complete=line_complete,
    )


@dataclass
class Selection:
    winner: AnswerCandidate
    tally: dict[str, int]
    agreement: str
    fallback: bool
    changed: bool


def _answer_key(candidate: AnswerCandidate) -> str:
    if candidate.normalized is not None:
        return candidate.normalized
    return candidate.answer_text.strip()


def _agreement(tally: dict[str, int], total: int) -> str:
    sizes = sorted(tally.values(), reverse=True)
    if not sizes:
        text = "0"
    elif len(sizes) == 1:
        text = f"{sizes[0]}-0"
    else:
        text = "-".join(str(size) for size in sizes)
    abstained = total - sum(sizes)
    return text + (f"+{abstained}abstain" if abstained else "")


def select_answer(
    candidates: Sequence[AnswerCandidate], rule: str, incumbent: str
) -> Selection:
    if rule not in RESELECT_RULES:
        raise ValueError(f"Unknown answer rule {rule!r}; choose from {RESELECT_RULES}")
    by_model = {candidate.model: candidate for candidate in candidates}
    if incumbent not in by_model:
        raise ValueError(f"Incumbent {incumbent!r} has no answer candidate")
    incumbent_candidate = by_model[incumbent]

    groups: dict[str, list[AnswerCandidate]] = {}
    for candidate in candidates:
        if candidate.normalized is not None:
            groups.setdefault(candidate.normalized, []).append(candidate)
    tally = {answer: len(members) for answer, members in groups.items()}

    def rank(candidate: AnswerCandidate) -> tuple[float, bool]:
        confidence = candidate.z if candidate.z is not None else -math.inf
        return confidence, candidate.model == incumbent

    fallback = False
    if rule == "incumbent":
        winner = incumbent_candidate
    elif not groups:
        winner = incumbent_candidate
        fallback = True
    elif rule == "max_conf":
        winner = max((c for members in groups.values() for c in members), key=rank)
    else:
        best_answer = max(
            groups,
            key=lambda answer: (len(groups[answer]), max(map(rank, groups[answer]))),
        )
        winner = max(groups[best_answer], key=rank)
    return Selection(
        winner=winner,
        tally=tally,
        agreement=_agreement(tally, len(candidates)),
        fallback=fallback,
        changed=_answer_key(winner) != _answer_key(incumbent_candidate),
    )


Normalizer = Callable[[str, str], "str | None"]


def _mmstar_normalizer() -> Normalizer:
    import eva_mmstar

    def normalize(prompt: str, answer_text: str) -> str | None:
        extracted = eva_mmstar.extract_from_pred("Answer:" + answer_text)
        if not extracted:
            return None
        letter = re.match(r"^\(?([a-z])\)?(?![a-z0-9])", extracted)
        return letter.group(1).upper() if letter else extracted

    return normalize


def _hallusionbench_normalizer() -> Normalizer:
    import eva_hallusionbench

    def normalize(prompt: str, answer_text: str) -> str | None:
        return eva_hallusionbench.extract_binary_answer("Answer:" + answer_text)["prediction"]

    return normalize


def _logicvista_normalizer() -> Normalizer:
    import eva_logicvista

    def normalize(prompt: str, answer_text: str) -> str | None:
        answer, _ = eva_logicvista.parse_compatible_answer("Answer:" + answer_text, {})
        return ",".join(sorted(answer)) if answer else None

    return normalize


_MATHVISTA_FLOAT_RE = re.compile(r"write only the final number rounded to (\d+) decimal")
_CHOICE_LINE_RE = re.compile(r"^[ \t]*\(([A-Z])\)[ \t]*(.*?)[ \t]*$", re.MULTILINE)


def mathvista_problem(prompt: str) -> dict[str, Any]:
    from benchmarks.mathvista import ANSWER_REQUIREMENTS

    problem: dict[str, Any] = {
        "question_type": "free_form",
        "answer_type": "text",
        "precision": None,
        "choices": [],
    }
    if ANSWER_REQUIREMENTS["multi_choice"] in prompt:
        problem["question_type"] = "multi_choice"
        _, _, choice_block = prompt.partition("Choices:")
        choices = [text for _, text in _CHOICE_LINE_RE.findall(choice_block)]
        problem["choices"] = choices or [""] * 26
    elif ANSWER_REQUIREMENTS["integer"] in prompt:
        problem["answer_type"] = "integer"
    elif ANSWER_REQUIREMENTS["list"] in prompt:
        problem["answer_type"] = "list"
    else:
        match = _MATHVISTA_FLOAT_RE.search(prompt)
        if match:
            problem["answer_type"] = "float"
            problem["precision"] = int(match.group(1))
    return problem


def _mathvista_normalizer() -> Normalizer:
    import eva_mathvista

    def normalize(prompt: str, answer_text: str) -> str | None:
        problem = mathvista_problem(prompt)
        status, extraction, _ = eva_mathvista.parse_candidate(answer_text, problem)
        if status != "ok" or extraction is None:
            return None
        if problem["answer_type"] == "float":
            value = float(extraction)
            precision = problem["precision"]
            return f"{value:.{precision}f}" if precision is not None else repr(value)
        return extraction

    return normalize


def _normalize_generic(prompt: str, answer_text: str) -> str | None:
    lines = [line for line in answer_text.splitlines() if line.strip()]
    if not lines:
        return None
    text = re.sub(r"[*_`$]", "", lines[0]).strip()
    text = text.strip(".,;:!?。，；：！？ \t").lower()
    return re.sub(r"\s+", " ", text) or None


_NORMALIZER_FACTORIES: dict[str, Callable[[], Normalizer]] = {
    "mmstar": _mmstar_normalizer,
    "mathvista": _mathvista_normalizer,
    "logicvista": _logicvista_normalizer,
    "hallusionbench": _hallusionbench_normalizer,
}


def get_normalizer(benchmark_name: str) -> Normalizer:
    factory = _NORMALIZER_FACTORIES.get(benchmark_name)
    normalize = factory() if factory is not None else _normalize_generic

    def safe_normalize(prompt: str, answer_text: str) -> str | None:
        if not answer_text.strip():
            return None
        try:
            value = normalize(prompt, answer_text)
        except Exception:
            return None
        return value or None

    return safe_normalize


def rules_sha256() -> str:
    parts = (
        ANSWER_MARKER_RE.pattern,
        repr(sorted(FORCED_MARKERS.items())),
        DEFAULT_FORCED_MARKER,
        _FORMAT_ONLY_RE.pattern,
        NORMALIZER_VERSION,
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
