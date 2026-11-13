from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


def common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def cache_length(cache: Any) -> int | None:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        return None
    try:
        return int(get_seq_length())
    except (TypeError, ValueError):
        return None


def supports_crop(cache: Any) -> bool:
    return callable(getattr(cache, "crop", None)) and cache_length(cache) is not None


@dataclass
class LedgerPlan:
    full_ids: list[int]
    keep: int | None = None
    cropped: int = 0

    @property
    def prefill_tokens(self) -> int:
        return len(self.full_ids) - (self.keep or 0)


@dataclass
class KVLedger:
    owner_model: str
    sample_fingerprint: str
    cache: Any
    token_ids: list[int]
    static_len: int
    payload: dict[str, Any] = field(default_factory=dict)
    epoch: int = 0
    released: bool = False

    def is_consistent(self) -> bool:
        return not self.released and cache_length(self.cache) == len(self.token_ids)

    def plan(self, target_ids: Sequence[int]) -> tuple[int | None, str | None]:
        if self.released or self.cache is None:
            return None, "ledger_released"
        if len(target_ids) <= self.static_len:
            return None, "target_within_static_prefix"
        keep = min(common_prefix_len(self.token_ids, target_ids), len(target_ids) - 1)
        if keep < self.static_len:
            return None, "static_prefix_mismatch"
        return keep, None

    def crop(self, keep: int) -> int:
        dropped = len(self.token_ids) - keep
        if keep < 0 or dropped < 0:
            raise ValueError(f"cannot keep {keep} of {len(self.token_ids)} cached positions")
        if dropped:
            self.cache.crop(keep)
            del self.token_ids[keep:]
        self.epoch += 1
        return dropped

    def reset(self, cache: Any, token_ids: Sequence[int]) -> None:
        self.cache = cache
        self.token_ids = [int(value) for value in token_ids]
        self.epoch += 1

    def append(self, cache: Any, token_id: int) -> None:
        self.cache = cache
        self.token_ids.append(int(token_id))
        self.epoch += 1

    def release(self) -> None:
        self.cache = None
        self.token_ids = []
        self.payload.clear()
        self.released = True
