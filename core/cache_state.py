from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


class CacheStateError(RuntimeError):
    pass


def history_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_sequence_length(past_key_values: Any) -> int | None:
    if past_key_values is None:
        return None
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length())
        except (TypeError, ValueError):
            return None
    try:
        return int(past_key_values[0][0].shape[-2])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


@dataclass
class PreparedSample:
    owner_model: str
    sample_fingerprint: str
    image_path: str
    prompt: str
    payload: dict[str, Any] = field(default_factory=dict)
    released: bool = False

    def assert_usable(self, owner_model: str, sample_fingerprint: str) -> None:
        if self.released:
            raise CacheStateError("prepared sample has already been released")
        if self.owner_model != owner_model:
            raise CacheStateError(
                f"prepared sample owner mismatch: {self.owner_model} != {owner_model}"
            )
        if self.sample_fingerprint != sample_fingerprint:
            raise CacheStateError("prepared sample fingerprint mismatch")

    def release(self) -> None:
        if self.released:
            return
        self.payload.clear()
        self.released = True


@dataclass
class ContinuationState:
    owner_model: str
    sample_fingerprint: str
    source_history_fingerprint: str
    committed_history_fingerprint: str
    past_key_values: Any
    pending_token_id: int | None
    cache_seq_len: int | None
    expected_cache_seq_len: int | None
    source: str
    adapter_payload: dict[str, Any] = field(default_factory=dict)
    pending_consumed: bool = False
    released: bool = False

    def validate(
        self,
        *,
        owner_model: str,
        sample_fingerprint: str,
        committed_history_fingerprint: str,
    ) -> tuple[bool, str | None]:
        if self.released:
            return False, "state_released"
        if self.pending_consumed:
            return False, "pending_token_already_consumed"
        if self.owner_model != owner_model:
            return False, "owner_mismatch"
        if self.sample_fingerprint != sample_fingerprint:
            return False, "sample_fingerprint_mismatch"
        if self.committed_history_fingerprint != committed_history_fingerprint:
            return False, "history_fingerprint_mismatch"
        if self.past_key_values is None:
            return False, "missing_past_key_values"
        if self.pending_token_id is None:
            return False, "missing_pending_token"
        if self.cache_seq_len is None:
            return False, "unknown_cache_length"
        actual_cache_seq_len = cache_sequence_length(self.past_key_values)
        if actual_cache_seq_len is None:
            return False, "unknown_live_cache_length"
        if actual_cache_seq_len != self.cache_seq_len:
            return False, "live_cache_length_mismatch"
        if (
            self.expected_cache_seq_len is not None
            and self.cache_seq_len != self.expected_cache_seq_len
        ):
            return False, "cache_length_mismatch"
        return True, None

    def claim_pending_token(self) -> int:
        if self.released:
            raise CacheStateError("cannot consume a released continuation state")
        if self.pending_consumed:
            raise CacheStateError("pending token has already been consumed")
        if self.pending_token_id is None:
            raise CacheStateError("continuation state has no pending token")
        self.pending_consumed = True
        return int(self.pending_token_id)

    def release(self) -> None:
        if self.released:
            return
        self.past_key_values = None
        self.adapter_payload.clear()
        self.released = True


@dataclass
class CandidateResult:
    model_name: str
    text: str
    log_probs: list[float]
    has_eos: bool
    token_details: list[tuple[str, int, float]]
    token_ids: list[int]
    continuation_state: ContinuationState | None
    z_score: float | None = None

    def release_state(self) -> bool:
        if self.continuation_state is None:
            return False
        was_released = self.continuation_state.released
        self.continuation_state.release()
        self.continuation_state = None
        return not was_released


class OptCacheStateManager:
    def __init__(self) -> None:
        self.active_state: ContinuationState | None = None
        self.promotions = 0
        self.invalid_promotions = 0
        self.released_states = 0
        self.fallback_reason_counts: dict[str, int] = {}

    def _record_invalid(self, reason: str) -> None:
        self.invalid_promotions += 1
        self.fallback_reason_counts[reason] = (
            self.fallback_reason_counts.get(reason, 0) + 1
        )

    def _release_active(self) -> None:
        if self.active_state is None:
            return
        was_released = self.active_state.released
        self.active_state.release()
        if not was_released:
            self.released_states += 1
        self.active_state = None

    def promote(
        self,
        candidate: CandidateResult,
        *,
        expected_owner: str,
        sample_fingerprint: str,
        committed_text: str,
        adapter_validator: Callable[[ContinuationState, str], tuple[bool, str | None]]
        | None = None,
    ) -> ContinuationState | None:
        state = candidate.continuation_state
        if state is None:
            self._record_invalid("missing_candidate_state")
            self._release_active()
            return None
        valid, reason = state.validate(
            owner_model=expected_owner,
            sample_fingerprint=sample_fingerprint,
            committed_history_fingerprint=history_fingerprint(committed_text),
        )
        if valid and adapter_validator is not None:
            valid, reason = adapter_validator(state, committed_text)
        if not valid:
            reason = reason or "adapter_validation_failed"
            self._record_invalid(reason)
            if candidate.release_state():
                self.released_states += 1
            self._release_active()
            return None

        old_state = self.active_state
        self.active_state = state
        candidate.continuation_state = None
        if old_state is not None and old_state is not state:
            was_released = old_state.released
            old_state.release()
            if not was_released:
                self.released_states += 1
        self.promotions += 1
        return state

    def replace_after_decode(
        self, old_state: ContinuationState | None, new_state: ContinuationState
    ) -> None:
        if old_state is not None and old_state is not new_state:
            was_released = old_state.released
            old_state.release()
            if not was_released:
                self.released_states += 1
        self.active_state = new_state

    def clear(self) -> None:
        self._release_active()


def build_sample_fingerprint(image_path: str, prompt: str) -> str:
    path = Path(image_path).resolve()
    stat = path.stat()
    payload = "\0".join(
        (str(path), str(stat.st_size), str(stat.st_mtime_ns), prompt)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pin_tensor_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().to("cpu", copy=True).contiguous()
        if torch.cuda.is_available() and not tensor.is_pinned():
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                pass
        return tensor
    if isinstance(value, dict):
        return {key: pin_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(pin_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [pin_tensor_tree(item) for item in value]
    return value


def to_device_tree(value: Any, device: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(
            device,
            non_blocking=bool(value.device.type == "cpu" and value.is_pinned()),
        )
    if isinstance(value, dict):
        return {key: to_device_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device_tree(item, device) for item in value)
    if isinstance(value, list):
        return [to_device_tree(item, device) for item in value]
    return value


def as_plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {key: item for key, item in value.items()}
    raise TypeError(f"Expected mapping-like model inputs, got {type(value).__name__}")


def single_row_token_ids(input_ids: Any) -> list[int]:
    if input_ids is None:
        return []
    if torch.is_tensor(input_ids):
        if input_ids.ndim == 2:
            input_ids = input_ids[0]
        return [int(value) for value in input_ids.detach().cpu().tolist()]
    return [int(value) for value in input_ids]


def greedy_cache_fastpath_is_neutral(
    generation_config: Any,
) -> tuple[bool, list[str]]:
    if generation_config is None:
        return False, ["missing_generation_config"]
    neutral_values: dict[str, tuple[Any, ...]] = {
        "repetition_penalty": (None, 1.0),
        "encoder_repetition_penalty": (None, 1.0),
        "no_repeat_ngram_size": (None, 0),
        "encoder_no_repeat_ngram_size": (None, 0),
        "bad_words_ids": (None, [], ()),
        "force_words_ids": (None, [], ()),
        "suppress_tokens": (None, [], ()),
        "begin_suppress_tokens": (None, [], ()),
        "forced_bos_token_id": (None,),
        "forced_eos_token_id": (None,),
        "exponential_decay_length_penalty": (None,),
        "sequence_bias": (None, {}),
        "watermarking_config": (None,),
        "token_healing": (None, False),
        "min_length": (None, 0),
        "min_new_tokens": (None, 0),
        "num_beams": (None, 1),
        "num_beam_groups": (None, 1),
        "diversity_penalty": (None, 0.0),
        "penalty_alpha": (None,),
        "guidance_scale": (None, 1.0),
        "dola_layers": (None,),
    }
    non_neutral: list[str] = []
    for field_name, allowed_values in neutral_values.items():
        value = getattr(generation_config, field_name, None)
        if not any(value == allowed for allowed in allowed_values):
            non_neutral.append(f"{field_name}={value!r}")
    return not non_neutral, non_neutral
