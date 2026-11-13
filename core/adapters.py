from __future__ import annotations

import importlib
import inspect
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from core.cache_state import (
    CacheStateError,
    CandidateResult,
    ContinuationState,
    PreparedSample,
    as_plain_mapping,
    build_sample_fingerprint,
    cache_sequence_length,
    greedy_cache_fastpath_is_neutral,
    history_fingerprint,
    pin_tensor_tree,
    single_row_token_ids,
    to_device_tree,
)
from core.kv_ledger import KVLedger, LedgerPlan, supports_crop
from core.runtime import require_flash_attn, timer

QWEN_MAX_PIXELS = 3_686_400
INTERNVL_INPUT_SIZE = 448
INTERNVL_MAX_NUM = 12
INTERNVL_USE_THUMBNAIL = True
MINICPM_MAX_SLICE_NUMS = 9
MINICPM_ENABLE_THINKING = False
MINICPM_ITERATION = True
FLASH_ATTN_IMPLEMENTATION = "flash_attention_2"


def kv_ledger_generation_check(generation_config: Any) -> tuple[bool, list[str]]:
    _, reasons = greedy_cache_fastpath_is_neutral(generation_config)
    cache_implementation = getattr(generation_config, "cache_implementation", None)
    if cache_implementation is not None:
        reasons = reasons + [f"cache_implementation={cache_implementation!r}"]
    return not reasons, reasons


class BaseVLModel:
    MODEL_NAME = "Base"

    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.prepared_sample: PreparedSample | None = None
        self.candidate_cache_fastpath_enabled = True
        self.candidate_cache_fastpath_disabled_reasons: list[str] = []
        self.ledger: KVLedger | None = None
        self.kv_ledger_enabled = True
        self.kv_ledger_disabled_reasons: list[str] = []
        self.sample_stats: dict[str, Any] = {}
        self.ori_input_token: dict[str, int | None] | None = None
        self._reset_sample_stats()

    def load(self) -> None:
        raise NotImplementedError

    def disable_candidate_cache(self) -> None:
        self.candidate_cache_fastpath_enabled = False
        if "operator_disabled" not in self.candidate_cache_fastpath_disabled_reasons:
            self.candidate_cache_fastpath_disabled_reasons.append("operator_disabled")

    def _set_ori_input_token(self, total: int, image: int | None) -> None:
        self.ori_input_token = {
            "total": int(total),
            "image": None if image is None else int(image),
            "text": None if image is None else int(total) - int(image),
        }

    def _reset_sample_stats(self) -> None:
        self.ori_input_token = None
        self.sample_stats = {
            "static_input_cache_hits": 0,
            "static_input_cache_misses": 0,
            "continuation_cache_fallbacks": 0,
            "fallback_reason_counts": {},
            "candidate_cache_fastpath_enabled": (
                self.candidate_cache_fastpath_enabled
            ),
            "candidate_cache_fastpath_disabled_reasons": list(
                self.candidate_cache_fastpath_disabled_reasons
            ),
            "kv_ledger_enabled": self.kv_ledger_enabled,
            "kv_ledger_disabled_reasons": list(self.kv_ledger_disabled_reasons),
            "ledger_cold_prefills": 0,
            "ledger_cold_prefill_tokens": 0,
            "ledger_reuse_calls": 0,
            "ledger_reused_tokens": 0,
            "ledger_prefilled_tokens": 0,
            "ledger_cropped_tokens": 0,
            "ledger_fallback_reason_counts": {},
            "ledger_calls": [],
        }

    def prepare_sample(
        self, image_path: str, prompt: str, sample_fingerprint: str
    ) -> None:
        raise NotImplementedError

    def release_sample(self) -> None:
        self._release_ledger()
        if self.prepared_sample is not None:
            self.prepared_sample.release()
        self.prepared_sample = None

    def _install_prepared_sample(
        self,
        *,
        image_path: str,
        prompt: str,
        sample_fingerprint: str,
        payload: dict[str, Any],
    ) -> None:
        self.release_sample()
        self._reset_sample_stats()
        self.prepared_sample = PreparedSample(
            owner_model=self.MODEL_NAME,
            sample_fingerprint=sample_fingerprint,
            image_path=image_path,
            prompt=prompt,
            payload=payload,
        )
        self.sample_stats["static_input_cache_misses"] = 1

    def _require_prepared(self, image_path: str, prompt: str) -> PreparedSample:
        prepared = self.prepared_sample
        if prepared is None:
            raise CacheStateError(
                f"{self.MODEL_NAME} has no prepared sample; call prepare_sample first"
            )
        expected = build_sample_fingerprint(image_path, prompt)
        prepared.assert_usable(self.MODEL_NAME, expected)
        self.sample_stats["static_input_cache_hits"] += 1
        return prepared

    def _record_cache_fallback(self, reason: str) -> None:
        self.sample_stats["continuation_cache_fallbacks"] += 1
        counts = self.sample_stats["fallback_reason_counts"]
        counts[reason] = counts.get(reason, 0) + 1

    def _validate_pending_history_payload(
        self, state: ContinuationState,
    ) -> tuple[bool, str | None]:
        history_ids = state.adapter_payload.get("history_token_ids")
        if not isinstance(history_ids, list) or not history_ids:
            return False, "missing_history_token_ids"
        if state.pending_token_id is None:
            return False, "missing_pending_token"
        if int(history_ids[-1]) != int(state.pending_token_id):
            return False, "pending_token_history_mismatch"
        return self._validate_ledger_binding(state)

    def _validate_ledger_binding(
        self, state: ContinuationState
    ) -> tuple[bool, str | None]:
        epoch = state.adapter_payload.get("ledger_epoch")
        if epoch is None:
            return True, None
        ledger = self.ledger
        if (
            ledger is None
            or ledger.cache is not state.past_key_values
            or ledger.epoch != epoch
        ):
            return False, "ledger_advanced"
        return True, None

    def _release_ledger(self) -> None:
        if self.ledger is not None:
            self.ledger.release()
        self.ledger = None

    def _disable_kv_ledger(self, reason: str) -> None:
        self.kv_ledger_enabled = False
        if reason not in self.kv_ledger_disabled_reasons:
            self.kv_ledger_disabled_reasons.append(reason)
        self.sample_stats["kv_ledger_enabled"] = False
        self.sample_stats["kv_ledger_disabled_reasons"] = list(
            self.kv_ledger_disabled_reasons
        )
        print(f"[KV-Ledger][{self.MODEL_NAME}] disabled; every later call uses a full prefill: {reason}")
        self._release_ledger()

    def _record_ledger_fallback(self, reason: str) -> None:
        counts = self.sample_stats["ledger_fallback_reason_counts"]
        counts[reason] = counts.get(reason, 0) + 1

    def _ledger_plan(self, full_ids: list[int]) -> LedgerPlan:
        plan = LedgerPlan(full_ids=full_ids)
        ledger = self.ledger
        if not self.kv_ledger_enabled or ledger is None:
            self._release_ledger()
            return plan
        prepared = self.prepared_sample
        keep: int | None = None
        if (
            prepared is None
            or prepared.released
            or ledger.sample_fingerprint != prepared.sample_fingerprint
        ):
            reason = "ledger_sample_mismatch"
        elif not ledger.is_consistent():
            reason = "ledger_length_mismatch"
        else:
            keep, reason = ledger.plan(full_ids)
        if keep is None:
            self._record_ledger_fallback(reason or "no_reusable_prefix")
            self._release_ledger()
            return plan
        try:
            plan.cropped = ledger.crop(keep)
        except Exception as error:
            self._disable_kv_ledger(f"cache_crop_failed:{type(error).__name__}")
            return plan
        plan.keep = keep
        return plan

    def _ledger_record(
        self,
        plan: LedgerPlan,
        outputs: Any,
        kind: str,
        *,
        static_len: int,
        cold_payload: dict[str, Any] | None = None,
    ) -> None:
        scores = getattr(outputs, "scores", None) or ()
        generated = (
            [int(value) for value in outputs.sequences[0, -len(scores) :].tolist()]
            if scores
            else []
        )
        stats = self.sample_stats
        if plan.keep is None:
            stats["ledger_cold_prefills"] += 1
            stats["ledger_cold_prefill_tokens"] += len(plan.full_ids)
        else:
            stats["ledger_reuse_calls"] += 1
            stats["ledger_reused_tokens"] += plan.keep
            stats["ledger_cropped_tokens"] += plan.cropped
            stats["ledger_prefilled_tokens"] += plan.prefill_tokens
        mode = "cold" if plan.keep is None else "reuse"
        stats["ledger_calls"].append(
            {
                "kind": kind,
                "mode": mode,
                "reused": plan.keep or 0,
                "cropped": plan.cropped,
                "prefilled": plan.prefill_tokens,
                "generated": len(generated),
            }
        )
        print(
            f" [KV-Ledger][{self.MODEL_NAME}] {kind}: {mode} reuse={plan.keep or 0} "
            f"crop={plan.cropped} prefill={plan.prefill_tokens} gen={len(generated)}"
        )
        if not self.kv_ledger_enabled:
            return

        cache = getattr(outputs, "past_key_values", None)
        token_ids = plan.full_ids + generated[:-1]
        if plan.keep is None:
            prepared = self.prepared_sample
            if not supports_crop(cache):
                self._disable_kv_ledger(
                    f"cache_not_croppable:{type(cache).__name__}"
                )
                return
            self.ledger = KVLedger(
                owner_model=self.MODEL_NAME,
                sample_fingerprint=prepared.sample_fingerprint,
                cache=cache,
                token_ids=token_ids,
                static_len=int(static_len),
                payload=dict(cold_payload or {}),
            )
        elif self.ledger is not None:
            self.ledger.reset(cache, token_ids)
        else:
            return
        if not self.ledger.is_consistent():
            self._record_ledger_fallback("ledger_length_mismatch")
            self._release_ledger()

    @staticmethod
    def _candidate_source(plan: LedgerPlan) -> str:
        return "candidate_full_prefill" if plan.keep is None else "candidate_ledger"

    def _ledger_advance(
        self, previous_cache: Any, cache: Any, token_id: int
    ) -> None:
        ledger = self.ledger
        if ledger is None or ledger.cache is not previous_cache:
            return
        ledger.append(cache, token_id)
        if not ledger.is_consistent():
            self._record_ledger_fallback("ledger_length_mismatch")
            self._release_ledger()

    def _history_token_ids(self, assistant_text: str) -> list[int]:
        encoded = self.tokenizer(
            assistant_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        return single_row_token_ids(encoded)

    def validate_continuation_state(
        self, state: ContinuationState, committed_text: str
    ) -> tuple[bool, str | None]:
        prepared = self.prepared_sample
        if prepared is None or prepared.released:
            return False, "missing_prepared_sample"
        if (
            state.source.startswith("candidate_")
            and not self.candidate_cache_fastpath_enabled
        ):
            return False, "candidate_cache_fastpath_disabled"
        valid, reason = state.validate(
            owner_model=self.MODEL_NAME,
            sample_fingerprint=prepared.sample_fingerprint,
            committed_history_fingerprint=history_fingerprint(committed_text),
        )
        if not valid:
            return valid, reason
        valid, reason = self._validate_pending_history_payload(state)
        if not valid:
            return valid, reason
        expected_history_ids = state.adapter_payload["history_token_ids"]
        actual_history_ids = self._history_token_ids(committed_text)
        if actual_history_ids != expected_history_ids:
            return False, "history_tokenization_mismatch"
        return True, None

    def _make_continuation_state(
        self,
        *,
        source_text: str,
        committed_text: str,
        past_key_values: Any,
        pending_token_id: int | None,
        expected_cache_seq_len: int | None,
        source: str,
        history_token_ids: list[int],
        adapter_payload: dict[str, Any] | None = None,
    ) -> ContinuationState:
        prepared = self.prepared_sample
        if prepared is None:
            raise CacheStateError("cannot create continuation state without sample context")
        payload = dict(adapter_payload or {})
        payload["history_token_ids"] = list(history_token_ids)
        ledger = self.ledger
        if past_key_values is not None and ledger is not None and past_key_values is ledger.cache:
            payload["ledger_epoch"] = ledger.epoch
        return ContinuationState(
            owner_model=self.MODEL_NAME,
            sample_fingerprint=prepared.sample_fingerprint,
            source_history_fingerprint=history_fingerprint(source_text),
            committed_history_fingerprint=history_fingerprint(committed_text),
            past_key_values=past_key_values,
            pending_token_id=pending_token_id,
            cache_seq_len=cache_sequence_length(past_key_values),
            expected_cache_seq_len=expected_cache_seq_len,
            source=source,
            adapter_payload=payload,
        )

    def safe_decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        if len(token_ids) > 1:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        dummy_ids = self.tokenizer.encode("a", add_special_tokens=False)
        full_text = self.tokenizer.decode(
            dummy_ids + token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        dummy_decoded = self.tokenizer.decode(
            dummy_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if full_text.startswith(dummy_decoded):
            return full_text[len(dummy_decoded) :]
        return full_text


class Qwen3VLAdapter(BaseVLModel):
    MODEL_NAME = "Qwen3-VL"

    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        super().__init__(model_path, device)
        self.processor = None

    def load(self) -> None:
        require_flash_attn()
        print(f"Loading Qwen3-VL to {self.device} (BF16, FlashAttention-2)...")
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=FLASH_ATTN_IMPLEMENTATION,
        ).eval()

    def prepare_sample(
        self, image_path: str, prompt: str, sample_fingerprint: str
    ) -> None:
        with timer("[Qwen3] static image preprocess"):
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
                try:
                    original_pixels = image.width * image.height
                    safe_max_pixels = min(original_pixels, QWEN_MAX_PIXELS)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                    "max_pixels": safe_max_pixels,
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    base_inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                finally:
                    image.close()
            cached_inputs = pin_tensor_tree(as_plain_mapping(base_inputs))
        self._install_prepared_sample(
            image_path=image_path,
            prompt=prompt,
            sample_fingerprint=sample_fingerprint,
            payload={
                "base_inputs": cached_inputs,
                "prompt_ids": single_row_token_ids(cached_inputs["input_ids"]),
            },
        )
        input_ids = cached_inputs["input_ids"]
        self._set_ori_input_token(
            total=int(input_ids.shape[-1]),
            image=int((input_ids == self._image_token_id()).sum().item()),
        )

    def _image_token_id(self) -> int:
        token_id = getattr(self.model.config, "image_token_id", None)
        if token_id is None:
            token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        return int(token_id)

    def _append_text_tokens(
        self, inputs: dict[str, Any], token_ids: list[int]
    ) -> dict[str, Any]:
        if not token_ids:
            return dict(inputs)
        result = dict(inputs)
        tokens = torch.tensor(
            [token_ids], device=self.model.device, dtype=torch.long
        )
        result["input_ids"] = torch.cat([result["input_ids"], tokens], dim=1)
        for key, fill_value in (
            ("attention_mask", 1),
            ("mm_token_type_ids", 0),
            ("token_type_ids", 0),
        ):
            value = result.get(key)
            if value is None:
                continue
            extension_shape = list(value.shape)
            extension_shape[-1] = len(token_ids)
            extension = torch.full(
                extension_shape,
                fill_value,
                device=value.device,
                dtype=value.dtype,
            )
            result[key] = torch.cat([value, extension], dim=-1)
        return result

    def _prepare_generation_inputs(
        self, image_path: str, prompt: str, assistant_text: str
    ) -> tuple[dict[str, Any], list[int], LedgerPlan]:
        prepared = self._require_prepared(image_path, prompt)
        history_ids = self._history_token_ids(assistant_text)
        plan = self._ledger_plan(list(prepared.payload["prompt_ids"]) + history_ids)
        base_inputs = prepared.payload["base_inputs"]
        if plan.keep is not None:
            base_inputs = {
                key: value
                for key, value in base_inputs.items()
                if key in self._TEXT_INPUT_KEYS
            }
        inputs = to_device_tree(base_inputs, self.model.device)
        if history_ids:
            inputs = self._append_text_tokens(inputs, history_ids)
        return inputs, history_ids, plan

    def _run_generate(
        self, inputs: dict[str, Any], plan: LedgerPlan, max_new_tokens: int
    ) -> Any:
        cache_kwargs: dict[str, Any] = {}
        if plan.keep is not None:
            self._restore_rope_deltas(self.ledger.payload.get("rope_deltas"))
            cache_kwargs["past_key_values"] = self.ledger.cache
        with torch.no_grad():
            return self.model.generate(
                **inputs,
                **cache_kwargs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=(
                    self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                ),
                do_sample=False,
                use_cache=True,
            )

    def _record_generation(self, plan: LedgerPlan, outputs: Any, kind: str) -> None:
        prepared = self.prepared_sample
        self._ledger_record(
            plan,
            outputs,
            kind,
            static_len=len(prepared.payload["prompt_ids"]),
            cold_payload={"rope_deltas": self._capture_rope_deltas()},
        )

    _TEXT_INPUT_KEYS = frozenset(
        ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids")
    )

    def _state_inputs_after_candidate(
        self, inputs: dict[str, Any], generated_ids: list[int]
    ) -> dict[str, Any]:
        state_inputs = {
            key: value for key, value in inputs.items() if key in self._TEXT_INPUT_KEYS
        }
        cached_generated_ids = generated_ids[:-1]
        if cached_generated_ids:
            state_inputs = self._append_text_tokens(
                state_inputs, cached_generated_ids
            )
        return state_inputs

    def _capture_rope_deltas(self) -> Any:
        qwen_core = getattr(self.model, "model", None)
        rope_deltas = getattr(qwen_core, "rope_deltas", None)
        if torch.is_tensor(rope_deltas):
            return rope_deltas.detach().clone()
        return rope_deltas

    def _restore_rope_deltas(self, rope_deltas: Any) -> None:
        qwen_core = getattr(self.model, "model", None)
        if qwen_core is not None and hasattr(qwen_core, "rope_deltas"):
            qwen_core.rope_deltas = rope_deltas

    def _validate_qwen_state_payload(
        self, state: ContinuationState,
    ) -> tuple[bool, str | None]:
        model_inputs = state.adapter_payload.get("model_inputs")
        if not isinstance(model_inputs, dict):
            return False, "missing_qwen_model_inputs"
        if any(
            key in model_inputs
            for key in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            )
        ):
            return False, "qwen_state_contains_visual_inputs"
        input_ids = model_inputs.get("input_ids")
        if input_ids is None or int(input_ids.shape[-1]) != state.cache_seq_len:
            return False, "qwen_input_cache_length_mismatch"
        input_length = int(input_ids.shape[-1])
        for key in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
            value = model_inputs.get(key)
            if value is not None and int(value.shape[-1]) != input_length:
                return False, f"qwen_{key}_length_mismatch"
        qwen_core = getattr(self.model, "model", None)
        if (
            qwen_core is not None
            and hasattr(qwen_core, "rope_deltas")
            and not torch.is_tensor(state.adapter_payload.get("rope_deltas"))
        ):
            return False, "missing_qwen_rope_deltas"
        return True, None

    def validate_continuation_state(
        self, state: ContinuationState, committed_text: str
    ) -> tuple[bool, str | None]:
        valid, reason = super().validate_continuation_state(
            state, committed_text
        )
        if not valid:
            return valid, reason
        return self._validate_qwen_state_payload(state)

    def _decode_window(
        self, outputs: Any
    ) -> tuple[list[int], list[float], bool, list[tuple[str, int, float]]]:
        score_count = len(outputs.scores)
        generated_ids = (
            outputs.sequences[0, -score_count:].tolist() if score_count else []
        )
        log_probs: list[float] = []
        valid_ids: list[int] = []
        token_details: list[tuple[str, int, float]] = []
        has_eos = False
        eos_value = self.model.generation_config.eos_token_id
        eos_ids = set(eos_value if isinstance(eos_value, list) else [eos_value])
        for step, token_id in enumerate(generated_ids):
            token_lp = F.log_softmax(outputs.scores[step][0], dim=-1)[
                token_id
            ].item()
            log_probs.append(token_lp)
            valid_ids.append(int(token_id))
            token_details.append((self.safe_decode([token_id]), token_id, token_lp))
            if token_id in eos_ids:
                has_eos = True
                break
        return valid_ids, log_probs, has_eos, token_details

    def generate_next_token(
        self,
        image_path: str,
        prompt: str,
        assistant_text: str,
        continuation_state: ContinuationState | None = None,
    ):
        if continuation_state is not None:
            valid, reason = continuation_state.validate(
                owner_model=self.MODEL_NAME,
                sample_fingerprint=self._require_prepared(
                    image_path, prompt
                ).sample_fingerprint,
                committed_history_fingerprint=history_fingerprint(assistant_text),
            )
            if valid:
                valid, reason = self._validate_pending_history_payload(
                    continuation_state
                )
            if valid:
                valid, reason = self._validate_qwen_state_payload(
                    continuation_state
                )
            if not valid:
                self._record_cache_fallback(reason or "invalid_state")
                continuation_state.release()
                continuation_state = None

        if continuation_state is None:
            with timer("[Qwen3] text and cache preprocess", silent=True):
                inputs, history_ids, plan = self._prepare_generation_inputs(
                    image_path, prompt, assistant_text
                )
            prefill_len = int(inputs["input_ids"].shape[1])
            with timer("[Qwen3] generation (1 token)", silent=True):
                outputs = self._run_generate(inputs, plan, 1)
            self._record_generation(plan, outputs, "next_token")
            state_inputs = self._state_inputs_after_candidate(inputs, [])
            source = (
                "continuation_full_prefill"
                if plan.keep is None
                else "continuation_ledger_sync"
            )
        else:
            history_ids = list(
                continuation_state.adapter_payload["history_token_ids"]
            )
            pending_token_id = continuation_state.claim_pending_token()
            state_inputs = self._append_text_tokens(
                continuation_state.adapter_payload["model_inputs"],
                [pending_token_id],
            )
            self._restore_rope_deltas(
                continuation_state.adapter_payload.get("rope_deltas")
            )
            prefill_len = int(continuation_state.cache_seq_len or 0) + 1
            previous_cache = continuation_state.past_key_values
            with timer("[Qwen3] generation (1 token, KV cache)", silent=True):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **state_inputs,
                        past_key_values=previous_cache,
                        use_cache=True,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=(
                            self.tokenizer.pad_token_id
                            or self.tokenizer.eos_token_id
                        ),
                        do_sample=False,
                    )
            self._ledger_advance(
                previous_cache,
                getattr(outputs, "past_key_values", None),
                pending_token_id,
            )
            source = "continuation_cache"
        next_token_id = int(outputs.sequences[0, -1].item())
        token_str = self.safe_decode([next_token_id])
        log_prob = F.log_softmax(outputs.scores[0][0], dim=-1)[
            next_token_id
        ].item()
        eos_value = self.model.generation_config.eos_token_id
        is_eos = (
            next_token_id in eos_value
            if isinstance(eos_value, list)
            else next_token_id == eos_value
        )
        history_after = history_ids + [next_token_id]
        new_state = self._make_continuation_state(
            source_text=assistant_text,
            committed_text=assistant_text + token_str,
            past_key_values=getattr(outputs, "past_key_values", None),
            pending_token_id=next_token_id,
            expected_cache_seq_len=prefill_len,
            source=source,
            history_token_ids=history_after,
            adapter_payload={
                "model_inputs": state_inputs,
                "rope_deltas": self._capture_rope_deltas(),
            },
        )
        return token_str, next_token_id, log_prob, is_eos, new_state

    def generate_bidding_window(
        self,
        image_path: str,
        prompt: str,
        assistant_text: str,
        window_size: int,
    ) -> CandidateResult:
        with timer("[Qwen3] bid window text and cache preprocess"):
            inputs, history_ids, plan = self._prepare_generation_inputs(
                image_path, prompt, assistant_text
            )
        prefill_len = int(inputs["input_ids"].shape[1])
        with timer(f"[Qwen3] bid window generation ({window_size} tokens)"):
            outputs = self._run_generate(inputs, plan, window_size)
        self._record_generation(plan, outputs, "window")
        valid_ids, log_probs, has_eos, token_details = self._decode_window(outputs)
        text = self.safe_decode(valid_ids)
        state = None
        if valid_ids:
            state = self._make_continuation_state(
                source_text=assistant_text,
                committed_text=assistant_text + text,
                past_key_values=getattr(outputs, "past_key_values", None),
                pending_token_id=valid_ids[-1],
                expected_cache_seq_len=prefill_len + len(valid_ids) - 1,
                source=self._candidate_source(plan),
                history_token_ids=history_ids + valid_ids,
                adapter_payload={
                    "model_inputs": self._state_inputs_after_candidate(
                        inputs, valid_ids
                    ),
                    "rope_deltas": self._capture_rope_deltas(),
                },
            )
        return CandidateResult(
            model_name=self.MODEL_NAME,
            text=text,
            log_probs=log_probs,
            has_eos=has_eos,
            token_details=token_details,
            token_ids=valid_ids,
            continuation_state=state,
        )


class MiniCPMVAdapter(BaseVLModel):
    MODEL_NAME = "MiniCPM-V"

    _GENERATION_ONLY_KEYS = {
        "cache_implementation",
        "do_sample",
        "early_stopping",
        "enable_thinking",
        "eos_token_id",
        "generation_config",
        "iteration",
        "logits_processor",
        "max_length",
        "max_new_tokens",
        "min_length",
        "min_new_tokens",
        "num_beams",
        "num_return_sequences",
        "output_attentions",
        "output_hidden_states",
        "output_scores",
        "pad_token_id",
        "past_key_values",
        "repetition_penalty",
        "return_dict_in_generate",
        "sampling",
        "stopping_criteria",
        "stream",
        "streamer",
        "synced_gpus",
        "temperature",
        "top_k",
        "top_p",
        "typical_p",
        "use_cache",
    }

    def __init__(
        self, model_path: str, device: str = "cuda:0", max_new_tokens: int = 4096
    ) -> None:
        super().__init__(model_path, device)
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        require_flash_attn()
        print(f"Loading MiniCPM-V to {self.device} (BF16, FlashAttention-2)...")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation=FLASH_ATTN_IMPLEMENTATION,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        target_obj = self._target_llm()
        (
            self.candidate_cache_fastpath_enabled,
            self.candidate_cache_fastpath_disabled_reasons,
        ) = greedy_cache_fastpath_is_neutral(
            getattr(target_obj, "generation_config", None)
        )
        if not self.candidate_cache_fastpath_enabled:
            print(
                "[MiniCPM] candidate KV promotion disabled; the next token "
                "falls back to a full generate. Non-neutral settings: "
                + ", ".join(self.candidate_cache_fastpath_disabled_reasons)
            )
        (
            self.kv_ledger_enabled,
            self.kv_ledger_disabled_reasons,
        ) = kv_ledger_generation_check(getattr(target_obj, "generation_config", None))
        if not self.kv_ledger_enabled:
            print(
                "[MiniCPM] KV ledger disabled; every bid uses a full prefill. Reasons: "
                + ", ".join(self.kv_ledger_disabled_reasons)
            )

    def _target_llm(self) -> Any:
        return self.model.llm if hasattr(self.model, "llm") else self.model

    def _target_device(self) -> Any:
        target_obj = self._target_llm()
        try:
            return next(target_obj.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device(self.device)

    def _intercept_static_inputs(
        self, image_path: str, prompt: str
    ) -> tuple[dict[str, Any], int | None]:
        intercepted_inputs: dict[str, Any] = {}
        captured_bounds: list[Any] = []
        target_obj = self._target_llm()
        original_generate = target_obj.generate
        original_get_vllm_embedding = getattr(self.model, "get_vllm_embedding", None)

        def capture_vllm_embedding(data: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(data, dict) and data.get("image_bound") is not None:
                captured_bounds.append(data["image_bound"])
            return original_get_vllm_embedding(data, *args, **kwargs)

        def hijack_generate(*generate_args: Any, **kwargs: Any) -> Any:
            if generate_args:
                raise TypeError(
                    "MiniCPM generate interception expected keyword-only model inputs"
                )
            intercepted_inputs.update(kwargs)
            fake_ids = torch.tensor(
                [[self.tokenizer.eos_token_id]], device=self._target_device()
            )
            if kwargs.get("return_dict_in_generate", False):
                from transformers.generation.utils import GenerateDecoderOnlyOutput

                return GenerateDecoderOnlyOutput(sequences=fake_ids)
            return fake_ids

        target_obj.generate = hijack_generate
        if callable(original_get_vllm_embedding):
            self.model.get_vllm_embedding = capture_vllm_embedding
        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
                try:
                    msgs = [{"role": "user", "content": [image, prompt]}]
                    chat_result = self.model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=self.tokenizer,
                        sampling=False,
                        max_new_tokens=self.max_new_tokens,
                        enable_thinking=MINICPM_ENABLE_THINKING,
                        max_slice_nums=MINICPM_MAX_SLICE_NUMS,
                        iteration=MINICPM_ITERATION,
                    )
                    if not isinstance(chat_result, str) and hasattr(
                        chat_result, "__iter__"
                    ):
                        list(chat_result)
                finally:
                    image.close()
        except Exception:
            if not intercepted_inputs:
                raise
        finally:
            target_obj.generate = original_generate
            if callable(original_get_vllm_embedding):
                self.model.get_vllm_embedding = original_get_vllm_embedding

        if not intercepted_inputs:
            raise RuntimeError("MiniCPM chat input interception produced no model inputs")

        for key in self._GENERATION_ONLY_KEYS:
            intercepted_inputs.pop(key, None)
        if not any(
            intercepted_inputs.get(key) is not None
            for key in ("inputs_embeds", "input_ids")
        ):
            raise KeyError("MiniCPM intercepted neither inputs_embeds nor input_ids")
        image_tokens = (
            self._count_image_bound(captured_bounds[-1]) if captured_bounds else None
        )
        return intercepted_inputs, image_tokens

    @staticmethod
    def _count_image_bound(image_bound: Any) -> int | None:
        try:
            total = 0
            for bounds in image_bound:
                if torch.is_tensor(bounds):
                    if bounds.numel() == 0:
                        continue
                    pairs = bounds.reshape(-1, 2)
                    total += int((pairs[:, 1] - pairs[:, 0]).sum().item())
                else:
                    for start, end in bounds:
                        total += int(end) - int(start)
            return total
        except (TypeError, ValueError):
            return None

    def prepare_sample(
        self, image_path: str, prompt: str, sample_fingerprint: str
    ) -> None:
        with timer("[MiniCPM] static image preprocess (input interception)"):
            with torch.inference_mode():
                base_inputs, image_tokens = self._intercept_static_inputs(
                    image_path, prompt
                )
                cached_inputs = pin_tensor_tree(base_inputs)
        prompt_len = self._input_length(cached_inputs)
        if cached_inputs.get("inputs_embeds") is None:
            prompt_ids = single_row_token_ids(cached_inputs["input_ids"])
        else:
            prompt_ids = [self._PROMPT_PLACEHOLDER_ID] * prompt_len
        self._install_prepared_sample(
            image_path=image_path,
            prompt=prompt,
            sample_fingerprint=sample_fingerprint,
            payload={"base_inputs": cached_inputs, "prompt_ids": prompt_ids},
        )
        self._set_ori_input_token(total=prompt_len, image=image_tokens)

    _PROMPT_PLACEHOLDER_ID = 0

    def _prepare_generation_inputs(
        self, image_path: str, prompt: str, assistant_text: str
    ) -> tuple[dict[str, Any], list[int], LedgerPlan]:
        prepared = self._require_prepared(image_path, prompt)
        target_obj = self._target_llm()
        target_device = self._target_device()
        history_ids = self._history_token_ids(assistant_text)
        plan = self._ledger_plan(list(prepared.payload["prompt_ids"]) + history_ids)
        base_inputs = prepared.payload["base_inputs"]
        if plan.keep is not None:
            base_inputs = {
                key: value
                for key, value in base_inputs.items()
                if key not in ("inputs_embeds", "input_ids")
            }
        inputs = to_device_tree(base_inputs, target_device)
        if plan.keep is not None:
            inputs["input_ids"] = torch.tensor(
                [plan.full_ids], device=target_device, dtype=torch.long
            )
            if inputs.get("attention_mask") is None:
                inputs["attention_mask"] = torch.ones(
                    (1, len(prepared.payload["prompt_ids"])),
                    device=target_device,
                    dtype=torch.long,
                )
        if history_ids and plan.keep is None:
            assistant_ids = torch.tensor(
                [history_ids], device=target_device, dtype=torch.long
            )
            if inputs.get("inputs_embeds") is not None:
                assistant_embeds = target_obj.get_input_embeddings()(assistant_ids)
                inputs["inputs_embeds"] = torch.cat(
                    [inputs["inputs_embeds"], assistant_embeds], dim=1
                )
            elif inputs.get("input_ids") is not None:
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], assistant_ids], dim=1
                )
            else:
                raise KeyError("MiniCPM cached neither inputs_embeds nor input_ids")
        if history_ids and inputs.get("attention_mask") is not None:
            mask_extension = torch.ones(
                (1, len(history_ids)),
                device=target_device,
                dtype=inputs["attention_mask"].dtype,
            )
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], mask_extension], dim=1
            )
        return inputs, history_ids, plan

    def _run_generate(
        self,
        inputs: dict[str, Any],
        plan: LedgerPlan,
        max_new_tokens: int,
        **extra_kwargs: Any,
    ) -> Any:
        cache_kwargs: dict[str, Any] = {}
        if plan.keep is not None:
            cache_kwargs["past_key_values"] = self.ledger.cache
        with torch.no_grad():
            return self._target_llm().generate(
                **inputs,
                **cache_kwargs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
                **extra_kwargs,
            )

    def _record_generation(self, plan: LedgerPlan, outputs: Any, kind: str) -> None:
        self._ledger_record(
            plan,
            outputs,
            kind,
            static_len=len(self.prepared_sample.payload["prompt_ids"]),
        )

    @staticmethod
    def _input_length(inputs: dict[str, Any]) -> int:
        for key in ("inputs_embeds", "input_ids"):
            value = inputs.get(key)
            if value is not None:
                return int(value.shape[1])
        raise KeyError("MiniCPM generation inputs have no sequence tensor")

    def _decode_window(
        self, outputs: Any
    ) -> tuple[list[int], list[float], bool, list[tuple[str, int, float]]]:
        score_count = len(outputs.scores)
        generated_ids = (
            outputs.sequences[0, -score_count:].tolist() if score_count else []
        )
        log_probs: list[float] = []
        valid_ids: list[int] = []
        token_details: list[tuple[str, int, float]] = []
        has_eos = False
        for step, token_id in enumerate(generated_ids):
            token_id = int(token_id)
            token_lp = F.log_softmax(outputs.scores[step][0], dim=-1)[
                token_id
            ].item()
            log_probs.append(token_lp)
            valid_ids.append(token_id)
            token_details.append((self.safe_decode([token_id]), token_id, token_lp))
            if token_id == self.tokenizer.eos_token_id:
                has_eos = True
                break
        return valid_ids, log_probs, has_eos, token_details

    def generate_next_token(
        self,
        image_path: str,
        prompt: str,
        assistant_text: str,
        continuation_state: ContinuationState | None = None,
    ):
        target_obj = self._target_llm()
        prepared = self._require_prepared(image_path, prompt)
        if continuation_state is not None:
            valid, reason = continuation_state.validate(
                owner_model=self.MODEL_NAME,
                sample_fingerprint=prepared.sample_fingerprint,
                committed_history_fingerprint=history_fingerprint(assistant_text),
            )
            if valid:
                valid, reason = self._validate_pending_history_payload(
                    continuation_state
                )
            if not valid:
                self._record_cache_fallback(reason or "invalid_state")
                continuation_state.release()
                continuation_state = None

        if continuation_state is None:
            with timer("[MiniCPM] text and cache preprocess", silent=True):
                inputs, history_ids, plan = self._prepare_generation_inputs(
                    image_path, prompt, assistant_text
                )
            prefill_len = self._input_length(inputs)
            with timer("[MiniCPM] generation (1 token)", silent=True):
                outputs = self._run_generate(inputs, plan, 1, num_beams=1)
            self._record_generation(plan, outputs, "next_token")
            next_token_id = int(outputs.sequences[0, -1].item())
            log_prob = F.log_softmax(outputs.scores[0][0], dim=-1)[
                next_token_id
            ].item()
            source = (
                "continuation_full_prefill"
                if plan.keep is None
                else "continuation_ledger_sync"
            )
        else:
            history_ids = list(
                continuation_state.adapter_payload["history_token_ids"]
            )
            pending_token_id = continuation_state.claim_pending_token()
            input_ids = torch.tensor(
                [[pending_token_id]], device=self._target_device(), dtype=torch.long
            )
            past_len = int(continuation_state.cache_seq_len or 0)
            attention_mask = torch.ones(
                (1, past_len + 1),
                dtype=torch.long,
                device=self._target_device(),
            )
            previous_cache = continuation_state.past_key_values
            with timer("[MiniCPM] generation (1 token, KV cache)", silent=True):
                with torch.no_grad():
                    outputs = target_obj(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=previous_cache,
                        use_cache=True,
                        return_dict=True,
                    )
            self._ledger_advance(
                previous_cache,
                getattr(outputs, "past_key_values", None),
                pending_token_id,
            )
            logits = outputs.logits[0, -1, :]
            next_token_id = int(torch.argmax(logits).item())
            log_prob = F.log_softmax(logits, dim=-1)[next_token_id].item()
            prefill_len = past_len + 1
            source = "continuation_cache"
        token_str = self.safe_decode([next_token_id])
        is_eos = next_token_id == self.tokenizer.eos_token_id
        new_state = self._make_continuation_state(
            source_text=assistant_text,
            committed_text=assistant_text + token_str,
            past_key_values=getattr(outputs, "past_key_values", None),
            pending_token_id=next_token_id,
            expected_cache_seq_len=prefill_len,
            source=source,
            history_token_ids=history_ids + [next_token_id],
        )
        return token_str, next_token_id, log_prob, is_eos, new_state

    def generate_bidding_window(
        self,
        image_path: str,
        prompt: str,
        assistant_text: str,
        window_size: int,
    ) -> CandidateResult:
        with timer("[MiniCPM] bid window text and cache preprocess"):
            inputs, history_ids, plan = self._prepare_generation_inputs(
                image_path, prompt, assistant_text
            )
        prefill_len = self._input_length(inputs)
        with timer(f"[MiniCPM] bid window generation ({window_size} tokens)"):
            outputs = self._run_generate(inputs, plan, window_size)
        self._record_generation(plan, outputs, "window")
        valid_ids, log_probs, has_eos, token_details = self._decode_window(outputs)
        text = self.safe_decode(valid_ids)
        state = None
        if valid_ids:
            state = self._make_continuation_state(
                source_text=assistant_text,
                committed_text=assistant_text + text,
                past_key_values=getattr(outputs, "past_key_values", None),
                pending_token_id=valid_ids[-1],
                expected_cache_seq_len=prefill_len + len(valid_ids) - 1,
                source=self._candidate_source(plan),
                history_token_ids=history_ids + valid_ids,
            )
        return CandidateResult(
            model_name=self.MODEL_NAME,
            text=text,
            log_probs=log_probs,
            has_eos=has_eos,
            token_details=token_details,
            token_ids=valid_ids,
            continuation_state=state,
        )


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLAdapter(BaseVLModel):
    MODEL_NAME = "InternVL"

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._relay_eos_token_id: int | None = None
        self._visual_feature_fastpath_enabled = False

    def load(self) -> None:
        require_flash_attn()
        print(f"Loading InternVL to {self.device} (BF16, FlashAttention enabled)...")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map={"": self.device},
        ).eval()
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            self.IMG_CONTEXT_TOKEN
        )
        generate_parameters = inspect.signature(self.model.generate).parameters
        self._visual_feature_fastpath_enabled = (
            "visual_features" in generate_parameters
        )
        if not self._visual_feature_fastpath_enabled:
            print(
                "[InternVL] remote generate() does not accept visual_features; "
                "the visual-feature fast path is off and only the tiled pixels are cached."
            )
        language_model = self.model.language_model
        (
            self.candidate_cache_fastpath_enabled,
            self.candidate_cache_fastpath_disabled_reasons,
        ) = greedy_cache_fastpath_is_neutral(
            getattr(language_model, "generation_config", None)
        )
        if not self.candidate_cache_fastpath_enabled:
            print(
                "[InternVL] candidate KV promotion disabled; the next token "
                "falls back to a full generate. Non-neutral settings: "
                + ", ".join(self.candidate_cache_fastpath_disabled_reasons)
            )
        (
            self.kv_ledger_enabled,
            self.kv_ledger_disabled_reasons,
        ) = kv_ledger_generation_check(
            getattr(language_model, "generation_config", None)
        )
        if not self.kv_ledger_enabled:
            print(
                "[InternVL] KV ledger disabled; every bid uses a full prefill. Reasons: "
                + ", ".join(self.kv_ledger_disabled_reasons)
            )

    def _build_transform(self, input_size: int) -> Any:
        return T.Compose(
            [
                T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: Iterable[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_ratio_diff, best_ratio = float("inf"), (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff, best_ratio = ratio_diff, ratio
            elif (
                ratio_diff == best_ratio_diff
                and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
            ):
                best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = INTERNVL_MAX_NUM,
        image_size: int = INTERNVL_INPUT_SIZE,
        use_thumbnail: bool = False,
    ) -> list[Image.Image]:
        original_width, original_height = image.size
        target_ratios = sorted(
            {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if min_num <= i * j <= max_num
            },
            key=lambda value: value[0] * value[1],
        )
        target_ratio = self._find_closest_aspect_ratio(
            original_width / original_height,
            target_ratios,
            original_width,
            original_height,
            image_size,
        )
        target_width = image_size * target_ratio[0]
        target_height = image_size * target_ratio[1]
        resized = image.resize((target_width, target_height))
        tiles = [
            resized.crop(
                (
                    (index % (target_width // image_size)) * image_size,
                    (index // (target_width // image_size)) * image_size,
                    ((index % (target_width // image_size)) + 1) * image_size,
                    ((index // (target_width // image_size)) + 1) * image_size,
                )
            )
            for index in range(target_ratio[0] * target_ratio[1])
        ]
        if use_thumbnail and len(tiles) != 1:
            tiles.append(image.resize((image_size, image_size)))
        return tiles

    def _get_conversation_builder(self) -> Any:
        module = importlib.import_module(self.model.__class__.__module__)
        builder = getattr(module, "get_conv_template", None)
        if builder is None:
            raise AttributeError(
                f"{self.model.__class__.__module__} does not expose "
                "get_conv_template; cannot reproduce InternVL chat tokenization"
            )
        return builder

    def _load_image(
        self,
        image_file: str,
        input_size: int = INTERNVL_INPUT_SIZE,
        max_num: int = INTERNVL_MAX_NUM,
    ) -> torch.Tensor:
        with Image.open(image_file) as opened_image:
            image = opened_image.convert("RGB")
        transform = self._build_transform(input_size=input_size)
        processed_images = self._dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=INTERNVL_USE_THUMBNAIL,
            max_num=max_num,
        )
        try:
            return torch.stack([transform(tile) for tile in processed_images])
        finally:
            for tile in processed_images:
                tile.close()
            image.close()

    def prepare_sample(
        self, image_path: str, prompt: str, sample_fingerprint: str
    ) -> None:
        with timer("[InternVL] static vision preprocess (tiling and encoding)"):
            pixel_values = self._load_image(
                image_path,
                input_size=INTERNVL_INPUT_SIZE,
                max_num=INTERNVL_MAX_NUM,
            ).to(torch.bfloat16)
            image_tiles = int(pixel_values.shape[0])
            visual_features = None
            if self._visual_feature_fastpath_enabled:
                with torch.inference_mode():
                    device_pixels = pixel_values.to(self.model.device)
                    visual_features = self.model.extract_feature(device_pixels)

        question = prompt if "<image>" in prompt else f"<image>\n{prompt}"
        template = self._get_conversation_builder()(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        expanded_query = template.get_prompt()

        visual_tokens = int(self.model.num_image_token) * image_tiles
        image_tokens = (
            self.IMG_START_TOKEN
            + self.IMG_CONTEXT_TOKEN * visual_tokens
            + self.IMG_END_TOKEN
        )
        expanded_query = expanded_query.replace("<image>", image_tokens, 1)
        if "<image>" in expanded_query:
            raise ValueError("Only one image placeholder is supported per sample")
        relay_eos_token_id = self.tokenizer.convert_tokens_to_ids(
            template.sep.strip()
        )
        cached_pixels = pin_tensor_tree(pixel_values)
        cached_features = pin_tensor_tree(visual_features)
        if self._visual_feature_fastpath_enabled:
            del device_pixels
        del visual_features
        self._relay_eos_token_id = int(relay_eos_token_id)
        self._install_prepared_sample(
            image_path=image_path,
            prompt=prompt,
            sample_fingerprint=sample_fingerprint,
            payload={
                "pixel_values": cached_pixels,
                "visual_features": cached_features,
                "expanded_query": expanded_query,
                "relay_eos_token_id": self._relay_eos_token_id,
            },
        )
        self.sample_stats["visual_feature_fastpath_enabled"] = (
            self._visual_feature_fastpath_enabled
        )
        self.sample_stats["vision_encoder_static_calls"] = int(
            self._visual_feature_fastpath_enabled
        )
        prompt_ids = self._history_token_ids("")
        image_positions = [
            index
            for index, token_id in enumerate(prompt_ids)
            if token_id == self.model.img_context_token_id
        ]
        self.prepared_sample.payload["static_len"] = (
            image_positions[-1] + 1 if image_positions else len(prompt_ids)
        )
        self._set_ori_input_token(total=len(prompt_ids), image=len(image_positions))

    def _history_token_ids(self, assistant_text: str) -> list[int]:
        prepared = self.prepared_sample
        if prepared is None or prepared.released:
            raise CacheStateError("InternVL has no prepared prompt tokenization")
        encoded = self.tokenizer(
            prepared.payload["expanded_query"] + assistant_text,
            return_tensors="pt",
        ).input_ids
        return single_row_token_ids(encoded)

    def _prepare_generation_inputs(
        self, image_path: str, prompt: str, assistant_text: str
    ) -> tuple[Any, Any, dict[str, Any], list[int], LedgerPlan]:
        prepared = self._require_prepared(image_path, prompt)
        self._relay_eos_token_id = int(prepared.payload["relay_eos_token_id"])
        expanded_query = prepared.payload["expanded_query"] + assistant_text
        model_inputs = as_plain_mapping(
            self.tokenizer(expanded_query, return_tensors="pt")
        )
        history_ids = single_row_token_ids(model_inputs["input_ids"])
        plan = self._ledger_plan(list(history_ids))
        pixel_values = visual_features = None
        if plan.keep is None:
            pixel_values = to_device_tree(
                prepared.payload["pixel_values"], self.model.device
            )
            visual_features = to_device_tree(
                prepared.payload["visual_features"], self.model.device
            )
        model_inputs = to_device_tree(model_inputs, self.model.device)
        return pixel_values, visual_features, model_inputs, history_ids, plan

    def _run_generate(
        self,
        model_inputs: dict[str, Any],
        pixel_values: Any,
        visual_features: Any,
        plan: LedgerPlan,
        max_new_tokens: int,
    ) -> Any:
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self._relay_eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )
        with torch.no_grad():
            if plan.keep is None:
                return self.model.generate(
                    **model_inputs,
                    **self._vision_generate_kwargs(pixel_values, visual_features),
                    **generate_kwargs,
                )
            return self.model.language_model.generate(
                **model_inputs,
                past_key_values=self.ledger.cache,
                use_cache=True,
                **generate_kwargs,
            )

    def _record_generation(self, plan: LedgerPlan, outputs: Any, kind: str) -> None:
        self._ledger_record(
            plan,
            outputs,
            kind,
            static_len=int(self.prepared_sample.payload["static_len"]),
        )

    def _vision_generate_kwargs(
        self, pixel_values: torch.Tensor, visual_features: Any
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"pixel_values": pixel_values}
        if self._visual_feature_fastpath_enabled:
            kwargs["visual_features"] = visual_features
        return kwargs

    def _is_relay_eos(self, token_id: int) -> bool:
        return token_id == self._relay_eos_token_id

    def generate_next_token(
        self,
        image_path: str,
        prompt: str,
        assistant_text: str,
        continuation_state: ContinuationState | None = None,
    ):
        prepared = self._require_prepared(image_path, prompt)
        if continuation_state is not None:
            valid, reason = continuation_state.validate(
                owner_model=self.MODEL_NAME,
                sample_fingerprint=prepared.sample_fingerprint,
                committed_history_fingerprint=history_fingerprint(assistant_text),
            )
            if valid:
                valid, reason = self._validate_pending_history_payload(
                    continuation_state
                )
            if not valid:
                self._record_cache_fallback(reason or "invalid_state")
                continuation_state.release()
                continuation_state = None

        if continuation_state is None:
            with timer("[InternVL] text and cache preprocess", silent=True):
                pixel_values, visual_features, model_inputs, history_ids, plan = (
                    self._prepare_generation_inputs(image_path, prompt, assistant_text)
                )
            prefill_len = int(model_inputs["input_ids"].shape[1])
            with timer("[InternVL] generation (1 token)", silent=True):
                outputs = self._run_generate(
                    model_inputs, pixel_values, visual_features, plan, 1
                )
            self._record_generation(plan, outputs, "next_token")
            next_token_id = int(outputs.sequences[0, -1].item())
            logits = outputs.scores[0][0]
            source = (
                "continuation_full_prefill"
                if plan.keep is None
                else "continuation_ledger_sync"
            )
        else:
            history_ids = list(
                continuation_state.adapter_payload["history_token_ids"]
            )
            pending_token_id = continuation_state.claim_pending_token()
            input_ids = torch.tensor(
                [[pending_token_id]], device=self.model.device, dtype=torch.long
            )
            past_len = int(continuation_state.cache_seq_len or 0)
            attention_mask = torch.ones(
                (1, past_len + 1), dtype=torch.long, device=self.model.device
            )
            position_ids = torch.tensor(
                [[past_len]], device=self.model.device, dtype=torch.long
            )
            previous_cache = continuation_state.past_key_values
            with timer("[InternVL] generation (1 token, KV cache)", silent=True):
                with torch.no_grad():
                    outputs = self.model.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=previous_cache,
                        use_cache=True,
                        return_dict=True,
                    )
            self._ledger_advance(
                previous_cache,
                getattr(outputs, "past_key_values", None),
                pending_token_id,
            )
            logits = outputs.logits[0, -1, :]
            next_token_id = int(torch.argmax(logits).item())
            prefill_len = past_len + 1
            source = "continuation_cache"

        token_str = self.safe_decode([next_token_id])
        log_prob = F.log_softmax(logits, dim=-1)[next_token_id].item()
        new_state = self._make_continuation_state(
            source_text=assistant_text,
            committed_text=assistant_text + token_str,
            past_key_values=getattr(outputs, "past_key_values", None),
            pending_token_id=next_token_id,
            expected_cache_seq_len=prefill_len,
            source=source,
            history_token_ids=history_ids + [next_token_id],
        )
        return (
            token_str,
            next_token_id,
            log_prob,
            self._is_relay_eos(next_token_id),
            new_state,
        )

    def generate_bidding_window(
        self,
        image_path: str,
        prompt: str,
        assistant_text: str,
        window_size: int,
    ) -> CandidateResult:
        with timer("[InternVL] bid window text and cache preprocess"):
            pixel_values, visual_features, model_inputs, history_ids, plan = (
                self._prepare_generation_inputs(image_path, prompt, assistant_text)
            )
        prefill_len = int(model_inputs["input_ids"].shape[1])
        with timer(f"[InternVL] bid window generation ({window_size} tokens)"):
            outputs = self._run_generate(
                model_inputs, pixel_values, visual_features, plan, window_size
            )
        self._record_generation(plan, outputs, "window")
        generated_ids = outputs.sequences[0, -len(outputs.scores) :].tolist()
        log_probs: list[float] = []
        valid_ids: list[int] = []
        token_details: list[tuple[str, int, float]] = []
        has_eos = False
        for step, token_id in enumerate(generated_ids):
            token_id = int(token_id)
            token_lp = F.log_softmax(outputs.scores[step][0], dim=-1)[
                token_id
            ].item()
            log_probs.append(token_lp)
            valid_ids.append(token_id)
            token_details.append((self.safe_decode([token_id]), token_id, token_lp))
            if self._is_relay_eos(token_id):
                has_eos = True
                break
        text = self.safe_decode(valid_ids)
        state = None
        if valid_ids:
            state = self._make_continuation_state(
                source_text=assistant_text,
                committed_text=assistant_text + text,
                past_key_values=getattr(outputs, "past_key_values", None),
                pending_token_id=valid_ids[-1],
                expected_cache_seq_len=prefill_len + len(valid_ids) - 1,
                source=self._candidate_source(plan),
                history_token_ids=history_ids + valid_ids,
            )
        return CandidateResult(
            model_name=self.MODEL_NAME,
            text=text,
            log_probs=log_probs,
            has_eos=has_eos,
            token_details=token_details,
            token_ids=valid_ids,
            continuation_state=state,
        )
