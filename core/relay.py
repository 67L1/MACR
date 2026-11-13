from __future__ import annotations

import concurrent.futures
import json
import math
import os
import time
from typing import Any

from core.adapters import BaseVLModel
from core.cache_state import (
    CacheStateError,
    CandidateResult,
    ContinuationState,
    OptCacheStateManager,
    build_sample_fingerprint,
)
from core.runtime import Profiler

OPTIMIZATION_PROFILE = "sample_static_inputs_kv_ledger_v1"


class ConfidenceBiddingRelay:
    def __init__(
        self,
        models_dict: dict[str, BaseVLModel],
        window_size: int = 16,
        z_threshold: float = -0.5,
        initial_trial_windows: int = 1,
        stats_dir: str | None = None,
    ) -> None:
        self.models = models_dict
        self.expert_pool = list(models_dict.keys())
        self.window_size = window_size
        self.initial_trial_windows = max(1, int(initial_trial_windows))
        self.last_relay_stats = self._new_relay_stats()
        self.z_threshold = z_threshold
        self.model_stats: dict[str, Any] = {}
        if not stats_dir or not os.path.isdir(stats_dir):
            raise FileNotFoundError(f"Z-score statistics directory not found: {stats_dir}")
        for expert_id in self.expert_pool:
            stat_file = os.path.join(stats_dir, f"{expert_id}_statistics.json")
            if not os.path.isfile(stat_file):
                raise FileNotFoundError(f"Missing z-score statistics for {expert_id}: {stat_file}")
            with open(stat_file, "r", encoding="utf-8") as handle:
                self.model_stats[expert_id] = json.load(handle)
            print(f"[stats] Loaded z-score statistics for {expert_id}")

    def _new_relay_stats(self) -> dict[str, Any]:
        return {
            "handoff_count": 0,
            "markov_trigger_count": 0,
            "zscore_trigger_count": 0,
            "both_trigger_count": 0,
            "initial_model": None,
            "final_model": None,
            "generated_tokens": 0,
            "trigger_events": [],
        }

    def calculate_z_score(self, model_name: str, log_probs: list[float]) -> float:
        if not log_probs:
            return float("-inf")
        average_log_probability = sum(log_probs) / len(log_probs)
        stats = self.model_stats.get(model_name)
        if not stats:
            return average_log_probability
        target_length = len(log_probs)
        available_lengths = [int(key) for key in stats.keys()]
        if not available_lengths:
            return average_log_probability
        closest_length = str(
            min(available_lengths, key=lambda value: abs(value - target_length))
        )
        mean = stats[closest_length]["mu"]
        standard_deviation = stats[closest_length]["sigma"] + 1e-9
        return (average_log_probability - mean) / standard_deviation


PHASES = ("initial_selection", "continuation", "bidding")


def _empty_usage(experts: list[str]) -> dict[str, Any]:
    return {
        "committed_generated_tokens_raw": 0,
        "attempted_generated_tokens_raw": 0,
        "discarded_generated_tokens_raw": 0,
        "compute_overhead_ratio": 0.0,
        "by_model": {
            expert: {"attempted": 0, "committed": 0, "discarded": 0}
            for expert in experts
        },
        "by_phase": {
            phase: {"attempted": 0, "committed": 0, "discarded": 0}
            for phase in PHASES
        },
        "by_model_and_phase": {
            expert: {
                phase: {"attempted": 0, "committed": 0, "discarded": 0}
                for phase in PHASES
            }
            for expert in experts
        },
    }


def printable_token(token_text: str) -> str:
    return token_text.replace("\n", "\\n")


def print_candidate_token_table(
    heading: str,
    expert_id: str,
    z_score: float,
    token_details: list[tuple[str, int, float]],
    token_column: str = "Token",
    score_label: str = "Z-Score",
) -> None:
    print(f"\n {heading} [{expert_id}] {score_label}: {z_score:.4f}")
    print(
        f"    {token_column:<18} | {'Token ID':<10} | "
        f"{'Raw Prob':<10} | {'Log Prob':<10}"
    )
    print("    " + "-" * 60)
    for token_text, token_id, log_prob in token_details:
        safe_text = printable_token(token_text)
        raw_probability = math.exp(log_prob)
        print(
            f"    '{safe_text:<16}' | {token_id:<10} | "
            f"{raw_probability:.4f}     | {log_prob:.4f}"
        )


class MACRRelay(ConfidenceBiddingRelay):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_usage = _empty_usage(self.expert_pool)
        self.last_timing: dict[str, float] = {}
        self.last_cache_stats: dict[str, Any] = {}
        self.last_ori_input_token: dict[str, Any] = {}
        self._initial_selection_wall_seconds = 0.0
        self._bidding_wall_seconds = 0.0
        self._static_prepare_wall_seconds = 0.0
        self._continuation_cache_reuse_steps = 0
        self._continuation_ledger_sync_steps = 0
        self._continuation_full_prefill_steps = 0

    def _new_relay_stats(self) -> dict[str, Any]:
        return {
            "handoff_count": 0,
            "zscore_trigger_count": 0,
            "initial_model": None,
            "final_model": None,
            "trigger_events": [],
        }

    def reset_sample_state(self) -> None:
        self.last_relay_stats = self._new_relay_stats()
        self.last_usage = _empty_usage(self.expert_pool)
        self.last_timing = {}
        self.last_cache_stats = {}
        self.last_ori_input_token = {}
        self._initial_selection_wall_seconds = 0.0
        self._bidding_wall_seconds = 0.0
        self._static_prepare_wall_seconds = 0.0
        self._continuation_cache_reuse_steps = 0
        self._continuation_ledger_sync_steps = 0
        self._continuation_full_prefill_steps = 0

    def _record_tokens(
        self,
        model_name: str,
        phase: str,
        attempted: int,
        committed: int = 0,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"Unknown generation phase: {phase}")
        if attempted < 0 or committed < 0 or committed > attempted:
            raise ValueError(
                f"Invalid token accounting: attempted={attempted}, committed={committed}"
            )
        self.last_usage["attempted_generated_tokens_raw"] += attempted
        self.last_usage["committed_generated_tokens_raw"] += committed
        model_usage = self.last_usage["by_model"][model_name]
        phase_usage = self.last_usage["by_phase"][phase]
        cross_usage = self.last_usage["by_model_and_phase"][model_name][phase]
        for target in (model_usage, phase_usage, cross_usage):
            target["attempted"] += attempted
            target["committed"] += committed

    def _commit_tokens(self, model_name: str, phase: str, committed: int) -> None:
        if committed < 0:
            raise ValueError("committed token count must be non-negative")
        self.last_usage["committed_generated_tokens_raw"] += committed
        self.last_usage["by_model"][model_name]["committed"] += committed
        self.last_usage["by_phase"][phase]["committed"] += committed
        self.last_usage["by_model_and_phase"][model_name][phase][
            "committed"
        ] += committed

    def _finalize_usage(self) -> None:
        attempted = int(self.last_usage["attempted_generated_tokens_raw"])
        committed = int(self.last_usage["committed_generated_tokens_raw"])
        if committed > attempted:
            raise RuntimeError("Committed tokens exceed attempted tokens")
        discarded = attempted - committed
        self.last_usage["discarded_generated_tokens_raw"] = discarded
        self.last_usage["compute_overhead_ratio"] = (
            attempted / committed if committed else 0.0
        )
        for key in ("by_model", "by_phase"):
            for values in self.last_usage[key].values():
                values["discarded"] = values["attempted"] - values["committed"]
        for model_phases in self.last_usage["by_model_and_phase"].values():
            for values in model_phases.values():
                values["discarded"] = values["attempted"] - values["committed"]
        self.last_usage["generated_tokens_raw"] = committed
        self.last_usage["output_tokens"] = committed
        self.last_usage["accounting"] = (
            "exact_generation_ids_attempted_committed_and_discarded"
        )

    def run_initial_selection_phase(
        self,
        image_path: str,
        user_text: str,
        max_tokens: int,
    ) -> CandidateResult:
        trial_token_count = min(
            self.window_size * self.initial_trial_windows, max_tokens
        )
        if trial_token_count <= 0:
            raise ValueError("Initial selection requires a positive token budget")
        print(
            f"\n[Initial Selection] {self.initial_trial_windows} window(s) x "
            f"{self.window_size}, capped at {trial_token_count} token(s) per model"
        )

        def trial_task(expert_id: str) -> CandidateResult:
            candidate = self.models[expert_id].generate_bidding_window(
                image_path, user_text, "", trial_token_count
            )
            if candidate.model_name != expert_id:
                candidate.release_state()
                raise CacheStateError(
                    f"candidate owner mismatch: {candidate.model_name} != {expert_id}"
                )
            candidate.z_score = self.calculate_z_score(
                expert_id, candidate.log_probs
            )
            return candidate

        started = time.perf_counter()
        results: list[CandidateResult] = []
        errors: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.expert_pool)
        ) as executor:
            futures = [executor.submit(trial_task, expert) for expert in self.expert_pool]
            for future in concurrent.futures.as_completed(futures):
                try:
                    candidate = future.result()
                except Exception as error:
                    errors.append(error)
                    continue
                results.append(candidate)
                self._record_tokens(
                    candidate.model_name,
                    "initial_selection",
                    attempted=len(candidate.token_details),
                )
                print_candidate_token_table(
                    "[Initial Trial]",
                    candidate.model_name,
                    float(candidate.z_score),
                    candidate.token_details,
                )
        self._initial_selection_wall_seconds += time.perf_counter() - started

        if errors:
            for candidate in results:
                candidate.release_state()
            raise errors[0]

        best = max(results, key=lambda item: float(item.z_score))
        for candidate in results:
            if candidate is not best:
                candidate.release_state()
        self._commit_tokens(
            best.model_name, "initial_selection", len(best.token_details)
        )
        print(
            f"[Initial Selection Done] model={best.model_name}, "
            f"z={float(best.z_score):.4f}"
        )
        return best

    def run_bidding_phase(
        self,
        current_model: str,
        image_path: str,
        user_text: str,
        assistant_text: str,
        token_budget: int,
    ) -> CandidateResult:
        bid_token_count = min(self.window_size, token_budget)
        if bid_token_count <= 0:
            raise ValueError("Bidding requires a positive remaining token budget")
        print(
            f"\n[Bidding] current={current_model}, threshold={self.z_threshold}, "
            f"window={bid_token_count}"
        )

        def bid_task(expert_id: str) -> CandidateResult:
            candidate = self.models[expert_id].generate_bidding_window(
                image_path, user_text, assistant_text, bid_token_count
            )
            if candidate.model_name != expert_id:
                candidate.release_state()
                raise CacheStateError(
                    f"candidate owner mismatch: {candidate.model_name} != {expert_id}"
                )
            candidate.z_score = self.calculate_z_score(
                expert_id, candidate.log_probs
            )
            return candidate

        started = time.perf_counter()
        results: list[CandidateResult] = []
        errors: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.expert_pool)
        ) as executor:
            futures = [executor.submit(bid_task, expert) for expert in self.expert_pool]
            for future in concurrent.futures.as_completed(futures):
                try:
                    candidate = future.result()
                except Exception as error:
                    errors.append(error)
                    continue
                results.append(candidate)
                self._record_tokens(
                    candidate.model_name,
                    "bidding",
                    attempted=len(candidate.token_details),
                )
                print_candidate_token_table(
                    "[Bid]",
                    candidate.model_name,
                    float(candidate.z_score),
                    candidate.token_details,
                    token_column="Draft token",
                    score_label="Bid Z-Score",
                )
        self._bidding_wall_seconds += time.perf_counter() - started

        if errors:
            for candidate in results:
                candidate.release_state()
            raise errors[0]

        best = max(results, key=lambda item: float(item.z_score))
        for candidate in results:
            if candidate is not best:
                candidate.release_state()
        self._commit_tokens(best.model_name, "bidding", len(best.token_details))
        print(
            f"[Bidding Done] model={best.model_name}, "
            f"z={float(best.z_score):.4f}"
        )
        return best

    def _snapshot_cache_stats(
        self,
        manager: OptCacheStateManager,
        sample_fingerprint: str | None,
    ) -> dict[str, Any]:
        by_model: dict[str, Any] = {}
        for model_name, adapter in self.models.items():
            stats = dict(adapter.sample_stats)
            stats["fallback_reason_counts"] = dict(
                adapter.sample_stats.get("fallback_reason_counts", {})
            )
            stats["ledger_fallback_reason_counts"] = dict(
                adapter.sample_stats.get("ledger_fallback_reason_counts", {})
            )
            stats["ledger_calls"] = list(adapter.sample_stats.get("ledger_calls", []))
            by_model[model_name] = stats
        static_preprocess_count = sum(
            int(stats.get("static_input_cache_misses", 0))
            for stats in by_model.values()
        )
        continuation_runtime_fallbacks = sum(
            int(stats.get("continuation_cache_fallbacks", 0))
            for stats in by_model.values()
        )
        return {
            "optimization_profile": OPTIMIZATION_PROFILE,
            "sample_fingerprint": sample_fingerprint,
            "sample_static_preprocess_count": static_preprocess_count,
            "expected_static_preprocess_count": len(self.models),
            "static_inputs_prepared_once": (
                static_preprocess_count == len(self.models)
                and all(
                    int(stats.get("static_input_cache_misses", 0)) == 1
                    for stats in by_model.values()
                )
            ),
            "static_input_cache_hits": sum(
                int(stats.get("static_input_cache_hits", 0))
                for stats in by_model.values()
            ),
            "continuation_cache_reuse_steps": (
                self._continuation_cache_reuse_steps
            ),
            "continuation_ledger_sync_steps": (
                self._continuation_ledger_sync_steps
            ),
            "continuation_full_prefill_steps": (
                self._continuation_full_prefill_steps
            ),
            "kv_ledger": {
                key: sum(int(stats.get(f"ledger_{key}", 0)) for stats in by_model.values())
                for key in (
                    "cold_prefills",
                    "cold_prefill_tokens",
                    "reuse_calls",
                    "reused_tokens",
                    "prefilled_tokens",
                    "cropped_tokens",
                )
            },
            "winner_state_promotions": manager.promotions,
            "winner_state_promotion_fallbacks": manager.invalid_promotions,
            "continuation_runtime_fallbacks": continuation_runtime_fallbacks,
            "total_cache_fallbacks": (
                manager.invalid_promotions + continuation_runtime_fallbacks
            ),
            "promotion_fallback_reason_counts": dict(
                manager.fallback_reason_counts
            ),
            "released_continuation_states": manager.released_states,
            "active_state_released": manager.active_state is None,
            "by_model": by_model,
        }

    def _promote_candidate(
        self,
        manager: OptCacheStateManager,
        candidate: CandidateResult,
        sample_fingerprint: str,
        committed_text: str,
    ) -> ContinuationState | None:
        try:
            promoted = manager.promote(
                candidate,
                expected_owner=candidate.model_name,
                sample_fingerprint=sample_fingerprint,
                committed_text=committed_text,
                adapter_validator=self.models[
                    candidate.model_name
                ].validate_continuation_state,
            )
            if promoted is None:
                manager.clear()
            return promoted
        finally:
            candidate.release_state()

    def generate_response(
        self, image_path: str, user_prompt: str, max_tokens: int = 1024
    ) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.reset_sample_state()
        manager = OptCacheStateManager()
        sample_fingerprint: str | None = None
        profiler_start = {
            "preprocess_seconds": Profiler.total_prep_time,
            "model_generation_seconds": Profiler.total_gen_time,
        }
        response_started = time.perf_counter()
        try:
            sample_fingerprint = build_sample_fingerprint(image_path, user_prompt)
            for adapter in self.models.values():
                adapter.release_sample()
                adapter._reset_sample_stats()

            prepare_started = time.perf_counter()
            prepare_errors: list[Exception] = []

            def prepare_task(expert_id: str) -> None:
                self.models[expert_id].prepare_sample(
                    image_path, user_prompt, sample_fingerprint
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.expert_pool)
            ) as executor:
                futures = [
                    executor.submit(prepare_task, expert_id)
                    for expert_id in self.expert_pool
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        prepare_errors.append(error)
            self._static_prepare_wall_seconds = (
                time.perf_counter() - prepare_started
            )
            if prepare_errors:
                raise prepare_errors[0]

            return self._generate_response_impl(
                image_path,
                user_prompt,
                max_tokens,
                manager,
                sample_fingerprint,
            )
        finally:
            self.last_ori_input_token = {
                model_name: (
                    dict(adapter.ori_input_token)
                    if adapter.ori_input_token is not None
                    else None
                )
                for model_name, adapter in self.models.items()
            }
            manager.clear()
            for adapter in self.models.values():
                adapter.release_sample()
            self.last_cache_stats = self._snapshot_cache_stats(
                manager, sample_fingerprint
            )
            self._finalize_usage()
            response_seconds = time.perf_counter() - response_started
            attempted = self.last_usage["attempted_generated_tokens_raw"]
            committed = self.last_usage["committed_generated_tokens_raw"]
            model_generation_seconds = max(
                0.0,
                Profiler.total_gen_time - profiler_start["model_generation_seconds"],
            )
            self.last_timing = {
                "preprocess_seconds": max(
                    0.0, Profiler.total_prep_time - profiler_start["preprocess_seconds"]
                ),
                "static_prepare_wall_seconds": self._static_prepare_wall_seconds,
                "model_generation_seconds": model_generation_seconds,
                "initial_selection_wall_seconds": self._initial_selection_wall_seconds,
                "bidding_wall_seconds": self._bidding_wall_seconds,
                "response_wall_seconds": response_seconds,
                "attempted_tokens_per_model_generation_second": (
                    attempted / model_generation_seconds
                    if model_generation_seconds
                    else 0.0
                ),
                "committed_tokens_per_response_second": (
                    committed / response_seconds if response_seconds else 0.0
                ),
            }

    def _generate_response_impl(
        self,
        image_path: str,
        user_prompt: str,
        max_tokens: int,
        manager: OptCacheStateManager,
        sample_fingerprint: str,
    ) -> str:
        assistant_text = ""
        rolling_window_logprobs: list[float] = []
        step_count = 0
        consecutive_empty = 0

        initial_candidate = self.run_initial_selection_phase(
            image_path, user_prompt, max_tokens
        )
        current_model = initial_candidate.model_name
        assistant_text = initial_candidate.text
        self._promote_candidate(
            manager,
            initial_candidate,
            sample_fingerprint,
            assistant_text,
        )
        self.last_relay_stats["initial_model"] = current_model
        self.last_relay_stats["final_model"] = current_model
        print(f"\n==== Generation started: initial model [{current_model}] ====\n")
        print(
            f"{'Model':<10} | {'Token':<18} | "
            f"{'Raw Prob':<10} | {'Log Prob':<10} | {'Z-Score':<12}"
        )
        print("-" * 75)
        rolling_window_logprobs = initial_candidate.log_probs[
            -self.window_size :
        ].copy()
        step_count += len(initial_candidate.token_details)
        for token_text, _, log_prob in initial_candidate.token_details:
            safe_text = printable_token(token_text)
            raw_probability = math.exp(log_prob)
            print(
                f"[{current_model:<8}] | '{safe_text:<16}' | "
                f"{raw_probability:.4f}     | {log_prob:.4f}     | (Initial)"
            )

        if initial_candidate.has_eos or step_count >= max_tokens:
            return assistant_text

        while step_count < max_tokens:
            expert = self.models[current_model]
            old_state = manager.active_state
            token_str, token_id, log_prob, is_eos, new_state = (
                expert.generate_next_token(
                    image_path,
                    user_prompt,
                    assistant_text,
                    continuation_state=old_state,
                )
            )
            manager.replace_after_decode(old_state, new_state)
            if new_state.source == "continuation_cache":
                self._continuation_cache_reuse_steps += 1
            elif new_state.source == "continuation_ledger_sync":
                self._continuation_ledger_sync_steps += 1
            else:
                self._continuation_full_prefill_steps += 1
            self._record_tokens(
                current_model, "continuation", attempted=1, committed=1
            )
            assistant_text += token_str
            rolling_window_logprobs.append(log_prob)
            step_count += 1
            if len(rolling_window_logprobs) > self.window_size:
                rolling_window_logprobs.pop(0)

            current_z_score = None
            if len(rolling_window_logprobs) == self.window_size:
                current_z_score = self.calculate_z_score(
                    current_model, rolling_window_logprobs
                )
            z_display = (
                f"{current_z_score:.4f}"
                if current_z_score is not None
                else "filling window..."
            )
            safe_text = printable_token(token_str)
            raw_probability = math.exp(log_prob)
            print(
                f"[{current_model:<8}] | '{safe_text:<16}' | "
                f"{raw_probability:.4f}     | {log_prob:.4f}     | {z_display}"
            )

            if not token_str.strip():
                consecutive_empty += 1
            else:
                consecutive_empty = 0
            if consecutive_empty >= 5 or is_eos or step_count >= max_tokens:
                break
            if current_z_score is None or current_z_score >= self.z_threshold:
                continue

            previous_model = current_model
            remaining_budget = max_tokens - step_count
            if remaining_budget <= 0:
                break
            self.last_relay_stats["zscore_trigger_count"] += 1
            winner = self.run_bidding_phase(
                current_model,
                image_path,
                user_prompt,
                assistant_text,
                remaining_budget,
            )
            winner_model = winner.model_name
            win_text = winner.text
            win_logprobs = winner.log_probs
            win_eos = winner.has_eos
            win_details = winner.token_details
            committed_after_bid = assistant_text + win_text
            self._promote_candidate(
                manager,
                winner,
                sample_fingerprint,
                committed_after_bid,
            )
            is_handoff = winner_model != previous_model
            if is_handoff:
                self.last_relay_stats["handoff_count"] += 1
            self.last_relay_stats["trigger_events"].append(
                {
                    "trigger": "zscore",
                    "from_model": previous_model,
                    "to_model": winner_model,
                    "is_handoff": is_handoff,
                    "step": step_count,
                    "z_score": current_z_score,
                    "threshold": self.z_threshold,
                    "committed_bid_tokens": len(win_details),
                }
            )
            print(f"\n--- {winner_model} takes over with its bid draft ---")
            print(
                f"{'Model':<10} | {'Token':<18} | "
                f"{'Raw Prob':<10} | {'Log Prob':<10} | {'Z-Score':<12}"
            )
            print("-" * 75)
            for token_text, _, draft_log_prob in win_details:
                safe_text = printable_token(token_text)
                raw_probability = math.exp(draft_log_prob)
                print(
                    f"[{winner_model:<8}] | '{safe_text:<16}' | "
                    f"{raw_probability:.4f}     | {draft_log_prob:.4f}     | (Draft)"
                )
            print("-" * 75)
            current_model = winner_model
            self.last_relay_stats["final_model"] = current_model
            assistant_text = committed_after_bid
            rolling_window_logprobs = win_logprobs[-self.window_size :].copy()
            step_count += len(win_details)
            if win_eos or step_count >= max_tokens:
                break

        self.last_relay_stats["final_model"] = current_model
        return assistant_text
