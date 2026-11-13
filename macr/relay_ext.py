from __future__ import annotations

import concurrent.futures
import math
import time
from dataclasses import dataclass, field
from typing import Any

from core.cache_state import CacheStateError, CandidateResult, OptCacheStateManager
from core.relay import MACRRelay, printable_token

from macr_ext import answer_bid
from macr_ext.boundary import BoundaryDetector
from macr_ext.config import ExtConfig

ANSWER_PHASE = "answer_bid"


@dataclass
class _Trigger:
    step: int
    z_score: float


@dataclass
class _Segment:
    model: str
    phase: str
    start_char: int
    token_texts: list[str]


@dataclass
class _Loop:
    image_path: str
    user_prompt: str
    max_tokens: int
    manager: OptCacheStateManager
    sample_fingerprint: str
    assistant_text: str = ""
    current_model: str = ""
    rolling_window_logprobs: list[float] = field(default_factory=list)
    step_count: int = 0
    consecutive_empty: int = 0
    last_z_score: float | None = None
    pending: _Trigger | None = None
    segment: _Segment | None = None
    marker_scan_from: int = 0
    stop: bool = False


def _print_stream_header() -> None:
    print(
        f"{'Model':<10} | {'Token':<18} | "
        f"{'Raw Prob':<10} | {'Log Prob':<10} | {'Z-Score':<12}"
    )
    print("-" * 75)


def _print_stream_row(model: str, token_text: str, log_prob: float, label: str) -> None:
    safe_text = printable_token(token_text)
    raw_probability = math.exp(log_prob)
    print(
        f"[{model:<8}] | '{safe_text:<16}' | "
        f"{raw_probability:.4f}     | {log_prob:.4f}     | {label}"
    )


def _print_answer_table(
    candidates: list[answer_bid.AnswerCandidate], winner: str | None
) -> None:
    print(
        f"    {'Model':<10} | {'Answer text':<24} | {'Normalized':<12} | "
        f"{'Z-Score':>8} | {'Sum LP':>8}"
    )
    print("    " + "-" * 74)
    for candidate in candidates:
        z = "n/a" if candidate.z is None else f"{candidate.z:.4f}"
        sum_lp = "n/a" if candidate.sum_logprob is None else f"{candidate.sum_logprob:.4f}"
        text = printable_token(candidate.answer_text)[:24]
        mark = "  <== winner" if candidate.model == winner else ""
        print(
            f"    {candidate.model:<10} | {text:<24} | {str(candidate.normalized):<12} | "
            f"{z:>8} | {sum_lp:>8}{mark}"
        )


class MACRRelayExt(MACRRelay):
    def __init__(
        self,
        *args: Any,
        ext_config: ExtConfig,
        benchmark_name: str,
        **kwargs: Any,
    ) -> None:
        self.ext = ext_config
        self.benchmark_name = benchmark_name
        self.boundary = (
            BoundaryDetector(ext_config.delay_bid) if ext_config.delay_enabled else None
        )
        self.normalize_answer = (
            answer_bid.get_normalizer(benchmark_name) if ext_config.answer_enabled else None
        )
        self._answer_bid_wall_seconds = 0.0
        self._rejected_markers: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)
        self._add_answer_phase()
        print(f"[macr_ext] {ext_config.describe()}")

    def _new_relay_stats(self) -> dict[str, Any]:
        stats = super()._new_relay_stats()
        stats["bid_count"] = 0
        if self.ext.delay_enabled:
            stats["delay_bidding"] = {
                "mode": self.ext.delay_bid,
                "max_bid_delay": self.ext.max_bid_delay,
                "fire_reasons": {},
                "dropped_pending": {},
                "lookahead_discarded_tokens": 0,
                "delay_tokens": [],
            }
        if self.ext.answer_enabled:
            stats["answer_bid"] = None
        return stats

    def reset_sample_state(self) -> None:
        super().reset_sample_state()
        self._answer_bid_wall_seconds = 0.0
        self._rejected_markers = []
        self._add_answer_phase()

    def _add_answer_phase(self) -> None:
        if not self.ext.answer_enabled:
            return
        usage = self.last_usage
        usage["by_phase"].setdefault(
            ANSWER_PHASE, {"attempted": 0, "committed": 0, "discarded": 0}
        )
        for model_phases in usage["by_model_and_phase"].values():
            model_phases.setdefault(
                ANSWER_PHASE, {"attempted": 0, "committed": 0, "discarded": 0}
            )

    def _record_tokens(
        self,
        model_name: str,
        phase: str,
        attempted: int,
        committed: int = 0,
    ) -> None:
        if phase != ANSWER_PHASE:
            super()._record_tokens(model_name, phase, attempted, committed)
            return
        if attempted < 0 or committed < 0 or committed > attempted:
            raise ValueError(
                f"Invalid token accounting: attempted={attempted}, committed={committed}"
            )
        usage = self.last_usage
        usage["attempted_generated_tokens_raw"] += attempted
        usage["committed_generated_tokens_raw"] += committed
        for target in (
            usage["by_model"][model_name],
            usage["by_phase"][phase],
            usage["by_model_and_phase"][model_name][phase],
        ):
            target["attempted"] += attempted
            target["committed"] += committed

    def _uncommit_tokens(self, model_name: str, phase: str, count: int) -> None:
        if count <= 0:
            return
        usage = self.last_usage
        targets = (
            usage["by_model"][model_name],
            usage["by_phase"][phase],
            usage["by_model_and_phase"][model_name][phase],
        )
        if any(target["committed"] < count for target in targets):
            raise RuntimeError("Cannot un-commit more tokens than were committed")
        usage["committed_generated_tokens_raw"] -= count
        for target in targets:
            target["committed"] -= count

    def generate_response(
        self, image_path: str, user_prompt: str, max_tokens: int = 1024
    ) -> str:
        try:
            return super().generate_response(image_path, user_prompt, max_tokens)
        finally:
            if self.ext.answer_enabled:
                self.last_timing["answer_bid_wall_seconds"] = self._answer_bid_wall_seconds

    def _generate_response_impl(
        self,
        image_path: str,
        user_prompt: str,
        max_tokens: int,
        manager: OptCacheStateManager,
        sample_fingerprint: str,
    ) -> str:
        loop = _Loop(
            image_path=image_path,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            manager=manager,
            sample_fingerprint=sample_fingerprint,
        )

        initial_candidate = self.run_initial_selection_phase(
            image_path, user_prompt, max_tokens
        )
        loop.current_model = initial_candidate.model_name
        loop.assistant_text = initial_candidate.text
        loop.segment = _Segment(
            loop.current_model,
            "initial_selection",
            0,
            [token_text for token_text, _, _ in initial_candidate.token_details],
        )
        self._promote_candidate(
            manager,
            initial_candidate,
            sample_fingerprint,
            loop.assistant_text,
        )
        self.last_relay_stats["initial_model"] = loop.current_model
        self.last_relay_stats["final_model"] = loop.current_model
        print(f"\n==== Generation started: initial model [{loop.current_model}] ====\n")
        _print_stream_header()
        loop.rolling_window_logprobs = initial_candidate.log_probs[
            -self.window_size :
        ].copy()
        loop.step_count += len(initial_candidate.token_details)
        for token_text, _, log_prob in initial_candidate.token_details:
            _print_stream_row(loop.current_model, token_text, log_prob, "(Initial)")

        response = self._maybe_answer_bid(loop)
        if response is not None:
            return response
        if initial_candidate.has_eos or loop.step_count >= max_tokens:
            return self._finish(loop)

        while loop.step_count < max_tokens:
            expert = self.models[loop.current_model]
            old_state = manager.active_state
            token_str, token_id, log_prob, is_eos, new_state = (
                expert.generate_next_token(
                    image_path,
                    user_prompt,
                    loop.assistant_text,
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

            if loop.pending is not None:
                rule = self.boundary.before_token(loop.assistant_text, token_str)
                if rule is not None:
                    self._record_tokens(
                        loop.current_model, "continuation", attempted=1, committed=0
                    )
                    self.last_relay_stats["delay_bidding"][
                        "lookahead_discarded_tokens"
                    ] += 1
                    print(
                        f"[Delay] '{printable_token(token_str)}' opens a new "
                        "sentence: not committed, bidding before it"
                    )
                    manager.clear()
                    response = self._fire_bid(loop, rule, loop.pending)
                    if response is not None:
                        return response
                    if loop.stop:
                        break
                    continue

            self._record_tokens(
                loop.current_model, "continuation", attempted=1, committed=1
            )
            loop.segment = _Segment(
                loop.current_model,
                "continuation",
                len(loop.assistant_text),
                [token_str],
            )
            loop.assistant_text += token_str
            loop.rolling_window_logprobs.append(log_prob)
            loop.step_count += 1
            if len(loop.rolling_window_logprobs) > self.window_size:
                loop.rolling_window_logprobs.pop(0)

            current_z_score = None
            if len(loop.rolling_window_logprobs) == self.window_size:
                current_z_score = self.calculate_z_score(
                    loop.current_model, loop.rolling_window_logprobs
                )
            loop.last_z_score = current_z_score
            z_display = (
                f"{current_z_score:.4f}"
                if current_z_score is not None
                else "filling window..."
            )
            _print_stream_row(loop.current_model, token_str, log_prob, z_display)

            response = self._maybe_answer_bid(loop)
            if response is not None:
                return response

            if not token_str.strip():
                loop.consecutive_empty += 1
            else:
                loop.consecutive_empty = 0
            if loop.consecutive_empty >= 5 or is_eos or loop.step_count >= max_tokens:
                if loop.consecutive_empty >= 5:
                    self._drop_pending(loop, "empty_run")
                elif is_eos:
                    self._drop_pending(loop, "eos")
                else:
                    self._drop_pending(loop, "max_tokens")
                break

            if loop.pending is not None:
                rule = self.boundary.after_commit(loop.assistant_text, token_str)
                if (
                    rule is None
                    and loop.step_count - loop.pending.step >= self.ext.max_bid_delay
                ):
                    rule = "max_delay"
                if rule is None:
                    continue
                response = self._fire_bid(loop, rule, loop.pending)
                if response is not None:
                    return response
                if loop.stop:
                    break
                continue

            if current_z_score is None or current_z_score >= self.z_threshold:
                continue

            self.last_relay_stats["zscore_trigger_count"] += 1
            trigger = _Trigger(step=loop.step_count, z_score=current_z_score)
            reason = "immediate"
            if self.ext.delay_enabled:
                if self.boundary.after_commit(loop.assistant_text, token_str) is None:
                    loop.pending = trigger
                    print(
                        f"[Delay] z={current_z_score:.4f} < {self.z_threshold} at "
                        f"step {loop.step_count}: bidding at the next "
                        f"{self.ext.delay_bid} boundary "
                        f"(at most {self.ext.max_bid_delay} tokens)"
                    )
                    continue
                reason = "at_boundary"
            response = self._fire_bid(loop, reason, trigger)
            if response is not None:
                return response
            if loop.stop:
                break

        return self._finish(loop)

    def _fire_bid(self, loop: _Loop, reason: str, trigger: _Trigger) -> str | None:
        loop.pending = None
        previous_model = loop.current_model
        remaining_budget = loop.max_tokens - loop.step_count
        if remaining_budget <= 0:
            loop.stop = True
            return None
        self.last_relay_stats["bid_count"] += 1
        delay_tokens = loop.step_count - trigger.step
        if self.ext.delay_enabled:
            delay_stats = self.last_relay_stats["delay_bidding"]
            delay_stats["fire_reasons"][reason] = (
                delay_stats["fire_reasons"].get(reason, 0) + 1
            )
            delay_stats["delay_tokens"].append(delay_tokens)
            print(
                f"[Delay] bidding now: rule={reason}, delayed {delay_tokens} "
                f"token(s) after the trigger at step {trigger.step}"
            )

        winner = self.run_bidding_phase(
            previous_model,
            loop.image_path,
            loop.user_prompt,
            loop.assistant_text,
            remaining_budget,
        )
        winner_model = winner.model_name
        win_text = winner.text
        win_logprobs = winner.log_probs
        win_eos = winner.has_eos
        win_details = winner.token_details
        committed_after_bid = loop.assistant_text + win_text
        self._promote_candidate(
            loop.manager,
            winner,
            loop.sample_fingerprint,
            committed_after_bid,
        )
        is_handoff = winner_model != previous_model
        if is_handoff:
            self.last_relay_stats["handoff_count"] += 1
        event = {
            "trigger": "zscore",
            "from_model": previous_model,
            "to_model": winner_model,
            "is_handoff": is_handoff,
            "step": loop.step_count,
            "z_score": trigger.z_score,
            "threshold": self.z_threshold,
            "committed_bid_tokens": len(win_details),
        }
        if self.ext.delay_enabled:
            event.update(
                {
                    "trigger_step": trigger.step,
                    "delay_tokens": delay_tokens,
                    "fire_reason": reason,
                    "z_at_bid": loop.last_z_score,
                }
            )
        self.last_relay_stats["trigger_events"].append(event)
        print(f"\n--- {winner_model} takes over with its bid draft ---")
        _print_stream_header()
        for token_text, _, draft_log_prob in win_details:
            _print_stream_row(winner_model, token_text, draft_log_prob, "(Draft)")
        print("-" * 75)
        loop.current_model = winner_model
        self.last_relay_stats["final_model"] = loop.current_model
        loop.segment = _Segment(
            winner_model,
            "bidding",
            len(loop.assistant_text),
            [token_text for token_text, _, _ in win_details],
        )
        loop.assistant_text = committed_after_bid
        loop.rolling_window_logprobs = win_logprobs[-self.window_size :].copy()
        loop.step_count += len(win_details)

        response = self._maybe_answer_bid(loop)
        if response is not None:
            return response
        if win_eos or loop.step_count >= loop.max_tokens:
            loop.stop = True
        return None

    def _drop_pending(self, loop: _Loop, reason: str) -> None:
        if loop.pending is None:
            return
        dropped = self.last_relay_stats["delay_bidding"]["dropped_pending"]
        dropped[reason] = dropped.get(reason, 0) + 1
        loop.pending = None

    def _finish(self, loop: _Loop) -> str:
        self.last_relay_stats["final_model"] = loop.current_model
        if self.ext.answer_enabled and self.last_relay_stats["answer_bid"] is None:
            if self.ext.answer_force_marker:
                prefix = loop.assistant_text.rstrip() + answer_bid.forced_marker(
                    self.benchmark_name
                )
                response = self._run_answer_bid(loop, prefix, trigger="forced")
                if response is not None:
                    return response
            self.last_relay_stats["answer_bid"] = {
                "status": "no_marker",
                "rule": self.ext.answer_bid,
                "rejected_markers": list(self._rejected_markers),
            }
        return loop.assistant_text

    def _maybe_answer_bid(self, loop: _Loop) -> str | None:
        if not self.ext.answer_enabled:
            return None
        while True:
            cut = answer_bid.find_answer_marker(
                loop.assistant_text, since=loop.marker_scan_from
            )
            if cut is None:
                loop.marker_scan_from = len(loop.assistant_text)
                return None
            response = self._run_answer_bid(
                loop, loop.assistant_text[:cut], trigger="marker"
            )
            if response is not None:
                return response
            loop.marker_scan_from = cut

    def _uncommit_after(self, loop: _Loop, cut: int) -> int:
        segment = loop.segment
        if segment is None:
            return 0
        offset = segment.start_char
        count = 0
        for token_text in segment.token_texts:
            if offset >= cut:
                count += 1
            offset += len(token_text)
        self._uncommit_tokens(segment.model, segment.phase, count)
        return count

    def _run_answer_bid(self, loop: _Loop, prefix: str, trigger: str) -> str | None:
        incumbent = loop.current_model
        budget = self.ext.answer_max_tokens
        print(
            f"\n[Answer Bid] trigger={trigger}, rule={self.ext.answer_bid}, "
            f"incumbent={incumbent}, max_tokens={budget}"
        )
        started = time.perf_counter()

        def answer_task(expert_id: str) -> CandidateResult:
            candidate = self.models[expert_id].generate_bidding_window(
                loop.image_path, loop.user_prompt, prefix, budget
            )
            candidate.release_state()
            if candidate.model_name != expert_id:
                raise CacheStateError(
                    f"candidate owner mismatch: {candidate.model_name} != {expert_id}"
                )
            return candidate

        results: dict[str, CandidateResult] = {}
        errors: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.expert_pool)
        ) as executor:
            futures = [executor.submit(answer_task, expert) for expert in self.expert_pool]
            for future in concurrent.futures.as_completed(futures):
                try:
                    candidate = future.result()
                except Exception as error:
                    errors.append(error)
                    continue
                results[candidate.model_name] = candidate
                self._record_tokens(
                    candidate.model_name,
                    ANSWER_PHASE,
                    attempted=len(candidate.token_details),
                )
        self._answer_bid_wall_seconds += time.perf_counter() - started
        if errors:
            raise errors[0]

        candidates = [
            answer_bid.build_candidate(
                model=expert_id,
                text=results[expert_id].text,
                token_details=results[expert_id].token_details,
                finished=results[expert_id].has_eos,
                z_score=lambda log_probs, model=expert_id: self.calculate_z_score(
                    model, log_probs
                ),
                normalize=lambda text: self.normalize_answer(loop.user_prompt, text),
            )
            for expert_id in self.expert_pool
        ]
        incumbent_candidate = next(c for c in candidates if c.model == incumbent)
        if trigger == "marker" and incumbent_candidate.normalized is None:
            self._rejected_markers.append(
                {
                    "step": loop.step_count,
                    "prefix_chars": len(prefix),
                    "incumbent_answer_text": incumbent_candidate.answer_text,
                }
            )
            _print_answer_table(candidates, None)
            print(
                f"[Answer Bid] rejected: {incumbent} wrote no recognizable answer "
                "after the marker; generation continues"
            )
            return None

        selection = answer_bid.select_answer(candidates, self.ext.answer_bid, incumbent)
        winner = selection.winner
        discarded = self._uncommit_after(loop, len(prefix)) if trigger == "marker" else 0
        self._drop_pending(loop, "answer_marker")
        loop.manager.clear()
        self._commit_tokens(winner.model, ANSWER_PHASE, winner.answer_tokens)
        self.last_relay_stats["answer_bid"] = {
            "status": "done",
            "trigger": trigger,
            "rule": self.ext.answer_bid,
            "incumbent_model": incumbent,
            "prefix_chars": len(prefix),
            "discarded_after_marker_tokens": discarded,
            "rejected_markers": list(self._rejected_markers),
            "candidates": [candidate.to_record() for candidate in candidates],
            "tally": selection.tally,
            "agreement": selection.agreement,
            "fallback": selection.fallback,
            "winner_model": winner.model,
            "winner_answer": winner.normalized,
            "incumbent_answer": incumbent_candidate.normalized,
            "changed": selection.changed,
        }
        _print_answer_table(candidates, winner.model)
        print(
            f"[Answer Bid Done] winner={winner.model}, answer={winner.normalized!r}, "
            f"agreement={selection.agreement}, changed={selection.changed}"
        )
        return prefix + winner.answer_text
