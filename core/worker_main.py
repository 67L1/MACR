from __future__ import annotations

import contextlib
import importlib
import io
import itertools
import traceback
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkerSpec:
    name: str
    factory: str
    model_path: str
    device: str
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    disable_candidate_cache: bool = False
    seed: int | None = None


def state_fields(state_id: int, state: Any) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "owner_model": state.owner_model,
        "sample_fingerprint": state.sample_fingerprint,
        "source_history_fingerprint": state.source_history_fingerprint,
        "committed_history_fingerprint": state.committed_history_fingerprint,
        "has_cache": state.past_key_values is not None,
        "pending_token_id": state.pending_token_id,
        "cache_seq_len": state.cache_seq_len,
        "expected_cache_seq_len": state.expected_cache_seq_len,
        "source": state.source,
        "pending_consumed": state.pending_consumed,
        "released": state.released,
    }


def _profiler_totals() -> tuple[float, float, float]:
    from core.runtime import Profiler

    return (Profiler.total_prep_time, Profiler.total_gen_time, Profiler.total_bidding_time)


class _WorkerServer:
    def __init__(self, spec: WorkerSpec, adapter: Any) -> None:
        self.spec = spec
        self.adapter = adapter
        self.states: dict[int, Any] = {}
        self._state_ids = itertools.count(1)
        self._handlers = {
            "prepare_sample": self.prepare_sample,
            "release_sample": self.release_sample,
            "reset_sample_stats": self.reset_sample_stats,
            "generate_bidding_window": self.generate_bidding_window,
            "generate_next_token": self.generate_next_token,
            "validate_continuation_state": self.validate_continuation_state,
            "get_sample_stats": self.get_sample_stats,
            "get_ori_input_token": self.get_ori_input_token,
            "reset_peak_memory": self.reset_peak_memory,
            "collect_peak_memory": self.collect_peak_memory,
            "state_count": self.state_count,
        }

    def _register(self, state: Any) -> dict[str, Any] | None:
        if state is None:
            return None
        state_id = next(self._state_ids)
        self.states[state_id] = state
        return state_fields(state_id, state)

    def _release(self, state_ids: list[int]) -> None:
        for state_id in state_ids:
            state = self.states.pop(state_id, None)
            if state is not None:
                state.release()

    def _clear(self) -> None:
        self._release(list(self.states))

    def prepare_sample(self, image_path: str, prompt: str, sample_fingerprint: str) -> None:
        self._clear()
        self.adapter.prepare_sample(image_path, prompt, sample_fingerprint)

    def release_sample(self) -> None:
        self._clear()
        self.adapter.release_sample()

    def reset_sample_stats(self) -> None:
        self.adapter._reset_sample_stats()

    def generate_bidding_window(
        self, image_path: str, prompt: str, assistant_text: str, window_size: int
    ) -> dict[str, Any]:
        candidate = self.adapter.generate_bidding_window(
            image_path, prompt, assistant_text, window_size
        )
        state = candidate.continuation_state
        candidate.continuation_state = None
        return {
            "model_name": candidate.model_name,
            "text": candidate.text,
            "log_probs": list(candidate.log_probs),
            "has_eos": candidate.has_eos,
            "token_details": [tuple(detail) for detail in candidate.token_details],
            "token_ids": list(candidate.token_ids),
            "z_score": candidate.z_score,
            "state": self._register(state),
        }

    def generate_next_token(
        self,
        image_path: str,
        prompt: str,
        assistant_text: str,
        state_id: int | None = None,
        missing_reason: str | None = None,
    ) -> tuple[str, int, float, bool, dict[str, Any] | None]:
        state = self.states.pop(state_id, None) if state_id is not None else None
        if state_id is not None and state is None:
            missing_reason = "remote_state_missing"
        if missing_reason is not None:
            self.adapter._record_cache_fallback(missing_reason)
        try:
            token_str, token_id, log_prob, is_eos, new_state = self.adapter.generate_next_token(
                image_path, prompt, assistant_text, continuation_state=state
            )
        finally:
            if state is not None:
                state.release()
        return token_str, token_id, log_prob, is_eos, self._register(new_state)

    def validate_continuation_state(
        self, state_id: int, committed_text: str
    ) -> tuple[bool, str | None]:
        state = self.states.get(state_id)
        if state is None:
            return False, "remote_state_missing"
        return self.adapter.validate_continuation_state(state, committed_text)

    def get_sample_stats(self) -> dict[str, Any]:
        return self.adapter.sample_stats

    def get_ori_input_token(self) -> dict[str, Any] | None:
        return self.adapter.ori_input_token

    def reset_peak_memory(self) -> None:
        from core.runtime import reset_peak_memory

        reset_peak_memory({self.spec.name: self.spec.device})

    def collect_peak_memory(self) -> dict[str, dict[str, Any]]:
        from core.runtime import collect_peak_memory

        return collect_peak_memory({self.spec.name: self.spec.device})

    def state_count(self) -> int:
        return len(self.states)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> tuple:
        before = _profiler_totals()
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                handler = self._handlers.get(method)
                if handler is None:
                    raise AttributeError(f"model worker has no method {method!r}")
                status, payload = "ok", handler(*args, **kwargs)
        except Exception as error:
            status, payload = "error", {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        deltas = tuple(after - start for after, start in zip(_profiler_totals(), before))
        return status, payload, deltas, out.getvalue(), err.getvalue()

    def serve(self, conn: Any) -> None:
        try:
            while True:
                try:
                    method, args, kwargs, release_ids = conn.recv()
                except (EOFError, OSError):
                    return
                self._release(release_ids)
                if method == "shutdown":
                    self.release_sample()
                    conn.send(("ok", None, (0.0, 0.0, 0.0), "", ""))
                    return
                reply = self._dispatch(method, args, kwargs)
                try:
                    conn.send(reply)
                except (BrokenPipeError, OSError):
                    return
                except Exception as error:
                    conn.send((
                        "error",
                        {
                            "type": type(error).__name__,
                            "message": f"cannot send the result of {method}: {error}",
                            "traceback": traceback.format_exc(),
                        },
                        reply[2],
                        reply[3],
                        reply[4],
                    ))
        finally:
            self._clear()


def run_worker(conn: Any, spec: WorkerSpec) -> None:
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            module_name, _, attribute = spec.factory.partition(":")
            factory = getattr(importlib.import_module(module_name), attribute)
            import torch

            from core.runtime import set_global_seed

            if spec.seed is not None:
                set_global_seed(spec.seed)
            if spec.device.startswith("cuda"):
                torch.cuda.set_device(torch.device(spec.device))
            adapter = factory(spec.model_path, spec.device, **spec.init_kwargs)
            adapter.load()
            if spec.disable_candidate_cache:
                adapter.disable_candidate_cache()
        info = {
            "kv_ledger_enabled": getattr(adapter, "kv_ledger_enabled", None),
            "kv_ledger_disabled_reasons": list(getattr(adapter, "kv_ledger_disabled_reasons", [])),
            "candidate_cache_fastpath_enabled": getattr(
                adapter, "candidate_cache_fastpath_enabled", None
            ),
        }
        conn.send(("ready", info, out.getvalue(), err.getvalue()))
    except Exception:
        conn.send(("load_error", traceback.format_exc(), out.getvalue(), err.getvalue()))
        return
    _WorkerServer(spec, adapter).serve(conn)
