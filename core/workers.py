from __future__ import annotations

import multiprocessing
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from core.cache_state import CandidateResult, ContinuationState
from core.runtime import add_profiler_time
from core.worker_main import WorkerSpec, run_worker


class WorkerError(RuntimeError):
    pass


class WorkerDiedError(RuntimeError):
    pass


class RemoteCacheRef:
    __slots__ = ("state_id", "seq_len")

    def __init__(self, state_id: int, seq_len: int | None) -> None:
        self.state_id = state_id
        self.seq_len = seq_len

    def get_seq_length(self) -> int | None:
        return self.seq_len


@dataclass
class RemoteContinuationState(ContinuationState):
    state_id: int = 0
    on_release: Callable[[int], None] | None = field(default=None, repr=False, compare=False)

    def release(self) -> None:
        if self.released:
            return
        super().release()
        if self.on_release is not None:
            self.on_release(self.state_id)


class RemoteAdapter:
    def __init__(self, spec: WorkerSpec, process: Any, conn: Any, info: dict[str, Any]) -> None:
        self.spec = spec
        self.MODEL_NAME = spec.name
        self.device = spec.device
        self.process = process
        self.info = info
        self._conn = conn
        self._call_lock = threading.Lock()
        self._release_lock = threading.Lock()
        self._pending_release: list[int] = []
        self._dead = False

    def _queue_release(self, state_id: int) -> None:
        with self._release_lock:
            self._pending_release.append(state_id)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._call_lock:
            if self._dead:
                raise WorkerDiedError(f"{self.MODEL_NAME} worker is no longer running")
            with self._release_lock:
                release_ids, self._pending_release = self._pending_release, []
            try:
                self._conn.send((method, args, kwargs, release_ids))
                status, payload, deltas, out, err = self._conn.recv()
            except (EOFError, OSError) as error:
                self._dead = True
                self.process.join(timeout=5)
                raise WorkerDiedError(
                    f"{self.MODEL_NAME} worker (pid {self.process.pid}) died during {method}; "
                    f"exitcode={self.process.exitcode}"
                ) from error
        add_profiler_time(*deltas)
        if out:
            sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        if status != "ok":
            raise WorkerError(
                f"[{self.MODEL_NAME} worker] {payload['type']}: {payload['message']}\n"
                f"--- remote traceback ---\n{payload['traceback']}"
            )
        return payload

    def _state(self, fields: dict[str, Any] | None) -> RemoteContinuationState | None:
        if fields is None:
            return None
        state_id = fields["state_id"]
        return RemoteContinuationState(
            owner_model=fields["owner_model"],
            sample_fingerprint=fields["sample_fingerprint"],
            source_history_fingerprint=fields["source_history_fingerprint"],
            committed_history_fingerprint=fields["committed_history_fingerprint"],
            past_key_values=(
                RemoteCacheRef(state_id, fields["cache_seq_len"]) if fields["has_cache"] else None
            ),
            pending_token_id=fields["pending_token_id"],
            cache_seq_len=fields["cache_seq_len"],
            expected_cache_seq_len=fields["expected_cache_seq_len"],
            source=fields["source"],
            pending_consumed=fields["pending_consumed"],
            released=fields["released"],
            state_id=state_id,
            on_release=self._queue_release,
        )

    def prepare_sample(self, image_path: str, prompt: str, sample_fingerprint: str) -> None:
        self._call("prepare_sample", image_path, prompt, sample_fingerprint)

    def release_sample(self) -> None:
        self._call("release_sample")

    def _reset_sample_stats(self) -> None:
        self._call("reset_sample_stats")

    def generate_bidding_window(
        self, image_path: str, prompt: str, assistant_text: str, window_size: int
    ) -> CandidateResult:
        result = self._call(
            "generate_bidding_window", image_path, prompt, assistant_text, window_size
        )
        return CandidateResult(
            model_name=result["model_name"],
            text=result["text"],
            log_probs=result["log_probs"],
            has_eos=result["has_eos"],
            token_details=result["token_details"],
            token_ids=result["token_ids"],
            continuation_state=self._state(result["state"]),
            z_score=result["z_score"],
        )

    def generate_next_token(
        self,
        image_path: str,
        prompt: str,
        assistant_text: str,
        continuation_state: ContinuationState | None = None,
    ):
        state_id = None
        missing_reason = None
        if continuation_state is not None:
            if not isinstance(continuation_state, RemoteContinuationState):
                missing_reason = "state_not_from_this_worker"
            elif continuation_state.released:
                missing_reason = "state_released"
            else:
                state_id = continuation_state.state_id
                continuation_state.on_release = None
        token_str, token_id, log_prob, is_eos, fields = self._call(
            "generate_next_token", image_path, prompt, assistant_text, state_id, missing_reason
        )
        return token_str, token_id, log_prob, is_eos, self._state(fields)

    def validate_continuation_state(
        self, state: ContinuationState, committed_text: str
    ) -> tuple[bool, str | None]:
        if not isinstance(state, RemoteContinuationState):
            return False, "state_not_from_this_worker"
        if state.released:
            return False, "state_released"
        valid, reason = self._call("validate_continuation_state", state.state_id, committed_text)
        return bool(valid), reason

    @property
    def sample_stats(self) -> dict[str, Any]:
        return self._call("get_sample_stats")

    @property
    def ori_input_token(self) -> dict[str, Any] | None:
        return self._call("get_ori_input_token")

    def reset_peak_memory(self) -> None:
        self._call("reset_peak_memory")

    def collect_peak_memory(self) -> dict[str, dict[str, Any]]:
        return self._call("collect_peak_memory")

    def state_count(self) -> int:
        return self._call("state_count")

    def close(self, timeout: float = 30.0) -> None:
        if not self._dead:
            self._dead = True
            try:
                with self._call_lock:
                    self._conn.send(("shutdown", (), {}, []))
                    self._conn.recv()
            except (EOFError, OSError):
                pass
        _stop_process(self.process, self._conn, timeout)


def _stop_process(process: Any, conn: Any, timeout: float) -> None:
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    try:
        conn.close()
    except OSError:
        pass


def start_model_workers(specs: list[WorkerSpec]) -> dict[str, RemoteAdapter]:
    context = multiprocessing.get_context("spawn")
    launched: list[tuple[WorkerSpec, Any, Any]] = []
    adapters: dict[str, RemoteAdapter] = {}
    try:
        for spec in specs:
            parent_conn, child_conn = context.Pipe()
            process = context.Process(
                target=run_worker,
                args=(child_conn, spec),
                name=f"macr-worker-{spec.name}",
                daemon=True,
            )
            process.start()
            child_conn.close()
            launched.append((spec, process, parent_conn))
        failures: list[str] = []
        for spec, process, conn in launched:
            try:
                status, payload, out, err = conn.recv()
            except (EOFError, OSError):
                process.join(timeout=5)
                failures.append(
                    f"{spec.name}: worker exited during start-up (exitcode={process.exitcode})"
                )
                continue
            if out:
                sys.stdout.write(out)
            if err:
                sys.stderr.write(err)
            if status != "ready":
                failures.append(f"{spec.name}:\n{payload}")
                continue
            adapters[spec.name] = RemoteAdapter(spec, process, conn, payload)
        if failures:
            raise WorkerError("Model worker start-up failed:\n" + "\n".join(failures))
        return adapters
    except BaseException:
        for spec, process, conn in launched:
            if spec.name in adapters:
                adapters[spec.name].close(timeout=10)
            else:
                _stop_process(process, conn, timeout=0)
        raise


def shutdown_workers(models: dict[str, Any]) -> None:
    for adapter in models.values():
        if isinstance(adapter, RemoteAdapter):
            adapter.close()
