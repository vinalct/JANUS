"""Worker result transport and crash evidence, without source execution."""

from __future__ import annotations

import json
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from janus.orchestration.results import SourceAttempt
from janus.orchestration.timing import ExecutionTiming

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from janus.runtime.parallel import WorkerTask

MAX_ENVELOPE_BYTES = 16 * 1024 * 1024
WORKER_CRASH_PHASE = "worker_crash"


@dataclass(slots=True)
class _WorkerHandle:
    request: WorkerTask
    process: BaseProcess
    reader: Connection
    envelope: dict[str, Any] | None = None
    transport_error: str | None = None
    result_received: bool = False

    @property
    def sentinel(self) -> int:
        return self.process.sentinel

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()

    def join(self, *, timeout: float | None = None) -> None:
        self.process.join(timeout)

    def close(self) -> None:
        self.reader.close()
        self.process.close()


def _read_envelope(handle: _WorkerHandle) -> None:
    """Read before joining so pipe-buffer-sized results cannot block the sender."""
    try:
        message = handle.reader.recv_bytes(MAX_ENVELOPE_BYTES)
        envelope = json.loads(message)
        if not isinstance(envelope, dict) or set(envelope) != {"attempt", "worker"}:
            raise ValueError("invalid worker envelope")
        worker = envelope["worker"]
        if not isinstance(worker, dict) or set(worker) != {
            "pid",
            "spark_started",
            "spark_application_id",
        }:
            raise ValueError("invalid worker diagnostics")
        attempt = SourceAttempt.from_summary(envelope["attempt"])
        task = handle.request
        if (attempt.source_id, attempt.run_id, attempt.attempt) != (
            task.source_id,
            task.run_id,
            task.attempt,
        ):
            raise ValueError("worker attempt identity differs from dispatched task")
        handle.envelope = envelope
    except EOFError:
        pass
    except Exception as error:
        handle.transport_error = (
            f"Worker envelope refused (cap {MAX_ENVELOPE_BYTES} bytes / 16 MiB): "
            f"{type(error).__name__}: {error}"
        )
    finally:
        handle.result_received = True
        handle.reader.close()


class WorkerCrashed(RuntimeError):
    """A spawned attempt exited without a valid result envelope."""


def _describe_exit(exitcode: int | None) -> str:
    if exitcode == 0:
        return "worker exited cleanly without sending a result"
    if exitcode is not None and exitcode < 0:
        try:
            name = signal.Signals(-exitcode).name
        except ValueError:
            name = f"signal {-exitcode}"
        return f"worker exited with code {exitcode} ({name}) and sent no result"
    return f"worker exited with code {exitcode} and sent no result"


def _attempt_from_worker(handle: _WorkerHandle, timing: ExecutionTiming) -> SourceAttempt:
    if handle.envelope is not None:
        return SourceAttempt.from_summary(handle.envelope["attempt"])
    task = handle.request
    reason = f"Worker for source {task.source_id!r}: {_describe_exit(handle.exitcode)}"
    if handle.transport_error:
        reason += f"; {handle.transport_error}"
    return SourceAttempt.failed_from_exception(
        source_id=task.source_id,
        run_id=task.run_id,
        attempt=task.attempt,
        timing=timing,
        error=WorkerCrashed(reason),
        phase=WORKER_CRASH_PHASE,
    )
