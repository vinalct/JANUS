"""Guarded lifecycle fan-out for additive, queryable run observability.

The total terminal-emission budget is five seconds by default. A daemon worker bounds the
entire operation rather than only the catalog append, so projection and future fan-out
destinations share one deadline. The runs-table sink receives only the remaining budget.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from janus.lineage.store import (
    NullRunEventEmitter,
    PersistedArtifacts,
    RunObserver,
)
from janus.models import ExecutionPlan
from janus.observability.iceberg_sink import (
    IcebergAppendOutcome,
    IcebergAppendResult,
    append_run_record,
)
from janus.observability.records import RunEvidencePaths, RunRecord
from janus.observability.runs_table import (
    DEFAULT_RUNS_TABLE_IDENTIFIER,
    resolve_runs_table_identifier,
)
from janus.utils.environment import RuntimeLocation
from janus.utils.logging import StructuredLogger

DEFAULT_EMISSION_TIMEOUT_SECONDS = 5.0
_LOG = logging.getLogger(__name__)


class RunEmissionOutcome(StrEnum):
    """Observable terminal outcomes without consulting the catalog."""

    EMITTED = "emitted"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RunEmissionResult:
    """One guarded terminal fan-out result suitable for logs and CLI summaries."""

    lifecycle: str
    outcome: RunEmissionOutcome
    table_identifier: str
    reason: str | None = None
    stage: str | None = None
    exception_type: str | None = None
    duration_seconds: float = 0.0

    @property
    def emitted(self) -> bool:
        return self.outcome is RunEmissionOutcome.EMITTED

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "lifecycle": self.lifecycle,
            "outcome": self.outcome.value,
            "table_identifier": self.table_identifier,
            "duration_seconds": self.duration_seconds,
        }
        if self.reason is not None:
            summary["reason"] = self.reason
        if self.stage is not None:
            summary["stage"] = self.stage
        if self.exception_type is not None:
            summary["exception_type"] = self.exception_type
        return summary


class _EmissionLogger(Protocol):
    def info(self, event: str, **fields: Any) -> None: ...

    def warning(self, event: str, **fields: Any) -> None: ...


class _RunRecordAppender(Protocol):
    def __call__(
        self,
        record: RunRecord,
        config: Mapping[str, Any],
        resolved_paths: Mapping[str, RuntimeLocation],
        *,
        logger: StructuredLogger | _EmissionLogger | None,
        timeout_seconds: float,
    ) -> IcebergAppendResult: ...


@dataclass(slots=True)
class GuardedRunEventEmitter:
    """Project once and append under one deadline; every ordinary failure is data."""

    config: Mapping[str, Any]
    resolved_paths: Mapping[str, RuntimeLocation]
    logger: StructuredLogger | _EmissionLogger | None = None
    timeout_seconds: float = DEFAULT_EMISSION_TIMEOUT_SECONDS
    projector: Callable[[PersistedArtifacts], RunRecord] = field(
        default=lambda persisted: _project_run_record(persisted)
    )
    runs_table_sink: _RunRecordAppender = append_run_record
    last_result: RunEmissionResult | None = field(default=None, init=False)

    def emit_started(self, plan: ExecutionPlan, persisted: PersistedArtifacts) -> None:
        del persisted
        _report_started(
            self.logger,
            run_id=plan.run_context.run_id,
            table_identifier=_safe_table_identifier(self.config),
        )

    def emit_succeeded(self, plan: ExecutionPlan, persisted: PersistedArtifacts) -> None:
        del plan
        self._emit_terminal("succeeded", persisted)

    def emit_failed(self, plan: ExecutionPlan, persisted: PersistedArtifacts) -> None:
        del plan
        self._emit_terminal("failed", persisted)

    def _emit_terminal(self, lifecycle: str, persisted: PersistedArtifacts) -> None:
        started_at = time.monotonic()
        table_identifier = _safe_table_identifier(self.config)
        if not _valid_timeout(self.timeout_seconds):
            result = _failed(
                lifecycle,
                table_identifier,
                stage="budget",
                exception_type="InvalidEmissionTimeout",
            )
            self._finish(result, started_at)
            return

        results: list[RunEmissionResult] = []
        fatal_errors: list[BaseException] = []
        try:
            worker = threading.Thread(
                target=self._run_terminal_worker_with_fatal_bridge,
                args=(
                    fatal_errors,
                    results,
                    lifecycle,
                    persisted,
                    started_at,
                    table_identifier,
                ),
                name="janus-run-event-emission",
                daemon=True,
            )
            worker.start()
            worker.join(self.timeout_seconds)
        except Exception as exc:
            result = _failed(
                lifecycle,
                table_identifier,
                stage="worker",
                exception_type=type(exc).__name__,
            )
            self._finish(result, started_at)
            return

        if worker.is_alive():
            result = _failed(
                lifecycle,
                table_identifier,
                stage="budget",
                exception_type="EmissionTimeoutError",
            )
        elif fatal_errors:
            raise fatal_errors[0]
        elif results:
            result = results[0]
        else:
            result = _failed(
                lifecycle,
                table_identifier,
                stage="worker",
                exception_type="WorkerExitedWithoutResult",
            )
        self._finish(result, started_at)

    def _run_terminal_worker_with_fatal_bridge(
        self,
        fatal_errors: list[BaseException],
        results: list[RunEmissionResult],
        lifecycle: str,
        persisted: PersistedArtifacts,
        started_at: float,
        table_identifier: str,
    ) -> None:
        try:
            self._run_terminal_worker(results, lifecycle, persisted, started_at, table_identifier)
        except BaseException as exc:
            fatal_errors.append(exc)

    def _run_terminal_worker(
        self,
        results: list[RunEmissionResult],
        lifecycle: str,
        persisted: PersistedArtifacts,
        started_at: float,
        table_identifier: str,
    ) -> None:
        try:
            record = self.projector(persisted)
        except Exception as exc:
            results.append(
                _failed(
                    lifecycle,
                    table_identifier,
                    stage="projection",
                    exception_type=type(exc).__name__,
                )
            )
            return

        remaining = self.timeout_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            results.append(
                _failed(
                    lifecycle,
                    table_identifier,
                    stage="budget",
                    exception_type="EmissionTimeoutError",
                )
            )
            return

        try:
            sink_result = self.runs_table_sink(
                record,
                self.config,
                self.resolved_paths,
                logger=self.logger,
                timeout_seconds=remaining,
            )
            results.append(_from_sink_result(lifecycle, sink_result))
        except Exception as exc:
            results.append(
                _failed(
                    lifecycle,
                    table_identifier,
                    stage="runs_table",
                    exception_type=type(exc).__name__,
                )
            )

    def _finish(self, result: RunEmissionResult, started_at: float) -> None:
        elapsed = round(time.monotonic() - started_at, 6)
        completed = replace(result, duration_seconds=elapsed)
        self.last_result = completed
        _report_terminal(self.logger, completed)


def build_run_event_emitter(
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    logger: StructuredLogger | _EmissionLogger | None = None,
    timeout_seconds: float = DEFAULT_EMISSION_TIMEOUT_SECONDS,
    projector: Callable[[PersistedArtifacts], RunRecord] | None = None,
    runs_table_sink: _RunRecordAppender | None = None,
) -> GuardedRunEventEmitter:
    """Build one isolated emitter from one execution's resolved environment profile."""
    return GuardedRunEventEmitter(
        config=dict(config),
        resolved_paths=dict(resolved_paths),
        logger=logger,
        timeout_seconds=timeout_seconds,
        projector=projector if projector is not None else _project_run_record,
        runs_table_sink=(runs_table_sink if runs_table_sink is not None else append_run_record),
    )


def wire_run_event_emitter(
    observer: RunObserver,
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    logger: StructuredLogger | None,
) -> RunObserver:
    """Inject production emission without mutating or overriding explicit collaborators."""
    if type(observer) is not RunObserver or not isinstance(observer.emitter, NullRunEventEmitter):
        return observer
    return replace(
        observer,
        emitter=build_run_event_emitter(config, resolved_paths, logger=logger),
    )


def latest_run_emission(observer: object) -> RunEmissionResult | None:
    """Read a concrete emitter result when something terminal was attempted."""
    emitter = getattr(observer, "emitter", None)
    result = getattr(emitter, "last_result", None)
    return result if isinstance(result, RunEmissionResult) else None


def _project_run_record(persisted: PersistedArtifacts) -> RunRecord:
    lineage_record = persisted.lineage_record
    if lineage_record is None:
        raise ValueError("terminal emission requires a lineage record")
    validation_report = (
        persisted.validation_report.report if persisted.validation_report is not None else None
    )
    return RunRecord.from_run(
        persisted.run_metadata,
        lineage_record,
        emitted_at=lineage_record.emitted_at,
        checkpoint_result=persisted.checkpoint_result,
        validation_report=validation_report,
        evidence=RunEvidencePaths(
            run_metadata_path=persisted.run_metadata_path,
            lineage_path=persisted.lineage_path,
            validation_report_path=(
                persisted.validation_report.path
                if persisted.validation_report is not None
                else None
            ),
        ),
    )


def _from_sink_result(
    lifecycle: str,
    result: IcebergAppendResult,
) -> RunEmissionResult:
    outcomes = {
        IcebergAppendOutcome.EMITTED: RunEmissionOutcome.EMITTED,
        IcebergAppendOutcome.SKIPPED: RunEmissionOutcome.SKIPPED,
        IcebergAppendOutcome.FAILED: RunEmissionOutcome.FAILED,
    }
    return RunEmissionResult(
        lifecycle=lifecycle,
        outcome=outcomes[result.outcome],
        table_identifier=result.table_identifier,
        reason=result.reason,
        stage=result.step,
        exception_type=result.exception_type,
    )


def _failed(
    lifecycle: str,
    table_identifier: str,
    *,
    stage: str,
    exception_type: str,
) -> RunEmissionResult:
    return RunEmissionResult(
        lifecycle=lifecycle,
        outcome=RunEmissionOutcome.FAILED,
        table_identifier=table_identifier,
        reason=f"{stage}_failed",
        stage=stage,
        exception_type=exception_type,
    )


def _safe_table_identifier(config: Mapping[str, Any]) -> str:
    try:
        return resolve_runs_table_identifier(config)
    except Exception:
        return DEFAULT_RUNS_TABLE_IDENTIFIER


def _valid_timeout(value: float) -> bool:
    try:
        return math.isfinite(value) and value > 0
    except Exception:
        return False


def _report_started(
    logger: StructuredLogger | _EmissionLogger | None,
    *,
    run_id: str,
    table_identifier: str,
) -> None:
    fields = {
        "lifecycle": "started",
        "outcome": "terminal_only",
        "run_id": run_id,
        "table_identifier": table_identifier,
    }
    _log(logger, "info", "run_event_emission_finished", fields)


def _report_terminal(
    logger: StructuredLogger | _EmissionLogger | None,
    result: RunEmissionResult,
) -> None:
    level = "info" if result.emitted else "warning"
    _log(logger, level, "run_event_emission_finished", result.to_summary())


def _log(
    logger: StructuredLogger | _EmissionLogger | None,
    level: str,
    event: str,
    fields: Mapping[str, Any],
) -> None:
    try:
        if logger is None:
            log_level = logging.INFO if level == "info" else logging.WARNING
            _LOG.log(log_level, event, extra={"event_fields": dict(fields)})
        else:
            getattr(logger, level)(event, **fields)
    except Exception:
        pass
