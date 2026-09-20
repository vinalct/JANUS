"""Guarded lifecycle fan-out for additive, queryable run observability.

The total terminal-emission budget is five seconds by default. A daemon worker bounds the
entire operation rather than only the catalog append, so every fan-out destination shares one
deadline: each sink receives only what is left of it, and the join is what actually bounds a
run's exposure whatever a transport was configured to wait for.
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

from janus.lineage.models import LineageRecord, RunMetadata
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
from janus.observability.openlineage.sink import (
    OpenLineageRunSink,
    build_openlineage_sink,
    disabled_openlineage_sink,
)
from janus.observability.openlineage.transport import (
    NOT_CONFIGURED_REASON,
    OpenLineageEmissionOutcome,
    OpenLineageEmissionResult,
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


_APPEND_OUTCOMES = {
    IcebergAppendOutcome.EMITTED: RunEmissionOutcome.EMITTED,
    IcebergAppendOutcome.SKIPPED: RunEmissionOutcome.SKIPPED,
    IcebergAppendOutcome.FAILED: RunEmissionOutcome.FAILED,
}

_QUIET_REASONS = frozenset({NOT_CONFIGURED_REASON})
_OPENLINEAGE_OUTCOMES = {
    OpenLineageEmissionOutcome.EMITTED: RunEmissionOutcome.EMITTED,
    OpenLineageEmissionOutcome.SKIPPED: RunEmissionOutcome.SKIPPED,
    OpenLineageEmissionOutcome.FAILED: RunEmissionOutcome.FAILED,
}


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
    openlineage: OpenLineageEmissionResult | None = None

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
        if self.openlineage is not None:
            summary["openlineage"] = self.openlineage.to_summary()
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
    openlineage_sink: OpenLineageRunSink = field(default_factory=disabled_openlineage_sink)
    last_result: RunEmissionResult | None = field(default=None, init=False)
    last_started_result: RunEmissionResult | None = field(default=None, init=False)

    def emit_started(self, plan: ExecutionPlan, persisted: PersistedArtifacts) -> None:
        """Emit START through the configured transport; the runs table stays terminal-only."""
        started_at = time.monotonic()
        table_identifier = _safe_table_identifier(self.config)
        result = self._bounded(
            "started",
            table_identifier,
            started_at,
            lambda results: self._run_started_worker(
                results, persisted, started_at, table_identifier
            ),
        )
        completed = replace(result, duration_seconds=_elapsed(started_at))
        self.last_started_result = completed
        _report_lifecycle(self.logger, completed, run_id=plan.run_context.run_id)

    def emit_succeeded(self, plan: ExecutionPlan, persisted: PersistedArtifacts) -> None:
        del plan
        self._emit_terminal("succeeded", persisted)

    def emit_failed(self, plan: ExecutionPlan, persisted: PersistedArtifacts) -> None:
        del plan
        self._emit_terminal("failed", persisted)

    def _emit_terminal(self, lifecycle: str, persisted: PersistedArtifacts) -> None:
        started_at = time.monotonic()
        table_identifier = _safe_table_identifier(self.config)
        result = self._bounded(
            lifecycle,
            table_identifier,
            started_at,
            lambda results: self._run_terminal_worker(
                results, lifecycle, persisted, started_at, table_identifier
            ),
        )
        self._finish(result, started_at)

    def _bounded(
        self,
        lifecycle: str,
        table_identifier: str,
        started_at: float,
        worker: Callable[[list[RunEmissionResult]], None],
    ) -> RunEmissionResult:
        """Run one fan-out inside the total budget, whatever its destinations cost."""
        if not _valid_timeout(self.timeout_seconds):
            return _failed(
                lifecycle,
                table_identifier,
                stage="budget",
                exception_type="InvalidEmissionTimeout",
            )

        results: list[RunEmissionResult] = []
        fatal_errors: list[BaseException] = []

        def bridge() -> None:
            # Anything the per-stage guards did not already turn into data crosses the
            # thread boundary: a KeyboardInterrupt during emission must still interrupt.
            try:
                worker(results)
            except BaseException as exc:
                fatal_errors.append(exc)

        try:
            thread = threading.Thread(
                target=bridge,
                name="janus-run-event-emission",
                daemon=True,
            )
            thread.start()
            thread.join(self.timeout_seconds)
        except Exception as exc:
            return _failed(
                lifecycle,
                table_identifier,
                stage="worker",
                exception_type=type(exc).__name__,
            )

        if thread.is_alive():
            return _failed(
                lifecycle,
                table_identifier,
                stage="budget",
                exception_type="EmissionTimeoutError",
            )
        if fatal_errors:
            raise fatal_errors[0]
        if results:
            return results[0]
        return _failed(
            lifecycle,
            table_identifier,
            stage="worker",
            exception_type="WorkerExitedWithoutResult",
        )

    def _run_started_worker(
        self,
        results: list[RunEmissionResult],
        persisted: PersistedArtifacts,
        started_at: float,
        table_identifier: str,
    ) -> None:
        openlineage = self._emit_openlineage(persisted.run_metadata, started_at)
        results.append(
            RunEmissionResult(
                lifecycle="started",
                outcome=_OPENLINEAGE_OUTCOMES[openlineage.outcome],
                table_identifier=table_identifier,
                reason=openlineage.reason,
                stage=openlineage.step,
                exception_type=openlineage.exception_type,
                openlineage=openlineage,
            )
        )

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

        # One projection, both destinations: the event and the row cannot disagree.
        openlineage = self._emit_openlineage(
            persisted.run_metadata,
            started_at,
            lineage_record=persisted.lineage_record,
            run_record=record,
        )

        remaining = self.timeout_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            results.append(
                _failed(
                    lifecycle,
                    table_identifier,
                    stage="budget",
                    exception_type="EmissionTimeoutError",
                    openlineage=openlineage,
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
            results.append(_from_sink_result(lifecycle, sink_result, openlineage))
        except Exception as exc:
            results.append(
                _failed(
                    lifecycle,
                    table_identifier,
                    stage="runs_table",
                    exception_type=type(exc).__name__,
                    openlineage=openlineage,
                )
            )

    def _emit_openlineage(
        self,
        run_metadata: RunMetadata,
        started_at: float,
        *,
        lineage_record: LineageRecord | None = None,
        run_record: RunRecord | None = None,
    ) -> OpenLineageEmissionResult:
        """Deliver one event with what is left of the budget; a defective sink is data too."""
        remaining = self.timeout_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            return _openlineage_failure(
                self.openlineage_sink,
                step="budget",
                exception_type="EmissionTimeoutError",
            )
        try:
            return self.openlineage_sink.emit(
                run_metadata,
                lineage_record=lineage_record,
                run_record=run_record,
                budget_seconds=remaining,
                logger=self.logger,
            )
        except Exception as exc:
            return _openlineage_failure(
                self.openlineage_sink,
                step="openlineage",
                exception_type=type(exc).__name__,
            )

    def _finish(self, result: RunEmissionResult, started_at: float) -> None:
        completed = replace(result, duration_seconds=_elapsed(started_at))
        self.last_result = completed
        _report_lifecycle(self.logger, completed)


def build_run_event_emitter(
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    logger: StructuredLogger | _EmissionLogger | None = None,
    timeout_seconds: float = DEFAULT_EMISSION_TIMEOUT_SECONDS,
    projector: Callable[[PersistedArtifacts], RunRecord] | None = None,
    runs_table_sink: _RunRecordAppender | None = None,
    openlineage_sink: OpenLineageRunSink | None = None,
) -> GuardedRunEventEmitter:
    """Build one isolated emitter from one execution's resolved environment profile.

    The transport is selected here, once per run, so a profile error is read and logged once
    rather than re-read on every lifecycle hook.
    """
    return GuardedRunEventEmitter(
        config=dict(config),
        resolved_paths=dict(resolved_paths),
        logger=logger,
        timeout_seconds=timeout_seconds,
        projector=projector if projector is not None else _project_run_record,
        runs_table_sink=(runs_table_sink if runs_table_sink is not None else append_run_record),
        openlineage_sink=(
            openlineage_sink
            if openlineage_sink is not None
            else build_openlineage_sink(config, resolved_paths, logger=logger)
        ),
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
    openlineage: OpenLineageEmissionResult | None = None,
) -> RunEmissionResult:
    return RunEmissionResult(
        lifecycle=lifecycle,
        outcome=_APPEND_OUTCOMES[result.outcome],
        table_identifier=result.table_identifier,
        reason=result.reason,
        stage=result.step,
        exception_type=result.exception_type,
        openlineage=openlineage,
    )


def _failed(
    lifecycle: str,
    table_identifier: str,
    *,
    stage: str,
    exception_type: str,
    openlineage: OpenLineageEmissionResult | None = None,
) -> RunEmissionResult:
    return RunEmissionResult(
        lifecycle=lifecycle,
        outcome=RunEmissionOutcome.FAILED,
        table_identifier=table_identifier,
        reason=f"{stage}_failed",
        stage=stage,
        exception_type=exception_type,
        openlineage=openlineage,
    )


def _openlineage_failure(
    sink: OpenLineageRunSink,
    *,
    step: str,
    exception_type: str,
) -> OpenLineageEmissionResult:
    return OpenLineageEmissionResult(
        outcome=OpenLineageEmissionOutcome.FAILED,
        transport=sink.transport.kind,
        reason=f"{step}_failed",
        step=step,
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


def _elapsed(started_at: float) -> float:
    return round(time.monotonic() - started_at, 6)


def _report_lifecycle(
    logger: StructuredLogger | _EmissionLogger | None,
    result: RunEmissionResult,
    *,
    run_id: str | None = None,
) -> None:
    """One event per lifecycle hook, at info when it landed and warning when it did not."""
    fields = result.to_summary()
    if run_id is not None:
        fields["run_id"] = run_id
    if result.lifecycle == "started":
        fields["runs_table"] = "terminal_only"
    level = "info" if result.emitted or result.reason in _QUIET_REASONS else "warning"
    _log(logger, level, "run_event_emission_finished", fields)


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
