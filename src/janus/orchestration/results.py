"""Frozen, versioned outcomes shared by the batch runner and adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Self

from janus.models.dependencies import SourceDependencyEdge
from janus.orchestration.plans import BatchPlan, PlannedSource
from janus.orchestration.result_validation import (
    _canonical_ids,
    _require_sorted_unique_ids,
    _validate_attempt_history,
    _validate_aware,
    _validate_pipeline_membership,
    _validate_source_identity,
)
from janus.orchestration.summary_values import (
    FailureDetails,
    _require_keys,
    _require_string,
    freeze_evidence,
    thaw_json,
)
from janus.orchestration.timing import ExecutionTiming

PIPELINE_SUMMARY_SCHEMA_VERSION = 1
SOURCE_TERMINAL_STATUSES = frozenset({"failed", "skipped", "succeeded"})
SOURCE_ATTEMPT_STATUSES = frozenset({"failed", "succeeded"})
SUMMARY_PERSISTENCE_STATUSES = frozenset({"failed", "pending", "succeeded"})
UPSTREAM_FAILED_REASON_CODE = "upstream_failed"


class ExecutedRunEvidence(Protocol):
    """The small source-result surface needed to build an attempt record."""

    @property
    def status(self) -> str: ...

    @property
    def failure_reason(self) -> str | None: ...

    @property
    def error_type(self) -> str | None: ...

    @property
    def is_successful(self) -> bool: ...

    def to_summary(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SkipExplanation:
    """Why a source did not execute, including direct and transitive causes."""

    direct_blocking_upstream_ids: tuple[str, ...]
    root_failed_source_ids: tuple[str, ...]
    reason_code: str = UPSTREAM_FAILED_REASON_CODE

    def __post_init__(self) -> None:
        if not self.reason_code.strip():
            raise ValueError("reason_code must not be empty")
        _require_sorted_unique_ids(
            "direct_blocking_upstream_ids", self.direct_blocking_upstream_ids
        )
        _require_sorted_unique_ids("root_failed_source_ids", self.root_failed_source_ids)
        if not self.direct_blocking_upstream_ids:
            raise ValueError("a skipped source must name at least one direct blocker")
        if not self.root_failed_source_ids:
            raise ValueError("a skipped source must name at least one root failure")

    @classmethod
    def create(
        cls,
        *,
        direct_blocking_upstream_ids: Sequence[str],
        root_failed_source_ids: Sequence[str],
        reason_code: str = UPSTREAM_FAILED_REASON_CODE,
    ) -> Self:
        return cls(
            direct_blocking_upstream_ids=_canonical_ids(direct_blocking_upstream_ids),
            root_failed_source_ids=_canonical_ids(root_failed_source_ids),
            reason_code=reason_code.strip(),
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "direct_blocking_upstream_ids": list(self.direct_blocking_upstream_ids),
            "root_failed_source_ids": list(self.root_failed_source_ids),
        }


@dataclass(frozen=True, slots=True)
class SourceAttempt:
    """One actual source execution attempt and the evidence it produced."""

    source_id: str
    run_id: str
    attempt: int
    status: str
    timing: ExecutionTiming
    evidence: Mapping[str, Any] = field(default_factory=dict)
    failure: FailureDetails | None = None

    def __post_init__(self) -> None:
        _validate_source_identity(self.source_id, self.run_id, self.attempt)
        if self.status not in SOURCE_ATTEMPT_STATUSES:
            allowed = ", ".join(sorted(SOURCE_ATTEMPT_STATUSES))
            raise ValueError(f"attempt status must be one of: {allowed}")
        if not self.timing.executed:
            raise ValueError("a source attempt must carry actual execution timestamps")
        if (self.status == "failed") != (self.failure is not None):
            raise ValueError("a failed attempt must carry failure details, and a success must not")
        object.__setattr__(self, "evidence", freeze_evidence(self.evidence))

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any]) -> Self:
        """Rebuild an attempt from its JSON wire form."""
        _require_keys(
            summary,
            {"source_id", "run_id", "attempt", "attempted", "status", "timing", "evidence"},
            optional=frozenset({"failure"}),
        )
        if summary["attempted"] is not True:
            raise ValueError("an attempt summary must carry attempted=true")
        for name in ("source_id", "run_id", "status"):
            _require_string(summary[name], name)
        if isinstance(summary["attempt"], bool) or not isinstance(summary["attempt"], int):
            raise ValueError("attempt must be an integer")
        if not isinstance(summary["timing"], Mapping):
            raise ValueError("timing must be a JSON object")
        timing = ExecutionTiming.from_summary(summary["timing"])
        if not timing.executed:
            raise ValueError("timing must carry actual execution timestamps")
        if not isinstance(summary["evidence"], Mapping):
            raise ValueError("evidence must be a JSON object")
        failure = summary.get("failure")
        if failure is not None and not isinstance(failure, Mapping):
            raise ValueError("failure must be a JSON object or null")
        try:
            return cls(
                source_id=summary["source_id"],
                run_id=summary["run_id"],
                attempt=summary["attempt"],
                status=summary["status"],
                timing=timing,
                evidence=summary["evidence"],
                failure=FailureDetails.from_summary(failure) if failure is not None else None,
            )
        except TypeError as exc:
            raise ValueError(f"evidence is invalid: {exc}") from exc

    @classmethod
    def from_executed_run(
        cls,
        executed_run: ExecutedRunEvidence,
        *,
        timing: ExecutionTiming,
        source_id: str | None = None,
        run_id: str | None = None,
        attempt: int | None = None,
        failure_phase: str = "execution",
    ) -> Self:
        """Translate the established source result without copying its evidence schema."""
        if source_id is None or run_id is None or attempt is None:
            derived = _executed_run_identity(executed_run)
            source_id = source_id or derived[0]
            run_id = run_id or derived[1]
            attempt = attempt if attempt is not None else derived[2]
        assert source_id is not None
        assert run_id is not None
        assert attempt is not None
        status = "succeeded" if executed_run.is_successful else "failed"
        failure = None
        if status == "failed":
            failure = FailureDetails(
                phase=failure_phase,
                error_type=executed_run.error_type or "ExecutionFailure",
                reason=executed_run.failure_reason or "Source execution returned a failed result",
            )
        return cls(
            source_id=source_id,
            run_id=run_id,
            attempt=attempt,
            status=status,
            timing=timing,
            evidence=executed_run.to_summary(),
            failure=failure,
        )

    @classmethod
    def failed_from_exception(
        cls,
        *,
        source_id: str,
        run_id: str,
        attempt: int,
        timing: ExecutionTiming,
        error: Exception,
        phase: str = "execution",
    ) -> Self:
        return cls(
            source_id=source_id,
            run_id=run_id,
            attempt=attempt,
            status="failed",
            timing=timing,
            failure=FailureDetails.from_exception(error, phase=phase),
        )

    def as_budget_breach(self, budget_seconds: int) -> Self:
        """Fail the batch's view of a succeeded run for its duration budget.

        Bronze, quality and checkpoint evidence stay intact. The run succeeded;
        only the batch attempt fails, at the operator's explicit request.
        """
        failure = FailureDetails(
            phase="budget",
            error_type="RunDurationBudgetExceeded",
            reason=f"Run exceeded its declared duration budget of {budget_seconds}s",
        )
        return replace(self, status="failed", failure=failure)

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "source_id": self.source_id,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "attempted": True,
            "status": self.status,
            "timing": self.timing.to_summary(),
            "evidence": thaw_json(self.evidence),
        }
        if self.failure is not None:
            summary["failure"] = self.failure.to_summary()
        return summary


@dataclass(frozen=True, slots=True)
class SourceOutcome:
    """The final status of one planned source, separate from its attempt history."""

    source_id: str
    run_id: str
    attempt: int
    selected_directly: bool
    upstream_ids: tuple[str, ...]
    config_version: str
    status: str
    attempts: tuple[SourceAttempt, ...] = ()
    failure: FailureDetails | None = None
    skip: SkipExplanation | None = None

    def __post_init__(self) -> None:
        _validate_source_identity(self.source_id, self.run_id, self.attempt)
        if not self.config_version.strip():
            raise ValueError("config_version must not be empty")
        _require_sorted_unique_ids("upstream_ids", self.upstream_ids)
        if self.status not in SOURCE_TERMINAL_STATUSES:
            allowed = ", ".join(sorted(SOURCE_TERMINAL_STATUSES))
            raise ValueError(f"source status must be one of: {allowed}")
        _validate_attempt_history(self)

    @classmethod
    def from_attempts(
        cls,
        planned_source: PlannedSource,
        attempts: Sequence[SourceAttempt],
    ) -> Self:
        if not planned_source.is_planned:
            raise ValueError(f"Source {planned_source.source_id!r} has no executable plan")
        history = tuple(attempts)
        if not history:
            raise ValueError("an attempted source must carry at least one attempt")
        final = history[-1]
        return cls(
            source_id=planned_source.source_id,
            run_id=final.run_id,
            attempt=final.attempt,
            selected_directly=planned_source.selected_directly,
            upstream_ids=planned_source.upstream_ids,
            config_version=planned_source.config_version,
            status=final.status,
            attempts=history,
            failure=final.failure,
        )

    @classmethod
    def from_planning_failure(cls, planned_source: PlannedSource) -> Self:
        if planned_source.failure is None:
            raise ValueError(f"Source {planned_source.source_id!r} has no planning failure")
        return cls(
            source_id=planned_source.source_id,
            run_id=planned_source.run_id,
            attempt=planned_source.attempt,
            selected_directly=planned_source.selected_directly,
            upstream_ids=planned_source.upstream_ids,
            config_version=planned_source.config_version,
            status="failed",
            failure=FailureDetails(
                phase=planned_source.failure.phase,
                error_type=planned_source.failure.error_type,
                reason=planned_source.failure.reason,
            ),
        )

    @classmethod
    def skipped(
        cls,
        planned_source: PlannedSource,
        *,
        direct_blocking_upstream_ids: Sequence[str],
        root_failed_source_ids: Sequence[str],
        reason_code: str = UPSTREAM_FAILED_REASON_CODE,
    ) -> Self:
        return cls(
            source_id=planned_source.source_id,
            run_id=planned_source.run_id,
            attempt=planned_source.attempt,
            selected_directly=planned_source.selected_directly,
            upstream_ids=planned_source.upstream_ids,
            config_version=planned_source.config_version,
            status="skipped",
            skip=SkipExplanation.create(
                direct_blocking_upstream_ids=direct_blocking_upstream_ids,
                root_failed_source_ids=root_failed_source_ids,
                reason_code=reason_code,
            ),
        )

    @property
    def attempted(self) -> bool:
        return bool(self.attempts)

    @property
    def timing(self) -> ExecutionTiming:
        if not self.attempts:
            return ExecutionTiming.not_executed()
        first = self.attempts[0].timing
        last = self.attempts[-1].timing
        return ExecutionTiming(
            started_at=first.started_at,
            ended_at=last.ended_at,
            duration_seconds=round(
                sum(attempt.timing.duration_seconds for attempt in self.attempts), 6
            ),
        )

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "source_id": self.source_id,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "attempted": self.attempted,
            "selected_directly": self.selected_directly,
            "upstream_ids": list(self.upstream_ids),
            "config_version": self.config_version,
            "status": self.status,
            "timing": self.timing.to_summary(),
            "attempts": [attempt.to_summary() for attempt in self.attempts],
        }
        if self.failure is not None:
            summary["failure"] = self.failure.to_summary()
        if self.skip is not None:
            summary["skip"] = self.skip.to_summary()
        return summary


@dataclass(frozen=True, slots=True)
class SummaryPersistence:
    """Whether and where the aggregate summary reached durable storage."""

    status: str = "pending"
    path: Path | None = None
    failure: FailureDetails | None = None

    def __post_init__(self) -> None:
        if self.status not in SUMMARY_PERSISTENCE_STATUSES:
            allowed = ", ".join(sorted(SUMMARY_PERSISTENCE_STATUSES))
            raise ValueError(f"summary persistence status must be one of: {allowed}")
        if self.status == "succeeded" and self.path is None:
            raise ValueError("successful summary persistence must carry its path")
        if (self.status == "failed") != (self.failure is not None):
            raise ValueError("failed summary persistence must carry failure details")

    @classmethod
    def succeeded(cls, path: Path) -> Self:
        return cls(status="succeeded", path=path)

    @classmethod
    def failed(cls, path: Path, failure: FailureDetails) -> Self:
        return cls(status="failed", path=path, failure=failure)

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "status": self.status,
            "path": str(self.path) if self.path is not None else None,
        }
        if self.failure is not None:
            summary["failure"] = self.failure.to_summary()
        return summary


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """One aggregate pipeline result, ordered exactly like its validated batch plan."""

    pipeline_run_id: str
    attempt: int
    trigger: str
    environment: str
    planned_at: datetime
    requested_tags: tuple[str, ...]
    requested_domains: tuple[str, ...]
    root_ids: tuple[str, ...]
    included_upstream_ids: tuple[str, ...]
    source_order: tuple[str, ...]
    edges: tuple[SourceDependencyEdge, ...]
    config_versions: tuple[tuple[str, str], ...]
    timing: ExecutionTiming
    sources: tuple[SourceOutcome, ...]
    summary_persistence: SummaryPersistence = field(default_factory=SummaryPersistence)
    schema_version: int = PIPELINE_SUMMARY_SCHEMA_VERSION
    max_parallel: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != PIPELINE_SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PIPELINE_SUMMARY_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        if not self.pipeline_run_id.strip():
            raise ValueError("pipeline_run_id must not be empty")
        if self.attempt < 1:
            raise ValueError("attempt must be 1 or greater")
        if not self.trigger.strip() or not self.environment.strip():
            raise ValueError("trigger and environment must not be empty")
        _validate_aware("planned_at", self.planned_at)
        if not self.timing.executed:
            raise ValueError("a completed pipeline outcome must carry actual timestamps")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be 1 or greater")
        _validate_pipeline_membership(self)

    @classmethod
    def from_plan(
        cls,
        plan: BatchPlan,
        *,
        timing: ExecutionTiming,
        sources: Sequence[SourceOutcome],
        summary_persistence: SummaryPersistence | None = None,
        max_parallel: int = 1,
    ) -> Self:
        return cls(
            pipeline_run_id=plan.request.pipeline_run_id,
            attempt=plan.request.attempt,
            trigger=plan.request.trigger,
            environment=plan.request.environment,
            planned_at=plan.request.planned_at,
            requested_tags=plan.request.selection.tags,
            requested_domains=plan.request.selection.domains,
            root_ids=plan.root_ids,
            included_upstream_ids=plan.included_upstream_ids,
            source_order=plan.source_ids,
            edges=plan.edges,
            config_versions=tuple(plan.config_versions().items()),
            timing=timing,
            sources=tuple(sources),
            summary_persistence=summary_persistence or SummaryPersistence(),
            max_parallel=max_parallel,
        )

    @property
    def status(self) -> str:
        if any(source.status != "succeeded" for source in self.sources):
            return "failed"
        if self.summary_persistence.status == "succeeded":
            return "succeeded"
        if self.summary_persistence.status == "failed":
            return "failed"
        return "pending"

    @property
    def is_successful(self) -> bool:
        return self.status == "succeeded"

    def with_summary_persistence(self, persistence: SummaryPersistence) -> Self:
        return replace(self, summary_persistence=persistence)

    def totals(self) -> dict[str, Any]:
        statuses = [source.status for source in self.sources]
        return {
            "selected": len(self.root_ids),
            "expanded": len(self.source_order),
            "attempted": sum(source.attempted for source in self.sources),
            "succeeded": statuses.count("succeeded"),
            "failed": statuses.count("failed"),
            "skipped": statuses.count("skipped"),
            "status": self.status,
            "duration_seconds": self.timing.duration_seconds,
        }

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline": {
                **({"max_parallel": self.max_parallel} if self.max_parallel > 1 else {}),
                "pipeline_run_id": self.pipeline_run_id,
                "attempt": self.attempt,
                "trigger": self.trigger,
                "environment": self.environment,
                "planned_at": self.planned_at.isoformat(),
                "started_at": (
                    self.timing.started_at.isoformat()
                    if self.timing.started_at is not None
                    else None
                ),
                "ended_at": (
                    self.timing.ended_at.isoformat() if self.timing.ended_at is not None else None
                ),
            },
            "selection": {
                "requested": {
                    "tags": list(self.requested_tags),
                    "domains": list(self.requested_domains),
                },
                "root_ids": list(self.root_ids),
                "included_upstream_ids": list(self.included_upstream_ids),
                "source_ids": list(self.source_order),
            },
            "graph": {
                "edges": [
                    {
                        "producer_id": edge.producer_id,
                        "consumer_id": edge.consumer_id,
                        "table": edge.table,
                        "input_paths": list(edge.input_paths),
                    }
                    for edge in self.edges
                ]
            },
            "config_versions": dict(self.config_versions),
            "sources": [source.to_summary() for source in self.sources],
            "totals": self.totals(),
            "summary_persistence": self.summary_persistence.to_summary(),
        }


# Result-oriented aliases make the records readable at runner call sites while preserving
# the outcome terminology used by the wire contract.
SourceAttemptResult = SourceAttempt
SourceResult = SourceOutcome
PipelineResult = PipelineOutcome


def _executed_run_identity(executed_run: Any) -> tuple[str, str, int]:
    try:
        plan = executed_run.planned_run.plan
        attributes = plan.run_context.attributes_as_dict()
        return (
            plan.source.source_id,
            plan.run_context.run_id,
            int(attributes.get("pipeline_attempt", "1")),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "source_id, run_id and attempt are required when source identity cannot be "
            "derived from the executed result"
        ) from exc
