"""Pure validation helpers for orchestration result records."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from janus.orchestration.results import PipelineOutcome, SourceOutcome


def _validate_source_identity(source_id: str, run_id: str, attempt: int) -> None:
    if not source_id.strip():
        raise ValueError("source_id must not be empty")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if attempt < 1:
        raise ValueError("attempt must be 1 or greater")


def _validate_attempt_history(outcome: SourceOutcome) -> None:
    if outcome.status == "skipped":
        if outcome.attempts or outcome.failure is not None or outcome.skip is None:
            raise ValueError(
                "a skipped source has no attempts or failure, and must explain its skip"
            )
        unknown_blockers = set(outcome.skip.direct_blocking_upstream_ids) - set(
            outcome.upstream_ids
        )
        if unknown_blockers:
            raise ValueError(f"skip blockers are not direct upstreams: {sorted(unknown_blockers)}")
        return
    if outcome.skip is not None:
        raise ValueError("only a skipped source may carry a skip explanation")
    if not outcome.attempts:
        if outcome.status != "failed" or outcome.failure is None:
            raise ValueError("a non-attempted source can only be a recorded planning failure")
        return

    attempt_numbers = [attempt.attempt for attempt in outcome.attempts]
    if attempt_numbers != sorted(set(attempt_numbers)):
        raise ValueError("attempt history must have increasing, unique attempt numbers")
    if any(attempt.source_id != outcome.source_id for attempt in outcome.attempts):
        raise ValueError("every attempt must belong to the outcome source")
    final = outcome.attempts[-1]
    if (outcome.run_id, outcome.attempt, outcome.status) != (
        final.run_id,
        final.attempt,
        final.status,
    ):
        raise ValueError("the terminal source identity and status must match its final attempt")
    if outcome.failure != final.failure:
        raise ValueError("the terminal failure must match the final attempt")


def _validate_pipeline_membership(outcome: PipelineOutcome) -> None:
    if len(set(outcome.source_order)) != len(outcome.source_order):
        raise ValueError("source_order must not repeat a source")
    if tuple(source.source_id for source in outcome.sources) != outcome.source_order:
        raise ValueError("sources must cover source_order exactly and in deterministic order")
    if set(outcome.root_ids) - set(outcome.source_order):
        raise ValueError("every selected root must be in source_order")
    if set(outcome.included_upstream_ids) != set(outcome.source_order) - set(outcome.root_ids):
        raise ValueError("included_upstream_ids must be exactly the non-root sources")
    if tuple(source_id for source_id, _version in outcome.config_versions) != outcome.source_order:
        raise ValueError("config_versions must cover sources in source_order")
    if any(not version.strip() for _source_id, version in outcome.config_versions):
        raise ValueError("config versions must not be empty")


def _canonical_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({value.strip() for value in values if value.strip()}))


def _require_sorted_unique_ids(field_name: str, values: tuple[str, ...]) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must not contain empty source ids")
    if list(values) != sorted(set(values)):
        raise ValueError(f"{field_name} must be sorted and free of repeats")


def _validate_aware(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
