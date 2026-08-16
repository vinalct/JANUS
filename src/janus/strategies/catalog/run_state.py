"""The state one catalog run carries across its request inputs.

What a resumed run *recovered* (:class:`ResumeState`, and the two functions that read the
raw zone back for it), what the loop *accumulated* (:class:`CatalogRunTotals`), and what it
*produced* for result assembly (:class:`CatalogExtractionOutcome`). The loop that drives
them lives in ``extraction.py``.

Mirrors ``strategies/api/run_state.py`` deliberately, so the two families' orchestration
reads as one pattern — see :class:`ResumeState` for the one place they genuinely differ.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from janus.checkpoints import (
    CheckpointState,
    DeadLetterState,
    DeadLetterStore,
    ExtractionProgressStore,
)
from janus.models import ExecutionPlan, ExtractedArtifact
from janus.strategies.api import PaginationState
from janus.strategies.api.pagination import ApiPaginator, _resume_pagination_state
from janus.strategies.http import ApiRequest
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout

from .artifacts import (
    _rediscover_catalog_input_artifacts,
    _rediscover_catalog_raw_artifacts,
    _replay_catalog_entities_from_dir,
)
from .entities import _empty_entity_records, _merge_catalog_entity_records
from .pagination_loop import RequestInputExtraction


@dataclass(frozen=True, slots=True)
class ResumeState:
    """What a ``--resume`` catalog run recovered, as one consistent object."""

    progress: dict[str, Any] | None
    dead_letter_state: DeadLetterState | None
    dead_letter_keys: frozenset[str]
    completed_by_key: Mapping[str, int]
    current_input_key: str | None

    @classmethod
    def load(
        cls,
        plan: ExecutionPlan,
        *,
        progress_store: ExtractionProgressStore,
        dead_letter_store: DeadLetterStore,
        logger: StructuredLogger | None,
    ) -> Self:
        if plan.run_context.attributes_as_dict().get("resume") != "true":
            progress_store.clear(plan)
            dead_letter_store.clear(plan)
            return cls(
                progress=None,
                dead_letter_state=None,
                dead_letter_keys=frozenset(),
                completed_by_key={},
                current_input_key=None,
            )

        progress = progress_store.load(plan)
        dead_letter_state = dead_letter_store.load(plan)
        if logger is not None:
            if progress is not None:
                logger.info(
                    "catalog_extraction_resuming",
                    last_page_number=progress.get("last_page_number"),
                    last_offset=progress.get("last_offset"),
                    prior_artifact_count=progress.get("artifact_count", 0),
                )
            else:
                logger.info("catalog_extraction_resume_requested_no_progress_found")

        completed_by_key: dict[str, int] = {}
        if progress is not None:
            for entry in progress.get("completed_inputs", []):
                completed_by_key[entry["key"]] = entry["index"]

        return cls(
            progress=progress,
            dead_letter_state=dead_letter_state,
            dead_letter_keys=(
                dead_letter_state.item_keys if dead_letter_state is not None else frozenset()
            ),
            completed_by_key=completed_by_key,
            current_input_key=(
                progress.get("current_input_key") if progress is not None else None
            ),
        )

    def is_resuming(self, request_input_key: str) -> bool:
        """Return whether this input is one a previous attempt may have stopped inside."""
        return self.progress is not None and (
            request_input_key == self.current_input_key or self.current_input_key is None
        )

    def progress_to_resume(self, request_input_key: str) -> dict[str, Any] | None:
        """Return the progress record to rehydrate for this input, or ``None``.

        Carrying the invariant in the return type is what removes the ``assert progress is
        not None`` the mid-input resume branch needed to convince ``mypy`` of something the
        code already knew.
        """
        return self.progress if self.is_resuming(request_input_key) else None


@dataclass(slots=True)
class CatalogRunTotals:
    """The run's accumulators, including the entity index every input merges into."""

    artifacts: list[ExtractedArtifact] = field(default_factory=list)
    successful_requests: int = 0
    total_attempts: int = 0
    page_record_total: int = 0
    checkpoint_value: str | None = None
    normalized_records: dict[str, list[dict[str, Any]]] = field(
        default_factory=_empty_entity_records
    )
    entity_indexes: dict[tuple[str, str], int] = field(default_factory=dict)

    def absorb(self, plan: ExecutionPlan, extraction: RequestInputExtraction) -> None:
        """Commit a finished request input's pages, entities and counters to the run.

        The merge is checkpoint-aware — it is what decides which of two records for the same
        entity key survives — which is why it needs the plan.
        """
        self.artifacts.extend(extraction.artifacts)
        _merge_catalog_entity_records(
            plan,
            self.normalized_records,
            self.entity_indexes,
            extraction.normalized_records,
        )
        self.checkpoint_value = extraction.checkpoint_value
        self.successful_requests += extraction.successful_requests
        self.total_attempts += extraction.total_attempts
        self.page_record_total += extraction.page_record_total

    def entity_count(self, entity_type: str) -> int:
        return len(self.normalized_records[entity_type])

    @property
    def records_extracted(self) -> int:
        return sum(len(records) for records in self.normalized_records.values())

    @property
    def retry_count(self) -> int:
        return max(self.total_attempts - self.successful_requests, 0)


@dataclass(frozen=True, slots=True)
class CatalogExtractionOutcome:
    """What the request-input loop produced, ready for result assembly."""

    totals: CatalogRunTotals
    normalized_artifacts: list[ExtractedArtifact]
    dead_letter_count: int
    dead_letter_skip_count: int
    raw_path_prefix: Path | None


def recover_completed_input(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    paginator: ApiPaginator,
    totals: CatalogRunTotals,
    *,
    per_input_request: ApiRequest,
    checkpoint_state: CheckpointState | None,
    completed_input_index: int,
    request_input_count: int,
) -> None:
    """Fold an input a previous attempt already finished back into the run.

    Its pages are re-read from the raw zone rather than re-requested, and — unlike a live
    input — its entities go straight into the run's index: they are already known to be
    complete, so there is nothing to hold back until the input succeeds.
    """
    totals.artifacts.extend(
        _rediscover_catalog_input_artifacts(
            plan,
            storage_layout,
            completed_input_index,
            request_input_count,
        )
    )
    totals.checkpoint_value = _replay_catalog_entities_from_dir(
        plan,
        storage_layout,
        per_input_request,
        paginator,
        completed_input_index,
        request_input_count,
        checkpoint_state,
        totals.normalized_records,
        totals.entity_indexes,
        totals.checkpoint_value,
    )


def rehydrate_resumed_input(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    paginator: ApiPaginator,
    resume: ResumeState,
    *,
    request_input_key: str,
    per_input_request: ApiRequest,
    request_input_index: int,
    request_input_count: int,
    logger: StructuredLogger | None,
) -> tuple[list[ExtractedArtifact], PaginationState]:
    """Recover a partially-extracted input's pages and the state to continue from.

    Returns the paginator's initial state and no artifacts for any input an interrupted
    attempt was *not* inside — :class:`ResumeState` owns that question, so no caller
    re-derives it.
    """
    initial_state = paginator.initial_state(per_input_request)
    resumed_progress = resume.progress_to_resume(request_input_key)
    if resumed_progress is None:
        return [], initial_state

    pre_artifacts = _rediscover_catalog_raw_artifacts(
        plan,
        storage_layout,
        resumed_progress,
        request_input_index,
        request_input_count,
    )
    resumed_state = _resume_pagination_state(
        paginator,
        initial_state,
        resumed_progress,
    )
    if logger is not None:
        logger.info(
            "catalog_extraction_resume_artifacts_recovered",
            recovered_artifact_count=len(pre_artifacts),
            resuming_at_page=resumed_state.page_number,
            resuming_at_offset=resumed_state.offset,
        )
    return pre_artifacts, resumed_state
