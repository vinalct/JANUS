"""The state one API run carries across its request inputs.

What a resumed run *recovered* (:class:`ResumeState`, and the two functions that read the
raw zone back for it), what the loop *accumulated* (:class:`RunTotals`), and what it
*produced* for result assembly (:class:`ApiExtractionOutcome`). The loop that drives them
lives in ``extraction.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from janus.checkpoints import DeadLetterState, DeadLetterStore, ExtractionProgressStore
from janus.models import ExecutionPlan, ExtractedArtifact
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout

from .artifacts import _rediscover_all_artifacts_for_input, _rediscover_raw_artifacts
from .pagination import ApiPaginator, PaginationState, _resume_pagination_state
from .pagination_loop import RequestInputExtraction


@dataclass(frozen=True, slots=True)
class ResumeState:
    """What a ``--resume`` run recovered, as one consistent object.

    These five facts are read together and are only meaningful together: whether an input is
    the one a previous attempt was mid-way through depends on ``progress`` *and*
    ``current_input_key``, and the file index to reuse depends on both plus
    ``current_input_index``. Held as loose locals, every read site had to re-establish that
    relationship — which is why the mid-input resume branch needed an ``assert progress is
    not None`` to convince ``mypy`` of something the code already knew.
    :meth:`progress_to_resume` returns the progress record *or* ``None``, so the invariant is
    carried by the type rather than asserted.
    """

    progress: dict[str, Any] | None
    dead_letter_state: DeadLetterState | None
    dead_letter_keys: frozenset[str]
    completed_by_key: Mapping[str, int]
    current_input_key: str | None
    current_input_index: int | None

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
                current_input_index=None,
            )

        progress = progress_store.load(plan)
        dead_letter_state = dead_letter_store.load(plan)
        if logger is not None:
            if progress is not None:
                logger.info(
                    "api_extraction_resuming",
                    current_input_key=progress.get("current_input_key"),
                    last_page_number=progress.get("last_page_number"),
                    last_offset=progress.get("last_offset"),
                    completed_input_count=len(progress.get("completed_inputs", [])),
                    prior_artifact_count=progress.get("artifact_count", 0),
                )
            else:
                logger.info("api_extraction_resume_requested_no_progress_found")

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
            current_input_index=(
                progress.get("current_input_index") if progress is not None else None
            ),
        )

    def is_resuming(self, request_input_key: str) -> bool:
        """Return whether this input is the one a previous attempt stopped inside."""
        return self.progress is not None and request_input_key == self.current_input_key

    def file_index_for(self, request_input_key: str, request_input_index: int) -> int:
        """Return the index this input's raw pages are filed under.

        A resumed input keeps the index the interrupted attempt used, so its new pages land
        beside the ones already on disk; anything else is filed under its position in the run.
        """
        if self.is_resuming(request_input_key) and self.current_input_index is not None:
            return self.current_input_index
        return request_input_index

    def progress_to_resume(self, request_input_key: str) -> dict[str, Any] | None:
        """Return the progress record to rehydrate for this input, or ``None``."""
        return self.progress if self.is_resuming(request_input_key) else None


@dataclass(slots=True)
class RunTotals:
    """The run's accumulators."""

    artifacts: list[ExtractedArtifact] = field(default_factory=list)
    records_extracted: int = 0
    successful_requests: int = 0
    total_attempts: int = 0
    checkpoint_value: str | None = None
    concurrent_pagination_used: bool = False
    speculative_requests: int = 0
    discarded_requests: int = 0
    past_end_terminated_count: int = 0
    past_end_status: int | None = None
    total_records_reported: int | None = None
    lookahead_ceiling_source: str | None = None

    def absorb(self, extraction: RequestInputExtraction) -> None:
        self.artifacts.extend(extraction.artifacts)
        self.records_extracted += extraction.records_extracted
        self.successful_requests += extraction.successful_requests
        self.total_attempts += extraction.total_attempts
        self.checkpoint_value = extraction.checkpoint_value
        self.speculative_requests += extraction.speculative_requests
        self.discarded_requests += extraction.discarded_requests
        if extraction.past_end_status is not None:
            self.past_end_terminated_count += 1
            self.past_end_status = extraction.past_end_status
        if extraction.total_records_reported is not None:
            self.total_records_reported = extraction.total_records_reported
            self.lookahead_ceiling_source = extraction.lookahead_ceiling_source

    @property
    def retry_count(self) -> int:
        return max(self.total_attempts - self.successful_requests, 0)


@dataclass(frozen=True, slots=True)
class ApiExtractionOutcome:
    """What the request-input loop produced, ready for result assembly."""

    totals: RunTotals
    request_input_count: int
    request_input_metadata: Mapping[str, str]
    dead_letter_count: int
    dead_letter_skip_count: int
    raw_path_prefix: Path | None


def recover_completed_input(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    *,
    request_input_key: str,
    file_index: int,
    request_input_count: int,
    logger: StructuredLogger | None,
) -> list[ExtractedArtifact]:
    """Return the artifacts a previous attempt already wrote for a finished input."""
    pre_artifacts = _rediscover_all_artifacts_for_input(
        plan,
        storage_layout,
        file_index,
        request_input_count,
    )
    if logger is not None:
        logger.info(
            "api_request_input_skipped_already_complete",
            recovered_artifact_count=len(pre_artifacts),
            input_key=request_input_key,
        )
    return pre_artifacts


def rehydrate_resumed_input(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    paginator: ApiPaginator,
    resume: ResumeState,
    *,
    request_input_key: str,
    pagination_state: PaginationState,
    file_index: int,
    request_input_count: int,
    logger: StructuredLogger | None,
) -> tuple[list[ExtractedArtifact], PaginationState]:
    """Recover a partially-extracted input's pages and the state to continue from.

    Returns ``pagination_state`` unchanged and no artifacts for any input an interrupted
    attempt was *not* inside — ``ResumeState`` owns that question, so no caller re-derives it.
    """
    resumed_progress = resume.progress_to_resume(request_input_key)
    if resumed_progress is None:
        return [], pagination_state

    pre_artifacts = _rediscover_raw_artifacts(
        plan,
        storage_layout,
        resumed_progress,
        file_index,
        request_input_count,
    )
    resumed_state = _resume_pagination_state(
        paginator,
        pagination_state,
        resumed_progress,
    )
    if logger is not None:
        logger.info(
            "api_extraction_resume_artifacts_recovered",
            recovered_artifact_count=len(pre_artifacts),
            resuming_at_page=resumed_state.page_number,
            resuming_at_offset=resumed_state.offset,
            input_key=request_input_key,
        )
    return pre_artifacts, resumed_state
