"""Orchestration for a file run: which candidate is next, and what a failure costs.

The file family's loop is over **discovered files**, not request inputs. Downloads may
run concurrently, but verification, persistence and archive expansion remain in candidate
order on the loop thread. Orchestration decides which candidate is next and what a failure
costs; ``download.py``, ``archives.py`` and ``artifacts.py`` provide the mechanics composed
by ``download_loop.py``. Nothing here speaks HTTP beyond handing
:class:`~janus.strategies.files.download.FileDownloader` the client it opened.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Self

from janus.checkpoints import (
    CheckpointState,
    CheckpointStore,
    DeadLetterStore,
)
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult
from janus.strategies.common import _max_checkpoint_value, _raw_run_path_prefix
from janus.strategies.http import HttpRequestThrottle
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

from .artifacts import _redacted_location
from .concurrent_extraction import (
    CandidateLedger,
    DownloadWindowStats,
    _consume_candidate,
    _run_concurrent_candidates,
    _skip_dead_lettered,
)
from .discovery import _discover_files, _select_files
from .download import FileDownloader
from .download_loop import FileCandidateResult, _extract_one_file

if TYPE_CHECKING:
    from .core import DiscoveredFile, FileHook
    from .download_window import DownloadOutcome


@dataclass(frozen=True, slots=True)
class FileExtractionContext:
    """Everything resolved once per run, before the first candidate is touched."""

    plan: ExecutionPlan
    file_hook: FileHook | None
    storage_layout: StorageLayout
    checkpoint_state: CheckpointState | None
    downloader: FileDownloader
    throttle: HttpRequestThrottle
    logger: StructuredLogger | None
    #: The writer needs the run path prefix, which the loop resolves — so the context carries
    #: the factory, not the writer.
    raw_writer_factory: Callable[[StorageLayout], RawArtifactWriter]

    @classmethod
    def build(
        cls,
        plan: ExecutionPlan,
        *,
        file_hook: FileHook | None,
        downloader: FileDownloader,
        storage_layout_factory: Callable[[ExecutionPlan], StorageLayout],
        raw_writer_factory: Callable[[StorageLayout], RawArtifactWriter],
        checkpoint_store: CheckpointStore,
        logger: StructuredLogger | None,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> Self:
        return cls(
            plan=plan,
            file_hook=file_hook,
            storage_layout=storage_layout_factory(plan),
            checkpoint_state=checkpoint_store.load(plan),
            downloader=downloader,
            throttle=HttpRequestThrottle(
                requests_per_minute=plan.source_config.access.rate_limit.requests_per_minute,
                clock=clock,
                sleeper=sleeper,
            ),
            logger=logger,
            raw_writer_factory=raw_writer_factory,
        )

    @property
    def download_concurrency(self) -> int:
        return self.plan.source_config.access.rate_limit.concurrency

    @property
    def dead_letter_max_items(self) -> int:
        return self.plan.source_config.extraction.dead_letter_max_items

    def raw_writer_for(self, raw_path_prefix: Path | None) -> RawArtifactWriter:
        return self.raw_writer_factory(self.storage_layout).with_raw_path_prefix(raw_path_prefix)

    def log_extraction_started(self) -> None:
        if self.logger is None:
            return
        plan = self.plan
        self.logger.info(
            "file_extraction_started",
            strategy_variant=plan.source.strategy_variant,
            extraction_mode=plan.extraction_mode,
            checkpoint_strategy=plan.checkpoint_strategy,
            checkpoint_loaded=self.checkpoint_state is not None,
            checkpoint_value=(
                self.checkpoint_state.checkpoint_value
                if self.checkpoint_state is not None
                else None
            ),
            access_format=plan.source_config.access.format,
            input_format=plan.source_config.spark.input_format,
            file_pattern=plan.source_config.access.file_pattern,
            timeout_seconds=plan.source_config.access.timeout_seconds,
            requests_per_minute=(plan.source_config.access.rate_limit.requests_per_minute),
            concurrency=self.download_concurrency,
            dead_letter_max_items=self.dead_letter_max_items,
        )

    def log_discovery_finished(
        self,
        discovered_files: Sequence[DiscoveredFile],
        selected_files: Sequence[DiscoveredFile],
    ) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "file_discovery_finished",
            discovered_file_count=len(discovered_files),
            selected_file_count=len(selected_files),
            checkpoint_loaded=self.checkpoint_state is not None,
            discovered_source_kinds=sorted({file.source_kind for file in discovered_files}),
            selected_versions=[file.version for file in selected_files],
        )

    def log_candidate_started(
        self,
        discovered_file: DiscoveredFile,
        *,
        file_index: int,
        selected_file_count: int,
    ) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "file_candidate_started",
            file_index=file_index,
            selected_file_count=selected_file_count,
            source_kind=discovered_file.source_kind,
            source_location=_redacted_location(discovered_file),
            filename=discovered_file.filename,
            candidate_format=discovered_file.format,
            candidate_version=discovered_file.version,
            size_bytes=discovered_file.size_bytes,
            modified_at=(
                discovered_file.modified_at.isoformat()
                if discovered_file.modified_at is not None
                else None
            ),
        )

    def log_download_dispatched(
        self,
        discovered_file: DiscoveredFile,
        *,
        file_index: int,
        selected_file_count: int,
        in_flight: int,
    ) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "file_download_dispatched",
            file_index=file_index,
            selected_file_count=selected_file_count,
            in_flight=in_flight,
            download_concurrency=self.download_concurrency,
            filename=discovered_file.filename,
            source_location=_redacted_location(discovered_file),
        )

    def log_download_completed(
        self,
        outcome: DownloadOutcome,
        *,
        in_flight: int,
        waited_seconds: float,
    ) -> None:
        if self.logger is None:
            return
        downloaded = outcome.downloaded
        error_fields = (
            {"error_type": type(outcome.error).__name__} if outcome.error is not None else {}
        )
        self.logger.info(
            "file_download_completed",
            file_index=outcome.file_index,
            outcome="failed" if outcome.error is not None else "downloaded",
            in_flight=in_flight,
            attempts_used=downloaded.attempts_used if downloaded is not None else None,
            payload_size_bytes=downloaded.payload.size_bytes if downloaded is not None else None,
            download_seconds=outcome.download_seconds,
            waited_seconds=waited_seconds,
            **error_fields,
        )

    def log_download_drained(self, outcome: DownloadOutcome) -> None:
        if self.logger is None:
            return
        error_fields = (
            {"error_type": type(outcome.error).__name__} if outcome.error is not None else {}
        )
        self.logger.info(
            "file_download_drained",
            file_index=outcome.file_index,
            outcome="failed" if outcome.error is not None else "downloaded",
            discarded=True,
            **error_fields,
        )

    def log_extraction_finished(
        self,
        totals: RunTotals,
        *,
        discovered_file_count: int,
        selected_file_count: int,
        normalization_candidate_count: int,
        dead_letter_count: int,
        dead_letter_skip_count: int,
        download_window: DownloadWindowStats | None = None,
    ) -> None:
        if self.logger is None:
            return
        download_fields = (
            {
                "download_concurrency": download_window.download_concurrency,
                "max_in_flight_observed": download_window.max_in_flight_observed,
            }
            if download_window is not None
            else {}
        )
        self.logger.info(
            "file_extraction_finished",
            discovered_file_count=discovered_file_count,
            selected_file_count=selected_file_count,
            load_count=totals.loaded_file_count,
            attempt_count=totals.total_attempts,
            retry_count=totals.retry_count,
            persisted_file_count=totals.persisted_file_count,
            archive_member_count=totals.archive_member_count,
            checksum_verified_count=totals.checksum_verified_count,
            skipped_file_count=totals.skipped_file_count,
            normalization_candidate_count=normalization_candidate_count,
            artifact_count=len(totals.artifacts),
            checkpoint_value=totals.checkpoint_value,
            dead_letter_count=dead_letter_count,
            dead_letter_skipped_count=dead_letter_skip_count,
            **download_fields,
        )


@dataclass(slots=True)
class RunTotals:
    """The run's accumulators."""

    artifacts: list[ExtractedArtifact] = field(default_factory=list)
    resolved_versions: list[str] = field(default_factory=list)
    persisted_file_count: int = 0
    archive_member_count: int = 0
    checksum_verified_count: int = 0
    skipped_file_count: int = 0
    loaded_file_count: int = 0
    total_attempts: int = 0
    checkpoint_value: str | None = None

    def absorb(self, candidate: FileCandidateResult) -> None:
        self.artifacts.extend(candidate.artifacts)
        self.total_attempts += candidate.total_attempts
        self.loaded_file_count += candidate.loaded_file_count
        self.checksum_verified_count += candidate.checksum_verified_count
        self.archive_member_count += candidate.archive_member_count
        self.persisted_file_count += 1
        self.resolved_versions.append(candidate.version)
        self.checkpoint_value = _max_checkpoint_value(self.checkpoint_value, candidate.version)

    @property
    def retry_count(self) -> int:
        return max(self.total_attempts - self.loaded_file_count, 0)


@dataclass(frozen=True, slots=True)
class FileExtractionOutcome:
    """What the candidate loop produced, ready for result assembly."""

    totals: RunTotals
    discovered_file_count: int
    selected_file_count: int
    normalization_candidate_count: int
    dead_letter_count: int
    dead_letter_skip_count: int
    raw_path_prefix: Path | None
    download_window: DownloadWindowStats | None = None


def _run_sequential_candidates(
    context: FileExtractionContext,
    ledger: CandidateLedger,
    selected_files: Sequence[DiscoveredFile],
    *,
    client,
    raw_writer: RawArtifactWriter,
    dead_letter_store: DeadLetterStore,
) -> None:
    """Keep the sequential loop's one client and candidate-by-candidate behavior."""
    for file_index, discovered_file in enumerate(selected_files, start=1):
        context.log_candidate_started(
            discovered_file, file_index=file_index, selected_file_count=len(selected_files),
        )

        if _skip_dead_lettered(context, ledger, discovered_file, file_index=file_index):
            continue
        _consume_candidate(
            context, ledger, discovered_file,
            file_index=file_index, selected_file_count=len(selected_files),
            dead_letter_store=dead_letter_store,
            complete=partial(
                _extract_one_file, context, discovered_file,
                client=client, raw_writer=raw_writer, file_index=file_index,
            ),
        )


def run_file_extraction(
    context: FileExtractionContext,
    *,
    dead_letter_store: DeadLetterStore,
) -> FileExtractionOutcome:
    """Download every selected file, recording failures as dead letters while the budget allows."""
    plan = context.plan
    raw_path_prefix = _raw_run_path_prefix(plan)
    raw_writer = context.raw_writer_for(raw_path_prefix)

    resume = plan.run_context.attributes_as_dict().get("resume") == "true"
    dead_letter_state = dead_letter_store.load(plan) if resume else None
    if not resume:
        dead_letter_store.clear(plan)

    context.log_extraction_started()

    ledger = CandidateLedger(
        totals=RunTotals(), dead_letter_state=dead_letter_state,
        dead_letter_keys=set(dead_letter_state.item_keys) if dead_letter_state else set(),
    )

    with context.downloader.open_client() as client:
        discovered_files = _discover_files(plan, context.file_hook, client.transport)
        selected_files = _select_files(plan, discovered_files, context.checkpoint_state)
        context.log_discovery_finished(discovered_files, selected_files)

        if context.download_concurrency > 1:
            window_stats = _run_concurrent_candidates(
                context, ledger, selected_files,
                raw_writer=raw_writer, dead_letter_store=dead_letter_store,
            )
        else:
            _run_sequential_candidates(
                context, ledger, selected_files,
                client=client, raw_writer=raw_writer, dead_letter_store=dead_letter_store,
            )
            window_stats = None

    totals = ledger.totals
    normalization_candidate_count = sum(
        1
        for artifact in totals.artifacts
        if artifact.format == plan.source_config.spark.input_format
    )
    dead_letter_count = ledger.dead_letter_state.entry_count if ledger.dead_letter_state else 0

    context.log_extraction_finished(
        totals,
        discovered_file_count=len(discovered_files),
        selected_file_count=len(selected_files),
        normalization_candidate_count=normalization_candidate_count,
        dead_letter_count=dead_letter_count,
        dead_letter_skip_count=ledger.dead_letter_skip_count,
        download_window=window_stats,
    )

    return FileExtractionOutcome(
        totals=totals,
        discovered_file_count=len(discovered_files),
        selected_file_count=len(selected_files),
        normalization_candidate_count=normalization_candidate_count,
        dead_letter_count=dead_letter_count,
        dead_letter_skip_count=ledger.dead_letter_skip_count,
        raw_path_prefix=raw_path_prefix,
        download_window=window_stats,
    )


def build_extraction_result(
    context: FileExtractionContext,
    outcome: FileExtractionOutcome,
    *,
    dead_letter_store: DeadLetterStore,
) -> ExtractionResult:
    """Assemble the run's ``ExtractionResult``.

    ``records_extracted`` counts the artifacts the bronze handoff will actually read — the
    ones matching ``spark.input_format`` — not the files downloaded: one archive can persist
    a hundred members, and the archive itself is never normalized.
    """
    plan = context.plan
    totals = outcome.totals

    extraction_metadata = {
        "discovered_file_count": str(outcome.discovered_file_count),
        "selected_file_count": str(outcome.selected_file_count),
        "persisted_file_count": str(totals.persisted_file_count),
        "archive_member_count": str(totals.archive_member_count),
        "checksum_verified_count": str(totals.checksum_verified_count),
        "skipped_file_count": str(totals.skipped_file_count),
        "checkpoint_loaded": str(context.checkpoint_state is not None).lower(),
        "normalization_candidate_count": str(outcome.normalization_candidate_count),
    }
    if totals.resolved_versions:
        extraction_metadata["resolved_versions"] = ",".join(totals.resolved_versions)
    extraction_metadata["raw_path_prefix"] = str(outcome.raw_path_prefix or "")
    if outcome.download_window is not None:
        stats = outcome.download_window
        extraction_metadata["download_concurrency"] = str(stats.download_concurrency)
        extraction_metadata["max_in_flight_observed"] = str(stats.max_in_flight_observed)
        extraction_metadata["concurrent_downloads_used"] = str(
            stats.max_in_flight_observed > 1
        ).lower()

    extraction_result = ExtractionResult.from_plan(
        plan,
        tuple(totals.artifacts),
        records_extracted=outcome.normalization_candidate_count,
        checkpoint_value=totals.checkpoint_value,
        metadata=extraction_metadata,
    )
    extraction_result = extraction_result.with_metadata(
        "dead_letter_count", str(outcome.dead_letter_count),
    ).with_metadata(
        "dead_letter_skipped_count", str(outcome.dead_letter_skip_count),
    )
    if outcome.dead_letter_count > 0:
        extraction_result = extraction_result.with_metadata(
            "dead_letter_path",
            str(dead_letter_store.path(plan)),
        )
    return extraction_result
