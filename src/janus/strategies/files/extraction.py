"""Orchestration for a file run: which candidate is next, and what a failure costs.

The file family's loop is over **discovered files**, not request inputs — one candidate is
downloaded, verified, persisted and (when it is an archive) expanded before the next one
starts. *Orchestration* is that loop, the dead-letter policy around it and the result it
assembles; *mechanics* — fetching bytes, expanding an archive, deciding where a payload lands
— are ``download.py``, ``archives.py`` and ``artifacts.py``, composed one candidate at a time
by ``download_loop.py``. Nothing here speaks HTTP beyond handing
:class:`~janus.strategies.files.download.FileDownloader` the client it opened.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

from janus.checkpoints import (
    CheckpointState,
    CheckpointStore,
    DeadLetterStore,
    can_continue_after_dead_letter,
)
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult
from janus.strategies.common import _max_checkpoint_value, _raw_run_path_prefix
from janus.strategies.http import HttpRequestThrottle
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

from .artifacts import (
    _discovered_file_dead_letter_key,
    _discovered_file_dead_letter_metadata,
    _redacted_location,
)
from .discovery import _discover_files, _select_files
from .download import FileDownloader
from .download_loop import FileCandidateResult, _extract_one_file

if TYPE_CHECKING:
    from .core import DiscoveredFile, FileHook


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
            concurrency=plan.source_config.access.rate_limit.concurrency,
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

    def log_extraction_finished(
        self,
        totals: RunTotals,
        *,
        discovered_file_count: int,
        selected_file_count: int,
        normalization_candidate_count: int,
        dead_letter_count: int,
        dead_letter_skip_count: int,
    ) -> None:
        if self.logger is None:
            return
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


def run_file_extraction(
    context: FileExtractionContext,
    *,
    dead_letter_store: DeadLetterStore,
) -> FileExtractionOutcome:
    """Download every selected file, recording failures as dead letters while the budget allows."""
    plan = context.plan
    logger = context.logger
    raw_path_prefix = _raw_run_path_prefix(plan)
    raw_writer = context.raw_writer_for(raw_path_prefix)

    resume = plan.run_context.attributes_as_dict().get("resume") == "true"
    dead_letter_state = dead_letter_store.load(plan) if resume else None
    if not resume:
        dead_letter_store.clear(plan)

    context.log_extraction_started()

    totals = RunTotals()
    dead_letter_keys = set(dead_letter_state.item_keys) if dead_letter_state is not None else set()
    dead_letter_skip_count = 0

    with context.downloader.open_client() as client:
        discovered_files = _discover_files(plan, context.file_hook, client.transport)
        selected_files = _select_files(plan, discovered_files, context.checkpoint_state)
        context.log_discovery_finished(discovered_files, selected_files)

        for file_index, discovered_file in enumerate(selected_files, start=1):
            candidate_key = _discovered_file_dead_letter_key(discovered_file)
            context.log_candidate_started(
                discovered_file,
                file_index=file_index,
                selected_file_count=len(selected_files),
            )

            if candidate_key in dead_letter_keys:
                dead_letter_skip_count += 1
                if logger is not None:
                    logger.info(
                        "file_candidate_skipped_dead_letter",
                        file_index=file_index,
                        filename=discovered_file.filename,
                        dead_letter_count=len(dead_letter_keys),
                    )
                continue

            try:
                candidate = _extract_one_file(
                    context,
                    discovered_file,
                    client=client,
                    raw_writer=raw_writer,
                    file_index=file_index,
                )
            except Exception as exc:
                if logger is not None:
                    logger.exception("file_candidate_execution_failed")
                dead_letter_state = dead_letter_store.record(
                    plan,
                    item_key=candidate_key,
                    item_type="file_candidate",
                    error=exc,
                    metadata=_discovered_file_dead_letter_metadata(
                        discovered_file,
                        file_index=file_index,
                        selected_file_count=len(selected_files),
                    ),
                )
                dead_letter_keys = set(dead_letter_state.item_keys)
                if logger is not None:
                    logger.info(
                        "file_candidate_dead_lettered",
                        file_index=file_index,
                        filename=discovered_file.filename,
                        dead_letter_count=dead_letter_state.entry_count,
                        dead_letter_max_items=context.dead_letter_max_items,
                        error_type=type(exc).__name__,
                    )
                if not can_continue_after_dead_letter(
                    total_item_count=len(selected_files),
                    dead_letter_count=dead_letter_state.entry_count,
                    dead_letter_max_items=context.dead_letter_max_items,
                ):
                    raise
                dead_letter_skip_count += 1
                continue

            if candidate is None:
                totals.skipped_file_count += 1
                continue

            totals.absorb(candidate)

    normalization_candidate_count = sum(
        1
        for artifact in totals.artifacts
        if artifact.format == plan.source_config.spark.input_format
    )
    dead_letter_count = dead_letter_state.entry_count if dead_letter_state is not None else 0

    context.log_extraction_finished(
        totals,
        discovered_file_count=len(discovered_files),
        selected_file_count=len(selected_files),
        normalization_candidate_count=normalization_candidate_count,
        dead_letter_count=dead_letter_count,
        dead_letter_skip_count=dead_letter_skip_count,
    )

    return FileExtractionOutcome(
        totals=totals,
        discovered_file_count=len(discovered_files),
        selected_file_count=len(selected_files),
        normalization_candidate_count=normalization_candidate_count,
        dead_letter_count=dead_letter_count,
        dead_letter_skip_count=dead_letter_skip_count,
        raw_path_prefix=raw_path_prefix,
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

    extraction_result = ExtractionResult.from_plan(
        plan,
        tuple(totals.artifacts),
        records_extracted=outcome.normalization_candidate_count,
        checkpoint_value=totals.checkpoint_value,
        metadata=extraction_metadata,
    )
    extraction_result = extraction_result.with_metadata(
        "dead_letter_count",
        str(outcome.dead_letter_count),
    ).with_metadata(
        "dead_letter_skipped_count",
        str(outcome.dead_letter_skip_count),
    )
    if outcome.dead_letter_count > 0:
        extraction_result = extraction_result.with_metadata(
            "dead_letter_path",
            str(dead_letter_store.path(plan)),
        )
    return extraction_result
