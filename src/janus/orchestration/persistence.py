"""Atomic filesystem persistence for the versioned pipeline aggregate."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from janus.orchestration.errors import DuplicatePipelineRunError, PipelineIdentityError
from janus.orchestration.identity import validate_pipeline_run_id
from janus.orchestration.results import FailureDetails, PipelineOutcome, SummaryPersistence
from janus.utils.storage import StorageLayout

PIPELINES_DIRECTORY = "pipelines"
PIPELINE_SUMMARY_FILENAME = "summary.json"


def _replace_file(source: Path, target: Path) -> None:
    source.replace(target)


@dataclass(frozen=True, slots=True)
class PipelineSummaryStore:
    """Resolve, preflight, and atomically publish summaries in the metadata zone."""

    storage_layout: StorageLayout
    atomic_replace: Callable[[Path, Path], None] = _replace_file

    def summary_path(self, pipeline_run_id: str) -> Path:
        """Return a validated path that cannot escape the configured metadata root."""
        safe_id = validate_pipeline_run_id(pipeline_run_id)
        metadata_root = self.storage_layout.metadata_dir.resolve()
        candidate = (
            metadata_root / PIPELINES_DIRECTORY / safe_id / PIPELINE_SUMMARY_FILENAME
        ).resolve()
        if not candidate.is_relative_to(metadata_root):
            raise PipelineIdentityError(
                f"Pipeline summary path {candidate} escapes metadata root {metadata_root}"
            )
        return candidate

    def assert_available(self, pipeline_run_id: str) -> Path:
        """Read-only preflight a fresh core invocation performs before source writes."""
        path = self.summary_path(pipeline_run_id)
        if path.exists():
            raise DuplicatePipelineRunError(
                f"Pipeline id {pipeline_run_id!r} already has a finalized summary at {path}; "
                "start a re-execution or backfill with a new pipeline identity"
            )
        return path

    def persist(
        self,
        outcome: PipelineOutcome,
        *,
        allow_existing: bool = False,
    ) -> PipelineOutcome:
        """Publish a final aggregate, returning the same outcome marked as persisted.

        allow_existing is reserved for an adapter updating one correlated pipeline with a
        new attempt record. A fresh core invocation keeps the default and must also call
        assert_available before it starts source work.
        """
        path = self.summary_path(outcome.pipeline_run_id)
        if path.exists() and not allow_existing:
            self.assert_available(outcome.pipeline_run_id)

        persisted = outcome.with_summary_persistence(SummaryPersistence.succeeded(path))
        temporary_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not allow_existing:
                self.assert_available(outcome.pipeline_run_id)
            temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
            _write_json_document(temporary_path, persisted.to_summary())
            self.atomic_replace(temporary_path, path)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            failed = outcome.with_summary_persistence(
                SummaryPersistence.failed(
                    path,
                    FailureDetails.from_exception(exc, phase="summary_persistence"),
                )
            )
            raise PipelineSummaryPersistenceError(path, failed) from exc
        return persisted


class PipelineSummaryPersistenceError(RuntimeError):
    """An operational failure that retains the complete in-memory aggregate."""

    def __init__(self, path: Path, pipeline_outcome: PipelineOutcome) -> None:
        self.path = path
        self.pipeline_outcome = pipeline_outcome
        failure = pipeline_outcome.summary_persistence.failure
        reason = failure.reason if failure is not None else "unknown persistence failure"
        super().__init__(f"Could not persist pipeline summary at {path}: {reason}")


def _write_json_document(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
