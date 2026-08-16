"""The file strategy façade, and the vocabulary the rest of the family is written in.

What remains here is the contract, not the machinery: the :class:`FileStrategy` lifecycle
methods the planner calls, the :class:`FileHook` extension points a source implements, the
:class:`DiscoveredFile` record every other module in the package passes around, the exception
hierarchy, and compatibility re-exports for names that were defined here before the package
was split.

Where the machinery went:

* ``discovery.py`` — what files exist, and which of them this run wants.
* ``download.py`` — fetch one file's bytes and prove they are the advertised bytes.
* ``archives.py`` — expand one archive safely.
* ``artifacts.py`` — raw-zone path layout, redaction and dead-letter metadata.
* ``extraction.py`` — orchestration: the candidate loop, dead-letter policy, totals.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from janus.checkpoints import CheckpointStore, DeadLetterStore
from janus.models import (
    ExecutionPlan,
    ExtractionResult,
    SourceConfig,
    WriteResult,
)
from janus.strategies.base import BaseStrategy, SourceHook
from janus.strategies.common import _default_storage_layout
from janus.strategies.files.formats import SUPPORTED_FILE_INPUT_FORMATS
from janus.strategies.http import (
    ApiResponse,
    ApiTransport,
    HttpStrategyError,
    UrllibApiTransport,
)
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter


class FileStrategyError(HttpStrategyError):
    """Base failure for file-strategy execution."""


class FileDiscoveryError(FileStrategyError):
    """Raised when the configured file source cannot be resolved deterministically."""


class FileDownloadError(FileStrategyError):
    """Raised when a remote file request fails."""


class FileIntegrityError(FileStrategyError):
    """Raised when an expected checksum does not match the downloaded payload."""


class ArchiveExtractionError(FileStrategyError):
    """Raised when an archive payload cannot be expanded safely."""


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """One discovered file candidate before raw persistence."""

    source_kind: str
    location: str
    filename: str
    format: str
    version: str | None = None
    size_bytes: int | None = None
    modified_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source_kind.strip():
            raise ValueError("source_kind must not be empty")
        if not self.location.strip():
            raise ValueError("location must not be empty")
        if not self.filename.strip():
            raise ValueError("filename must not be empty")
        if not self.format.strip():
            raise ValueError("format must not be empty")
        if self.modified_at is not None and (
            self.modified_at.tzinfo is None or self.modified_at.utcoffset() is None
        ):
            raise ValueError("modified_at must be timezone-aware")


class FileHook(SourceHook):
    """File-strategy hook points for source-specific layout and version quirks."""

    def discovered_files(
        self,
        plan: ExecutionPlan,
        files: Sequence[DiscoveredFile],
    ) -> Sequence[DiscoveredFile]:
        del plan
        return files

    def resolve_links(
        self,
        plan: ExecutionPlan,
        url: str,
        formato: str | None,
        transport: ApiTransport,
    ) -> Sequence[DiscoveredFile]:
        from janus.strategies.files.resolvers import build_resolver_chain, resolve_link

        return resolve_link(
            url, formato, transport, build_resolver_chain(plan.source_config.access.link_resolver)
        )

    def resolve_version(
        self,
        plan: ExecutionPlan,
        discovered_file: DiscoveredFile,
    ) -> str | None:
        del plan
        del discovered_file
        return None

    def expected_checksum(
        self,
        plan: ExecutionPlan,
        discovered_file: DiscoveredFile,
    ) -> str | None:
        del plan
        del discovered_file
        return None

    def prepare_download(
        self,
        plan: ExecutionPlan,
        discovered_file: DiscoveredFile,
        payload: bytes,
        *,
        response: ApiResponse | None = None,
    ) -> bytes:
        del plan
        del discovered_file
        del response
        return payload

    def archive_members(
        self,
        plan: ExecutionPlan,
        archive_file: DiscoveredFile,
        members: Sequence[DiscoveredFile],
    ) -> Sequence[DiscoveredFile]:
        del plan
        del archive_file
        return members


from janus.strategies.files.archives import (
    _archive_member_payloads,
    _filter_members,
)
from janus.strategies.files.artifacts import (
    _infer_handoff_format,
    _raw_extracted_relative_path,
)
from janus.strategies.files.download import (
    _RETRY_POLICY,
    FileDownloader,
    _read_checksum_sidecar,
)
from janus.strategies.files.extraction import (
    FileExtractionContext,
    build_extraction_result,
    run_file_extraction,
)

__all__ = [
    "_RETRY_POLICY",
    "ArchiveExtractionError",
    "DiscoveredFile",
    "FileDiscoveryError",
    "FileDownloadError",
    "FileDownloader",
    "FileHook",
    "FileIntegrityError",
    "FileStrategy",
    "FileStrategyError",
    "_archive_member_payloads",
    "_filter_members",
    "_infer_handoff_format",
    "_raw_extracted_relative_path",
    "_read_checksum_sidecar",
]


@dataclass(slots=True)
class FileStrategy(BaseStrategy):
    """Reusable bulk-file strategy for public datasets delivered as files or archives."""

    transport_factory: Callable[[], ApiTransport] = UrllibApiTransport
    storage_layout_factory: Callable[[ExecutionPlan], StorageLayout] = field(
        default_factory=lambda: _default_storage_layout
    )
    raw_writer_factory: Callable[[StorageLayout], RawArtifactWriter] = RawArtifactWriter
    checkpoint_store: CheckpointStore = field(default_factory=CheckpointStore)
    dead_letter_store: DeadLetterStore = field(default_factory=DeadLetterStore)
    env_reader: Callable[[str], str | None] = os.getenv
    sleeper: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    logger: StructuredLogger | None = None

    @property
    def strategy_family(self) -> str:
        return "file"

    def plan(
        self,
        source_config: SourceConfig,
        run_context,
        hook: SourceHook | None = None,
    ) -> ExecutionPlan:
        self._validate_source_config(source_config)
        plan = ExecutionPlan.from_source_config(source_config, run_context)
        plan = plan.with_note("strategy_family:file")
        plan = plan.with_note(f"strategy_variant:{source_config.strategy_variant}")
        if hook is not None:
            return hook.on_plan(plan)
        return plan

    def extract(
        self,
        plan: ExecutionPlan,
        hook: SourceHook | None = None,
        *,
        spark=None,
    ) -> ExtractionResult:
        del spark
        context = FileExtractionContext.build(
            plan,
            file_hook=hook if isinstance(hook, FileHook) else None,
            downloader=self._build_downloader(),
            storage_layout_factory=self.storage_layout_factory,
            raw_writer_factory=self.raw_writer_factory,
            checkpoint_store=self.checkpoint_store,
            logger=self._bind_logger(plan),
            clock=self.clock,
            sleeper=self.sleeper,
        )
        outcome = run_file_extraction(context, dead_letter_store=self.dead_letter_store)
        extraction_result = build_extraction_result(
            context,
            outcome,
            dead_letter_store=self.dead_letter_store,
        )
        if hook is not None:
            return hook.on_extraction_result(plan, extraction_result)
        return extraction_result

    def build_normalization_handoff(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        hook: SourceHook | None = None,
    ) -> ExtractionResult:
        handoff_artifacts = tuple(
            artifact
            for artifact in extraction_result.artifacts
            if artifact.format == plan.source_config.spark.input_format
        )
        if not handoff_artifacts:
            raise FileStrategyError(
                "No extracted artifacts match the configured "
                f"spark.input_format {plan.source_config.spark.input_format!r}"
            )

        handoff = replace(extraction_result, artifacts=handoff_artifacts).with_metadata(
            "normalization_artifact_count",
            str(len(handoff_artifacts)),
        )
        if hook is not None:
            return hook.on_normalization_handoff(plan, handoff)
        return handoff

    def emit_metadata(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
        hook: SourceHook | None = None,
    ) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "strategy_family": self.strategy_family,
            "strategy_variant": plan.source.strategy_variant,
            "input_format": plan.source_config.spark.input_format,
            "raw_persistence_format": "binary",
            "artifact_count": len(extraction_result.artifacts),
            "records_extracted": extraction_result.records_extracted or 0,
            "checkpoint_value": extraction_result.checkpoint_value or "",
            "write_result_count": len(write_results),
        }
        metadata.update(extraction_result.metadata_as_dict())
        if hook is not None:
            metadata.update(hook.metadata_fields(plan, extraction_result, write_results))
        return metadata

    def _build_downloader(self) -> FileDownloader:
        return FileDownloader(
            transport_factory=self.transport_factory,
            sleeper=self.sleeper,
            env_reader=self.env_reader,
        )

    def _validate_source_config(self, source_config: SourceConfig) -> None:
        if source_config.outputs.raw.format != "binary":
            raise ValueError("File strategy requires outputs.raw.format='binary'")
        if source_config.access.pagination.type != "none":
            raise ValueError("File strategy requires access.pagination.type='none'")
        if source_config.spark.input_format not in SUPPORTED_FILE_INPUT_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_FILE_INPUT_FORMATS))
            raise ValueError(f"File strategy spark.input_format must be one of: {allowed}")
        if (
            source_config.strategy_variant == "archive_package"
            and source_config.access.format != "binary"
        ):
            raise ValueError("archive_package requires access.format='binary'")

    def _bind_logger(self, plan: ExecutionPlan) -> StructuredLogger | None:
        if self.logger is None:
            return None
        return self.logger.bind(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            strategy_family=self.strategy_family,
        )
