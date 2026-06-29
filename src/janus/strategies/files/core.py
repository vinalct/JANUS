from __future__ import annotations

import fnmatch
import glob
import os
import re
import tarfile
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from janus.checkpoints import (
    CheckpointState,
    CheckpointStore,
    DeadLetterStore,
    can_continue_after_dead_letter,
)
from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    SourceConfig,
    WriteResult,
)
from janus.strategies.api import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiTransport,
    ApiTransportError,
    AuthResolutionError,
    UrllibApiTransport,
    inject_auth,
)
from janus.strategies.base import BaseStrategy, SourceHook
from janus.strategies.common import (
    _compare_checkpoint_values,
    _default_storage_layout,
    _freeze_string_mapping,
    _max_checkpoint_value,
    _parse_datetime,
    _raw_run_path_prefix,
    _retry_delay_seconds,
)
from janus.strategies.files.formats import (
    ARCHIVE_FILE_SUFFIXES,
    SUPPORTED_FILE_INPUT_FORMATS,
    _filename_from_content_disposition,
    _filename_from_url,
    _infer_format_name,
    _is_tarball_filename,
    _safe_path_segment,
)
from janus.utils.environment import resolve_project_path
from janus.utils.logging import StructuredLogger, redact_url
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
CHECKSUM_HEADER_CANDIDATES = ("x-checksum-sha256", "x-amz-checksum-sha256")
VERSION_TOKEN_PATTERN = re.compile(r"(\d{4}(?:[-_]\d{2}(?:[-_]\d{2})?)?)")


class FileStrategyError(RuntimeError):
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


@dataclass(slots=True)
class FileRequestThrottle:
    """Single-threaded rate limiter reused for remote file downloads."""

    requests_per_minute: int | None
    clock: Callable[[], float]
    sleeper: Callable[[float], None]
    _next_allowed_at: float | None = None

    def wait_for_turn(self) -> None:
        if self.requests_per_minute is None:
            return

        interval_seconds = 60 / self.requests_per_minute
        now = self.clock()
        if self._next_allowed_at is not None and now < self._next_allowed_at:
            self.sleeper(self._next_allowed_at - now)
            now = self._next_allowed_at
        self._next_allowed_at = now + interval_seconds


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
        file_hook = hook if isinstance(hook, FileHook) else None
        storage_layout = self.storage_layout_factory(plan)
        checkpoint_state = self.checkpoint_store.load(plan)
        raw_path_prefix = _raw_run_path_prefix(plan)
        raw_writer = self.raw_writer_factory(storage_layout).with_raw_path_prefix(raw_path_prefix)
        logger = self._bind_logger(plan)
        throttle = FileRequestThrottle(
            requests_per_minute=plan.source_config.access.rate_limit.requests_per_minute,
            clock=self.clock,
            sleeper=self.sleeper,
        )
        dead_letter_max_items = plan.source_config.extraction.dead_letter_max_items
        resume = plan.run_context.attributes_as_dict().get("resume") == "true"
        dead_letter_state = self.dead_letter_store.load(plan) if resume else None
        if not resume:
            self.dead_letter_store.clear(plan)

        if logger is not None:
            logger.info(
                "file_extraction_started",
                strategy_variant=plan.source.strategy_variant,
                extraction_mode=plan.extraction_mode,
                checkpoint_strategy=plan.checkpoint_strategy,
                checkpoint_loaded=checkpoint_state is not None,
                checkpoint_value=(
                    checkpoint_state.checkpoint_value if checkpoint_state is not None else None
                ),
                access_format=plan.source_config.access.format,
                input_format=plan.source_config.spark.input_format,
                file_pattern=plan.source_config.access.file_pattern,
                timeout_seconds=plan.source_config.access.timeout_seconds,
                requests_per_minute=(
                    plan.source_config.access.rate_limit.requests_per_minute
                ),
                concurrency=plan.source_config.access.rate_limit.concurrency,
                dead_letter_max_items=dead_letter_max_items,
            )

        artifacts: list[ExtractedArtifact] = []
        persisted_file_count = 0
        archive_member_count = 0
        checksum_verified_count = 0
        skipped_file_count = 0
        total_attempts = 0
        loaded_file_count = 0
        checkpoint_value: str | None = None
        resolved_versions: list[str] = []
        dead_letter_keys = (
            set(dead_letter_state.item_keys) if dead_letter_state is not None else set()
        )
        dead_letter_skip_count = 0

        with ApiClient(self.transport_factory()) as client:
            discovered_files = self._discover_files(plan, file_hook, client.transport)
            selected_files = self._select_files(plan, discovered_files, checkpoint_state)
            if logger is not None:
                logger.info(
                    "file_discovery_finished",
                    discovered_file_count=len(discovered_files),
                    selected_file_count=len(selected_files),
                    checkpoint_loaded=checkpoint_state is not None,
                    discovered_source_kinds=sorted(
                        {file.source_kind for file in discovered_files}
                    ),
                    selected_versions=[file.version for file in selected_files],
                )

            for file_index, discovered_file in enumerate(selected_files, start=1):
                candidate_key = _discovered_file_dead_letter_key(discovered_file)
                if logger is not None:
                    logger.info(
                        "file_candidate_started",
                        file_index=file_index,
                        selected_file_count=len(selected_files),
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

                candidate_artifacts: list[ExtractedArtifact] = []
                candidate_loaded_file_count = 0
                candidate_total_attempts = 0
                candidate_archive_member_count = 0
                candidate_checksum_verified_count = 0

                try:
                    payload, response, attempts_used = self._load_payload(
                        plan,
                        discovered_file,
                        client=client,
                        throttle=throttle,
                        logger=logger,
                    )
                    candidate_total_attempts += attempts_used
                    candidate_loaded_file_count += 1
                    if logger is not None:
                        logger.info(
                            "file_payload_loaded",
                            file_index=file_index,
                            source_kind=discovered_file.source_kind,
                            filename=discovered_file.filename,
                            status_code=response.status_code if response is not None else None,
                            attempts_used=attempts_used,
                            payload_size_bytes=len(payload),
                        )

                    _validate_remote_payload(discovered_file, payload, response)

                    if file_hook is not None:
                        payload = file_hook.prepare_download(
                            plan,
                            discovered_file,
                            payload,
                            response=response,
                        )

                    resolved_filename = _resolved_filename(discovered_file.filename, response)
                    resolved_file = replace(
                        discovered_file,
                        filename=resolved_filename,
                        format=_infer_format_name(
                            resolved_filename,
                            fallback=plan.source_config.access.format,
                        ),
                    )
                    version = self._resolve_version(
                        plan,
                        resolved_file,
                        payload,
                        response,
                        file_hook,
                    )

                    if _should_skip_for_checkpoint(plan, checkpoint_state, version):
                        skipped_file_count += 1
                        if logger is not None:
                            logger.info(
                                "file_candidate_skipped",
                                file_index=file_index,
                                filename=resolved_filename,
                                resolved_version=version,
                                checkpoint_value=(
                                    checkpoint_state.checkpoint_value
                                    if checkpoint_state is not None
                                    else None
                                ),
                                reason="checkpoint_not_newer",
                            )
                        continue

                    expected_checksum = self._expected_checksum(
                        plan,
                        resolved_file,
                        response=response,
                        file_hook=file_hook,
                    )
                    actual_checksum = sha256(payload).hexdigest()
                    checksum_verified = expected_checksum is not None
                    if expected_checksum is not None:
                        if actual_checksum.lower() != expected_checksum.lower():
                            raise FileIntegrityError(
                                f"Checksum mismatch for {resolved_filename!r}: "
                                f"expected {expected_checksum.lower()} got "
                                f"{actual_checksum.lower()}"
                            )
                        candidate_checksum_verified_count += 1

                    is_archive = _is_archive_file(plan, resolved_file, payload)
                    artifact_format = (
                        "binary"
                        if is_archive
                        else _infer_handoff_format(
                            resolved_file,
                            fallback=plan.source_config.spark.input_format,
                        )
                    )
                    persisted_original = raw_writer.write_bytes(
                        plan,
                        _raw_download_relative_path(version, resolved_filename),
                        payload,
                        metadata={
                            "source_kind": resolved_file.source_kind,
                            "source_location": _redacted_location(resolved_file),
                            "resolved_version": version,
                            "attempt_count": str(attempts_used),
                        },
                    )
                    candidate_artifacts.append(
                        ExtractedArtifact(
                            path=persisted_original.artifact.path,
                            format=artifact_format,
                            checksum=persisted_original.artifact.checksum,
                        )
                    )
                    if logger is not None:
                        logger.info(
                            "file_payload_persisted",
                            file_index=file_index,
                            filename=resolved_filename,
                            source_kind=resolved_file.source_kind,
                            resolved_version=version,
                            artifact_path=persisted_original.artifact.path,
                            artifact_format=artifact_format,
                            checksum_verified=checksum_verified,
                            is_archive=is_archive,
                            payload_size_bytes=len(payload),
                        )

                    if is_archive:
                        extracted_artifacts = self._extract_archive(
                            plan,
                            raw_writer,
                            resolved_file,
                            payload,
                            version=version,
                            file_hook=file_hook,
                        )
                        candidate_artifacts.extend(extracted_artifacts)
                        candidate_archive_member_count += len(extracted_artifacts)
                        if logger is not None:
                            logger.info(
                                "file_archive_extracted",
                                file_index=file_index,
                                archive_filename=resolved_file.filename,
                                resolved_version=version,
                                archive_member_count=len(extracted_artifacts),
                                first_member_artifact_path=(
                                    extracted_artifacts[0].path if extracted_artifacts else None
                                ),
                            )
                except Exception as exc:
                    if logger is not None:
                        logger.exception("file_candidate_execution_failed")
                    dead_letter_state = self.dead_letter_store.record(
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
                            dead_letter_max_items=dead_letter_max_items,
                            error_type=type(exc).__name__,
                        )
                    if not can_continue_after_dead_letter(
                        total_item_count=len(selected_files),
                        dead_letter_count=dead_letter_state.entry_count,
                        dead_letter_max_items=dead_letter_max_items,
                    ):
                        raise
                    dead_letter_skip_count += 1
                    continue

                artifacts.extend(candidate_artifacts)
                total_attempts += candidate_total_attempts
                loaded_file_count += candidate_loaded_file_count
                checksum_verified_count += candidate_checksum_verified_count
                archive_member_count += candidate_archive_member_count
                persisted_file_count += 1
                resolved_versions.append(version)
                checkpoint_value = _max_checkpoint_value(checkpoint_value, version)

        normalization_candidate_count = sum(
            1 for artifact in artifacts if artifact.format == plan.source_config.spark.input_format
        )
        extraction_metadata = {
            "discovered_file_count": str(len(discovered_files)),
            "selected_file_count": str(len(selected_files)),
            "persisted_file_count": str(persisted_file_count),
            "archive_member_count": str(archive_member_count),
            "checksum_verified_count": str(checksum_verified_count),
            "skipped_file_count": str(skipped_file_count),
            "checkpoint_loaded": str(checkpoint_state is not None).lower(),
            "normalization_candidate_count": str(normalization_candidate_count),
        }
        if resolved_versions:
            extraction_metadata["resolved_versions"] = ",".join(resolved_versions)
        extraction_metadata["raw_path_prefix"] = str(raw_path_prefix or "")

        if logger is not None:
            logger.info(
                "file_extraction_finished",
                discovered_file_count=len(discovered_files),
                selected_file_count=len(selected_files),
                load_count=loaded_file_count,
                attempt_count=total_attempts,
                retry_count=max(total_attempts - loaded_file_count, 0),
                persisted_file_count=persisted_file_count,
                archive_member_count=archive_member_count,
                checksum_verified_count=checksum_verified_count,
                skipped_file_count=skipped_file_count,
                normalization_candidate_count=normalization_candidate_count,
                artifact_count=len(artifacts),
                checkpoint_value=checkpoint_value,
                dead_letter_count=(dead_letter_state.entry_count if dead_letter_state else 0),
                dead_letter_skipped_count=dead_letter_skip_count,
            )

        dead_letter_count = dead_letter_state.entry_count if dead_letter_state is not None else 0

        extraction_result = ExtractionResult.from_plan(
            plan,
            tuple(artifacts),
            records_extracted=normalization_candidate_count,
            checkpoint_value=checkpoint_value,
            metadata=extraction_metadata,
        )
        extraction_result = extraction_result.with_metadata(
            "dead_letter_count",
            str(dead_letter_count),
        ).with_metadata(
            "dead_letter_skipped_count",
            str(dead_letter_skip_count),
        )
        if dead_letter_count > 0:
            extraction_result = extraction_result.with_metadata(
                "dead_letter_path",
                str(self.dead_letter_store.path(plan)),
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

    def _discover_files(
        self,
        plan: ExecutionPlan,
        file_hook: FileHook | None,
        transport: ApiTransport,
    ) -> tuple[DiscoveredFile, ...]:
        from janus.strategies.files.resolvers import build_resolver_chain, resolve_link

        access = plan.source_config.access
        discovered: list[DiscoveredFile] = []

        if access.url:
            if file_hook is not None:
                resolved_from_url: Sequence[DiscoveredFile] = file_hook.resolve_links(
                    plan, access.url, access.format, transport
                )
            else:
                chain = build_resolver_chain(access.link_resolver)
                resolved_from_url = resolve_link(access.url, access.format, transport, chain)

            if resolved_from_url:
                discovered.extend(
                    _filter_discovered_files(
                        resolved_from_url,
                        access.remote_file_pattern,
                    )
                )
            else:
                # Fallback: treat URL as direct so the download loop can dead-letter it on failure
                discovered.extend(
                    _filter_discovered_files(
                        (
                            DiscoveredFile(
                                source_kind="remote",
                                location=access.url,
                                filename=_filename_from_url(access.url),
                                format=_infer_format_name(
                                    access.url,
                                    fallback=access.format,
                                ),
                            ),
                        ),
                        access.remote_file_pattern,
                    )
                )

        if access.path:
            discovered.extend(self._discover_local_path(plan, access.path, access.file_pattern))

        if access.discovery_pattern:
            discovered.extend(self._discover_local_pattern(plan, access.discovery_pattern))

        deduplicated = {
            (item.source_kind, item.location): item
            for item in sorted(discovered, key=lambda item: (item.source_kind, item.location))
        }
        resolved = tuple(deduplicated.values())
        if file_hook is not None:
            resolved = tuple(file_hook.discovered_files(plan, resolved))
        return resolved

    def _discover_local_path(
        self,
        plan: ExecutionPlan,
        configured_path: str,
        file_pattern: str | None,
    ) -> list[DiscoveredFile]:
        path = resolve_project_path(plan.run_context.project_root, configured_path)
        if not path.exists():
            raise FileDiscoveryError(f"Configured file path does not exist: {path}")

        if path.is_file():
            return [self._local_discovered_file(plan, path)]

        matched_paths = (
            sorted(file_path for file_path in path.glob(file_pattern) if file_path.is_file())
            if file_pattern
            else sorted(file_path for file_path in path.iterdir() if file_path.is_file())
        )
        return [self._local_discovered_file(plan, file_path) for file_path in matched_paths]

    def _discover_local_pattern(
        self,
        plan: ExecutionPlan,
        discovery_pattern: str,
    ) -> list[DiscoveredFile]:
        expanded_pattern = str(
            resolve_project_path(plan.run_context.project_root, discovery_pattern)
        )
        matched_paths = sorted(
            Path(match)
            for match in glob.glob(expanded_pattern, recursive=True)
            if Path(match).is_file()
        )
        return [self._local_discovered_file(plan, file_path) for file_path in matched_paths]

    def _local_discovered_file(self, plan: ExecutionPlan, file_path: Path) -> DiscoveredFile:
        stat = file_path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        version = _default_discovered_version(
            plan,
            file_path.name,
            modified_at=modified_at,
        )
        return DiscoveredFile(
            source_kind="local",
            location=str(file_path.resolve()),
            filename=file_path.name,
            format=_infer_format_name(file_path.name, fallback=plan.source_config.access.format),
            version=version,
            size_bytes=stat.st_size,
            modified_at=modified_at,
        )

    def _select_files(
        self,
        plan: ExecutionPlan,
        discovered_files: Sequence[DiscoveredFile],
        checkpoint_state: CheckpointState | None,
    ) -> tuple[DiscoveredFile, ...]:
        if not discovered_files:
            return ()

        if plan.source.strategy_variant == "static_file":
            return tuple(discovered_files)

        if len(discovered_files) > 1 and any(item.version is None for item in discovered_files):
            raise FileDiscoveryError(
                "Versioned file selection requires a deterministic version for each discovered "
                "candidate when more than one file is present"
            )

        ordered_files = tuple(sorted(discovered_files, key=_version_sort_key))
        if checkpoint_state is not None:
            ordered_files = tuple(
                item
                for item in ordered_files
                if item.version is None
                or _compare_checkpoint_values(item.version, checkpoint_state.checkpoint_value) > 0
            )

        if plan.extraction_mode == "incremental":
            return ordered_files
        if not ordered_files:
            return ()
        return (ordered_files[-1],)

    def _load_payload(
        self,
        plan: ExecutionPlan,
        discovered_file: DiscoveredFile,
        *,
        client: ApiClient,
        throttle: FileRequestThrottle,
        logger: StructuredLogger | None,
    ) -> tuple[bytes, ApiResponse | None, int]:
        if discovered_file.source_kind == "local":
            return Path(discovered_file.location).read_bytes(), None, 1

        request = ApiRequest(
            method=plan.source_config.access.method,
            url=discovered_file.location,
            timeout_seconds=plan.source_config.access.timeout_seconds,
            headers=_freeze_string_mapping(plan.source_config.access.headers or {}),
            params=_freeze_string_mapping(plan.source_config.access.params or {}),
        )
        request = inject_auth(
            request,
            plan.source_config.access.auth,
            env_reader=self._resolve_env_var,
        )
        response, attempts_used = self._send_with_retries(
            plan,
            client,
            request,
            throttle,
            logger,
        )
        return response.body, response, attempts_used

    def _send_with_retries(
        self,
        plan: ExecutionPlan,
        client: ApiClient,
        request: ApiRequest,
        throttle: FileRequestThrottle,
        logger: StructuredLogger | None,
    ) -> tuple[ApiResponse, int]:
        retry_config = plan.source_config.extraction.retry
        last_error: Exception | None = None

        for attempt in range(1, retry_config.max_attempts + 1):
            throttle.wait_for_turn()
            try:
                response = client.send(request)
            except (ApiTransportError, AuthResolutionError) as exc:
                last_error = exc
                if attempt == retry_config.max_attempts:
                    raise FileDownloadError(str(exc)) from exc
                self._sleep_for_retry(plan, attempt, response=None, logger=logger)
                continue

            if 200 <= response.status_code < 300:
                return response, attempt

            if (
                response.status_code not in RETRYABLE_STATUS_CODES
                or attempt == retry_config.max_attempts
            ):
                raise FileDownloadError(
                    "File request failed with status "
                    f"{response.status_code} for {redact_url(response.request.full_url())}"
                )

            self._sleep_for_retry(plan, attempt, response=response, logger=logger)

        if last_error is not None:
            raise FileDownloadError(str(last_error)) from last_error
        raise FileDownloadError("Retry loop exited without a response or error")

    def _sleep_for_retry(
        self,
        plan: ExecutionPlan,
        attempt: int,
        *,
        response: ApiResponse | None,
        logger: StructuredLogger | None,
    ) -> None:
        delay = _retry_delay_seconds(plan, attempt, response)
        if logger is not None:
            logger.warning(
                "file_retry_scheduled",
                attempt=attempt,
                delay_seconds=delay,
                status_code=response.status_code if response is not None else None,
            )
        self.sleeper(delay)

    def _resolve_version(
        self,
        plan: ExecutionPlan,
        discovered_file: DiscoveredFile,
        payload: bytes,
        response: ApiResponse | None,
        file_hook: FileHook | None,
    ) -> str:
        if file_hook is not None:
            hook_version = file_hook.resolve_version(plan, discovered_file)
            if hook_version is not None and hook_version.strip():
                return hook_version.strip()

        if discovered_file.version is not None:
            return discovered_file.version

        if response is not None:
            headers = {
                key.lower(): value.strip()
                for key, value in response.headers_as_dict().items()
                if isinstance(value, str) and value.strip()
            }
            etag = headers.get("etag")
            if etag:
                return etag.strip('"')
            last_modified = headers.get("last-modified")
            if last_modified:
                return last_modified

        if plan.source.strategy_variant == "static_file":
            return "current"
        return sha256(payload).hexdigest()

    def _expected_checksum(
        self,
        plan: ExecutionPlan,
        discovered_file: DiscoveredFile,
        *,
        response: ApiResponse | None,
        file_hook: FileHook | None,
    ) -> str | None:
        if file_hook is not None:
            hook_checksum = file_hook.expected_checksum(plan, discovered_file)
            if hook_checksum:
                return hook_checksum.strip()

        if discovered_file.source_kind == "local":
            sidecar = Path(discovered_file.location).with_name(f"{discovered_file.filename}.sha256")
            if sidecar.exists():
                return _read_checksum_sidecar(sidecar)

        if response is None:
            return None

        headers = {key.lower(): value for key, value in response.headers_as_dict().items()}
        for header_name in CHECKSUM_HEADER_CANDIDATES:
            value = headers.get(header_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _extract_archive(
        self,
        plan: ExecutionPlan,
        raw_writer: RawArtifactWriter,
        archive_file: DiscoveredFile,
        payload: bytes,
        *,
        version: str,
        file_hook: FileHook | None,
    ) -> tuple[ExtractedArtifact, ...]:
        member_payloads = _archive_member_payloads(payload, archive_file.filename)
        members = tuple(
            DiscoveredFile(
                source_kind="archive",
                location=member_name,
                filename=PurePosixPath(member_name).name,
                format=_infer_format_name(member_name, fallback="binary"),
                version=version,
            )
            for member_name in member_payloads
        )
        members = _filter_members(members, plan.source_config.access.file_pattern)
        if file_hook is not None:
            members = tuple(file_hook.archive_members(plan, archive_file, members))

        extracted_artifacts: list[ExtractedArtifact] = []
        for member in members:
            member_payload = member_payloads.get(member.location)
            if member_payload is None:
                raise ArchiveExtractionError(
                    f"Archive member {member.location!r} was selected but is not available"
                )
            persisted = raw_writer.write_bytes(
                plan,
                _raw_extracted_relative_path(version, archive_file.filename, member.location),
                member_payload,
                metadata={
                    "archive_filename": archive_file.filename,
                    "archive_member": member.location,
                    "resolved_version": version,
                },
            )
            extracted_artifacts.append(
                ExtractedArtifact(
                    path=persisted.artifact.path,
                    format=_infer_handoff_format(
                        member,
                        fallback=plan.source_config.spark.input_format,
                    ),
                    checksum=persisted.artifact.checksum,
                )
            )
        return tuple(extracted_artifacts)

    def _bind_logger(self, plan: ExecutionPlan) -> StructuredLogger | None:
        if self.logger is None:
            return None
        return self.logger.bind(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            strategy_family=self.strategy_family,
        )

    def _resolve_env_var(self, name: str) -> str | None:
        value = self.env_reader(name)
        if value is not None:
            return value
        return os.getenv(name)


def _raw_download_relative_path(version: str, filename: str) -> Path:
    return Path("downloads") / _safe_path_segment(version) / _safe_filename(filename)


def _raw_extracted_relative_path(
    version: str,
    archive_filename: str,
    member_location: str,
) -> Path:
    return (
        Path("extracted")
        / _safe_path_segment(version)
        / _safe_path_segment(Path(archive_filename).stem or archive_filename)
        / _safe_archive_member_path(member_location)
    )


def _safe_archive_member_path(member_location: str) -> Path:
    pure_path = PurePosixPath(member_location)
    if pure_path.is_absolute():
        raise ArchiveExtractionError("Archive members must not use absolute paths")
    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ArchiveExtractionError("Archive members must not contain unsafe path segments")
    return Path(*pure_path.parts)


def _safe_filename(value: str) -> str:
    filename = Path(value).name.strip()
    if not filename:
        raise FileDiscoveryError("Resolved filename must not be empty")
    return filename


def _archive_member_payloads(payload: bytes, filename: str = "") -> dict[str, bytes]:
    if _is_tarball_filename(filename):
        return _tarball_member_payloads(payload)
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members: dict[str, bytes] = {}
            for member_name in archive.namelist():
                if member_name.endswith("/"):
                    continue
                safe_member_path = _safe_archive_member_path(member_name)
                with archive.open(member_name) as stream:
                    members[str(PurePosixPath(*safe_member_path.parts))] = stream.read()
            return members
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ArchiveExtractionError("Could not extract ZIP archive payload") from exc


def _tarball_member_payloads(payload: bytes) -> dict[str, bytes]:
    try:
        with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
            members: dict[str, bytes] = {}
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                safe_member_path = _safe_archive_member_path(member.name)
                f = archive.extractfile(member)
                if f is not None:
                    members[str(PurePosixPath(*safe_member_path.parts))] = f.read()
            return members
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveExtractionError("Could not extract tar.gz archive payload") from exc


def _filter_discovered_files(
    files: Sequence[DiscoveredFile],
    file_pattern: str | None,
) -> tuple[DiscoveredFile, ...]:
    if not file_pattern:
        return tuple(files)
    return tuple(
        item
        for item in files
        if fnmatch.fnmatch(item.location, file_pattern)
        or fnmatch.fnmatch(item.filename, file_pattern)
    )


def _filter_members(
    members: Sequence[DiscoveredFile],
    file_pattern: str | None,
) -> tuple[DiscoveredFile, ...]:
    return _filter_discovered_files(members, file_pattern)


def _is_archive_file(
    plan: ExecutionPlan,
    discovered_file: DiscoveredFile,
    payload: bytes,
) -> bool:
    if plan.source.strategy_variant == "archive_package":
        return True
    if _is_tarball_filename(discovered_file.filename):
        try:
            return tarfile.is_tarfile(BytesIO(payload))
        except (OSError, tarfile.TarError):
            return False
    suffix = Path(discovered_file.filename).suffix.lower()
    return suffix in ARCHIVE_FILE_SUFFIXES and zipfile.is_zipfile(BytesIO(payload))


def _resolved_filename(filename: str, response: ApiResponse | None) -> str:
    if response is None:
        return _safe_filename(filename)

    content_disposition = response.headers_as_dict().get("Content-Disposition")
    if isinstance(content_disposition, str):
        cd_name = _filename_from_content_disposition(content_disposition)
        if cd_name is not None:
            return _safe_filename(cd_name)
    return _safe_filename(filename)


def _validate_remote_payload(
    discovered_file: DiscoveredFile,
    payload: bytes,
    response: ApiResponse | None,
) -> None:
    if discovered_file.source_kind != "remote" or response is None:
        return

    content_type = response.headers_as_dict().get("Content-Type", "")
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type in {"text/html", "application/xhtml+xml"}:
        raise FileDownloadError(
            f"Expected a file payload for {discovered_file.filename!r} but received "
            f"{normalized_content_type or 'HTML'} from "
            f"{redact_url(response.request.full_url())}"
        )

    if (
        not payload
        and discovered_file.size_bytes is not None
        and discovered_file.size_bytes > 0
    ):
        raise FileDownloadError(
            f"Expected a non-empty payload for {discovered_file.filename!r} "
            f"({discovered_file.size_bytes} bytes discovered) but received 0 bytes"
        )


def _infer_handoff_format(discovered_file: DiscoveredFile, *, fallback: str) -> str:
    format_name = _infer_format_name(discovered_file.filename, fallback=fallback)
    if format_name == "binary" and fallback in SUPPORTED_FILE_INPUT_FORMATS:
        return fallback
    return format_name


def _default_discovered_version(
    plan: ExecutionPlan,
    filename: str,
    *,
    modified_at: datetime,
) -> str | None:
    if plan.source.strategy_variant == "static_file":
        return "current"

    version_match = VERSION_TOKEN_PATTERN.findall(filename)
    if version_match:
        return version_match[-1].replace("_", "-")
    return modified_at.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _version_sort_key(discovered_file: DiscoveredFile) -> tuple[str, Any]:
    version = discovered_file.version
    if version is None:
        return ("missing", discovered_file.location)

    normalized = version.strip()
    parsed_datetime = _parse_datetime(normalized)
    if parsed_datetime is not None:
        return ("datetime", parsed_datetime.astimezone(UTC))

    try:
        return ("decimal", Decimal(normalized))
    except InvalidOperation:
        return ("text", normalized)


def _should_skip_for_checkpoint(
    plan: ExecutionPlan,
    checkpoint_state: CheckpointState | None,
    candidate_version: str,
) -> bool:
    if checkpoint_state is None or plan.extraction_mode != "incremental":
        return False
    return _compare_checkpoint_values(candidate_version, checkpoint_state.checkpoint_value) <= 0


def _read_checksum_sidecar(path: Path) -> str:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise FileIntegrityError(f"Checksum sidecar is empty: {path}")
    return content.split()[0]


def _redacted_location(discovered_file: DiscoveredFile) -> str:
    if discovered_file.source_kind == "remote":
        return redact_url(discovered_file.location)
    return discovered_file.location


def _discovered_file_dead_letter_key(discovered_file: DiscoveredFile) -> str:
    return f"{discovered_file.source_kind}:{discovered_file.location}"


def _discovered_file_dead_letter_metadata(
    discovered_file: DiscoveredFile,
    *,
    file_index: int,
    selected_file_count: int,
) -> dict[str, str]:
    return {
        "file_index": str(file_index),
        "selected_file_count": str(selected_file_count),
        "source_kind": discovered_file.source_kind,
        "source_location": _redacted_location(discovered_file),
        "filename": discovered_file.filename,
    }
