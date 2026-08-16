"""Rebuild an ``ExtractionResult`` from a raw zone written by a previous run.

The mirror image of the strategies' raw *write* path: these functions infer artifact
identity, format, ordering and (for the file family) archive structure from the on-disk
layout produced by ``strategies/files/artifacts.py``, ``strategies/catalog/artifacts.py``
and ``strategies/api/artifacts.py``.

**That layout is a compatibility surface.** Changing a raw path template on the write
side without changing it here silently breaks ``--ingest-raw-to-bronze`` for every zone
written before the change — and the failure surfaces as "no artifacts found", not as an
error.

Rehydration is filesystem work: it never starts a Spark session of its own. The single
exception is inherited, not introduced here — an ``iceberg_rows`` request input has to be
read to know which contexts a catalog replay must walk, and that read goes through the
same ``scoped_request_input_session`` seam the live strategies use, which opens and closes
a session around that one call and nothing else.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult
from janus.planner import PlannedRun
from janus.runtime.spark_lifecycle import SparkSessionProvider, scoped_request_input_session
from janus.scripts.checksums import _artifact_format_for_path, _resolve_raw_checksum
from janus.scripts.replay_plan import _freeze_string_mapping, _plan_with_active_raw_root
from janus.strategies.api import ApiRequest, build_paginator
from janus.strategies.api.request_inputs import load_request_inputs
from janus.strategies.catalog.artifacts import (
    _persist_normalized_records,
    _rediscover_catalog_input_artifacts,
    _replay_catalog_entities_from_dir,
)
from janus.strategies.catalog.core import (
    ENTITY_TYPE_ORDER,
    CatalogStrategy,
)
from janus.strategies.catalog.metadata import _apply_per_input_params
from janus.strategies.http import resolve_url
from janus.utils.storage import StorageLayout
from janus.writers import SIDECAR_SUFFIX, RawArtifactWriter

_VERSIONED_DOWNLOAD_PATH_PARTS = 3


def _build_extraction_result_from_raw(
    planned_run: PlannedRun,
    plan: ExecutionPlan,
    spark_provider: SparkSessionProvider,
    storage_layout: StorageLayout,
) -> ExtractionResult:
    if isinstance(planned_run.strategy, CatalogStrategy):
        return _build_catalog_extraction_result_from_raw(
            planned_run,
            plan,
            spark_provider,
            storage_layout,
        )

    raw_plan = _plan_with_active_raw_root(plan)
    raw_artifacts = _rediscover_raw_artifacts(raw_plan)
    raw_artifacts = _rehydrate_file_raw_artifacts(
        planned_run,
        raw_plan,
        raw_artifacts,
        storage_layout,
    )
    return ExtractionResult.from_plan(
        plan,
        raw_artifacts,
        metadata={
            "raw_to_bronze": "true",
            "rediscovered_raw_artifact_count": str(len(raw_artifacts)),
            "raw_artifact_root": raw_plan.raw_output.path,
        },
    )


def _rediscover_raw_artifacts(
    plan: ExecutionPlan, *, verify_checksums: bool = False
) -> tuple[ExtractedArtifact, ...]:
    raw_root = Path(plan.raw_output.path)
    if not raw_root.exists():
        raise FileNotFoundError(f"Configured raw output path does not exist: {raw_root}")

    artifacts = tuple(
        ExtractedArtifact(
            path=str(path),
            format=_artifact_format_for_path(path, fallback=plan.source_config.spark.input_format),
            checksum=_resolve_raw_checksum(path, verify=verify_checksums),
        )
        for path in sorted(
            candidate
            for candidate in raw_root.rglob("*")
            if candidate.is_file() and candidate.suffix != SIDECAR_SUFFIX
        )
    )
    if not artifacts:
        raise FileNotFoundError(f"No raw artifacts were found under {raw_root}")
    return artifacts


def _rehydrate_file_raw_artifacts(
    planned_run: PlannedRun,
    plan: ExecutionPlan,
    raw_artifacts: tuple[ExtractedArtifact, ...],
    storage_layout: StorageLayout,
) -> tuple[ExtractedArtifact, ...]:
    if getattr(planned_run.strategy, "strategy_family", None) != "file":
        return raw_artifacts

    from janus.strategies.files.core import (
        DiscoveredFile,
        FileHook,
        _archive_member_payloads,
        _filter_members,
        _infer_handoff_format,
        _raw_extracted_relative_path,
    )

    artifacts_by_path = {artifact.path: artifact for artifact in raw_artifacts}
    raw_writer = RawArtifactWriter(storage_layout)
    file_hook = planned_run.hook if isinstance(planned_run.hook, FileHook) else None
    raw_root = Path(plan.raw_output.path)

    for artifact in raw_artifacts:
        artifact_path = Path(artifact.path)
        if not _is_archive_download_path(raw_root, artifact_path):
            continue

        archive_file = DiscoveredFile(
            source_kind="local",
            location=str(artifact_path),
            filename=artifact_path.name,
            format="binary",
            version=_download_version(raw_root, artifact_path),
        )
        member_payloads = _archive_member_payloads(
            artifact_path.read_bytes(),
            archive_file.filename,
        )
        members = tuple(
            DiscoveredFile(
                source_kind="archive",
                location=member_name,
                filename=PurePosixPath(member_name).name,
                format=_artifact_format_for_path(
                    Path(member_name),
                    fallback=plan.source_config.spark.input_format,
                ),
                version=archive_file.version,
            )
            for member_name in member_payloads
        )
        members = _filter_members(members, plan.source_config.access.file_pattern)
        if file_hook is not None:
            members = tuple(file_hook.archive_members(plan, archive_file, members))

        for member in members:
            member_payload = member_payloads.get(member.location)
            if member_payload is None:
                continue
            persisted = raw_writer.write_bytes(
                plan,
                _raw_extracted_relative_path(
                    archive_file.version or "current",
                    archive_file.filename,
                    member.location,
                ),
                member_payload,
                mode="ignore",
                metadata={
                    "archive_filename": archive_file.filename,
                    "archive_member": member.location,
                    "resolved_version": archive_file.version or "current",
                },
            )
            artifacts_by_path[str(persisted.artifact.path)] = ExtractedArtifact(
                path=str(persisted.artifact.path),
                format=_infer_handoff_format(
                    member,
                    fallback=plan.source_config.spark.input_format,
                ),
                checksum=persisted.artifact.checksum,
            )

    return tuple(sorted(artifacts_by_path.values(), key=lambda artifact: artifact.path))


def _is_archive_download_path(raw_root: Path, artifact_path: Path) -> bool:
    try:
        relative_path = artifact_path.relative_to(raw_root)
    except ValueError:
        return False
    if (
        len(relative_path.parts) < _VERSIONED_DOWNLOAD_PATH_PARTS
        or relative_path.parts[0] != "downloads"
    ):
        return False
    lower_name = artifact_path.name.lower()
    return lower_name.endswith((".zip", ".tar.gz", ".tgz"))


def _download_version(raw_root: Path, artifact_path: Path) -> str:
    try:
        relative_path = artifact_path.relative_to(raw_root)
    except ValueError:
        return "current"
    if (
        len(relative_path.parts) >= _VERSIONED_DOWNLOAD_PATH_PARTS
        and relative_path.parts[0] == "downloads"
    ):
        return relative_path.parts[1]
    return "current"


def _build_catalog_extraction_result_from_raw(
    planned_run: PlannedRun,
    plan: ExecutionPlan,
    spark_provider: SparkSessionProvider,
    storage_layout: StorageLayout,
) -> ExtractionResult:
    strategy = planned_run.strategy
    assert isinstance(strategy, CatalogStrategy)

    # Same scoped seam the live catalog strategy uses: only `iceberg_rows` inputs
    # open a session here, and it is stopped before the replay walk begins.
    request_inputs_config = plan.source_config.access.request_inputs
    with scoped_request_input_session(spark_provider, request_inputs_config) as session:
        request_inputs = tuple(load_request_inputs(request_inputs_config, spark=session))
    if not request_inputs:
        request_inputs = (None,)

    raw_plan = _plan_with_active_raw_root(plan)

    raw_artifacts = _rediscover_catalog_raw_artifacts(
        raw_plan,
        storage_layout,
        request_input_count=len(request_inputs),
    )
    if not raw_artifacts:
        raise FileNotFoundError(f"No raw artifacts were found under {raw_plan.raw_output.path}")

    normalized_records: dict[str, list[dict[str, Any]]] = {
        entity_type: [] for entity_type in ENTITY_TYPE_ORDER
    }
    entity_indexes: dict[tuple[str, str], int] = {}
    paginator = build_paginator(plan.source_config.access.pagination)
    base_request = _catalog_base_request(plan)
    checkpoint_value: str | None = None

    for request_input_index, request_input in enumerate(request_inputs, start=1):
        per_input_request = _apply_per_input_params(
            base_request,
            plan.source_config.access.parameter_bindings,
            request_input,
        )
        checkpoint_value = _replay_catalog_entities_from_dir(
            raw_plan,
            storage_layout,
            per_input_request,
            paginator,
            request_input_index,
            len(request_inputs),
            checkpoint_state=None,
            normalized_records=normalized_records,
            entity_indexes=entity_indexes,
            current_checkpoint_value=checkpoint_value,
        )

    raw_writer = RawArtifactWriter(storage_layout)
    normalized_artifacts = _persist_normalized_records(
        raw_plan,
        raw_writer,
        normalized_records,
    )
    all_artifacts = tuple(raw_artifacts + tuple(normalized_artifacts))
    records_extracted = sum(len(records) for records in normalized_records.values())
    normalized_artifact_count = len(normalized_artifacts)

    return ExtractionResult.from_plan(
        plan,
        all_artifacts,
        records_extracted=records_extracted,
        checkpoint_value=checkpoint_value,
        metadata={
            "raw_to_bronze": "true",
            "rediscovered_raw_artifact_count": str(len(raw_artifacts)),
            "normalized_artifact_count": str(normalized_artifact_count),
            "raw_artifact_root": raw_plan.raw_output.path,
            "organizations_extracted": str(len(normalized_records["organization"])),
            "groups_extracted": str(len(normalized_records["group"])),
            "datasets_extracted": str(len(normalized_records["dataset"])),
            "resources_extracted": str(len(normalized_records["resource"])),
        },
    )


def _rediscover_catalog_raw_artifacts(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    *,
    request_input_count: int,
) -> tuple[ExtractedArtifact, ...]:
    artifacts: list[ExtractedArtifact] = []
    for request_input_index in range(1, request_input_count + 1):
        artifacts.extend(
            _rediscover_catalog_input_artifacts(
                plan,
                storage_layout,
                request_input_index,
                request_input_count,
            )
        )

    if artifacts:
        return tuple(artifacts)

    return _rediscover_raw_artifacts(plan)


def _catalog_base_request(plan: ExecutionPlan) -> ApiRequest:
    source_access = plan.source_config.access
    return ApiRequest(
        method=source_access.method,
        url=resolve_url(plan.source_config, family_label="Catalog"),
        timeout_seconds=source_access.timeout_seconds,
        headers=_freeze_string_mapping(source_access.headers or {}),
        params=_freeze_string_mapping(source_access.params or {}),
    )
