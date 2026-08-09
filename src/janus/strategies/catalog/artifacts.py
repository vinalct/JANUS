"""Where a catalog page lands in the raw zone, how it is found again, and what is written back."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from janus.checkpoints import CheckpointState
from janus.models import ExecutionPlan, ExtractedArtifact
from janus.strategies.api import PaginationState
from janus.strategies.catalog.document import (
    CATALOG_EDGES_FILE,
    CATALOG_NODES_FILE,
    ENTITY_FILE_NAMES,
    ENTITY_TYPE_ORDER,
    CatalogParseSummary,
    _build_generic_catalog_edge,
    _build_generic_catalog_node,
    _compute_parse_summary,
)
from janus.strategies.catalog.entities import _collect_catalog_entities
from janus.strategies.common import _raw_page_path
from janus.strategies.http import ApiRequest, ApiResponse
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter


def _raw_relative_path(
    pagination_state: PaginationState,
    *,
    request_input_index: int = 1,
    request_input_count: int = 1,
) -> Path:
    return _raw_page_path(
        pagination_state,
        ".json",
        request_input_index=request_input_index,
        request_input_count=request_input_count,
    )


def _normalized_relative_path(entity_type: str) -> Path:
    return Path("normalized") / f"{ENTITY_FILE_NAMES[entity_type]}.jsonl"


def _catalog_input_dir(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    request_input_index: int,
    request_input_count: int,
) -> Path:
    raw_dir = storage_layout.resolve_output(plan, "raw").resolved_path
    if request_input_count > 1:
        return raw_dir / f"request-input-{request_input_index:06d}"
    return raw_dir / "pages"


def _sorted_catalog_pages(directory: Path) -> list[Path]:
    """Return all catalog page JSON files sorted by sequence number."""
    if not directory.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("*.json"):
        stem = path.stem
        for prefix, start in (
            ("page-", 5),
            ("offset-", 7),
            ("cursor-", 7),
            ("response-", 9),
        ):
            if stem.startswith(prefix):
                with contextlib.suppress(ValueError):
                    candidates.append((int(stem[start:]), path))
                break
    return [p for _, p in sorted(candidates)]


def _catalog_pagination_state_from_path(path: Path) -> PaginationState:
    """Reconstruct a PaginationState from a raw page filename."""
    stem = path.stem
    if stem.startswith("page-"):
        try:
            num = int(stem[5:])
            return PaginationState(request_index=num, page_number=num)
        except ValueError:
            pass
    if stem.startswith("offset-"):
        try:
            return PaginationState(request_index=1, offset=int(stem[7:]))
        except ValueError:
            pass
    if stem.startswith("cursor-"):
        try:
            return PaginationState(request_index=int(stem[7:]), cursor="")
        except ValueError:
            pass
    if stem.startswith("response-"):
        try:
            return PaginationState(request_index=int(stem[9:]))
        except ValueError:
            pass
    return PaginationState(request_index=1)


def _rediscover_catalog_raw_artifacts(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    progress: dict[str, Any],
    request_input_index: int = 1,
    request_input_count: int = 1,
) -> list[ExtractedArtifact]:
    """Re-discover raw JSON artifacts written by a previous partial catalog run."""
    directory = _catalog_input_dir(plan, storage_layout, request_input_index, request_input_count)
    if not directory.exists():
        return []

    last_page = progress.get("last_page_number")
    last_offset = progress.get("last_offset")
    artifacts: list[ExtractedArtifact] = []

    if last_page is not None:
        candidates: list[tuple[int, Path]] = []
        for path in directory.glob("page-*.json"):
            stem = path.stem
            try:
                num = int(stem[5:])
            except ValueError:
                continue
            if num <= last_page:
                candidates.append((num, path))
        for _, path in sorted(candidates):
            checksum = sha256(path.read_bytes()).hexdigest()
            artifacts.append(ExtractedArtifact(path=str(path), format="json", checksum=checksum))

    elif last_offset is not None:
        candidates = []
        for path in directory.glob("offset-*.json"):
            stem = path.stem
            try:
                num = int(stem[7:])
            except ValueError:
                continue
            if num <= last_offset:
                candidates.append((num, path))
        for _, path in sorted(candidates):
            checksum = sha256(path.read_bytes()).hexdigest()
            artifacts.append(ExtractedArtifact(path=str(path), format="json", checksum=checksum))

    return artifacts


def _rediscover_catalog_input_artifacts(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    request_input_index: int,
    request_input_count: int,
) -> list[ExtractedArtifact]:
    """Re-discover all raw JSON artifacts for a fully completed catalog input."""
    artifacts: list[ExtractedArtifact] = []
    for path in _sorted_catalog_pages(
        _catalog_input_dir(plan, storage_layout, request_input_index, request_input_count)
    ):
        checksum = sha256(path.read_bytes()).hexdigest()
        artifacts.append(ExtractedArtifact(path=str(path), format="json", checksum=checksum))
    return artifacts


def _replay_catalog_entities_from_dir(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    per_input_request: ApiRequest,
    paginator: Any,
    request_input_index: int,
    request_input_count: int,
    checkpoint_state: CheckpointState | None,
    normalized_records: dict[str, list[dict[str, Any]]],
    entity_indexes: dict[tuple[str, str], int],
    current_checkpoint_value: str | None,
) -> str | None:
    """Re-collect catalog entities from the raw JSON files of a completed input."""
    directory = _catalog_input_dir(plan, storage_layout, request_input_index, request_input_count)
    checkpoint_value = current_checkpoint_value
    for path in _sorted_catalog_pages(directory):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pagination_state = _catalog_pagination_state_from_path(path)
        request = paginator.apply(per_input_request, pagination_state)
        file_mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        response = ApiResponse(
            request=request,
            status_code=200,
            body=b"",
            received_at=file_mtime,
        )
        checksum = sha256(path.read_bytes()).hexdigest()
        raw_artifact = ExtractedArtifact(path=str(path), format="json", checksum=checksum)
        checkpoint_value = _collect_catalog_entities(
            plan,
            payload=payload,
            request=request,
            response=response,
            pagination_state=pagination_state,
            raw_artifact=raw_artifact,
            checkpoint_state=checkpoint_state,
            normalized_records=normalized_records,
            entity_indexes=entity_indexes,
            current_checkpoint_value=checkpoint_value,
        )
    return checkpoint_value


def _persist_normalized_records(
    plan: ExecutionPlan,
    raw_writer: RawArtifactWriter,
    normalized_records: Mapping[str, Sequence[dict[str, Any]]],
) -> list[ExtractedArtifact]:
    artifacts: list[ExtractedArtifact] = []
    for entity_type in ENTITY_TYPE_ORDER:
        records = normalized_records[entity_type]
        if not records:
            continue
        persisted = raw_writer.write_json_lines(
            plan,
            _normalized_relative_path(entity_type),
            records,
            metadata={
                "entity_type": entity_type,
                "record_count": str(len(records)),
            },
        )
        artifacts.append(persisted.artifact)
    return artifacts


def _persist_generic_artifacts(
    plan: ExecutionPlan,
    raw_writer: RawArtifactWriter,
    normalized_records: Mapping[str, Sequence[dict[str, Any]]],
) -> tuple[list[ExtractedArtifact], CatalogParseSummary]:
    all_records = [
        record
        for entity_type in ENTITY_TYPE_ORDER
        for record in normalized_records[entity_type]
    ]
    if not all_records:
        return [], _compute_parse_summary([], [])

    path_by_key: dict[str, str] = {
        record["entity_key"]: record["catalog_record_path"]
        for record in all_records
        if record.get("entity_key")
    }

    variant = plan.source.strategy_variant
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for record in all_records:
        parent_key = record.get("parent_entity_key")
        parent_node_path = path_by_key.get(parent_key) if parent_key else None
        nodes.append(
            _build_generic_catalog_node(
                record,
                parent_node_path=parent_node_path,
                variant=variant,
            )
        )
        if parent_key:
            edges.append(_build_generic_catalog_edge(record))

    parse_summary = _compute_parse_summary(nodes, edges)

    artifacts: list[ExtractedArtifact] = []
    persisted = raw_writer.write_json_lines(
        plan,
        Path("normalized") / f"{CATALOG_NODES_FILE}.jsonl",
        nodes,
        metadata={"record_count": str(len(nodes))},
    )
    artifacts.append(persisted.artifact)
    if edges:
        persisted = raw_writer.write_json_lines(
            plan,
            Path("normalized") / f"{CATALOG_EDGES_FILE}.jsonl",
            edges,
            metadata={"record_count": str(len(edges))},
        )
        artifacts.append(persisted.artifact)
    return artifacts, parse_summary
