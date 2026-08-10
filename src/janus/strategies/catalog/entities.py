"""Catalog entity normalization and merge helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from janus.checkpoints import CheckpointState
from janus.models import ExecutionPlan, ExtractedArtifact
from janus.strategies.api import PaginationState
from janus.strategies.catalog.document import (
    ENTITY_TYPE_ORDER,
    CatalogEntityReference,
    _build_entity_reference,
    _nested_batches,
    _root_batches,
)
from janus.strategies.common import (
    _compare_checkpoint_values,
    _max_checkpoint_value,
)
from janus.strategies.http import ApiRequest, ApiResponse


def _empty_entity_records() -> dict[str, list[dict[str, Any]]]:
    """One empty bucket per generic entity type, in the contract's order."""
    return {entity_type: [] for entity_type in ENTITY_TYPE_ORDER}


def _collect_catalog_entities(
    plan: ExecutionPlan,
    *,
    payload: Any,
    request: ApiRequest,
    response: ApiResponse,
    pagination_state: PaginationState,
    raw_artifact: ExtractedArtifact,
    checkpoint_state: CheckpointState | None,
    normalized_records: dict[str, list[dict[str, Any]]],
    entity_indexes: dict[tuple[str, str], int],
    current_checkpoint_value: str | None,
) -> str | None:
    checkpoint_value = current_checkpoint_value
    for batch in _root_batches(payload, plan.source.strategy_variant, path="payload"):
        for index, record in enumerate(batch.records):
            checkpoint_value = _collect_entity_tree(
                plan,
                entity_type=batch.entity_type,
                record=record,
                collection_path=batch.collection_path,
                record_path=f"{batch.collection_path}[{index}]",
                request=request,
                response=response,
                pagination_state=pagination_state,
                raw_artifact=raw_artifact,
                normalized_records=normalized_records,
                entity_indexes=entity_indexes,
                checkpoint_state=checkpoint_state,
                current_checkpoint_value=checkpoint_value,
            )
    return checkpoint_value


def _collect_entity_tree(
    plan: ExecutionPlan,
    *,
    entity_type: str,
    record: dict[str, Any],
    collection_path: str,
    record_path: str,
    request: ApiRequest,
    response: ApiResponse,
    pagination_state: PaginationState,
    raw_artifact: ExtractedArtifact,
    normalized_records: dict[str, list[dict[str, Any]]],
    entity_indexes: dict[tuple[str, str], int],
    checkpoint_state: CheckpointState | None,
    current_checkpoint_value: str | None,
    parent: CatalogEntityReference | None = None,
) -> str | None:
    checkpoint_value = current_checkpoint_value
    entity_reference = _build_entity_reference(entity_type, record, record_path)
    checkpoint_candidate = _checkpoint_candidate(plan, record)

    if not _should_skip_for_checkpoint(plan, checkpoint_state, checkpoint_candidate):
        normalized_record = _normalize_catalog_record(
            entity_type=entity_type,
            record=record,
            collection_path=collection_path,
            record_path=record_path,
            request=request,
            response=response,
            pagination_state=pagination_state,
            raw_artifact=raw_artifact,
            parent=parent,
        )
        _upsert_entity_record(
            plan,
            normalized_records,
            entity_indexes,
            entity_reference.entity_key,
            normalized_record,
        )
        if checkpoint_candidate is not None:
            checkpoint_value = _max_checkpoint_value(checkpoint_value, checkpoint_candidate)

    for child_batch in _nested_batches(
        record,
        variant=plan.source.strategy_variant,
        parent_type=entity_type,
        path=record_path,
    ):
        for index, child_record in enumerate(child_batch.records):
            checkpoint_value = _collect_entity_tree(
                plan,
                entity_type=child_batch.entity_type,
                record=child_record,
                collection_path=child_batch.collection_path,
                record_path=f"{child_batch.collection_path}[{index}]",
                request=request,
                response=response,
                pagination_state=pagination_state,
                raw_artifact=raw_artifact,
                normalized_records=normalized_records,
                entity_indexes=entity_indexes,
                checkpoint_state=checkpoint_state,
                current_checkpoint_value=checkpoint_value,
                parent=entity_reference,
            )
    return checkpoint_value


def _normalize_catalog_record(
    *,
    entity_type: str,
    record: Mapping[str, Any],
    collection_path: str,
    record_path: str,
    request: ApiRequest,
    response: ApiResponse,
    pagination_state: PaginationState,
    raw_artifact: ExtractedArtifact,
    parent: CatalogEntityReference | None,
) -> dict[str, Any]:
    entity_reference = _build_entity_reference(entity_type, record, record_path)
    return {
        "entity_type": entity_type,
        "entity_key": entity_reference.entity_key,
        "entity_id": entity_reference.entity_id,
        "parent_entity_type": parent.entity_type if parent is not None else None,
        "parent_entity_key": parent.entity_key if parent is not None else None,
        "parent_entity_id": parent.entity_id if parent is not None else None,
        "catalog_collection_path": collection_path,
        "catalog_record_path": record_path,
        "catalog_request_url": request.full_url(),
        "catalog_request_index": pagination_state.request_index,
        "catalog_page_number": pagination_state.page_number,
        "catalog_offset": pagination_state.offset,
        "catalog_cursor": pagination_state.cursor,
        "catalog_received_at": response.received_at.isoformat(),
        "catalog_raw_artifact_path": raw_artifact.path,
        "payload": json.dumps(dict(record), sort_keys=True, ensure_ascii=False),
    }

def _upsert_entity_record(
    plan: ExecutionPlan,
    normalized_records: dict[str, list[dict[str, Any]]],
    entity_indexes: dict[tuple[str, str], int],
    entity_key: str,
    candidate: dict[str, Any],
) -> None:
    entity_type = candidate["entity_type"]
    key = (entity_type, entity_key)
    existing_index = entity_indexes.get(key)
    if existing_index is None:
        entity_indexes[key] = len(normalized_records[entity_type])
        normalized_records[entity_type].append(candidate)
        return

    existing = normalized_records[entity_type][existing_index]
    if _prefer_candidate_record(plan, existing, candidate):
        normalized_records[entity_type][existing_index] = candidate


def _prefer_candidate_record(
    plan: ExecutionPlan,
    existing: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    existing_value = _payload_checkpoint_value(existing, plan.checkpoint_field)
    candidate_value = _payload_checkpoint_value(candidate, plan.checkpoint_field)
    if candidate_value is None:
        return False
    if existing_value is None:
        return True
    return _compare_checkpoint_values(candidate_value, existing_value) > 0


def _merge_catalog_entity_records(
    plan: ExecutionPlan,
    normalized_records: dict[str, list[dict[str, Any]]],
    entity_indexes: dict[tuple[str, str], int],
    request_input_records: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    for entity_type in ENTITY_TYPE_ORDER:
        for candidate in request_input_records.get(entity_type, ()):  # pragma: no branch
            entity_key = _string_value(candidate.get("entity_key"))
            if entity_key is None:
                continue
            _upsert_entity_record(
                plan,
                normalized_records,
                entity_indexes,
                entity_key,
                candidate,
            )


def _catalog_total_entity_count(
    normalized_records: Mapping[str, Sequence[dict[str, Any]]],
    request_input_records: Mapping[str, Sequence[dict[str, Any]]],
) -> int:
    return sum(
        len(normalized_records[entity_type]) + len(request_input_records[entity_type])
        for entity_type in ENTITY_TYPE_ORDER
    )


def _payload_checkpoint_value(
    record: Mapping[str, Any],
    checkpoint_field: str | None,
) -> str | None:
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return None
    return _string_value(_lookup_field(payload, checkpoint_field))


def _checkpoint_candidate(plan: ExecutionPlan, record: Mapping[str, Any]) -> str | None:
    if plan.checkpoint_field is None:
        return None
    return _string_value(_lookup_field(record, plan.checkpoint_field))


def _should_skip_for_checkpoint(
    plan: ExecutionPlan,
    checkpoint_state: CheckpointState | None,
    checkpoint_value: str | None,
) -> bool:
    if (
        checkpoint_state is None
        or checkpoint_value is None
        or plan.extraction_mode != "incremental"
    ):
        return False
    return _compare_checkpoint_values(checkpoint_value, checkpoint_state.checkpoint_value) <= 0


# Deliberately NOT shared with the identically named helpers in ``strategies/api/records.py``:
# this ``_lookup_field`` accepts ``str | None`` and returns ``None`` for a missing path, while
# the API one requires a ``str``. Merging them would widen the API contract silently.
def _lookup_field(record: Mapping[str, Any], field_path: str | None) -> Any:
    if field_path is None:
        return None

    current: Any = record
    for segment in field_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized
