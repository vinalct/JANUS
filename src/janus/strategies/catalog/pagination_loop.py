"""The page loop that walks one catalog request input to the end of its stream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janus.checkpoints import CheckpointState, ExtractionProgressStore
from janus.models import ExecutionPlan, ExtractedArtifact
from janus.strategies.api import PaginationState
from janus.strategies.api.pagination import ApiPaginator
from janus.strategies.http import ApiRequest, HttpRequestThrottle
from janus.utils.logging import StructuredLogger
from janus.writers import RawArtifactWriter

from .document import ENTITY_TYPE_ORDER
from .entities import (
    _catalog_total_entity_count,
    _collect_catalog_entities,
    _empty_entity_records,
)
from .requests import CatalogRequestExecutor, CatalogRequestHook, CatalogRequestSession


@dataclass(frozen=True, slots=True)
class RequestInputExtraction:
    """Everything one catalog request input contributed to the run.

    Where the API family's counterpart of this name carries speculation counters, the
    catalog's carries entities: the records this input produced, held apart from the run's
    index until the input is known to have finished.
    """

    artifacts: list[ExtractedArtifact]
    normalized_records: dict[str, list[dict[str, Any]]]
    successful_requests: int
    total_attempts: int
    page_record_total: int
    checkpoint_value: str | None


def extract_request_input_pages(
    executor: CatalogRequestExecutor,
    session: CatalogRequestSession,
    plan: ExecutionPlan,
    *,
    catalog_hook: CatalogRequestHook | None,
    per_input_request: ApiRequest,
    paginator: ApiPaginator,
    pagination_state: PaginationState | None,
    pre_artifacts: Sequence[ExtractedArtifact],
    checkpoint_state: CheckpointState | None,
    checkpoint_value: str | None,
    raw_writer: RawArtifactWriter,
    throttle: HttpRequestThrottle,
    logger: StructuredLogger | None,
    request_input_index: int,
    request_input_count: int,
    completed_inputs: list[tuple[str, int]],
    request_input_key: str,
    progress_store: ExtractionProgressStore,
    run_records: Mapping[str, Sequence[dict[str, Any]]],
    prior_page_record_total: int,
    prior_artifact_count: int,
    raw_path_prefix: Path | None,
) -> RequestInputExtraction:
    """Walk one request input's pages, returning what it contributed.

    ``run_records`` and the two ``prior_*`` counts are read-only: the per-page log reports
    run-wide totals, and the run absorbs what this returns only once the input finished.
    """
    artifacts: list[ExtractedArtifact] = list(pre_artifacts)
    normalized_records = _empty_entity_records()
    entity_indexes: dict[tuple[str, str], int] = {}
    successful_requests = 0
    total_attempts = 0
    page_record_total = 0

    while pagination_state is not None:
        request = executor.prepare_request(
            plan,
            per_input_request,
            pagination_state,
            paginator,
            catalog_hook=catalog_hook,
            checkpoint_state=checkpoint_state,
            logger=logger,
        )

        response, payload, attempts_used = session.send(
            plan,
            request,
            throttle,
            logger,
        )
        total_attempts += attempts_used
        successful_requests += 1

        response, payload = executor.apply_response_hooks(
            plan,
            request,
            response,
            payload,
            catalog_hook=catalog_hook,
        )

        persisted = executor.persist_raw_payload(
            plan,
            raw_writer,
            response=response,
            payload=payload,
            pagination_state=pagination_state,
            request_input_index=request_input_index,
            request_input_count=request_input_count,
        )
        artifacts.append(persisted.artifact)

        primary_batch_size = executor.page_record_count(
            plan,
            request,
            response,
            payload,
            hook=catalog_hook,
        )
        page_record_total += primary_batch_size
        entity_counts_before = {
            entity_type: len(records) for entity_type, records in normalized_records.items()
        }
        checkpoint_value = _collect_catalog_entities(
            plan,
            payload=payload,
            request=request,
            response=response,
            pagination_state=pagination_state,
            raw_artifact=persisted.artifact,
            checkpoint_state=checkpoint_state,
            normalized_records=normalized_records,
            entity_indexes=entity_indexes,
            current_checkpoint_value=checkpoint_value,
        )
        entity_counts_after = {
            entity_type: len(records) for entity_type, records in normalized_records.items()
        }
        page_entity_count = sum(
            entity_counts_after[entity_type] - entity_counts_before[entity_type]
            for entity_type in ENTITY_TYPE_ORDER
        )
        next_pagination_state = paginator.next_state(
            pagination_state,
            records_extracted=primary_batch_size,
            payload=payload,
        )
        if logger is not None:
            logger.info(
                "catalog_request_finished",
                request_index=pagination_state.request_index,
                page_number=pagination_state.page_number,
                offset=pagination_state.offset,
                cursor=pagination_state.cursor,
                status_code=response.status_code,
                attempts_used=attempts_used,
                records_extracted=primary_batch_size,
                entities_extracted=page_entity_count,
                total_records=prior_page_record_total + page_record_total,
                total_entities=_catalog_total_entity_count(run_records, normalized_records),
                organizations_extracted=(
                    len(run_records["organization"]) + len(normalized_records["organization"])
                ),
                groups_extracted=(
                    len(run_records["group"]) + len(normalized_records["group"])
                ),
                datasets_extracted=(
                    len(run_records["dataset"]) + len(normalized_records["dataset"])
                ),
                resources_extracted=(
                    len(run_records["resource"]) + len(normalized_records["resource"])
                ),
                artifact_path=persisted.artifact.path,
                has_next_page=next_pagination_state is not None,
                next_page_number=(
                    next_pagination_state.page_number
                    if next_pagination_state is not None
                    else None
                ),
                next_offset=(
                    next_pagination_state.offset if next_pagination_state is not None else None
                ),
            )
        progress_store.save(
            plan,
            page_number=pagination_state.page_number,
            offset=pagination_state.offset,
            cursor=pagination_state.cursor,
            request_index=pagination_state.request_index,
            artifact_count=prior_artifact_count + len(artifacts),
            completed_inputs=completed_inputs,
            current_input_key=request_input_key,
            current_input_index=request_input_index,
            request_input_count=request_input_count,
            raw_path_prefix=(str(raw_path_prefix) if raw_path_prefix is not None else None),
        )
        pagination_state = next_pagination_state

    return RequestInputExtraction(
        artifacts=artifacts,
        normalized_records=normalized_records,
        successful_requests=successful_requests,
        total_attempts=total_attempts,
        page_record_total=page_record_total,
        checkpoint_value=checkpoint_value,
    )
