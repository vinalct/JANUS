from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from janus.checkpoints import (
    CheckpointState,
    CheckpointStore,
    DeadLetterStore,
    ExtractionProgressStore,
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
    PaginationState,
    build_paginator,
)
from janus.strategies.api.pagination import _resume_pagination_state
from janus.strategies.api.request_inputs import (
    ApiRequestInputLoadError,
    load_request_inputs,
    resolve_parameter_bindings,
)
from janus.strategies.base import BaseStrategy, SourceHook
from janus.strategies.catalog.document import (  # noqa: F401
    CATALOG_EDGES_FILE,
    CATALOG_NODES_FILE,
    COLLECTION_ALIASES,
    CREATED_AT_KEYS,
    DATASET_HINT_KEYS,
    DESCRIPTION_KEYS,
    ENTITY_FILE_NAMES,
    ENTITY_HINT_KEYS_BY_TYPE,
    ENTITY_TYPE_ORDER,
    FORMAT_KEYS,
    GENERIC_COLLECTION_KEYS,
    GROUP_HINT_KEYS,
    IDENTIFIER_KEYS,
    NAME_KEYS,
    ORGANIZATION_HINT_KEYS,
    RESOURCE_HINT_KEYS,
    ROOT_ENTITY_PRIORITY,
    STATE_KEYS,
    TITLE_KEYS,
    UNKNOWN_ENTITY_TYPE,
    UPDATED_AT_KEYS,
    URL_KEYS,
    WRAPPER_CONTAINER_KEYS,
    CatalogBatch,
    CatalogEntityReference,
    CatalogParseSummary,
    DocumentNode,
    NodeClassification,
    _batches_from_mapping,
    _build_entity_reference,
    _build_generic_catalog_edge,
    _build_generic_catalog_node,
    _classification_confidence,
    _coerce_records,
    _collect_document_nodes,
    _compute_parse_summary,
    _first_string,
    _infer_entity_type,
    _matched_signals,
    _nested_batches,
    _normalize_mapping,
    _payload_hash,
    _root_batches,
    _score_record,
    classify_catalog_node,
    walk_document,
)
from janus.strategies.common import (
    _compare_checkpoint_values,
    _default_storage_layout,
    _format_datetime,
    _freeze_string_mapping,
    _max_checkpoint_value,
    _parse_datetime,
    _raw_page_path,
    _raw_run_path_prefix,
    _request_input_key,
    _retry_delay_seconds,
    _stringify_mapping,
)
from janus.strategies.http import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiTransport,
    ApiTransportError,
    AuthResolutionError,
    HttpRequestThrottle,
    UrllibApiTransport,
    inject_auth,
)
from janus.utils.logging import StructuredLogger, redact_url
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
SUPPORTED_CATALOG_PAYLOAD_FORMATS = frozenset({"json"})
SUPPORTED_CATALOG_INPUT_FORMATS = frozenset({"jsonl"})


class CatalogStrategyError(RuntimeError):
    """Base failure for metadata/catalog strategy execution."""


class CatalogResponseError(CatalogStrategyError):
    """Raised when a catalog request finished with a non-success response."""

    def __init__(self, response: ApiResponse) -> None:
        self.response = response
        message = (
            f"Catalog request failed with status {response.status_code} for "
            f"{redact_url(response.request.full_url())}"
        )
        super().__init__(message)


class CatalogPayloadError(CatalogStrategyError):
    """Raised when a catalog payload cannot be decoded or traversed safely."""


class CatalogHook(SourceHook):
    """Catalog-specific hook points for wrapper quirks and source-local iteration details."""

    def prepare_request(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        *,
        checkpoint_state: CheckpointState | None = None,
        pagination_state: PaginationState | None = None,
    ) -> ApiRequest:
        del plan
        del checkpoint_state
        del pagination_state
        return request

    def handle_response(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
    ) -> ApiResponse:
        del plan
        del request
        return response

    def transform_payload(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> Any:
        del plan
        del request
        del response
        return payload

    def checkpoint_params(
        self,
        plan: ExecutionPlan,
        checkpoint_value: str,
    ) -> Mapping[str, str] | None:
        del plan
        del checkpoint_value
        return None

    def page_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> Sequence[Any] | None:
        del plan
        del request
        del response
        del payload
        return None

@dataclass(slots=True)
class CatalogStrategy(BaseStrategy):
    """Reusable metadata-first strategy for public dataset catalogs."""

    transport_factory: Callable[[], ApiTransport] = UrllibApiTransport
    storage_layout_factory: Callable[[ExecutionPlan], StorageLayout] = field(
        default_factory=lambda: _default_storage_layout
    )
    raw_writer_factory: Callable[[StorageLayout], RawArtifactWriter] = RawArtifactWriter
    checkpoint_store: CheckpointStore = field(default_factory=CheckpointStore)
    progress_store: ExtractionProgressStore = field(default_factory=ExtractionProgressStore)
    dead_letter_store: DeadLetterStore = field(default_factory=DeadLetterStore)
    env_reader: Callable[[str], str | None] = os.getenv
    sleeper: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    logger: StructuredLogger | None = None

    @property
    def strategy_family(self) -> str:
        return "catalog"

    def plan(
        self,
        source_config: SourceConfig,
        run_context,
        hook: SourceHook | None = None,
    ) -> ExecutionPlan:
        self._validate_source_config(source_config)
        plan = ExecutionPlan.from_source_config(source_config, run_context)
        plan = plan.with_note("strategy_family:catalog")
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
        catalog_hook = hook if isinstance(hook, CatalogHook) else None
        storage_layout = self.storage_layout_factory(plan)
        checkpoint_state = self.checkpoint_store.load(plan)
        base_request = self._build_base_request(plan, checkpoint_state, catalog_hook)
        paginator = build_paginator(plan.source_config.access.pagination)
        throttle = HttpRequestThrottle(
            requests_per_minute=plan.source_config.access.rate_limit.requests_per_minute,
            clock=self.clock,
            sleeper=self.sleeper,
        )
        logger = self._bind_logger(plan)
        dead_letter_max_items = plan.source_config.extraction.dead_letter_max_items

        request_inputs_config = plan.source_config.access.request_inputs
        parameter_bindings = plan.source_config.access.parameter_bindings

        try:
            request_inputs = load_request_inputs(request_inputs_config, spark=spark)
        except ApiRequestInputLoadError:
            if logger is not None:
                logger.exception(
                    "catalog_request_input_loading_failed",
                    request_input_type=request_inputs_config.type,
                )
            raise

        if logger is not None:
            logger.info(
                "catalog_extraction_started",
                request_url=base_request.full_url(),
                method=base_request.method,
                pagination_type=plan.source_config.access.pagination.type,
                page_size=plan.source_config.access.pagination.page_size,
                checkpoint_loaded=checkpoint_state is not None,
                timeout_seconds=base_request.timeout_seconds,
                request_input_type=request_inputs_config.type,
                request_input_count=len(request_inputs),
                dead_letter_max_items=dead_letter_max_items,
            )

        resume = plan.run_context.attributes_as_dict().get("resume") == "true"
        progress = None
        dead_letter_state = None
        if resume:
            progress = self.progress_store.load(plan)
            dead_letter_state = self.dead_letter_store.load(plan)
            if logger is not None:
                if progress is not None:
                    logger.info(
                        "catalog_extraction_resuming",
                        last_page_number=progress.get("last_page_number"),
                        last_offset=progress.get("last_offset"),
                        prior_artifact_count=progress.get("artifact_count", 0),
                    )
                else:
                    logger.info("catalog_extraction_resume_requested_no_progress_found")
        else:
            self.progress_store.clear(plan)
            self.dead_letter_store.clear(plan)

        raw_path_prefix = _raw_run_path_prefix(plan, progress)
        raw_writer = self.raw_writer_factory(storage_layout).with_raw_path_prefix(raw_path_prefix)

        raw_artifacts: list[ExtractedArtifact] = []

        completed_by_key: dict[str, int] = {}
        if progress is not None:
            for entry in progress.get("completed_inputs", []):
                completed_by_key[entry["key"]] = entry["index"]
        current_key_from_progress: str | None = (
            progress.get("current_input_key") if progress is not None else None
        )
        completed_inputs: list[tuple[str, int]] = list(completed_by_key.items())
        request_input_count = len(request_inputs)
        dead_letter_keys = (
            set(dead_letter_state.item_keys) if dead_letter_state is not None else set()
        )
        dead_letter_skip_count = 0

        normalized_records: dict[str, list[dict[str, Any]]] = {
            entity_type: [] for entity_type in ENTITY_TYPE_ORDER
        }
        entity_indexes: dict[tuple[str, str], int] = {}
        checkpoint_request_value = _checkpoint_request_value(plan, checkpoint_state)
        checkpoint_value: str | None = None
        successful_requests = 0
        total_attempts = 0
        page_record_total = 0

        with ApiClient(self.transport_factory()) as client:
            for request_input_index, request_input in enumerate(request_inputs, start=1):
                request_input_key = _request_input_key(request_input)
                per_input_request = _apply_per_input_params(
                    base_request,
                    parameter_bindings,
                    request_input,
                    checkpoint_value=checkpoint_request_value,
                )

                if request_input_key in completed_by_key:
                    completed_input_index = completed_by_key[request_input_key]
                    pre_artifacts = _rediscover_catalog_input_artifacts(
                        plan, storage_layout, completed_input_index, request_input_count
                    )
                    raw_artifacts.extend(pre_artifacts)
                    checkpoint_value = _replay_catalog_entities_from_dir(
                        self,
                        plan,
                        storage_layout,
                        per_input_request,
                        paginator,
                        completed_input_index,
                        request_input_count,
                        checkpoint_state,
                        normalized_records,
                        entity_indexes,
                        checkpoint_value,
                    )
                    continue

                if request_input_key in dead_letter_keys:
                    dead_letter_skip_count += 1
                    if logger is not None:
                        logger.info(
                            "catalog_request_input_skipped_dead_letter",
                            input_key=request_input_key,
                            dead_letter_count=len(dead_letter_keys),
                        )
                    continue

                is_resuming = progress is not None and (
                    request_input_key == current_key_from_progress
                    or current_key_from_progress is None
                )
                request_input_raw_artifacts: list[ExtractedArtifact] = []
                request_input_records: dict[str, list[dict[str, Any]]] = {
                    entity_type: [] for entity_type in ENTITY_TYPE_ORDER
                }
                request_input_indexes: dict[tuple[str, str], int] = {}
                request_input_checkpoint_value = checkpoint_value
                request_input_successful_requests = 0
                request_input_total_attempts = 0
                request_input_page_record_total = 0

                pagination_state: PaginationState | None
                if is_resuming:
                    # is_resuming implies progress was loaded (see its definition above).
                    assert progress is not None
                    pre_artifacts = _rediscover_catalog_raw_artifacts(
                        plan, storage_layout, progress, request_input_index, request_input_count
                    )
                    request_input_raw_artifacts.extend(pre_artifacts)
                    pagination_state = _resume_pagination_state(
                        paginator, paginator.initial_state(per_input_request), progress
                    )
                    if logger is not None:
                        logger.info(
                            "catalog_extraction_resume_artifacts_recovered",
                            recovered_artifact_count=len(pre_artifacts),
                            resuming_at_page=pagination_state.page_number,
                            resuming_at_offset=pagination_state.offset,
                        )
                else:
                    pagination_state = paginator.initial_state(per_input_request)

                try:
                    while pagination_state is not None:
                        request = paginator.apply(per_input_request, pagination_state)
                        if catalog_hook is not None:
                            request = catalog_hook.prepare_request(
                                plan,
                                request,
                                checkpoint_state=checkpoint_state,
                                pagination_state=pagination_state,
                            )

                        if logger is not None:
                            logger.info(
                                "catalog_request_started",
                                request_index=pagination_state.request_index,
                                page_number=pagination_state.page_number,
                                offset=pagination_state.offset,
                                cursor=pagination_state.cursor,
                                request_url=request.full_url(),
                            )

                        response, payload, attempts_used = self._send_with_retries(
                            plan,
                            client,
                            request,
                            throttle,
                            logger,
                        )
                        request_input_total_attempts += attempts_used
                        request_input_successful_requests += 1

                        if catalog_hook is not None:
                            response = catalog_hook.handle_response(plan, request, response)

                        if catalog_hook is not None:
                            payload = catalog_hook.transform_payload(
                                plan,
                                request,
                                response,
                                payload,
                            )

                        persisted = self._persist_raw_payload(
                            plan,
                            raw_writer,
                            response=response,
                            payload=payload,
                            pagination_state=pagination_state,
                            request_input_index=request_input_index,
                            request_input_count=request_input_count,
                        )
                        request_input_raw_artifacts.append(persisted.artifact)

                        primary_batch_size = self._page_record_count(
                            plan,
                            request,
                            response,
                            payload,
                            hook=catalog_hook,
                        )
                        request_input_page_record_total += primary_batch_size
                        entity_counts_before = {
                            entity_type: len(records)
                            for entity_type, records in request_input_records.items()
                        }
                        request_input_checkpoint_value = self._collect_catalog_entities(
                            plan,
                            payload=payload,
                            request=request,
                            response=response,
                            pagination_state=pagination_state,
                            raw_artifact=persisted.artifact,
                            checkpoint_state=checkpoint_state,
                            normalized_records=request_input_records,
                            entity_indexes=request_input_indexes,
                            current_checkpoint_value=request_input_checkpoint_value,
                        )
                        entity_counts_after = {
                            entity_type: len(records)
                            for entity_type, records in request_input_records.items()
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
                                total_records=page_record_total + request_input_page_record_total,
                                total_entities=_catalog_total_entity_count(
                                    normalized_records,
                                    request_input_records,
                                ),
                                organizations_extracted=(
                                    len(normalized_records["organization"])
                                    + len(request_input_records["organization"])
                                ),
                                groups_extracted=(
                                    len(normalized_records["group"])
                                    + len(request_input_records["group"])
                                ),
                                datasets_extracted=(
                                    len(normalized_records["dataset"])
                                    + len(request_input_records["dataset"])
                                ),
                                resources_extracted=(
                                    len(normalized_records["resource"])
                                    + len(request_input_records["resource"])
                                ),
                                artifact_path=persisted.artifact.path,
                                has_next_page=next_pagination_state is not None,
                                next_page_number=(
                                    next_pagination_state.page_number
                                    if next_pagination_state is not None
                                    else None
                                ),
                                next_offset=(
                                    next_pagination_state.offset
                                    if next_pagination_state is not None
                                    else None
                                ),
                            )
                        self.progress_store.save(
                            plan,
                            page_number=pagination_state.page_number,
                            offset=pagination_state.offset,
                            cursor=pagination_state.cursor,
                            request_index=pagination_state.request_index,
                            artifact_count=len(raw_artifacts) + len(request_input_raw_artifacts),
                            completed_inputs=completed_inputs,
                            current_input_key=request_input_key,
                            current_input_index=request_input_index,
                            request_input_count=request_input_count,
                            raw_path_prefix=(
                                str(raw_path_prefix) if raw_path_prefix is not None else None
                            ),
                        )
                        pagination_state = next_pagination_state
                except Exception as exc:
                    if logger is not None:
                        logger.exception("catalog_request_execution_failed")
                    dead_letter_state = self.dead_letter_store.record(
                        plan,
                        item_key=request_input_key,
                        item_type="request_input",
                        error=exc,
                        metadata=_catalog_request_input_dead_letter_metadata(
                            request_input=request_input,
                            request_input_index=request_input_index,
                            request_input_count=request_input_count,
                            request=per_input_request,
                        ),
                    )
                    dead_letter_keys = set(dead_letter_state.item_keys)
                    if logger is not None:
                        logger.info(
                            "catalog_request_input_dead_lettered",
                            input_key=request_input_key,
                            dead_letter_count=dead_letter_state.entry_count,
                            dead_letter_max_items=dead_letter_max_items,
                            error_type=type(exc).__name__,
                        )
                    if not can_continue_after_dead_letter(
                        total_item_count=request_input_count,
                        dead_letter_count=dead_letter_state.entry_count,
                        dead_letter_max_items=dead_letter_max_items,
                    ):
                        raise
                    dead_letter_skip_count += 1
                    continue

                raw_artifacts.extend(request_input_raw_artifacts)
                _merge_catalog_entity_records(
                    plan,
                    normalized_records,
                    entity_indexes,
                    request_input_records,
                )
                checkpoint_value = request_input_checkpoint_value
                successful_requests += request_input_successful_requests
                total_attempts += request_input_total_attempts
                page_record_total += request_input_page_record_total

                completed_inputs.append((request_input_key, request_input_index))

        self.progress_store.clear(plan)
        normalized_artifacts = self._persist_normalized_records(
            plan,
            raw_writer,
            normalized_records,
        )
        records_extracted = sum(len(records) for records in normalized_records.values())
        if logger is not None:
            logger.info(
                "catalog_extraction_finished",
                request_count=successful_requests,
                retry_count=max(total_attempts - successful_requests, 0),
                attempt_count=total_attempts,
                records_extracted=records_extracted,
                page_record_count=page_record_total,
                raw_artifact_count=len(raw_artifacts),
                normalized_artifact_count=len(normalized_artifacts),
                artifact_count=len(raw_artifacts) + len(normalized_artifacts),
                checkpoint_value=checkpoint_value,
                organizations_extracted=len(normalized_records["organization"]),
                groups_extracted=len(normalized_records["group"]),
                datasets_extracted=len(normalized_records["dataset"]),
                resources_extracted=len(normalized_records["resource"]),
                dead_letter_count=(dead_letter_state.entry_count if dead_letter_state else 0),
                dead_letter_skipped_count=dead_letter_skip_count,
            )

        dead_letter_count = dead_letter_state.entry_count if dead_letter_state is not None else 0

        extraction_result = ExtractionResult.from_plan(
            plan,
            tuple(raw_artifacts + normalized_artifacts),
            records_extracted=records_extracted,
            checkpoint_value=checkpoint_value,
            metadata={
                "request_count": str(successful_requests),
                "retry_count": str(max(total_attempts - successful_requests, 0)),
                "attempt_count": str(total_attempts),
                "pagination_type": plan.source_config.access.pagination.type,
                "checkpoint_loaded": str(checkpoint_state is not None).lower(),
                "raw_page_count": str(len(raw_artifacts)),
                "page_record_count": str(page_record_total),
                "normalized_artifact_count": str(len(normalized_artifacts)),
                "organizations_extracted": str(len(normalized_records["organization"])),
                "groups_extracted": str(len(normalized_records["group"])),
                "datasets_extracted": str(len(normalized_records["dataset"])),
                "resources_extracted": str(len(normalized_records["resource"])),
                "dead_letter_count": str(dead_letter_count),
                "dead_letter_skipped_count": str(dead_letter_skip_count),
                "entity_types_emitted": ",".join(
                    entity_type
                    for entity_type in ENTITY_TYPE_ORDER
                    if normalized_records[entity_type]
                ) or "none",
                **(
                    {"raw_path_prefix": str(raw_path_prefix)}
                    if raw_path_prefix is not None
                    else {}
                ),
            },
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
        entity_artifact_names = {
            f"{ENTITY_FILE_NAMES[entity_type]}.jsonl"
            for entity_type in ENTITY_TYPE_ORDER
        }
        handoff_artifacts = tuple(
            artifact
            for artifact in extraction_result.artifacts
            if artifact.format == plan.source_config.spark.input_format
            and Path(artifact.path).name in entity_artifact_names
        )
        if not handoff_artifacts:
            raise CatalogStrategyError(
                "No normalized catalog entity artifacts match the configured "
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
            "pagination_type": plan.source_config.access.pagination.type,
            "request_timeout_seconds": plan.source_config.access.timeout_seconds,
            "input_format": plan.source_config.spark.input_format,
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
        if source_config.access.format not in SUPPORTED_CATALOG_PAYLOAD_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_CATALOG_PAYLOAD_FORMATS))
            raise ValueError(f"Catalog access.format must be one of: {allowed}")
        if source_config.outputs.raw.format not in SUPPORTED_CATALOG_PAYLOAD_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_CATALOG_PAYLOAD_FORMATS))
            raise ValueError(f"Catalog outputs.raw.format must be one of: {allowed}")
        if source_config.spark.input_format not in SUPPORTED_CATALOG_INPUT_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_CATALOG_INPUT_FORMATS))
            raise ValueError(f"Catalog spark.input_format must be one of: {allowed}")

    def _build_base_request(
        self,
        plan: ExecutionPlan,
        checkpoint_state: CheckpointState | None,
        catalog_hook: CatalogHook | None,
    ) -> ApiRequest:
        source_access = plan.source_config.access
        request = ApiRequest(
            method=source_access.method,
            url=_resolve_url(plan.source_config),
            timeout_seconds=source_access.timeout_seconds,
            headers=_freeze_string_mapping(source_access.headers or {}),
            params=_freeze_string_mapping(source_access.params or {}),
        )
        request = inject_auth(request, source_access.auth, env_reader=self._resolve_env_var)

        checkpoint_value = _checkpoint_request_value(plan, checkpoint_state)
        checkpoint_params = _default_checkpoint_params(plan, checkpoint_value)
        if catalog_hook is not None and checkpoint_value is not None:
            hook_checkpoint_params = catalog_hook.checkpoint_params(plan, checkpoint_value)
            if hook_checkpoint_params is not None:
                checkpoint_params = _stringify_mapping(hook_checkpoint_params)

        if checkpoint_params:
            path_params, query_checkpoint_params = _split_path_and_query_params(
                request.url, checkpoint_params
            )
            if path_params:
                request = request.with_url(request.url.format_map(path_params))
            if query_checkpoint_params:
                request = request.with_params(query_checkpoint_params)
        return request

    def _send_with_retries(
        self,
        plan: ExecutionPlan,
        client: ApiClient,
        request: ApiRequest,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
    ) -> tuple[ApiResponse, Any, int]:
        retry_config = plan.source_config.extraction.retry
        last_transport_error: Exception | None = None

        for attempt in range(1, retry_config.max_attempts + 1):
            throttle.wait_for_turn()
            try:
                response = client.send(request)
            except (ApiTransportError, AuthResolutionError) as exc:
                last_transport_error = exc
                if attempt == retry_config.max_attempts:
                    raise CatalogStrategyError(str(exc)) from exc
                self._sleep_for_retry(plan, attempt, response=None, logger=logger)
                continue

            if 200 <= response.status_code < 300:
                try:
                    payload = self._decode_payload(plan, response)
                except CatalogPayloadError:
                    if attempt == retry_config.max_attempts:
                        raise
                    self._sleep_for_retry(plan, attempt, response=response, logger=logger)
                    continue
                return response, payload, attempt

            if (
                response.status_code not in RETRYABLE_STATUS_CODES
                or attempt == retry_config.max_attempts
            ):
                raise CatalogResponseError(response)

            self._sleep_for_retry(plan, attempt, response=response, logger=logger)

        if last_transport_error is not None:
            raise CatalogStrategyError(str(last_transport_error)) from last_transport_error
        raise CatalogStrategyError("Retry loop exited without a response or error")

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
                "catalog_retry_scheduled",
                attempt=attempt,
                delay_seconds=delay,
                status_code=response.status_code if response is not None else None,
            )
        self.sleeper(delay)

    def _decode_payload(self, plan: ExecutionPlan, response: ApiResponse) -> Any:
        if plan.source_config.access.format != "json":
            raise CatalogPayloadError(
                f"Unsupported catalog payload format: {plan.source_config.access.format}"
            )

        try:
            return response.json()
        except (UnicodeDecodeError, ValueError) as exc:
            raise CatalogPayloadError(
                f"Failed to decode catalog payload from {redact_url(response.request.full_url())}"
            ) from exc

    def _persist_raw_payload(
        self,
        plan: ExecutionPlan,
        raw_writer: RawArtifactWriter,
        *,
        response: ApiResponse,
        payload: Any,
        pagination_state: PaginationState,
        request_input_index: int = 1,
        request_input_count: int = 1,
    ):
        return raw_writer.write_json(
            plan,
            _raw_relative_path(
                pagination_state,
                request_input_index=request_input_index,
                request_input_count=request_input_count,
            ),
            payload,
            metadata={
                "request_url": response.request.full_url(),
                "status_code": str(response.status_code),
                "request_index": str(pagination_state.request_index),
            },
        )

    def _page_record_count(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
        *,
        hook: CatalogHook | None,
    ) -> int:
        if hook is not None:
            hook_records = hook.page_records(plan, request, response, payload)
            if hook_records is not None:
                return len(tuple(hook_records))

        batches = _root_batches(payload, plan.source.strategy_variant, path="payload")
        if not batches:
            return 0
        return len(batches[0].records)

    def _collect_catalog_entities(
        self,
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
                checkpoint_value = self._collect_entity_tree(
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
        self,
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
                checkpoint_value = self._collect_entity_tree(
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

    def _persist_normalized_records(
        self,
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


def _split_path_and_query_params(
    url: str,
    bound_params: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Separate bound params into path params (referenced as {name} in the URL) and query params."""
    placeholders = set(re.findall(r"\{(\w+)\}", url))
    path_params = {k: v for k, v in bound_params.items() if k in placeholders}
    query_params = {k: v for k, v in bound_params.items() if k not in placeholders}
    return path_params, query_params


def _apply_per_input_params(
    base_request: ApiRequest,
    parameter_bindings: Any,
    request_input: dict[str, Any] | None,
    *,
    checkpoint_value: str | None = None,
) -> ApiRequest:
    """Apply per-request-input parameter bindings to the base request."""
    if not parameter_bindings:
        return base_request
    bound_params = resolve_parameter_bindings(
        parameter_bindings,
        request_input=request_input,
        checkpoint_value=checkpoint_value,
    )
    if not bound_params:
        return base_request
    path_params, query_params = _split_path_and_query_params(base_request.url, bound_params)
    request = base_request
    if path_params:
        request = request.with_url(base_request.url.format_map(path_params))
    if query_params:
        request = request.with_params(query_params)
    return request


def _resolve_url(source_config: SourceConfig) -> str:
    if source_config.access.url:
        return source_config.access.url

    base_url = (source_config.access.base_url or "").strip()
    path = (source_config.access.path or "").strip()
    if not base_url:
        raise ValueError("Catalog source requires access.base_url or access.url")
    if not path:
        return base_url
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _default_checkpoint_params(
    plan: ExecutionPlan,
    checkpoint_value: str | None,
) -> dict[str, str]:
    if checkpoint_value is None or plan.checkpoint_field is None:
        return {}
    return {plan.checkpoint_field: checkpoint_value}


def _checkpoint_request_value(
    plan: ExecutionPlan,
    checkpoint_state: CheckpointState | None,
) -> str | None:
    if (
        plan.extraction_mode != "incremental"
        or plan.checkpoint_field is None
        or checkpoint_state is None
    ):
        return None

    value = checkpoint_state.checkpoint_value
    lookback_days = plan.source_config.extraction.lookback_days
    if not lookback_days:
        return value

    parsed_datetime = _parse_datetime(value)
    if parsed_datetime is None:
        return value
    return _format_datetime(parsed_datetime - timedelta(days=lookback_days))


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
                try:
                    candidates.append((int(stem[start:]), path))
                except ValueError:
                    pass
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
    strategy: CatalogStrategy,
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
        checkpoint_value = strategy._collect_catalog_entities(
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


def _catalog_request_input_dead_letter_metadata(
    *,
    request_input: Mapping[str, Any] | None,
    request_input_index: int,
    request_input_count: int,
    request: ApiRequest,
) -> dict[str, str]:
    metadata = {
        "request_input_index": str(request_input_index),
        "request_input_count": str(request_input_count),
        "request_url": request.full_url(),
    }
    if request_input:
        metadata["request_input_field_names"] = ",".join(sorted(str(key) for key in request_input))
    return metadata


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
