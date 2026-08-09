"""The catalog strategy façade.

What remains here is the contract, not the machinery: the :class:`CatalogStrategy` lifecycle
methods the planner calls, the :class:`CatalogHook` extension points a source implements, and
compatibility re-exports for names that were defined in this module before the package was
split.

Where the machinery went:

* ``errors.py`` — the exception hierarchy.
* ``entities.py`` — record shaping, the governed normalized-record contract, entity merging.
* ``artifacts.py`` — raw/normalized path layout, page rediscovery, replay, persistence.
* ``metadata.py`` — dead-letter metadata and per-input parameter binding.
* ``document.py`` — the document walker.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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
from janus.runtime.spark_lifecycle import scoped_request_input_session
from janus.strategies.api import (
    PaginationState,
    build_paginator,
)
from janus.strategies.api.pagination import _resume_pagination_state
from janus.strategies.api.request_inputs import (
    ApiRequestInputLoadError,
    load_request_inputs,
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
    _default_storage_layout,
    _freeze_string_mapping,
    _raw_run_path_prefix,
    _request_input_key,
    _stringify_mapping,
)
from janus.strategies.http import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiTransport,
    HttpRequestThrottle,
    PayloadDecodeError,
    RetryErrorPolicy,
    UrllibApiTransport,
    checkpoint_request_value,
    decode_payload,
    default_checkpoint_params,
    inject_auth,
    resolve_url,
    send_with_retries,
    split_path_and_query_params,
)
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

from .artifacts import (
    _persist_generic_artifacts,
    _persist_normalized_records,
    _raw_relative_path,
    _rediscover_catalog_input_artifacts,
    _rediscover_catalog_raw_artifacts,
    _replay_catalog_entities_from_dir,
)
from .entities import (
    _catalog_total_entity_count,
    _collect_catalog_entities,
    _merge_catalog_entity_records,
    _normalize_catalog_record,
)
from .errors import CatalogPayloadError, CatalogResponseError, CatalogStrategyError
from .metadata import _apply_per_input_params, _catalog_request_input_dead_letter_metadata

__all__ = [
    "SUPPORTED_CATALOG_INPUT_FORMATS",
    "SUPPORTED_CATALOG_PAYLOAD_FORMATS",
    "CatalogHook",
    "CatalogPayloadError",
    "CatalogResponseError",
    "CatalogStrategy",
    "CatalogStrategyError",
    "_apply_per_input_params",
    "_normalize_catalog_record",
    "_persist_generic_artifacts",
    "_rediscover_catalog_input_artifacts",
    "_replay_catalog_entities_from_dir",
]

SUPPORTED_CATALOG_PAYLOAD_FORMATS = frozenset({"json"})
SUPPORTED_CATALOG_INPUT_FORMATS = frozenset({"jsonl"})

_RETRY_POLICY = RetryErrorPolicy(
    transport_error_factory=CatalogStrategyError,
    response_error_factory=CatalogResponseError,
    retry_log_event="catalog_retry_scheduled",
)


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
            with scoped_request_input_session(
                spark,
                request_inputs_config,
            ) as request_input_session:
                request_inputs = load_request_inputs(
                    request_inputs_config,
                    spark=request_input_session,
                )
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
        request_checkpoint_value = checkpoint_request_value(plan, checkpoint_state)
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
                    checkpoint_value=request_checkpoint_value,
                )

                if request_input_key in completed_by_key:
                    completed_input_index = completed_by_key[request_input_key]
                    pre_artifacts = _rediscover_catalog_input_artifacts(
                        plan, storage_layout, completed_input_index, request_input_count
                    )
                    raw_artifacts.extend(pre_artifacts)
                    checkpoint_value = _replay_catalog_entities_from_dir(
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

                        response, payload, attempts_used = send_with_retries(
                            plan,
                            client,
                            request,
                            throttle,
                            logger,
                            policy=_RETRY_POLICY,
                            sleeper=self.sleeper,
                            decode=lambda response: self._decode_payload(plan, response),
                            payload_error_types=(CatalogPayloadError,),
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
                        request_input_checkpoint_value = _collect_catalog_entities(
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
        normalized_artifacts = _persist_normalized_records(
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
            url=resolve_url(plan.source_config, family_label="Catalog"),
            timeout_seconds=source_access.timeout_seconds,
            headers=_freeze_string_mapping(source_access.headers or {}),
            params=_freeze_string_mapping(source_access.params or {}),
        )
        request = inject_auth(request, source_access.auth, env_reader=self._resolve_env_var)

        checkpoint_value = checkpoint_request_value(plan, checkpoint_state)
        checkpoint_params = default_checkpoint_params(plan, checkpoint_value)
        if catalog_hook is not None and checkpoint_value is not None:
            hook_checkpoint_params = catalog_hook.checkpoint_params(plan, checkpoint_value)
            if hook_checkpoint_params is not None:
                checkpoint_params = _stringify_mapping(hook_checkpoint_params)

        if checkpoint_params:
            path_params, query_checkpoint_params = split_path_and_query_params(
                request.url, checkpoint_params
            )
            if path_params:
                request = request.with_url(request.url.format_map(path_params))
            if query_checkpoint_params:
                request = request.with_params(query_checkpoint_params)
        return request

    def _decode_payload(self, plan: ExecutionPlan, response: ApiResponse) -> Any:
        try:
            return decode_payload(
                plan.source_config.access.format,
                response,
                allowed_formats=frozenset({"json"}),
                family_label="catalog",
                failure_label="catalog",
            )
        except PayloadDecodeError as exc:
            raise CatalogPayloadError(str(exc)) from exc.__cause__

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
