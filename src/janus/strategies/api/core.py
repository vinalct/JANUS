from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeGuard
from urllib.parse import urljoin

from janus.checkpoints import (
    CheckpointState,
    CheckpointStore,
    DeadLetterStore,
    ExtractionProgressStore,
    can_continue_after_dead_letter,
)
from janus.models import (
    CombinedRequestInputsConfig,
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    IcebergRowsRequestInputsConfig,
    SourceConfig,
    WriteResult,
)
from janus.strategies.base import BaseStrategy, SourceHook
from janus.strategies.common import (
    _compare_checkpoint_values,
    _default_storage_layout,
    _format_datetime,
    _freeze_string_mapping,
    _parse_datetime,
    _raw_page_path,
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
    HttpStrategyError,
    PayloadDecodeError,
    RetryErrorPolicy,
    UrllibApiTransport,
    decode_payload,
    inject_auth,
    send_with_retries,
)
from janus.utils.logging import StructuredLogger, redact_url
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

from .pagination import (
    OffsetPaginator,
    PageNumberPaginator,
    PaginationState,
    _resume_pagination_state,
    build_paginator,
)
from .request_inputs import (
    ApiParameterBindingError,
    ApiRequestInputLoadError,
    load_request_inputs,
    merge_request_params,
    resolve_parameter_bindings,
)

SUPPORTED_API_PAYLOAD_FORMATS = frozenset({"binary", "json", "jsonl", "text"})
RAW_FILE_SUFFIXES = {
    "binary": ".bin",
    "json": ".json",
    "jsonl": ".jsonl",
    "text": ".txt",
}
DEFAULT_RECORD_KEYS = ("records", "items", "results", "data", "value")


class ApiStrategyError(HttpStrategyError):
    """Base failure for API strategy execution."""


class ApiResponseError(ApiStrategyError):
    """Raised when an API call finished with a non-success response."""

    def __init__(self, response: ApiResponse) -> None:
        self.response = response
        message = (
            f"API request failed with status {response.status_code} for "
            f"{redact_url(response.request.full_url())}"
        )
        super().__init__(message)


class ApiPayloadError(ApiStrategyError):
    """Raised when the configured payload format cannot be decoded."""


_RETRY_POLICY = RetryErrorPolicy(
    transport_error_factory=ApiStrategyError,
    response_error_factory=ApiResponseError,
    retry_log_event="api_retry_scheduled",
)


class ApiHook(SourceHook):
    """API-specific hook points layered on top of the generic source-hook contract."""

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

    def extract_records(
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

    def resolve_next_cursor(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> str | None:
        del plan
        del request
        del response
        del payload
        return None

    def checkpoint_params(
        self,
        plan: ExecutionPlan,
        checkpoint_value: str,
    ) -> Mapping[str, str] | None:
        del plan
        del checkpoint_value
        return None


@dataclass(frozen=True, slots=True)
class SubmittedApiRequest:
    pagination_state: PaginationState
    request: ApiRequest
    future: Future[tuple[ApiResponse, Any, int]]


@dataclass(frozen=True, slots=True)
class ProcessedApiRequest:
    artifact: ExtractedArtifact
    records_extracted: int
    checkpoint_value: str | None
    next_pagination_state: PaginationState | None


@dataclass(slots=True)
class ApiStrategy(BaseStrategy):
    """Reusable HTTP strategy for public federal API integrations."""

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
        return "api"

    def plan(
        self,
        source_config: SourceConfig,
        run_context,
        hook: SourceHook | None = None,
    ) -> ExecutionPlan:
        self._validate_source_config(source_config)
        plan = ExecutionPlan.from_source_config(source_config, run_context)
        plan = plan.with_note("strategy_family:api")
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
        api_hook = hook if isinstance(hook, ApiHook) else None
        storage_layout = self.storage_layout_factory(plan)
        checkpoint_state = self.checkpoint_store.load(plan)
        checkpoint_request_value = _checkpoint_request_value(plan, checkpoint_state)
        base_request = self._build_base_request(plan, checkpoint_state, api_hook)
        paginator = build_paginator(plan.source_config.access.pagination)
        throttle = HttpRequestThrottle(
            requests_per_minute=plan.source_config.access.rate_limit.requests_per_minute,
            clock=self.clock,
            sleeper=self.sleeper,
        )
        logger = self._bind_logger(plan)
        request_input_type = plan.source_config.access.request_inputs.type
        parameter_bindings = plan.source_config.access.parameter_bindings
        dead_letter_max_items = plan.source_config.extraction.dead_letter_max_items

        try:
            request_inputs = load_request_inputs(
                plan.source_config.access.request_inputs,
                spark=spark,
            )
        except ApiRequestInputLoadError:
            if logger is not None:
                logger.exception(
                    "api_request_input_loading_failed",
                    request_input_type=request_input_type,
                )
            raise

        request_input_count = len(request_inputs)
        request_input_metadata = _request_input_metadata(plan, request_input_count)
        bound_parameter_names = sorted(parameter_bindings or {})
        if logger is not None:
            logger.info(
                "api_extraction_started",
                request_url=base_request.full_url(),
                method=base_request.method,
                pagination_type=plan.source_config.access.pagination.type,
                page_size=plan.source_config.access.pagination.page_size,
                checkpoint_loaded=checkpoint_state is not None,
                timeout_seconds=base_request.timeout_seconds,
                request_input_type=request_input_type,
                request_input_count=request_input_count,
                bound_parameter_names=bound_parameter_names,
                dead_letter_max_items=dead_letter_max_items,
                upstream_namespace=request_input_metadata.get("upstream_namespace"),
                upstream_table_name=request_input_metadata.get("upstream_table_name"),
                upstream_column_names=(
                    request_input_metadata["upstream_column_names"].split(",")
                    if request_input_metadata.get("upstream_column_names")
                    else []
                ),
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
                        "api_extraction_resuming",
                        current_input_key=progress.get("current_input_key"),
                        last_page_number=progress.get("last_page_number"),
                        last_offset=progress.get("last_offset"),
                        completed_input_count=len(progress.get("completed_inputs", [])),
                        prior_artifact_count=progress.get("artifact_count", 0),
                    )
                else:
                    logger.info("api_extraction_resume_requested_no_progress_found")
        else:
            self.progress_store.clear(plan)
            self.dead_letter_store.clear(plan)

        raw_path_prefix = _raw_run_path_prefix(plan, progress)
        raw_writer = self.raw_writer_factory(storage_layout).with_raw_path_prefix(raw_path_prefix)

        artifacts: list[ExtractedArtifact] = []
        total_records = 0
        successful_requests = 0
        total_attempts = 0
        checkpoint_value: str | None = None

        completed_by_key: dict[str, int] = {}
        if progress is not None:
            for entry in progress.get("completed_inputs", []):
                completed_by_key[entry["key"]] = entry["index"]
        current_key_from_progress: str | None = (
            progress.get("current_input_key") if progress is not None else None
        )
        current_index_from_progress: int | None = (
            progress.get("current_input_index") if progress is not None else None
        )

        completed_inputs: list[tuple[str, int]] = list(completed_by_key.items())
        dead_letter_keys = (
            set(dead_letter_state.item_keys) if dead_letter_state is not None else set()
        )
        dead_letter_skip_count = 0

        for request_input_index, request_input in enumerate(request_inputs, start=1):
            request_input_key = _request_input_key(request_input)

            request_input_logger = logger
            if request_input_logger is not None:
                request_input_logger = request_input_logger.bind(
                    request_input_index=request_input_index,
                    request_input_count=request_input_count,
                    request_input_type=request_input_type,
                )

            if request_input_key in completed_by_key:
                file_index = completed_by_key[request_input_key]
                pre_artifacts = _rediscover_all_artifacts_for_input(
                    plan, storage_layout, file_index, request_input_count
                )
                artifacts.extend(pre_artifacts)
                if request_input_logger is not None:
                    request_input_logger.info(
                        "api_request_input_skipped_already_complete",
                        recovered_artifact_count=len(pre_artifacts),
                        input_key=request_input_key,
                    )
                continue

            if request_input_key in dead_letter_keys:
                dead_letter_skip_count += 1
                if request_input_logger is not None:
                    request_input_logger.info(
                        "api_request_input_skipped_dead_letter",
                        input_key=request_input_key,
                        dead_letter_count=len(dead_letter_keys),
                    )
                continue

            is_resuming = progress is not None and request_input_key == current_key_from_progress
            file_index = (
                current_index_from_progress
                if is_resuming and current_index_from_progress is not None
                else request_input_index
            )

            if request_input_logger is not None:
                request_input_logger.info(
                    "api_request_input_started",
                    request_input_field_names=_request_input_field_names(request_input),
                )

            try:
                bound_params = resolve_parameter_bindings(
                    parameter_bindings,
                    request_input=request_input,
                    checkpoint_value=checkpoint_request_value,
                )
                path_params, query_bound_params = _split_path_and_query_params(
                    base_request.url, bound_params
                )
                request_params = merge_request_params(
                    plan.source_config.access.params,
                    query_bound_params,
                )
            except ApiParameterBindingError:
                if request_input_logger is not None:
                    request_input_logger.exception(
                        "api_parameter_binding_failed",
                        request_input_field_names=_request_input_field_names(request_input),
                    )
                raise

            request_input_base_request = base_request
            if path_params:
                request_input_base_request = request_input_base_request.with_url(
                    base_request.url.format_map(path_params)
                )
            if request_params:
                request_input_base_request = request_input_base_request.with_params(request_params)

            if request_input_logger is not None:
                request_input_logger.info(
                    "api_request_input_bound",
                    bound_parameter_names=sorted(bound_params),
                    request_parameter_names=sorted(request_params),
                )

            pagination_state = paginator.initial_state(request_input_base_request)

            if is_resuming:
                # is_resuming implies progress was loaded (see its definition above).
                assert progress is not None
                pre_artifacts = _rediscover_raw_artifacts(
                    plan, storage_layout, progress, file_index, request_input_count
                )
                artifacts.extend(pre_artifacts)
                pagination_state = _resume_pagination_state(paginator, pagination_state, progress)
                if request_input_logger is not None:
                    request_input_logger.info(
                        "api_extraction_resume_artifacts_recovered",
                        recovered_artifact_count=len(pre_artifacts),
                        resuming_at_page=pagination_state.page_number,
                        resuming_at_offset=pagination_state.offset,
                        input_key=request_input_key,
                    )
            try:
                if _supports_concurrent_pagination(
                    paginator,
                    plan.source_config.access.rate_limit.concurrency,
                ):
                    (
                        request_input_artifacts,
                        request_input_total_records,
                        request_input_successful_requests,
                        request_input_total_attempts,
                        request_input_checkpoint_value,
                    ) = self._extract_concurrent_pages(
                        plan,
                        api_hook=api_hook,
                        base_request=request_input_base_request,
                        paginator=paginator,
                        pagination_state=pagination_state,
                        checkpoint_state=checkpoint_state,
                        checkpoint_value=checkpoint_value,
                        raw_writer=raw_writer,
                        throttle=throttle,
                        logger=request_input_logger,
                        request_input_index=file_index,
                        request_input_count=request_input_count,
                        completed_inputs=completed_inputs,
                        current_input_key=request_input_key,
                        current_input_index=file_index,
                        progress_store=self.progress_store,
                        prior_artifact_count=len(artifacts),
                        raw_path_prefix=raw_path_prefix,
                    )
                else:
                    (
                        request_input_artifacts,
                        request_input_total_records,
                        request_input_successful_requests,
                        request_input_total_attempts,
                        request_input_checkpoint_value,
                    ) = self._extract_sequential_pages(
                        plan,
                        api_hook=api_hook,
                        base_request=request_input_base_request,
                        paginator=paginator,
                        pagination_state=pagination_state,
                        checkpoint_state=checkpoint_state,
                        checkpoint_value=checkpoint_value,
                        raw_writer=raw_writer,
                        throttle=throttle,
                        logger=request_input_logger,
                        request_input_index=file_index,
                        request_input_count=request_input_count,
                        completed_inputs=completed_inputs,
                        current_input_key=request_input_key,
                        current_input_index=file_index,
                        progress_store=self.progress_store,
                        prior_artifact_count=len(artifacts),
                        raw_path_prefix=raw_path_prefix,
                    )
            except Exception as exc:
                if request_input_logger is not None:
                    request_input_logger.exception("api_request_execution_failed")
                dead_letter_state = self.dead_letter_store.record(
                    plan,
                    item_key=request_input_key,
                    item_type="request_input",
                    error=exc,
                    metadata=_request_input_dead_letter_metadata(
                        request_input=request_input,
                        request_input_index=request_input_index,
                        request_input_count=request_input_count,
                        request=request_input_base_request,
                    ),
                )
                dead_letter_keys = set(dead_letter_state.item_keys)
                if request_input_logger is not None:
                    request_input_logger.info(
                        "api_request_input_dead_lettered",
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

            artifacts.extend(request_input_artifacts)
            total_records += request_input_total_records
            successful_requests += request_input_successful_requests
            total_attempts += request_input_total_attempts
            checkpoint_value = request_input_checkpoint_value
            completed_inputs.append((request_input_key, file_index))

            if request_input_logger is not None:
                request_input_logger.info(
                    "api_request_input_finished",
                    request_count=request_input_successful_requests,
                    retry_count=max(
                        request_input_total_attempts - request_input_successful_requests,
                        0,
                    ),
                    attempt_count=request_input_total_attempts,
                    records_extracted=request_input_total_records,
                    artifact_count=len(request_input_artifacts),
                    checkpoint_value=request_input_checkpoint_value,
                )

        if logger is not None:
            logger.info(
                "api_extraction_finished",
                request_count=successful_requests,
                retry_count=max(total_attempts - successful_requests, 0),
                attempt_count=total_attempts,
                records_extracted=total_records,
                artifact_count=len(artifacts),
                checkpoint_value=checkpoint_value,
                request_input_type=request_input_type,
                request_input_count=request_input_count,
                dead_letter_count=(dead_letter_state.entry_count if dead_letter_state else 0),
                dead_letter_skipped_count=dead_letter_skip_count,
            )

        self.progress_store.clear(plan)

        dead_letter_count = dead_letter_state.entry_count if dead_letter_state is not None else 0

        extraction_result = ExtractionResult.from_plan(
            plan,
            tuple(artifacts),
            records_extracted=total_records,
            checkpoint_value=checkpoint_value,
            metadata={
                "request_count": str(successful_requests),
                "retry_count": str(max(total_attempts - successful_requests, 0)),
                "attempt_count": str(total_attempts),
                "pagination_type": plan.source_config.access.pagination.type,
                "auth_type": plan.source_config.access.auth.type,
                "checkpoint_loaded": str(checkpoint_state is not None).lower(),
                "records_extracted": str(total_records),
                "dead_letter_count": str(dead_letter_count),
                "dead_letter_skipped_count": str(dead_letter_skip_count),
                **(
                    {"raw_path_prefix": str(raw_path_prefix)}
                    if raw_path_prefix is not None
                    else {}
                ),
                **request_input_metadata,
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

    def _extract_sequential_pages(
        self,
        plan: ExecutionPlan,
        *,
        api_hook: ApiHook | None,
        base_request: ApiRequest,
        paginator,
        pagination_state: PaginationState | None,
        checkpoint_state: CheckpointState | None,
        checkpoint_value: str | None,
        raw_writer: RawArtifactWriter,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
        request_input_index: int,
        request_input_count: int,
        completed_inputs: list[tuple[str, int]],
        current_input_key: str | None,
        current_input_index: int,
        progress_store: ExtractionProgressStore,
        prior_artifact_count: int = 0,
        raw_path_prefix: Path | None = None,
    ) -> tuple[list[ExtractedArtifact], int, int, int, str | None]:
        artifacts: list[ExtractedArtifact] = []
        total_records = 0
        successful_requests = 0
        total_attempts = 0

        with ApiClient(self.transport_factory()) as client:
            while pagination_state is not None:
                request = self._prepare_request(
                    plan,
                    base_request,
                    pagination_state,
                    paginator,
                    api_hook=api_hook,
                    checkpoint_state=checkpoint_state,
                    logger=logger,
                )
                response, payload, attempts_used = self._send_request(
                    plan,
                    client,
                    request,
                    throttle,
                    logger,
                )
                successful_requests += 1
                total_attempts += attempts_used
                processed = self._process_response(
                    plan,
                    raw_writer,
                    paginator,
                    pagination_state,
                    request,
                    response,
                    payload,
                    attempts_used,
                    checkpoint_value,
                    api_hook=api_hook,
                    logger=logger,
                    total_records=total_records,
                    request_input_index=request_input_index,
                    request_input_count=request_input_count,
                )
                artifacts.append(processed.artifact)
                total_records += processed.records_extracted
                checkpoint_value = processed.checkpoint_value
                progress_store.save(
                    plan,
                    page_number=pagination_state.page_number,
                    offset=pagination_state.offset,
                    cursor=pagination_state.cursor,
                    request_index=pagination_state.request_index,
                    artifact_count=prior_artifact_count + len(artifacts),
                    completed_inputs=completed_inputs,
                    current_input_key=current_input_key,
                    current_input_index=current_input_index,
                    request_input_count=request_input_count,
                    raw_path_prefix=str(raw_path_prefix) if raw_path_prefix is not None else None,
                )
                pagination_state = processed.next_pagination_state

        return (
            artifacts,
            total_records,
            successful_requests,
            total_attempts,
            checkpoint_value,
        )

    def _extract_concurrent_pages(
        self,
        plan: ExecutionPlan,
        *,
        api_hook: ApiHook | None,
        base_request: ApiRequest,
        paginator: PageNumberPaginator | OffsetPaginator,
        pagination_state: PaginationState | None,
        checkpoint_state: CheckpointState | None,
        checkpoint_value: str | None,
        raw_writer: RawArtifactWriter,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
        request_input_index: int,
        request_input_count: int,
        completed_inputs: list[tuple[str, int]],
        current_input_key: str | None,
        current_input_index: int,
        progress_store: ExtractionProgressStore,
        prior_artifact_count: int = 0,
        raw_path_prefix: Path | None = None,
    ) -> tuple[list[ExtractedArtifact], int, int, int, str | None]:
        concurrency = plan.source_config.access.rate_limit.concurrency
        artifacts: list[ExtractedArtifact] = []
        total_records = 0
        successful_requests = 0
        total_attempts = 0
        next_request_index = pagination_state.request_index if pagination_state is not None else 1
        predicted_state = pagination_state
        pending: dict[int, SubmittedApiRequest] = {}
        stop_submitting = False

        executor = ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="janus-api",
        )
        try:
            while predicted_state is not None or pending:
                while (
                    predicted_state is not None
                    and not stop_submitting
                    and len(pending) < concurrency
                ):
                    request = self._prepare_request(
                        plan,
                        base_request,
                        predicted_state,
                        paginator,
                        api_hook=api_hook,
                        checkpoint_state=checkpoint_state,
                        logger=logger,
                    )
                    future = executor.submit(
                        self._fetch_with_dedicated_client,
                        plan,
                        request,
                        throttle,
                        logger,
                    )
                    pending[predicted_state.request_index] = SubmittedApiRequest(
                        pagination_state=predicted_state,
                        request=request,
                        future=future,
                    )
                    predicted_state = _predicted_next_pagination_state(
                        paginator,
                        predicted_state,
                    )

                if next_request_index not in pending:
                    break

                submitted = pending.pop(next_request_index)
                response, payload, attempts_used = submitted.future.result()
                successful_requests += 1
                total_attempts += attempts_used
                processed = self._process_response(
                    plan,
                    raw_writer,
                    paginator,
                    submitted.pagination_state,
                    submitted.request,
                    response,
                    payload,
                    attempts_used,
                    checkpoint_value,
                    api_hook=api_hook,
                    logger=logger,
                    total_records=total_records,
                    request_input_index=request_input_index,
                    request_input_count=request_input_count,
                )
                artifacts.append(processed.artifact)
                total_records += processed.records_extracted
                checkpoint_value = processed.checkpoint_value
                progress_store.save(
                    plan,
                    page_number=submitted.pagination_state.page_number,
                    offset=submitted.pagination_state.offset,
                    cursor=submitted.pagination_state.cursor,
                    request_index=submitted.pagination_state.request_index,
                    artifact_count=prior_artifact_count + len(artifacts),
                    completed_inputs=completed_inputs,
                    current_input_key=current_input_key,
                    current_input_index=current_input_index,
                    request_input_count=request_input_count,
                    raw_path_prefix=str(raw_path_prefix) if raw_path_prefix is not None else None,
                )
                next_request_index += 1
                if processed.next_pagination_state is None:
                    stop_submitting = True
                    for pending_request in pending.values():
                        pending_request.future.cancel()
                    break
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        return (
            artifacts,
            total_records,
            successful_requests,
            total_attempts,
            checkpoint_value,
        )

    def build_normalization_handoff(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        hook: SourceHook | None = None,
    ) -> ExtractionResult:
        if hook is not None:
            return hook.on_normalization_handoff(plan, extraction_result)
        return extraction_result

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
            "auth_type": plan.source_config.access.auth.type,
            "request_timeout_seconds": plan.source_config.access.timeout_seconds,
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
        access_format = source_config.access.format
        raw_format = source_config.outputs.raw.format
        if access_format not in SUPPORTED_API_PAYLOAD_FORMATS:
            allowed_formats = ", ".join(sorted(SUPPORTED_API_PAYLOAD_FORMATS))
            raise ValueError(f"API access.format must be one of: {allowed_formats}")
        if raw_format not in SUPPORTED_API_PAYLOAD_FORMATS:
            allowed_formats = ", ".join(sorted(SUPPORTED_API_PAYLOAD_FORMATS))
            raise ValueError(f"API outputs.raw.format must be one of: {allowed_formats}")

        variant = source_config.strategy_variant
        pagination_type = source_config.access.pagination.type
        if variant == "page_number_api" and pagination_type != "page_number":
            raise ValueError("page_number_api requires access.pagination.type='page_number'")
        if variant == "offset_api" and pagination_type != "offset":
            raise ValueError("offset_api requires access.pagination.type='offset'")
        if variant == "cursor_api" and pagination_type != "cursor":
            raise ValueError("cursor_api requires access.pagination.type='cursor'")
        if variant == "date_window_api" and source_config.extraction.mode != "incremental":
            raise ValueError("date_window_api requires extraction.mode='incremental'")

    def _build_base_request(
        self,
        plan: ExecutionPlan,
        checkpoint_state: CheckpointState | None,
        api_hook: ApiHook | None,
    ) -> ApiRequest:
        source_access = plan.source_config.access
        request = ApiRequest(
            method=source_access.method,
            url=_resolve_url(plan.source_config),
            timeout_seconds=source_access.timeout_seconds,
            headers=_freeze_string_mapping(source_access.headers or {}),
            params=(),
        )
        request = inject_auth(request, source_access.auth, env_reader=self._resolve_env_var)

        checkpoint_value = _checkpoint_request_value(plan, checkpoint_state)
        checkpoint_params = _default_checkpoint_params(plan, checkpoint_value)
        if api_hook is not None and checkpoint_value is not None:
            hook_checkpoint_params = api_hook.checkpoint_params(plan, checkpoint_value)
            if hook_checkpoint_params is not None:
                checkpoint_params = _stringify_mapping(hook_checkpoint_params)

        if checkpoint_params:
            request = request.with_params(checkpoint_params)
        return request

    def _prepare_request(
        self,
        plan: ExecutionPlan,
        base_request: ApiRequest,
        pagination_state: PaginationState,
        paginator,
        *,
        api_hook: ApiHook | None,
        checkpoint_state: CheckpointState | None,
        logger: StructuredLogger | None,
    ) -> ApiRequest:
        request = paginator.apply(base_request, pagination_state)
        if api_hook is not None:
            request = api_hook.prepare_request(
                plan,
                request,
                checkpoint_state=checkpoint_state,
                pagination_state=pagination_state,
            )

        if logger is not None:
            logger.info(
                "api_request_started",
                request_index=pagination_state.request_index,
                page_number=pagination_state.page_number,
                offset=pagination_state.offset,
                cursor=pagination_state.cursor,
                request_url=request.full_url(),
            )
        return request

    def _fetch_with_dedicated_client(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
    ) -> tuple[ApiResponse, Any, int]:
        with ApiClient(self.transport_factory()) as client:
            return self._send_request(plan, client, request, throttle, logger)

    def _send_request(
        self,
        plan: ExecutionPlan,
        client: ApiClient,
        request: ApiRequest,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
    ) -> tuple[ApiResponse, Any, int]:
        return send_with_retries(
            plan,
            client,
            request,
            throttle,
            logger,
            policy=_RETRY_POLICY,
            sleeper=self.sleeper,
            decode=lambda response: self._decode_payload(plan, response),
            payload_error_types=(ApiPayloadError,),
        )

    def _process_response(
        self,
        plan: ExecutionPlan,
        raw_writer: RawArtifactWriter,
        paginator,
        pagination_state: PaginationState,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
        attempts_used: int,
        checkpoint_value: str | None,
        *,
        api_hook: ApiHook | None,
        logger: StructuredLogger | None,
        total_records: int,
        request_input_index: int,
        request_input_count: int,
    ) -> ProcessedApiRequest:
        if api_hook is not None:
            response = api_hook.handle_response(plan, request, response)

        if api_hook is not None:
            payload = api_hook.transform_payload(plan, request, response, payload)

        records = self._extract_records(plan, request, response, payload, api_hook)
        resolved_checkpoint_value = self._resolve_checkpoint_value(
            checkpoint_value,
            records,
            checkpoint_field=plan.checkpoint_field,
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

        next_cursor = None
        if api_hook is not None:
            next_cursor = api_hook.resolve_next_cursor(plan, request, response, payload)
        next_pagination_state = paginator.next_state(
            pagination_state,
            records_extracted=len(records),
            payload=payload,
            next_cursor=next_cursor,
        )
        if logger is not None:
            logger.info(
                "api_request_finished",
                request_index=pagination_state.request_index,
                page_number=pagination_state.page_number,
                offset=pagination_state.offset,
                cursor=pagination_state.cursor,
                status_code=response.status_code,
                attempts_used=attempts_used,
                records_extracted=len(records),
                total_records=total_records + len(records),
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

        return ProcessedApiRequest(
            artifact=persisted.artifact,
            records_extracted=len(records),
            checkpoint_value=resolved_checkpoint_value,
            next_pagination_state=next_pagination_state,
        )

    def _decode_payload(self, plan: ExecutionPlan, response: ApiResponse) -> Any:
        try:
            return decode_payload(
                plan.source_config.access.format,
                response,
                family_label="API",
            )
        except PayloadDecodeError as exc:
            raise ApiPayloadError(str(exc)) from exc.__cause__

    def _extract_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
        api_hook: ApiHook | None,
    ) -> tuple[Any, ...]:
        if api_hook is not None:
            hook_records = api_hook.extract_records(plan, request, response, payload)
            if hook_records is not None:
                return tuple(hook_records)
        return tuple(_default_records_from_payload(payload))

    def _persist_raw_payload(
        self,
        plan: ExecutionPlan,
        raw_writer: RawArtifactWriter,
        *,
        response: ApiResponse,
        payload: Any,
        pagination_state: PaginationState,
        request_input_index: int,
        request_input_count: int,
    ):
        relative_path = _raw_relative_path(
            plan.source_config.outputs.raw.format,
            pagination_state,
            request_input_index=request_input_index,
            request_input_count=request_input_count,
        )
        metadata = {
            "request_url": response.request.full_url(),
            "status_code": str(response.status_code),
            "request_index": str(pagination_state.request_index),
            "request_input_index": str(request_input_index),
            "request_input_count": str(request_input_count),
            "request_input_type": plan.source_config.access.request_inputs.type,
        }
        bound_parameter_names = sorted(plan.source_config.access.parameter_bindings or {})
        if bound_parameter_names:
            metadata["bound_parameter_names"] = ",".join(bound_parameter_names)
        if pagination_state.page_number is not None:
            metadata["page_number"] = str(pagination_state.page_number)
        if pagination_state.offset is not None:
            metadata["offset"] = str(pagination_state.offset)
        if pagination_state.cursor is not None:
            metadata["cursor"] = pagination_state.cursor

        raw_format = plan.source_config.outputs.raw.format
        if raw_format == "json":
            return raw_writer.write_json(plan, relative_path, payload, metadata=metadata)
        if raw_format == "jsonl":
            return raw_writer.write_json_lines(
                plan,
                relative_path,
                _default_records_from_payload(payload),
                metadata=metadata,
            )
        if raw_format == "text":
            return raw_writer.write_text(plan, relative_path, response.text(), metadata=metadata)
        if raw_format == "binary":
            return raw_writer.write_bytes(plan, relative_path, response.body, metadata=metadata)
        raise ValueError(f"Unsupported raw output format for API strategy: {raw_format}")

    def _resolve_checkpoint_value(
        self,
        current_value: str | None,
        records: Sequence[Any],
        *,
        checkpoint_field: str | None,
    ) -> str | None:
        if checkpoint_field is None:
            return current_value

        resolved_value = current_value
        for record in records:
            if not isinstance(record, Mapping):
                continue
            candidate = _string_value(_lookup_field(record, checkpoint_field))
            if candidate is None:
                continue
            if resolved_value is None or _compare_checkpoint_values(candidate, resolved_value) > 0:
                resolved_value = candidate
        return resolved_value

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


def _resolve_url(source_config: SourceConfig) -> str:
    if source_config.access.url:
        return source_config.access.url

    base_url = (source_config.access.base_url or "").strip()
    path = (source_config.access.path or "").strip()
    if not base_url:
        raise ValueError("API source requires access.base_url or access.url")
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


def _supports_concurrent_pagination(
    paginator: Any,
    concurrency: int,
) -> TypeGuard[PageNumberPaginator | OffsetPaginator]:
    return concurrency > 1 and isinstance(paginator, PageNumberPaginator | OffsetPaginator)


def _predicted_next_pagination_state(
    paginator: PageNumberPaginator | OffsetPaginator,
    pagination_state: PaginationState,
) -> PaginationState | None:
    return paginator.next_state(
        pagination_state,
        records_extracted=paginator.page_size,
        payload=None,
    )


def _raw_relative_path(
    raw_format: str,
    pagination_state: PaginationState,
    *,
    request_input_index: int,
    request_input_count: int,
) -> Path:
    return _raw_page_path(
        pagination_state,
        RAW_FILE_SUFFIXES[raw_format],
        request_input_index=request_input_index,
        request_input_count=request_input_count,
    )


def _request_input_metadata(plan: ExecutionPlan, request_input_count: int) -> dict[str, str]:
    parameter_bindings = plan.source_config.access.parameter_bindings or {}
    request_inputs = plan.source_config.access.request_inputs
    metadata = {
        "request_input_type": request_inputs.type,
        "request_input_count": str(request_input_count),
    }
    if parameter_bindings:
        metadata["bound_parameter_names"] = ",".join(sorted(parameter_bindings))
    if request_inputs.type == "iceberg_rows" and isinstance(
        request_inputs, IcebergRowsRequestInputsConfig
    ):
        metadata["upstream_namespace"] = request_inputs.namespace
        metadata["upstream_table_name"] = request_inputs.table_name
        metadata["upstream_column_names"] = ",".join(
            sorted(
                {
                    str(column).strip()
                    for column in request_inputs.columns.values()
                    if str(column).strip()
                }
            )
        )
    if request_inputs.type == "combined" and isinstance(
        request_inputs, CombinedRequestInputsConfig
    ):
        for sub_ri in request_inputs.inputs:
            if isinstance(sub_ri, IcebergRowsRequestInputsConfig):
                metadata["upstream_namespace"] = sub_ri.namespace
                metadata["upstream_table_name"] = sub_ri.table_name
                metadata["upstream_column_names"] = ",".join(
                    sorted(
                        {
                            str(column).strip()
                            for column in sub_ri.columns.values()
                            if str(column).strip()
                        }
                    )
                )
                break
    return metadata



def _request_input_dead_letter_metadata(
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
    field_names = _request_input_field_names(request_input)
    if field_names:
        metadata["request_input_field_names"] = ",".join(field_names)
    return metadata


def _request_input_field_names(
    request_input: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if not request_input:
        return ()
    return tuple(sorted(str(field_name) for field_name in request_input))




def _pages_dir(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    request_input_index: int,
    request_input_count: int,
) -> Path:
    """Return the raw subdirectory for one request input, mirroring _raw_relative_path."""
    raw_dir = storage_layout.resolve_output(plan, "raw").resolved_path
    if request_input_count > 1:
        return raw_dir / f"request-input-{request_input_index:06d}"
    return raw_dir / "pages"


def _rediscover_all_artifacts_for_input(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    request_input_index: int,
    request_input_count: int,
) -> list[ExtractedArtifact]:
    """Return all raw artifacts written for a fully completed request input."""
    raw_format = plan.source_config.outputs.raw.format
    suffix = RAW_FILE_SUFFIXES.get(raw_format, "")
    directory = _pages_dir(plan, storage_layout, request_input_index, request_input_count)

    if not directory.exists():
        return []

    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(f"*{suffix}"):
        stem = path.stem
        for prefix, start in (("page-", 5), ("offset-", 7), ("cursor-", 7), ("response-", 9)):
            if stem.startswith(prefix):
                try:
                    candidates.append((int(stem[start:]), path))
                except ValueError:
                    pass
                break

    artifacts = []
    for _, path in sorted(candidates):
        checksum = sha256(path.read_bytes()).hexdigest()
        artifacts.append(ExtractedArtifact(path=str(path), format=raw_format, checksum=checksum))
    return artifacts


def _rediscover_raw_artifacts(
    plan: ExecutionPlan,
    storage_layout: StorageLayout,
    progress: dict[str, Any],
    request_input_index: int = 1,
    request_input_count: int = 1,
) -> list[ExtractedArtifact]:
    """Re-discover raw artifact files written by a previous partial run."""
    raw_format = plan.source_config.outputs.raw.format
    suffix = RAW_FILE_SUFFIXES.get(raw_format, "")
    directory = _pages_dir(plan, storage_layout, request_input_index, request_input_count)

    if not directory.exists():
        return []

    last_page = progress.get("last_page_number")
    last_offset = progress.get("last_offset")
    artifacts: list[ExtractedArtifact] = []

    if last_page is not None:
        candidates: list[tuple[int, Path]] = []
        for path in directory.glob(f"page-*{suffix}"):
            stem = path.stem
            if not stem.startswith("page-"):
                continue
            try:
                num = int(stem[5:])
            except ValueError:
                continue
            if num <= last_page:
                candidates.append((num, path))
        for _, path in sorted(candidates):
            checksum = sha256(path.read_bytes()).hexdigest()
            artifacts.append(
                ExtractedArtifact(path=str(path), format=raw_format, checksum=checksum)
            )

    elif last_offset is not None:
        candidates = []
        for path in directory.glob(f"offset-*{suffix}"):
            stem = path.stem
            if not stem.startswith("offset-"):
                continue
            try:
                num = int(stem[7:])
            except ValueError:
                continue
            if num <= last_offset:
                candidates.append((num, path))
        for _, path in sorted(candidates):
            checksum = sha256(path.read_bytes()).hexdigest()
            artifacts.append(
                ExtractedArtifact(path=str(path), format=raw_format, checksum=checksum)
            )

    return artifacts


def _default_records_from_payload(payload: Any) -> Sequence[Any]:
    if payload is None:
        return ()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return payload
    if isinstance(payload, Mapping):
        for key in DEFAULT_RECORD_KEYS:
            nested = payload.get(key)
            if isinstance(nested, list):
                return nested
        return (payload,)
    return ()


def _lookup_field(record: Mapping[str, Any], field_path: str) -> Any:
    current: Any = record
    for segment in field_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    return rendered
