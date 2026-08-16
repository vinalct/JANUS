"""The two page loops that walk one request input to the end of its stream."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from janus.checkpoints import CheckpointState, ExtractionProgressStore
from janus.models import ExecutionPlan, ExtractedArtifact
from janus.strategies.http import ApiClient, ApiRequest, HttpRequestThrottle
from janus.utils.logging import StructuredLogger, redact_url
from janus.writers import RawArtifactWriter

from .errors import ApiResponseError
from .pagination import ApiPaginator, OffsetPaginator, PageNumberPaginator, PaginationState
from .requests import ApiRequestExecutor, ApiRequestHook
from .speculation import (
    SpeculativePaginationPolicy,
    SubmittedApiRequest,
    _cancel_pending,
    _may_submit,
    _predicted_next_pagination_state,
    _raise_on_past_end_conflict,
    resolve_speculative_policy,
)


@dataclass(frozen=True, slots=True)
class RequestInputExtraction:
    """Everything one request input contributed to the run.

    The trailing fields describe *speculation*, so they stay at their defaults for the
    sequential path — which never guesses, never over-fetches, and never infers the end of a
    stream from a status code.
    """

    artifacts: list[ExtractedArtifact]
    records_extracted: int
    successful_requests: int
    total_attempts: int
    checkpoint_value: str | None
    speculative_requests: int = 0
    discarded_requests: int = 0
    past_end_status: int | None = None
    past_end_request_index: int | None = None
    total_records_reported: int | None = None
    lookahead_ceiling_source: str | None = None


def extract_sequential_pages(
    executor: ApiRequestExecutor,
    plan: ExecutionPlan,
    *,
    api_hook: ApiRequestHook | None,
    base_request: ApiRequest,
    paginator: ApiPaginator,
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
) -> RequestInputExtraction:
    artifacts: list[ExtractedArtifact] = []
    total_records = 0
    successful_requests = 0
    total_attempts = 0

    with ApiClient(executor.transport_factory()) as client:
        while pagination_state is not None:
            request = executor.prepare_request(
                plan,
                base_request,
                pagination_state,
                paginator,
                api_hook=api_hook,
                checkpoint_state=checkpoint_state,
                logger=logger,
            )
            response, payload, attempts_used = executor.send_request(
                plan,
                client,
                request,
                throttle,
                logger,
            )
            successful_requests += 1
            total_attempts += attempts_used
            processed = executor.process_response(
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

    return RequestInputExtraction(
        artifacts=artifacts,
        records_extracted=total_records,
        successful_requests=successful_requests,
        total_attempts=total_attempts,
        checkpoint_value=checkpoint_value,
    )


def extract_concurrent_pages(
    executor: ApiRequestExecutor,
    plan: ExecutionPlan,
    *,
    api_hook: ApiRequestHook | None,
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
) -> RequestInputExtraction:
    concurrency = plan.source_config.access.rate_limit.concurrency
    artifacts: list[ExtractedArtifact] = []
    total_records = 0
    successful_requests = 0
    total_attempts = 0
    next_request_index = pagination_state.request_index if pagination_state is not None else 1
    predicted_state = pagination_state
    pending: dict[int, SubmittedApiRequest] = {}
    stop_submitting = False
    speculative_requests = 0
    discarded_requests = 0
    past_end_status: int | None = None
    past_end_request_index: int | None = None

    policy: SpeculativePaginationPolicy | None = (
        resolve_speculative_policy(plan, paginator, pagination_state)
        if pagination_state is not None
        else None
    )
    terminal_statuses = policy.past_end_status_codes if policy is not None else frozenset()

    # ``None`` is "no ceiling known": speculation runs on the structural bound alone (the
    # in-flight window) until a page reports how many records this request input holds.
    lookahead_ceiling: int | None = None
    ceiling_logged = False
    total_records_reported: int | None = None
    lookahead_ceiling_source: str | None = None

    pool = ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="janus-api",
    )
    try:
        while predicted_state is not None or pending:
            while (
                predicted_state is not None
                and not stop_submitting
                and len(pending) < concurrency
                and _may_submit(
                    predicted_state.request_index,
                    lookahead_ceiling,
                    next_request_index,
                )
            ):
                request = executor.prepare_request(
                    plan,
                    base_request,
                    predicted_state,
                    paginator,
                    api_hook=api_hook,
                    checkpoint_state=checkpoint_state,
                    logger=logger,
                )
                future = pool.submit(
                    executor.fetch_with_dedicated_client,
                    plan,
                    request,
                    throttle,
                    logger,
                    terminal_status_codes=terminal_statuses,
                )
                pending[predicted_state.request_index] = SubmittedApiRequest(
                    pagination_state=predicted_state,
                    request=request,
                    future=future,
                )
                if policy is not None and policy.is_speculative(predicted_state.request_index):
                    speculative_requests += 1
                predicted_state = _predicted_next_pagination_state(
                    paginator,
                    predicted_state,
                )

            if next_request_index not in pending:
                break

            submitted = pending.pop(next_request_index)
            response, payload, attempts_used = submitted.future.result()

            if policy is not None and policy.is_past_end_status(response.status_code):
                index = submitted.pagination_state.request_index
                if not policy.is_speculative(index):
                    # The first request of this input was never a guess: a 404 here is a
                    # broken endpoint, and must fail exactly as it does today.
                    raise ApiResponseError(response)
                _raise_on_past_end_conflict(pending, response, index, logger)
                past_end_status = response.status_code
                past_end_request_index = index
                if logger is not None:
                    logger.warning(
                        "api_pagination_past_end_detected",
                        request_index=index,
                        page_number=submitted.pagination_state.page_number,
                        offset=submitted.pagination_state.offset,
                        status_code=response.status_code,
                        request_url=redact_url(submitted.request.full_url()),
                        committed_request_count=successful_requests,
                    )
                stop_submitting = True
                discarded_requests += _cancel_pending(
                    pending,
                    logger,
                    next_request_index=index + 1,
                )
                break

            successful_requests += 1
            total_attempts += attempts_used
            processed = executor.process_response(
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
                speculation_policy=policy,
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

            if policy is not None and processed.total_records_reported is not None:
                total_records_reported = processed.total_records_reported
                lookahead_ceiling_source = processed.total_records_source
                candidate = policy.last_request_index_for_total(total_records_reported)
                if candidate is not None:
                    lookahead_ceiling = (
                        candidate
                        if lookahead_ceiling is None
                        else max(lookahead_ceiling, candidate)
                    )
                    if not ceiling_logged and logger is not None:
                        logger.info(
                            "api_pagination_lookahead_bounded",
                            total_records=total_records_reported,
                            last_request_index=lookahead_ceiling,
                            concurrency=concurrency,
                            ceiling_source=lookahead_ceiling_source,
                        )
                    ceiling_logged = True

            if processed.next_pagination_state is None:
                stop_submitting = True
                discarded_requests += _cancel_pending(
                    pending,
                    logger,
                    next_request_index=next_request_index,
                )
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    return RequestInputExtraction(
        artifacts=artifacts,
        records_extracted=total_records,
        successful_requests=successful_requests,
        total_attempts=total_attempts,
        checkpoint_value=checkpoint_value,
        speculative_requests=speculative_requests,
        discarded_requests=discarded_requests,
        past_end_status=past_end_status,
        past_end_request_index=past_end_request_index,
        total_records_reported=total_records_reported,
        lookahead_ceiling_source=lookahead_ceiling_source,
    )
