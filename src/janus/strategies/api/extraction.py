"""Orchestration for an API run: what to request next, and what to do when it fails.

*Orchestration* decides which request input is next, whether
a previous attempt already finished it, and whether a failure ends the run or is recorded and
skipped. *Mechanics* — performing one request, turning the answer into an artifact — are
``requests.py``, and the page loops beneath them ``pagination_loop.py``. Nothing here speaks
HTTP, which ``tests/unit/strategies/api/test_extraction_boundary.py`` enforces.
:class:`ApiExtractionContext` holds everything resolved once per run; the state the loop
carries across inputs is ``run_state.py``.

The Spark lifecycle is untouched by all of it: ``spark`` is a provider, never a session, and
:func:`load_scoped_request_inputs` delegates to the shared ``scoped_request_input_session``
rather than reimplementing the scope.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from janus.checkpoints import (
    CheckpointState,
    CheckpointStore,
    DeadLetterStore,
    ExtractionProgressStore,
    can_continue_after_dead_letter,
)
from janus.models import ExecutionPlan, ExtractionResult
from janus.runtime.spark_lifecycle import scoped_request_input_session
from janus.strategies.common import _raw_run_path_prefix, _request_input_key
from janus.strategies.http import (
    ApiRequest,
    HttpRequestThrottle,
    checkpoint_request_value,
    split_path_and_query_params,
)
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

from .metadata import (
    _request_input_dead_letter_metadata,
    _request_input_field_names,
    _request_input_metadata,
    _speculation_metadata,
)
from .pagination import (
    ApiPaginator,
    build_paginator,
)
from .pagination_loop import (
    RequestInputExtraction,
    extract_concurrent_pages,
    extract_sequential_pages,
)
from .request_inputs import (
    ApiParameterBindingError,
    ApiRequestInputLoadError,
    load_request_inputs,
    merge_request_params,
    resolve_parameter_bindings,
)
from .requests import ApiRequestExecutor, ApiRequestHook
from .run_state import (
    ApiExtractionOutcome,
    ResumeState,
    RunTotals,
    recover_completed_input,
    rehydrate_resumed_input,
)
from .speculation import _supports_concurrent_pagination


@dataclass(frozen=True, slots=True)
class ApiExtractionContext:
    """Everything resolved once per run, before the first request input is touched."""

    plan: ExecutionPlan
    api_hook: ApiRequestHook | None
    storage_layout: StorageLayout
    checkpoint_state: CheckpointState | None
    request_checkpoint_value: str | None
    base_request: ApiRequest
    paginator: ApiPaginator
    throttle: HttpRequestThrottle
    logger: StructuredLogger | None
    #: The writer needs the resume-derived path prefix, known only after the
    #: ``api_extraction_started`` log — so the context carries the factory, not the writer.
    raw_writer_factory: Callable[[StorageLayout], RawArtifactWriter]

    @classmethod
    def build(
        cls,
        plan: ExecutionPlan,
        *,
        api_hook: ApiRequestHook | None,
        executor: ApiRequestExecutor,
        storage_layout_factory: Callable[[ExecutionPlan], StorageLayout],
        raw_writer_factory: Callable[[StorageLayout], RawArtifactWriter],
        checkpoint_store: CheckpointStore,
        logger: StructuredLogger | None,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> Self:
        checkpoint_state = checkpoint_store.load(plan)
        return cls(
            plan=plan,
            api_hook=api_hook,
            storage_layout=storage_layout_factory(plan),
            checkpoint_state=checkpoint_state,
            request_checkpoint_value=checkpoint_request_value(plan, checkpoint_state),
            base_request=executor.build_base_request(plan, checkpoint_state, api_hook),
            paginator=build_paginator(plan.source_config.access.pagination),
            throttle=HttpRequestThrottle(
                requests_per_minute=plan.source_config.access.rate_limit.requests_per_minute,
                clock=clock,
                sleeper=sleeper,
            ),
            logger=logger,
            raw_writer_factory=raw_writer_factory,
        )

    @property
    def request_input_type(self) -> str:
        return self.plan.source_config.access.request_inputs.type

    @property
    def dead_letter_max_items(self) -> int:
        return self.plan.source_config.extraction.dead_letter_max_items

    @property
    def concurrency(self) -> int:
        return self.plan.source_config.access.rate_limit.concurrency

    def raw_writer_for(self, raw_path_prefix: Path | None) -> RawArtifactWriter:
        return self.raw_writer_factory(self.storage_layout).with_raw_path_prefix(raw_path_prefix)

    def bind_request_input_logger(
        self,
        request_input_index: int,
        request_input_count: int,
    ) -> StructuredLogger | None:
        if self.logger is None:
            return None
        return self.logger.bind(
            request_input_index=request_input_index,
            request_input_count=request_input_count,
            request_input_type=self.request_input_type,
        )

    def log_extraction_started(
        self,
        *,
        request_input_count: int,
        request_input_metadata: Mapping[str, str],
    ) -> None:
        if self.logger is None:
            return
        plan = self.plan
        self.logger.info(
            "api_extraction_started",
            request_url=self.base_request.full_url(),
            method=self.base_request.method,
            pagination_type=plan.source_config.access.pagination.type,
            page_size=plan.source_config.access.pagination.page_size,
            checkpoint_loaded=self.checkpoint_state is not None,
            timeout_seconds=self.base_request.timeout_seconds,
            request_input_type=self.request_input_type,
            request_input_count=request_input_count,
            bound_parameter_names=sorted(plan.source_config.access.parameter_bindings or {}),
            dead_letter_max_items=self.dead_letter_max_items,
            upstream_namespace=request_input_metadata.get("upstream_namespace"),
            upstream_table_name=request_input_metadata.get("upstream_table_name"),
            upstream_column_names=(
                request_input_metadata["upstream_column_names"].split(",")
                if request_input_metadata.get("upstream_column_names")
                else []
            ),
        )

    def log_extraction_finished(
        self,
        totals: RunTotals,
        *,
        request_input_count: int,
        dead_letter_count: int,
        dead_letter_skip_count: int,
    ) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "api_extraction_finished",
            request_count=totals.successful_requests,
            retry_count=totals.retry_count,
            attempt_count=totals.total_attempts,
            records_extracted=totals.records_extracted,
            artifact_count=len(totals.artifacts),
            checkpoint_value=totals.checkpoint_value,
            request_input_type=self.request_input_type,
            request_input_count=request_input_count,
            dead_letter_count=dead_letter_count,
            dead_letter_skipped_count=dead_letter_skip_count,
        )


def load_scoped_request_inputs(
    context: ApiExtractionContext,
    *,
    spark: Any = None,
) -> Sequence[dict[str, Any] | None]:
    """Load the run's request inputs inside the shared session scope.

    ``spark`` is a session *provider*, never a session: ``scoped_request_input_session`` is the
    single place extraction acquires one, and only ``iceberg_rows`` inputs ever need it.
    """
    try:
        with scoped_request_input_session(
            spark,
            context.plan.source_config.access.request_inputs,
        ) as request_input_session:
            return load_request_inputs(
                context.plan.source_config.access.request_inputs,
                spark=request_input_session,
            )
    except ApiRequestInputLoadError:
        if context.logger is not None:
            context.logger.exception(
                "api_request_input_loading_failed",
                request_input_type=context.request_input_type,
            )
        raise


def bind_request_input(
    context: ApiExtractionContext,
    request_input: dict[str, Any] | None,
    *,
    logger: StructuredLogger | None,
) -> ApiRequest:
    """Resolve one request input's parameter bindings onto the run's base request.

    The ``ApiParameterBindingError`` is logged and re-raised, and this call sits *outside* the
    caller's dead-letter ``try`` on purpose: a binding failure is a config bug affecting every
    input, so dead-lettering it would turn one fail-fast error into N silent skips.
    """
    try:
        bound_params = resolve_parameter_bindings(
            context.plan.source_config.access.parameter_bindings,
            request_input=request_input,
            checkpoint_value=context.request_checkpoint_value,
        )
        path_params, query_bound_params = split_path_and_query_params(
            context.base_request.url, bound_params
        )
        request_params = merge_request_params(
            context.plan.source_config.access.params,
            query_bound_params,
        )
    except ApiParameterBindingError:
        if logger is not None:
            logger.exception(
                "api_parameter_binding_failed",
                request_input_field_names=_request_input_field_names(request_input),
            )
        raise

    request = context.base_request
    if path_params:
        request = request.with_url(context.base_request.url.format_map(path_params))
    if request_params:
        request = request.with_params(request_params)

    if logger is not None:
        logger.info(
            "api_request_input_bound",
            bound_parameter_names=sorted(bound_params),
            request_parameter_names=sorted(request_params),
        )
    return request


def dispatch_pages(
    executor: ApiRequestExecutor,
    context: ApiExtractionContext,
    *,
    totals: RunTotals,
    **loop_kwargs: Any,
) -> RequestInputExtraction:
    """Walk one request input's pages, concurrently when the config and paginator allow it."""
    paginator = context.paginator
    if _supports_concurrent_pagination(paginator, context.concurrency):
        # Set before the call, so a run that dead-letters mid-input still reports the
        # concurrency-only metadata block for the requests it did make.
        totals.concurrent_pagination_used = True
        return extract_concurrent_pages(
            executor,
            context.plan,
            api_hook=context.api_hook,
            paginator=paginator,
            checkpoint_state=context.checkpoint_state,
            throttle=context.throttle,
            **loop_kwargs,
        )
    return extract_sequential_pages(
        executor,
        context.plan,
        api_hook=context.api_hook,
        paginator=paginator,
        checkpoint_state=context.checkpoint_state,
        throttle=context.throttle,
        **loop_kwargs,
    )


def run_api_extraction(
    executor: ApiRequestExecutor,
    context: ApiExtractionContext,
    *,
    request_inputs: Sequence[dict[str, Any] | None],
    progress_store: ExtractionProgressStore,
    dead_letter_store: DeadLetterStore,
) -> ApiExtractionOutcome:
    """Run every request input, recording failures as dead letters while the budget allows."""
    plan = context.plan
    request_input_count = len(request_inputs)
    request_input_metadata = _request_input_metadata(plan, request_input_count)
    context.log_extraction_started(
        request_input_count=request_input_count,
        request_input_metadata=request_input_metadata,
    )

    resume = ResumeState.load(
        plan,
        progress_store=progress_store,
        dead_letter_store=dead_letter_store,
        logger=context.logger,
    )
    raw_path_prefix = _raw_run_path_prefix(plan, resume.progress)
    raw_writer = context.raw_writer_for(raw_path_prefix)

    totals = RunTotals()
    completed_inputs: list[tuple[str, int]] = list(resume.completed_by_key.items())
    dead_letter_state = resume.dead_letter_state
    dead_letter_keys = set(resume.dead_letter_keys)
    dead_letter_skip_count = 0

    for request_input_index, request_input in enumerate(request_inputs, start=1):
        request_input_key = _request_input_key(request_input)
        request_input_logger = context.bind_request_input_logger(
            request_input_index,
            request_input_count,
        )

        if request_input_key in resume.completed_by_key:
            totals.artifacts.extend(
                recover_completed_input(
                    plan,
                    context.storage_layout,
                    request_input_key=request_input_key,
                    file_index=resume.completed_by_key[request_input_key],
                    request_input_count=request_input_count,
                    raw_path_prefix=raw_path_prefix,
                    logger=request_input_logger,
                )
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

        file_index = resume.file_index_for(request_input_key, request_input_index)

        if request_input_logger is not None:
            request_input_logger.info(
                "api_request_input_started",
                request_input_field_names=_request_input_field_names(request_input),
            )

        request_input_base_request = bind_request_input(
            context,
            request_input,
            logger=request_input_logger,
        )

        pagination_state = context.paginator.initial_state(request_input_base_request)

        pre_artifacts, pagination_state = rehydrate_resumed_input(
            plan,
            context.storage_layout,
            context.paginator,
            resume,
            request_input_key=request_input_key,
            pagination_state=pagination_state,
            file_index=file_index,
            request_input_count=request_input_count,
            logger=request_input_logger,
        )
        totals.artifacts.extend(pre_artifacts)

        try:
            extracted = dispatch_pages(
                executor,
                context,
                totals=totals,
                base_request=request_input_base_request,
                pagination_state=pagination_state,
                checkpoint_value=totals.checkpoint_value,
                raw_writer=raw_writer,
                logger=request_input_logger,
                request_input_index=file_index,
                request_input_count=request_input_count,
                completed_inputs=completed_inputs,
                current_input_key=request_input_key,
                current_input_index=file_index,
                progress_store=progress_store,
                prior_artifact_count=len(totals.artifacts),
                raw_path_prefix=raw_path_prefix,
            )
        except Exception as exc:
            if request_input_logger is not None:
                request_input_logger.exception("api_request_execution_failed")
            dead_letter_state = dead_letter_store.record(
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
                    dead_letter_max_items=context.dead_letter_max_items,
                    error_type=type(exc).__name__,
                )
            if not can_continue_after_dead_letter(
                total_item_count=request_input_count,
                dead_letter_count=dead_letter_state.entry_count,
                dead_letter_max_items=context.dead_letter_max_items,
            ):
                raise
            dead_letter_skip_count += 1
            continue

        totals.absorb(extracted)
        completed_inputs.append((request_input_key, file_index))

        if request_input_logger is not None:
            request_input_logger.info(
                "api_request_input_finished",
                request_count=extracted.successful_requests,
                retry_count=max(
                    extracted.total_attempts - extracted.successful_requests,
                    0,
                ),
                attempt_count=extracted.total_attempts,
                records_extracted=extracted.records_extracted,
                artifact_count=len(extracted.artifacts),
                checkpoint_value=extracted.checkpoint_value,
                speculative_request_count=extracted.speculative_requests,
                past_end_status=extracted.past_end_status,
            )

    context.log_extraction_finished(
        totals,
        request_input_count=request_input_count,
        dead_letter_count=(dead_letter_state.entry_count if dead_letter_state else 0),
        dead_letter_skip_count=dead_letter_skip_count,
    )

    _clear_progress_if_run_is_complete(
        plan,
        progress_store,
        dead_letter_skip_count=dead_letter_skip_count,
        logger=context.logger,
    )

    return ApiExtractionOutcome(
        totals=totals,
        request_input_count=request_input_count,
        request_input_metadata=request_input_metadata,
        dead_letter_count=(
            dead_letter_state.entry_count if dead_letter_state is not None else 0
        ),
        dead_letter_skip_count=dead_letter_skip_count,
        raw_path_prefix=raw_path_prefix,
    )


def _clear_progress_if_run_is_complete(
    plan: ExecutionPlan,
    progress_store: ExtractionProgressStore,
    *,
    dead_letter_skip_count: int,
    logger: StructuredLogger | None,
) -> None:
    """Discard resume state only when there is nothing left to resume.

    Falling out of the request-input loop is not the same as having extracted anything. An
    input that was skipped because a previous run dead-lettered it — or dead-lettered in this
    run and continued past — leaves the loop by the same door as a completed one, and the
    unconditional clear that used to live here then deleted the progress record on its way
    out. That is how a run extracting *zero* records destroyed the position a three-hour
    extraction had reached: the dead-letter check runs before rehydration, so the skip
    happened first and the clear finished the job.

    Keeping the record costs nothing when it is stale — a later successful run clears it —
    while deleting it early is unrecoverable.
    """
    if dead_letter_skip_count:
        if logger is not None:
            logger.info(
                "api_extraction_progress_retained",
                dead_letter_skipped_count=dead_letter_skip_count,
                reason="request inputs were skipped or dead-lettered; the run is incomplete",
            )
        return

    progress_store.clear(plan)


def build_extraction_result(
    context: ApiExtractionContext,
    outcome: ApiExtractionOutcome,
    *,
    dead_letter_store: DeadLetterStore,
) -> ExtractionResult:
    """Assemble the run's ``ExtractionResult``.

    The metadata mapping is a contract — ``test_concurrent_sequential_equivalence`` diffs it
    key by key — so it stays one literal rather than being split across sub-builders.
    """
    plan = context.plan
    totals = outcome.totals

    extraction_result = ExtractionResult.from_plan(
        plan,
        tuple(totals.artifacts),
        records_extracted=totals.records_extracted,
        checkpoint_value=totals.checkpoint_value,
        metadata={
            "request_count": str(totals.successful_requests),
            "retry_count": str(totals.retry_count),
            "attempt_count": str(totals.total_attempts),
            "pagination_type": plan.source_config.access.pagination.type,
            "auth_type": plan.source_config.access.auth.type,
            "checkpoint_loaded": str(context.checkpoint_state is not None).lower(),
            "records_extracted": str(totals.records_extracted),
            "dead_letter_count": str(outcome.dead_letter_count),
            "dead_letter_skipped_count": str(outcome.dead_letter_skip_count),
            "pagination_concurrency": str(context.concurrency),
            **(
                {"raw_path_prefix": str(outcome.raw_path_prefix)}
                if outcome.raw_path_prefix is not None
                else {}
            ),
            **_speculation_metadata(
                concurrent_pagination_used=totals.concurrent_pagination_used,
                speculative_requests=totals.speculative_requests,
                discarded_requests=totals.discarded_requests,
                past_end_terminated_count=totals.past_end_terminated_count,
                past_end_status=totals.past_end_status,
                lookahead_ceiling_source=totals.lookahead_ceiling_source,
                total_records_reported=totals.total_records_reported,
            ),
            **outcome.request_input_metadata,
        },
    )
    if outcome.dead_letter_count > 0:
        extraction_result = extraction_result.with_metadata(
            "dead_letter_path",
            str(dead_letter_store.path(plan)),
        )
    return extraction_result
