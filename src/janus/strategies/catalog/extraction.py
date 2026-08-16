"""Orchestration for a catalog run: what to request next, and what to do when it fails.

*Orchestration* decides which request input is next, whether a previous attempt already
finished it, and whether a failure ends the run or is recorded and skipped. *Mechanics* —
performing one request, turning the answer into an artifact — are ``requests.py``, and the
page loop beneath them ``pagination_loop.py``. Nothing here speaks HTTP, which
``tests/unit/strategies/catalog/test_catalog_boundary.py`` enforces.
:class:`CatalogExtractionContext` holds everything resolved once per run; the state the loop
carries across inputs is ``run_state.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from janus.models import ExecutionPlan, ExtractedArtifact, ExtractionResult
from janus.runtime.spark_lifecycle import scoped_request_input_session
from janus.strategies.api import build_paginator
from janus.strategies.api.pagination import ApiPaginator
from janus.strategies.api.request_inputs import (
    ApiRequestInputLoadError,
    load_request_inputs,
)
from janus.strategies.common import _raw_run_path_prefix, _request_input_key
from janus.strategies.http import (
    ApiRequest,
    HttpRequestThrottle,
    checkpoint_request_value,
)
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

from .artifacts import _persist_normalized_records
from .document import ENTITY_TYPE_ORDER
from .metadata import _apply_per_input_params, _catalog_request_input_dead_letter_metadata
from .pagination_loop import extract_request_input_pages
from .requests import CatalogRequestExecutor, CatalogRequestHook
from .run_state import (
    CatalogExtractionOutcome,
    CatalogRunTotals,
    ResumeState,
    recover_completed_input,
    rehydrate_resumed_input,
)


@dataclass(frozen=True, slots=True)
class CatalogExtractionContext:
    """Everything resolved once per run, before the first request input is touched."""

    plan: ExecutionPlan
    catalog_hook: CatalogRequestHook | None
    storage_layout: StorageLayout
    checkpoint_state: CheckpointState | None
    request_checkpoint_value: str | None
    base_request: ApiRequest
    paginator: ApiPaginator
    throttle: HttpRequestThrottle
    logger: StructuredLogger | None
    raw_writer_factory: Callable[[StorageLayout], RawArtifactWriter]

    @classmethod
    def build(
        cls,
        plan: ExecutionPlan,
        *,
        catalog_hook: CatalogRequestHook | None,
        executor: CatalogRequestExecutor,
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
            catalog_hook=catalog_hook,
            storage_layout=storage_layout_factory(plan),
            checkpoint_state=checkpoint_state,
            request_checkpoint_value=checkpoint_request_value(plan, checkpoint_state),
            base_request=executor.build_base_request(plan, checkpoint_state, catalog_hook),
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
    def parameter_bindings(self) -> Any:
        return self.plan.source_config.access.parameter_bindings

    @property
    def dead_letter_max_items(self) -> int:
        return self.plan.source_config.extraction.dead_letter_max_items

    def raw_writer_for(self, raw_path_prefix: Path | None) -> RawArtifactWriter:
        return self.raw_writer_factory(self.storage_layout).with_raw_path_prefix(raw_path_prefix)

    def log_extraction_started(self, *, request_input_count: int) -> None:
        if self.logger is None:
            return
        plan = self.plan
        self.logger.info(
            "catalog_extraction_started",
            request_url=self.base_request.full_url(),
            method=self.base_request.method,
            pagination_type=plan.source_config.access.pagination.type,
            page_size=plan.source_config.access.pagination.page_size,
            checkpoint_loaded=self.checkpoint_state is not None,
            timeout_seconds=self.base_request.timeout_seconds,
            request_input_type=self.request_input_type,
            request_input_count=request_input_count,
            dead_letter_max_items=self.dead_letter_max_items,
        )

    def log_extraction_finished(
        self,
        totals: CatalogRunTotals,
        *,
        normalized_artifacts: Sequence[ExtractedArtifact],
        dead_letter_count: int,
        dead_letter_skip_count: int,
    ) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "catalog_extraction_finished",
            request_count=totals.successful_requests,
            retry_count=totals.retry_count,
            attempt_count=totals.total_attempts,
            records_extracted=totals.records_extracted,
            page_record_count=totals.page_record_total,
            raw_artifact_count=len(totals.artifacts),
            normalized_artifact_count=len(normalized_artifacts),
            artifact_count=len(totals.artifacts) + len(normalized_artifacts),
            checkpoint_value=totals.checkpoint_value,
            organizations_extracted=totals.entity_count("organization"),
            groups_extracted=totals.entity_count("group"),
            datasets_extracted=totals.entity_count("dataset"),
            resources_extracted=totals.entity_count("resource"),
            dead_letter_count=dead_letter_count,
            dead_letter_skipped_count=dead_letter_skip_count,
        )


def load_scoped_request_inputs(
    context: CatalogExtractionContext,
    *,
    spark: Any = None,
) -> Sequence[dict[str, Any] | None]:
    """Load the run's request inputs inside the shared session scope.

    ``spark`` is a session *provider*, never a session: ``scoped_request_input_session`` is
    the single place extraction acquires one, and only ``iceberg_rows`` inputs ever need it.
    """
    request_inputs_config = context.plan.source_config.access.request_inputs
    try:
        with scoped_request_input_session(
            spark,
            request_inputs_config,
        ) as request_input_session:
            return load_request_inputs(
                request_inputs_config,
                spark=request_input_session,
            )
    except ApiRequestInputLoadError:
        if context.logger is not None:
            context.logger.exception(
                "catalog_request_input_loading_failed",
                request_input_type=request_inputs_config.type,
            )
        raise


def bind_request_input(
    context: CatalogExtractionContext,
    request_input: dict[str, Any] | None,
) -> ApiRequest:
    """Resolve one request input's parameter bindings onto the run's base request.

    Called before the skip checks and *outside* the loop's dead-letter ``try``, for two
    reasons: replaying an already-finished input needs the same request the original attempt
    sent, and a binding failure is a config bug affecting every input — dead-lettering it
    would turn one fail-fast error into N silent skips.
    """
    return _apply_per_input_params(
        context.base_request,
        context.parameter_bindings,
        request_input,
        checkpoint_value=context.request_checkpoint_value,
    )


def run_catalog_extraction(
    executor: CatalogRequestExecutor,
    context: CatalogExtractionContext,
    *,
    request_inputs: Sequence[dict[str, Any] | None],
    progress_store: ExtractionProgressStore,
    dead_letter_store: DeadLetterStore,
) -> CatalogExtractionOutcome:
    """Run every request input, recording failures as dead letters while the budget allows."""
    plan = context.plan
    request_input_count = len(request_inputs)
    context.log_extraction_started(request_input_count=request_input_count)

    resume = ResumeState.load(
        plan,
        progress_store=progress_store,
        dead_letter_store=dead_letter_store,
        logger=context.logger,
    )
    raw_path_prefix = _raw_run_path_prefix(plan, resume.progress)
    raw_writer = context.raw_writer_for(raw_path_prefix)

    totals = CatalogRunTotals()
    completed_inputs: list[tuple[str, int]] = list(resume.completed_by_key.items())
    dead_letter_state = resume.dead_letter_state
    dead_letter_keys = set(resume.dead_letter_keys)
    dead_letter_skip_count = 0

    with executor.open_session() as session:
        for request_input_index, request_input in enumerate(request_inputs, start=1):
            request_input_key = _request_input_key(request_input)
            per_input_request = bind_request_input(context, request_input)

            if request_input_key in resume.completed_by_key:
                recover_completed_input(
                    plan,
                    context.storage_layout,
                    context.paginator,
                    totals,
                    per_input_request=per_input_request,
                    checkpoint_state=context.checkpoint_state,
                    completed_input_index=resume.completed_by_key[request_input_key],
                    request_input_count=request_input_count,
                )
                continue

            if request_input_key in dead_letter_keys:
                dead_letter_skip_count += 1
                if context.logger is not None:
                    context.logger.info(
                        "catalog_request_input_skipped_dead_letter",
                        input_key=request_input_key,
                        dead_letter_count=len(dead_letter_keys),
                    )
                continue

            pre_artifacts, pagination_state = rehydrate_resumed_input(
                plan,
                context.storage_layout,
                context.paginator,
                resume,
                request_input_key=request_input_key,
                per_input_request=per_input_request,
                request_input_index=request_input_index,
                request_input_count=request_input_count,
                logger=context.logger,
            )

            try:
                extracted = extract_request_input_pages(
                    executor,
                    session,
                    plan,
                    catalog_hook=context.catalog_hook,
                    per_input_request=per_input_request,
                    paginator=context.paginator,
                    pagination_state=pagination_state,
                    pre_artifacts=pre_artifacts,
                    checkpoint_state=context.checkpoint_state,
                    checkpoint_value=totals.checkpoint_value,
                    raw_writer=raw_writer,
                    throttle=context.throttle,
                    logger=context.logger,
                    request_input_index=request_input_index,
                    request_input_count=request_input_count,
                    completed_inputs=completed_inputs,
                    request_input_key=request_input_key,
                    progress_store=progress_store,
                    run_records=totals.normalized_records,
                    prior_page_record_total=totals.page_record_total,
                    prior_artifact_count=len(totals.artifacts),
                    raw_path_prefix=raw_path_prefix,
                )
            except Exception as exc:
                if context.logger is not None:
                    context.logger.exception("catalog_request_execution_failed")
                dead_letter_state = dead_letter_store.record(
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
                if context.logger is not None:
                    context.logger.info(
                        "catalog_request_input_dead_lettered",
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

            totals.absorb(plan, extracted)
            completed_inputs.append((request_input_key, request_input_index))

    progress_store.clear(plan)
    # Persist → log → assemble, in that order: the second pass over the accumulated entities
    # produces the artifacts the finish log counts and the ``ExtractionResult`` carries.
    normalized_artifacts = _persist_normalized_records(
        plan,
        raw_writer,
        totals.normalized_records,
    )
    context.log_extraction_finished(
        totals,
        normalized_artifacts=normalized_artifacts,
        dead_letter_count=(dead_letter_state.entry_count if dead_letter_state else 0),
        dead_letter_skip_count=dead_letter_skip_count,
    )

    return CatalogExtractionOutcome(
        totals=totals,
        normalized_artifacts=normalized_artifacts,
        dead_letter_count=(
            dead_letter_state.entry_count if dead_letter_state is not None else 0
        ),
        dead_letter_skip_count=dead_letter_skip_count,
        raw_path_prefix=raw_path_prefix,
    )


def build_extraction_result(
    context: CatalogExtractionContext,
    outcome: CatalogExtractionOutcome,
    *,
    dead_letter_store: DeadLetterStore,
) -> ExtractionResult:
    """Assemble the run's ``ExtractionResult``.

    The per-entity-type counts and ``entity_types_emitted`` are part of the *generic* catalog
    contract rather than a source-specific list — they move as one literal and stay computed
    from ``ENTITY_TYPE_ORDER``.
    """
    plan = context.plan
    totals = outcome.totals

    extraction_result = ExtractionResult.from_plan(
        plan,
        tuple(totals.artifacts + outcome.normalized_artifacts),
        records_extracted=totals.records_extracted,
        checkpoint_value=totals.checkpoint_value,
        metadata={
            "request_count": str(totals.successful_requests),
            "retry_count": str(totals.retry_count),
            "attempt_count": str(totals.total_attempts),
            "pagination_type": plan.source_config.access.pagination.type,
            "checkpoint_loaded": str(context.checkpoint_state is not None).lower(),
            "raw_page_count": str(len(totals.artifacts)),
            "page_record_count": str(totals.page_record_total),
            "normalized_artifact_count": str(len(outcome.normalized_artifacts)),
            "organizations_extracted": str(totals.entity_count("organization")),
            "groups_extracted": str(totals.entity_count("group")),
            "datasets_extracted": str(totals.entity_count("dataset")),
            "resources_extracted": str(totals.entity_count("resource")),
            "dead_letter_count": str(outcome.dead_letter_count),
            "dead_letter_skipped_count": str(outcome.dead_letter_skip_count),
            "entity_types_emitted": ",".join(
                entity_type
                for entity_type in ENTITY_TYPE_ORDER
                if totals.normalized_records[entity_type]
            ) or "none",
            **(
                {"raw_path_prefix": str(outcome.raw_path_prefix)}
                if outcome.raw_path_prefix is not None
                else {}
            ),
        },
    )
    if outcome.dead_letter_count > 0:
        extraction_result = extraction_result.with_metadata(
            "dead_letter_path",
            str(dead_letter_store.path(plan)),
        )
    return extraction_result
