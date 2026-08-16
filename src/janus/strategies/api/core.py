"""The API strategy façade.

What remains here is the contract, not the machinery: the :class:`ApiStrategy` lifecycle
methods the planner calls, the :class:`ApiHook` extension points a source implements,
``CONCURRENCY_ONLY_METADATA_KEYS`` declaration, and compatibility re-exports for names that
were defined in this module before the package was split.

Where the machinery went:

* ``requests.py`` — per-request mechanics (prepare → send → persist → count).
* ``pagination_loop.py`` — the sequential and concurrent page loops.
* ``extraction.py`` — orchestration: request-input iteration, resume, dead-letters, totals.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from janus.checkpoints import (
    CheckpointState,
    CheckpointStore,
    DeadLetterStore,
    ExtractionProgressStore,
)
from janus.models import (
    ExecutionPlan,
    ExtractionResult,
    SourceConfig,
    WriteResult,
)
from janus.strategies.base import BaseStrategy, SourceHook
from janus.strategies.common import _default_storage_layout, _request_input_key
from janus.strategies.http import (
    ApiRequest,
    ApiResponse,
    ApiTransport,
    UrllibApiTransport,
)
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

# Compatibility re-exports: every name below was defined in this module before the package
# was split, and is imported from here by tests, hooks or downstream code (FR-3).
from .artifacts import _raw_relative_path
from .errors import (
    ApiPastEndConflictError,
    ApiPayloadError,
    ApiResponseError,
    ApiStrategyError,
)
from .extraction import (
    ApiExtractionContext,
    build_extraction_result,
    load_scoped_request_inputs,
    run_api_extraction,
)
from .pagination import PaginationState
from .pagination_loop import RequestInputExtraction
from .requests import _RETRY_POLICY, ApiRequestExecutor, ProcessedApiRequest
from .speculation import SubmittedApiRequest

__all__ = [
    "CONCURRENCY_ONLY_METADATA_KEYS",
    "SUPPORTED_API_PAYLOAD_FORMATS",
    "_RETRY_POLICY",
    "ApiHook",
    "ApiPastEndConflictError",
    "ApiPayloadError",
    "ApiRequestExecutor",
    "ApiResponseError",
    "ApiStrategy",
    "ApiStrategyError",
    "ProcessedApiRequest",
    "RequestInputExtraction",
    "SubmittedApiRequest",
    "_raw_relative_path",
    "_request_input_key",
]

SUPPORTED_API_PAYLOAD_FORMATS = frozenset({"binary", "json", "jsonl", "text"})

CONCURRENCY_ONLY_METADATA_KEYS = frozenset(
    {
        "speculative_request_count",
        "speculative_discarded_count",
        "past_end_terminated_count",
        "past_end_status",
        "lookahead_ceiling_source",
        "total_records_reported",
    }
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

    def resolve_total_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> int | None:
        """Return the total record count for this request input, when the API exposes one.

        Overriding this caps concurrent look-ahead exactly, removing all speculative
        over-fetch. Returning ``None`` (the default) falls back to the generic payload
        discovery in ``speculation.py``.
        """
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
    #: Per-request mechanics. ``None`` builds one from this strategy's own dependencies, so
    #: every constructor keyword above keeps working untouched; the explicit field exists so
    #: a test can inject a fake executor.
    request_executor: ApiRequestExecutor | None = None

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
        executor = self._build_request_executor()
        context = ApiExtractionContext.build(
            plan,
            api_hook=hook if isinstance(hook, ApiHook) else None,
            executor=executor,
            storage_layout_factory=self.storage_layout_factory,
            raw_writer_factory=self.raw_writer_factory,
            checkpoint_store=self.checkpoint_store,
            logger=self._bind_logger(plan),
            clock=self.clock,
            sleeper=self.sleeper,
        )
        request_inputs = load_scoped_request_inputs(context, spark=spark)
        outcome = run_api_extraction(
            executor,
            context,
            request_inputs=request_inputs,
            progress_store=self.progress_store,
            dead_letter_store=self.dead_letter_store,
        )
        extraction_result = build_extraction_result(
            context,
            outcome,
            dead_letter_store=self.dead_letter_store,
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

    def _build_request_executor(self) -> ApiRequestExecutor:
        return self.request_executor or ApiRequestExecutor(
            transport_factory=self.transport_factory,
            sleeper=self.sleeper,
            env_reader=self.env_reader,
        )

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

    def _bind_logger(self, plan: ExecutionPlan) -> StructuredLogger | None:
        if self.logger is None:
            return None
        return self.logger.bind(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            strategy_family=self.strategy_family,
        )
