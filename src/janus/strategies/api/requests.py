"""Per-request mechanics for the API family: prepare → send → persist → count.

``ApiRequestExecutor`` performs *one* request end to end. It owns no loop and no policy:
it does not know about pagination, request inputs, resume or dead-letters — those are
orchestration and live in ``extraction.py`` / ``pagination_loop.py``.

It is the family-specific *composition* of the shared HTTP layer, not a second copy of it.
The retry loop lives in :mod:`janus.strategies.http.retry`, the throttle in
``strategies/http/throttle.py``, payload decoding in ``strategies/http/payload.py`` and the
URL/param binding helpers in ``strategies/http/binding.py``. ``send_request`` is a thin call
into ``send_with_retries`` carrying this family's decode callback, error factories and
``terminal_status_codes``; status-code classification belongs to that loop and must not be
re-derived here.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from janus.checkpoints import CheckpointState
from janus.models import ExecutionPlan, ExtractedArtifact
from janus.strategies.common import (
    _compare_checkpoint_values,
    _freeze_string_mapping,
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
)
from janus.utils.logging import StructuredLogger
from janus.writers import PersistedArtifact, RawArtifactWriter

from .artifacts import _raw_relative_path
from .errors import ApiPayloadError, ApiResponseError, ApiStrategyError
from .pagination import ApiPaginator, PaginationState
from .records import _default_records_from_payload, _lookup_field, _string_value
from .speculation import (
    SpeculativePaginationPolicy,
    TotalRecordsResolver,
    _resolve_total_records,
)

#: Configuration of the *one* call this module makes into the shared retry loop.
_RETRY_POLICY = RetryErrorPolicy(
    transport_error_factory=ApiStrategyError,
    response_error_factory=ApiResponseError,
    retry_log_event="api_retry_scheduled",
)


class ApiRequestHook(TotalRecordsResolver, Protocol):
    """The hook methods one request consults, as a structural type.

    ``ApiHook`` itself is declared in ``core.py``, which imports this module — naming the
    class here would close the cycle ``errors.py`` was split out to break. ``ApiHook``
    satisfies this protocol as written, and so does any subclass of it.
    """

    def prepare_request(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        *,
        checkpoint_state: CheckpointState | None = None,
        pagination_state: PaginationState | None = None,
    ) -> ApiRequest: ...

    def handle_response(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
    ) -> ApiResponse: ...

    def transform_payload(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> Any: ...

    def extract_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> Sequence[Any] | None: ...

    def resolve_next_cursor(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> str | None: ...

    def checkpoint_params(
        self,
        plan: ExecutionPlan,
        checkpoint_value: str,
    ) -> Mapping[str, str] | None: ...


@dataclass(frozen=True, slots=True)
class ProcessedApiRequest:
    """What one answered request contributed, before any loop decides what to do with it."""

    artifact: ExtractedArtifact
    records_extracted: int
    checkpoint_value: str | None
    next_pagination_state: PaginationState | None
    total_records_reported: int | None = None
    total_records_source: str | None = None


@dataclass(frozen=True, slots=True)
class ApiRequestExecutor:
    """Performs one API request end to end: prepare → send → persist → count.

    Depends on exactly three of the strategy's collaborators, which is what makes it a
    small object rather than a bag of functions taking ``self``.
    """

    transport_factory: Callable[[], ApiTransport] = UrllibApiTransport
    sleeper: Callable[[float], None] = time.sleep
    env_reader: Callable[[str], str | None] = os.getenv

    def build_base_request(
        self,
        plan: ExecutionPlan,
        checkpoint_state: CheckpointState | None,
        api_hook: ApiRequestHook | None,
    ) -> ApiRequest:
        source_access = plan.source_config.access
        request = ApiRequest(
            method=source_access.method,
            url=resolve_url(plan.source_config, family_label="API"),
            timeout_seconds=source_access.timeout_seconds,
            headers=_freeze_string_mapping(source_access.headers or {}),
            params=(),
        )
        request = inject_auth(request, source_access.auth, env_reader=self.resolve_env_var)

        checkpoint_value = checkpoint_request_value(plan, checkpoint_state)
        checkpoint_params = default_checkpoint_params(plan, checkpoint_value)
        if api_hook is not None and checkpoint_value is not None:
            hook_checkpoint_params = api_hook.checkpoint_params(plan, checkpoint_value)
            if hook_checkpoint_params is not None:
                checkpoint_params = _stringify_mapping(hook_checkpoint_params)

        if checkpoint_params:
            request = request.with_params(checkpoint_params)
        return request

    def prepare_request(
        self,
        plan: ExecutionPlan,
        base_request: ApiRequest,
        pagination_state: PaginationState,
        paginator: ApiPaginator,
        *,
        api_hook: ApiRequestHook | None,
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

    def fetch_with_dedicated_client(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
        *,
        terminal_status_codes: frozenset[int] = frozenset(),
    ) -> tuple[ApiResponse, Any, int]:
        with ApiClient(self.transport_factory()) as client:
            return self.send_request(
                plan,
                client,
                request,
                throttle,
                logger,
                terminal_status_codes=terminal_status_codes,
            )

    def send_request(
        self,
        plan: ExecutionPlan,
        client: ApiClient,
        request: ApiRequest,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
        *,
        terminal_status_codes: frozenset[int] = frozenset(),
    ) -> tuple[ApiResponse, Any, int]:
        """Send one request; ``terminal_status_codes`` returns instead of raising.

        A terminal response carries a ``None`` payload — callers passing a non-empty
        set must check ``response.status_code`` before consuming it.
        """
        return send_with_retries(
            plan,
            client,
            request,
            throttle,
            logger,
            policy=_RETRY_POLICY,
            sleeper=self.sleeper,
            decode=lambda response: self.decode_payload(plan, response),
            payload_error_types=(ApiPayloadError,),
            terminal_status_codes=terminal_status_codes,
        )

    def process_response(
        self,
        plan: ExecutionPlan,
        raw_writer: RawArtifactWriter,
        paginator: ApiPaginator,
        pagination_state: PaginationState,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
        attempts_used: int,
        checkpoint_value: str | None,
        *,
        api_hook: ApiRequestHook | None,
        logger: StructuredLogger | None,
        total_records: int,
        request_input_index: int,
        request_input_count: int,
        speculation_policy: SpeculativePaginationPolicy | None = None,
    ) -> ProcessedApiRequest:
        if api_hook is not None:
            response = api_hook.handle_response(plan, request, response)

        if api_hook is not None:
            payload = api_hook.transform_payload(plan, request, response, payload)

        records = self.extract_records(plan, request, response, payload, api_hook)
        resolved_checkpoint_value = self.resolve_checkpoint_value(
            checkpoint_value,
            records,
            checkpoint_field=plan.checkpoint_field,
        )

        persisted = self.persist_raw_payload(
            plan,
            raw_writer,
            response=response,
            payload=payload,
            pagination_state=pagination_state,
            request_input_index=request_input_index,
            request_input_count=request_input_count,
        )

        total_records_reported, total_records_source = _resolve_total_records(
            plan,
            request,
            response,
            payload,
            api_hook=api_hook,
            policy=speculation_policy,
            logger=logger,
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
            total_records_reported=total_records_reported,
            total_records_source=total_records_source,
        )

    def decode_payload(self, plan: ExecutionPlan, response: ApiResponse) -> Any:
        try:
            return decode_payload(
                plan.source_config.access.format,
                response,
                family_label="API",
            )
        except PayloadDecodeError as exc:
            raise ApiPayloadError(str(exc)) from exc.__cause__

    def extract_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
        api_hook: ApiRequestHook | None,
    ) -> tuple[Any, ...]:
        if api_hook is not None:
            hook_records = api_hook.extract_records(plan, request, response, payload)
            if hook_records is not None:
                return tuple(hook_records)
        return tuple(_default_records_from_payload(payload))

    def persist_raw_payload(
        self,
        plan: ExecutionPlan,
        raw_writer: RawArtifactWriter,
        *,
        response: ApiResponse,
        payload: Any,
        pagination_state: PaginationState,
        request_input_index: int,
        request_input_count: int,
    ) -> PersistedArtifact:
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

    def resolve_checkpoint_value(
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

    def resolve_env_var(self, name: str) -> str | None:
        value = self.env_reader(name)
        if value is not None:
            return value
        return os.getenv(name)
