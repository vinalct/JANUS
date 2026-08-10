"""Per-request mechanics for the catalog family: build → send → persist → count."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from janus.checkpoints import CheckpointState
from janus.models import ExecutionPlan
from janus.strategies.api import PaginationState
from janus.strategies.api.pagination import ApiPaginator
from janus.strategies.common import _freeze_string_mapping, _stringify_mapping
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
from janus.writers import PersistedArtifact, RawArtifactWriter

from .artifacts import _raw_relative_path
from .document import _root_batches
from .errors import CatalogPayloadError, CatalogResponseError, CatalogStrategyError

#: Configuration of the *one* call this module makes into the shared retry loop.
_RETRY_POLICY = RetryErrorPolicy(
    transport_error_factory=CatalogStrategyError,
    response_error_factory=CatalogResponseError,
    retry_log_event="catalog_retry_scheduled",
)


class CatalogRequestHook(Protocol):
    """The hook methods one catalog request consults, as a structural type.

    ``CatalogHook`` itself is declared in ``core.py``, which imports this module — naming
    the class here would close an import cycle. ``CatalogHook`` satisfies this protocol as
    written, and so does any subclass of it.
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

    def checkpoint_params(
        self,
        plan: ExecutionPlan,
        checkpoint_value: str,
    ) -> Mapping[str, str] | None: ...

    def page_records(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
    ) -> Sequence[Any] | None: ...


@dataclass(frozen=True, slots=True)
class CatalogRequestExecutor:
    """Performs one catalog request end to end: build → send → persist → count.

    Depends on exactly three of the strategy's collaborators, which is what makes it a
    small object rather than a bag of functions taking ``self``.
    """

    transport_factory: Callable[[], ApiTransport] = UrllibApiTransport
    sleeper: Callable[[float], None] = time.sleep
    env_reader: Callable[[str], str | None] = os.getenv

    @contextmanager
    def open_session(self) -> Iterator[CatalogRequestSession]:
        """Open the one client a catalog run sends every request through.

        A catalog run holds a single connection for all of its request inputs, where the
        API family opens one per page loop — the difference is real and is preserved here.
        Yielding a session rather than the client itself is what lets the orchestration
        layer own that lifetime without ever holding something HTTP-shaped.
        """
        with ApiClient(self.transport_factory()) as client:
            yield CatalogRequestSession(executor=self, client=client)

    def build_base_request(
        self,
        plan: ExecutionPlan,
        checkpoint_state: CheckpointState | None,
        catalog_hook: CatalogRequestHook | None,
    ) -> ApiRequest:
        source_access = plan.source_config.access
        request = ApiRequest(
            method=source_access.method,
            url=resolve_url(plan.source_config, family_label="Catalog"),
            timeout_seconds=source_access.timeout_seconds,
            headers=_freeze_string_mapping(source_access.headers or {}),
            params=_freeze_string_mapping(source_access.params or {}),
        )
        request = inject_auth(request, source_access.auth, env_reader=self.resolve_env_var)

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

    def prepare_request(
        self,
        plan: ExecutionPlan,
        per_input_request: ApiRequest,
        pagination_state: PaginationState,
        paginator: ApiPaginator,
        *,
        catalog_hook: CatalogRequestHook | None,
        checkpoint_state: CheckpointState | None,
        logger: StructuredLogger | None,
    ) -> ApiRequest:
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
        return request

    def apply_response_hooks(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
        *,
        catalog_hook: CatalogRequestHook | None,
    ) -> tuple[ApiResponse, Any]:
        """Let the source's hook reshape the answer before anything is derived from it."""
        if catalog_hook is not None:
            response = catalog_hook.handle_response(plan, request, response)

        if catalog_hook is not None:
            payload = catalog_hook.transform_payload(
                plan,
                request,
                response,
                payload,
            )
        return response, payload

    def decode_payload(self, plan: ExecutionPlan, response: ApiResponse) -> Any:
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

    def persist_raw_payload(
        self,
        plan: ExecutionPlan,
        raw_writer: RawArtifactWriter,
        *,
        response: ApiResponse,
        payload: Any,
        pagination_state: PaginationState,
        request_input_index: int = 1,
        request_input_count: int = 1,
    ) -> PersistedArtifact:
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

    def page_record_count(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        response: ApiResponse,
        payload: Any,
        *,
        hook: CatalogRequestHook | None,
    ) -> int:
        """Count the records on one page: the hook's answer first, the generic walk second.

        ``CatalogHook.page_records`` is the sanctioned place for a source whose envelope the
        structural walk cannot count. The fallback below stays generic.
        """
        if hook is not None:
            hook_records = hook.page_records(plan, request, response, payload)
            if hook_records is not None:
                return len(tuple(hook_records))

        batches = _root_batches(payload, plan.source.strategy_variant, path="payload")
        if not batches:
            return 0
        return len(batches[0].records)

    def resolve_env_var(self, name: str) -> str | None:
        value = self.env_reader(name)
        if value is not None:
            return value
        return os.getenv(name)


@dataclass(frozen=True, slots=True)
class CatalogRequestSession:
    """The open connection a catalog run sends every one of its requests through."""

    executor: CatalogRequestExecutor
    client: ApiClient

    def send(
        self,
        plan: ExecutionPlan,
        request: ApiRequest,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
    ) -> tuple[ApiResponse, Any, int]:
        """Send one request through the shared retry loop, decoding its payload.

        No ``terminal_status_codes``: a catalog request has no past-end status to read as a
        clean end-of-stream, so every non-success status is classified by the shared loop
        exactly as it is today.
        """
        return send_with_retries(
            plan,
            self.client,
            request,
            throttle,
            logger,
            policy=_RETRY_POLICY,
            sleeper=self.executor.sleeper,
            decode=lambda response: self.executor.decode_payload(plan, response),
            payload_error_types=(CatalogPayloadError,),
        )
