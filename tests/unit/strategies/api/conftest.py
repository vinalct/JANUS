"""Shared scripted transport for the concurrent-pagination test suite.

The fixtures in `test_api_strategy.py` cannot express the failure modes this suite
needs: `FakeTransport` answers by call order (meaningless once several pages are in
flight) and `ConcurrentPageTransport` hard-codes `200` for every page. The transport
here answers by *pagination key*, so a test can script "page 4 is a 404" or "offset 6
is a 416" without knowing which thread gets there first.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from janus.checkpoints import ExtractionProgressStore
from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api import ApiStrategy
from janus.strategies.http.transport import ApiRequest, ApiResponse
from janus.utils.logging import StructuredLogger
from janus.utils.storage import StorageLayout


@dataclass(frozen=True, slots=True)
class PageScript:
    """One scripted response for one pagination index."""

    status_code: int = 200
    records: tuple[dict[str, str], ...] = ()
    extra_payload: Mapping[str, Any] = field(default_factory=dict)
    body: bytes | None = None
    latency_seconds: float = 0.0

    def encoded_body(self) -> bytes:
        if self.body is not None:
            return self.body
        if not 200 <= self.status_code < 300:
            return json.dumps(
                {"error": "scripted failure", "status": self.status_code}
            ).encode("utf-8")
        payload: dict[str, Any] = {"records": [dict(record) for record in self.records]}
        payload.update(self.extra_payload)
        return json.dumps(payload).encode("utf-8")


#: Answer for an unscripted key: an empty ``200``, i.e. the well-behaved past-end API.
EMPTY_PAGE_SCRIPT = PageScript()


class ScriptedPageTransport:
    """Fake transport that answers by *pagination key*, not by call order."""

    def __init__(
        self,
        scripts: Mapping[int, PageScript],
        *,
        key_param: str = "page",
        default_script: PageScript = EMPTY_PAGE_SCRIPT,
        scope_param: str | None = None,
        scoped_scripts: Mapping[str, Mapping[int, PageScript]] | None = None,
    ) -> None:
        self._scripts = dict(scripts)
        self._key_param = key_param
        self._default_script = default_script
        self._scope_param = scope_param
        self._scoped_scripts = {
            scope: dict(scope_scripts)
            for scope, scope_scripts in (scoped_scripts or {}).items()
        }
        self._lock = threading.Lock()
        self._active_requests = 0
        self.requests: list[ApiRequest] = []
        self.requested_keys: list[int] = []
        self.max_active_requests = 0
        self.opened = False
        self.closed = False

    @property
    def send_count(self) -> int:
        with self._lock:
            return len(self.requested_keys)

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request: ApiRequest) -> ApiResponse:
        query = parse_qs(urlsplit(request.full_url()).query)
        if self._key_param not in query:
            raise AssertionError(
                f"Scripted transport expected a '{self._key_param}' query parameter, "
                f"got {sorted(query)}"
            )
        key = int(query[self._key_param][0])
        script = self._resolve_script(key, query)

        with self._lock:
            self.requests.append(request)
            self.requested_keys.append(key)
            self._active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self._active_requests)
        try:
            if script.latency_seconds > 0:
                time.sleep(script.latency_seconds)
            return ApiResponse(
                request=request,
                status_code=script.status_code,
                body=script.encoded_body(),
            )
        finally:
            with self._lock:
                self._active_requests -= 1

    def _resolve_script(self, key: int, query: Mapping[str, list[str]]) -> PageScript:
        if self._scope_param is not None:
            scope_values = query.get(self._scope_param)
            if scope_values:
                scope_scripts = self._scoped_scripts.get(scope_values[0], {})
                scoped = scope_scripts.get(key)
                if scoped is not None:
                    return scoped
        return self._scripts.get(key, self._default_script)


class RecordingProgressStore(ExtractionProgressStore):
    """Progress store that remembers every saved position.

    ``extract()`` calls ``clear()`` on success, so the on-disk file is gone by the time a test
    could read it. The spy keeps the saved rows, which is what the resume path would replay.
    """

    def __init__(self) -> None:
        super().__init__()
        self.saved_request_indexes: list[int] = []

    def save(self, plan: ExecutionPlan, **kwargs: Any) -> Path:
        self.saved_request_indexes.append(int(kwargs.get("request_index", 0)))
        return super().save(plan, **kwargs)


def build_transport_factory(
    transport: ScriptedPageTransport,
) -> Callable[[], ScriptedPageTransport]:
    """Return a factory handing back the *same* transport on every call.

    ``ApiStrategy`` builds one ``ApiClient(self.transport_factory())`` per concurrent
    request; a factory returning a fresh instance would split the request log across
    instances and silently weaken every over-fetch assertion.
    """

    def factory() -> ScriptedPageTransport:
        return transport

    return factory


def build_concurrent_strategy(
    tmp_path: Path,
    transport: ScriptedPageTransport,
    *,
    sleeper: Callable[[float], None] | None = None,
    logger: StructuredLogger | None = None,
    progress_store: ExtractionProgressStore | None = None,
) -> ApiStrategy:
    """Build an ``ApiStrategy`` wired to ``transport`` with a real (unfrozen) clock.

    The clock is left at the default on purpose: the shared throttle paces against it,
    and a frozen clock plus a rate limit stalls pacing. Concurrency tests pass
    ``requests_per_minute=None`` instead.
    """
    return ApiStrategy(
        transport_factory=build_transport_factory(transport),
        storage_layout_factory=lambda plan: build_storage_layout(tmp_path),
        sleeper=sleeper or (lambda seconds: None),
        logger=logger,
        **({"progress_store": progress_store} if progress_store is not None else {}),
    )


def build_concurrent_plan(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "page_number_api",
    pagination_type: str = "page_number",
    page_size: int = 2,
    concurrency: int = 3,
    requests_per_minute: int | None = None,
    retry_max_attempts: int = 2,
    retry_backoff_seconds: int = 1,
    request_inputs: dict[str, Any] | None = None,
    parameter_bindings: dict[str, Any] | None = None,
    dead_letter_max_items: int = 0,
    past_end_status_codes: list[int] | None = None,
    total_count_field: str | None = None,
    checkpoint_field: str | None = None,
    run_id: str | None = None,
    attributes: Mapping[str, str] | None = None,
) -> ExecutionPlan:
    """Build an execution plan for a concurrency-capable API source.

    ``run_id`` and ``attributes`` exist for suites that run the *same* source twice — the
    equivalence harness needs a distinct run per leg, and ``attributes={"resume": "true"}``
    is the only way to reach the resume branch of ``extract()``.
    """
    source_config = build_concurrent_source_config(
        tmp_path,
        source_id=source_id,
        variant=variant,
        pagination_type=pagination_type,
        page_size=page_size,
        concurrency=concurrency,
        requests_per_minute=requests_per_minute,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        request_inputs=request_inputs,
        parameter_bindings=parameter_bindings,
        dead_letter_max_items=dead_letter_max_items,
        past_end_status_codes=past_end_status_codes,
        total_count_field=total_count_field,
        checkpoint_field=checkpoint_field,
    )
    run_context = RunContext.create(
        run_id=run_id or f"run-{source_id}",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        attributes=attributes,
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def build_concurrent_source_config(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "page_number_api",
    pagination_type: str = "page_number",
    page_size: int = 2,
    concurrency: int = 3,
    requests_per_minute: int | None = None,
    retry_max_attempts: int = 2,
    retry_backoff_seconds: int = 1,
    request_inputs: dict[str, Any] | None = None,
    parameter_bindings: dict[str, Any] | None = None,
    dead_letter_max_items: int = 0,
    past_end_status_codes: list[int] | None = None,
    total_count_field: str | None = None,
    checkpoint_field: str | None = None,
) -> SourceConfig:
    """Build a concurrency-capable API source config."""
    extraction: dict[str, Any] = {
        "mode": "full_refresh",
        "checkpoint_strategy": "none",
        "dead_letter_max_items": dead_letter_max_items,
        "retry": {
            "max_attempts": retry_max_attempts,
            "backoff_strategy": "fixed",
            "backoff_seconds": retry_backoff_seconds,
        },
    }
    quality: dict[str, Any] = {"allow_schema_evolution": True}
    if checkpoint_field is not None:
        extraction["mode"] = "incremental"
        extraction["checkpoint_field"] = checkpoint_field
        extraction["checkpoint_strategy"] = "max_value"
        quality["unique_fields"] = ["id"]

    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "api",
            "strategy": "api",
            "strategy_variant": variant,
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "base_url": "https://example.invalid",
                "path": "/records",
                "method": "GET",
                "format": "json",
                "timeout_seconds": 30,
                "auth": {"type": "none"},
                "request_inputs": request_inputs,
                "parameter_bindings": parameter_bindings,
                "pagination": _pagination_block(
                    pagination_type,
                    page_size,
                    past_end_status_codes=past_end_status_codes,
                    total_count_field=total_count_field,
                ),
                "rate_limit": {
                    "requests_per_minute": requests_per_minute,
                    "concurrency": concurrency,
                    "backoff_seconds": None,
                },
            },
            "extraction": extraction,
            "schema": {"mode": "infer"},
            "spark": {
                "input_format": "json",
                "write_mode": "append",
            },
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": quality,
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def build_storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": "runtime",
                "raw_dir": "runtime/raw",
                "bronze_dir": "runtime/bronze",
                "metadata_dir": "runtime/metadata",
            }
        },
        tmp_path,
    )


def _pagination_block(
    pagination_type: str,
    page_size: int,
    *,
    past_end_status_codes: list[int] | None = None,
    total_count_field: str | None = None,
) -> dict[str, Any]:
    """Build a pagination block; ``None`` keys are omitted so the contract default applies."""
    if pagination_type == "page_number":
        block: dict[str, Any] = {
            "type": "page_number",
            "page_param": "page",
            "size_param": "page_size",
            "page_size": page_size,
        }
    elif pagination_type == "offset":
        block = {
            "type": "offset",
            "offset_param": "offset",
            "limit_param": "limit",
            "page_size": page_size,
        }
    else:
        raise AssertionError(
            f"Unsupported pagination type for concurrency tests: {pagination_type}"
        )

    if past_end_status_codes is not None:
        block["past_end_status_codes"] = list(past_end_status_codes)
    if total_count_field is not None:
        block["total_count_field"] = total_count_field
    return block
