"""Shared harness for the HTTP characterization suite.

These tests pin the current HTTP behavior of the api, catalog, and file
strategy families so the shared-extraction-layer refactor can prove it changed
structure without changing behavior.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from itertools import count
from pathlib import Path
from typing import Any

import pytest

import janus.strategies.api.core as api_core
import janus.strategies.catalog.core as catalog_core
import janus.strategies.files.core as files_core
from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api.http import ApiClient, ApiRequest, ApiResponse
from janus.utils.logging import StructuredLogger, build_structured_logger

# ---------------------------------------------------------------------------
# Per-family symbol mappings — the single place later refactor steps update.



@dataclass(frozen=True, slots=True)
class FamilyUnderTest:
    """Current per-family HTTP symbols and the observable contract they carry."""

    name: str
    strategy_cls: type
    throttle_cls: type
    retryable_status_codes: frozenset[int]
    transport_exhaustion_error: type[Exception]
    response_error: type[Exception]
    payload_error: type[Exception] | None
    retry_log_event: str
    returns_payload: bool
    response_error_has_response: bool
    status_error_message: Callable[[int, str], str]
    send_with_retries: Callable[..., Any]
    decode_payload: Callable[..., Any] | None


@dataclass(frozen=True, slots=True)
class BindingUnderTest:
    """Current per-family module-level binding helpers."""

    name: str
    split_path_and_query_params: Callable[..., Any]
    resolve_url: Callable[[SourceConfig], str]
    default_checkpoint_params: Callable[..., dict[str, str]]
    checkpoint_request_value: Callable[..., str | None]
    resolve_url_missing_base_message: str


def _send_via_strategy_method(strategy, plan, client, request, throttle, logger):
    return strategy._send_with_retries(plan, client, request, throttle, logger)


def _decode_via_strategy_method(strategy, plan, response):
    return strategy._decode_payload(plan, response)


FAMILIES: dict[str, FamilyUnderTest] = {
    "api": FamilyUnderTest(
        name="api",
        strategy_cls=api_core.ApiStrategy,
        throttle_cls=api_core.ApiRequestThrottle,
        retryable_status_codes=api_core.RETRYABLE_STATUS_CODES,
        transport_exhaustion_error=api_core.ApiStrategyError,
        response_error=api_core.ApiResponseError,
        payload_error=api_core.ApiPayloadError,
        retry_log_event="api_retry_scheduled",
        returns_payload=True,
        response_error_has_response=True,
        status_error_message=(
            lambda status, url: f"API request failed with status {status} for {url}"
        ),
        send_with_retries=_send_via_strategy_method,
        decode_payload=_decode_via_strategy_method,
    ),
    "catalog": FamilyUnderTest(
        name="catalog",
        strategy_cls=catalog_core.CatalogStrategy,
        throttle_cls=catalog_core.CatalogRequestThrottle,
        retryable_status_codes=catalog_core.RETRYABLE_STATUS_CODES,
        transport_exhaustion_error=catalog_core.CatalogStrategyError,
        response_error=catalog_core.CatalogResponseError,
        payload_error=catalog_core.CatalogPayloadError,
        retry_log_event="catalog_retry_scheduled",
        returns_payload=True,
        response_error_has_response=True,
        status_error_message=(
            lambda status, url: f"Catalog request failed with status {status} for {url}"
        ),
        send_with_retries=_send_via_strategy_method,
        decode_payload=_decode_via_strategy_method,
    ),
    "file": FamilyUnderTest(
        name="file",
        strategy_cls=files_core.FileStrategy,
        throttle_cls=files_core.FileRequestThrottle,
        retryable_status_codes=files_core.RETRYABLE_STATUS_CODES,
        transport_exhaustion_error=files_core.FileDownloadError,
        response_error=files_core.FileDownloadError,
        payload_error=None,
        retry_log_event="file_retry_scheduled",
        returns_payload=False,
        response_error_has_response=False,
        status_error_message=(
            lambda status, url: f"File request failed with status {status} for {url}"
        ),
        send_with_retries=_send_via_strategy_method,
        decode_payload=None,
    ),
}

BINDINGS: dict[str, BindingUnderTest] = {
    "api": BindingUnderTest(
        name="api",
        split_path_and_query_params=api_core._split_path_and_query_params,
        resolve_url=api_core._resolve_url,
        default_checkpoint_params=api_core._default_checkpoint_params,
        checkpoint_request_value=api_core._checkpoint_request_value,
        resolve_url_missing_base_message="API source requires access.base_url or access.url",
    ),
    "catalog": BindingUnderTest(
        name="catalog",
        split_path_and_query_params=catalog_core._split_path_and_query_params,
        resolve_url=catalog_core._resolve_url,
        default_checkpoint_params=catalog_core._default_checkpoint_params,
        checkpoint_request_value=catalog_core._checkpoint_request_value,
        resolve_url_missing_base_message="Catalog source requires access.base_url or access.url",
    ),
}


@pytest.fixture(params=sorted(FAMILIES))
def family(request) -> FamilyUnderTest:
    return FAMILIES[request.param]


@pytest.fixture(params=[name for name in sorted(FAMILIES) if FAMILIES[name].returns_payload])
def payload_family(request) -> FamilyUnderTest:
    return FAMILIES[request.param]


@pytest.fixture(params=sorted(BINDINGS))
def binding(request) -> BindingUnderTest:
    return BINDINGS[request.param]


# ---------------------------------------------------------------------------
# Deterministic fakes



@dataclass(frozen=True, slots=True)
class ResponseSpec:
    status_code: int
    body: bytes = b"{}"
    headers: tuple[tuple[str, str], ...] = ()


def json_response(payload: Any, *, status_code: int = 200) -> ResponseSpec:
    return ResponseSpec(status_code=status_code, body=json.dumps(payload).encode("utf-8"))


class FakeTransport:
    """Scripted transport: each send pops the next ResponseSpec or raises the exception."""

    def __init__(self, script: list[ResponseSpec | Exception]) -> None:
        self._script = list(script)
        self.requests: list[ApiRequest] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def send(self, request: ApiRequest) -> ApiResponse:
        self.requests.append(request)
        if not self._script:
            raise AssertionError("No scripted responses remain for this transport")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ApiResponse(
            request=request,
            status_code=item.status_code,
            body=item.body,
            headers=item.headers,
        )


class CountingThrottle:
    """Inert stand-in for the per-family throttles that counts consultations."""

    def __init__(self) -> None:
        self.calls = 0

    def wait_for_turn(self) -> None:
        self.calls += 1


class ScriptedClock:
    """Monotonic clock that pops pre-scripted instants and counts reads."""

    def __init__(self, instants: list[float] | None = None) -> None:
        self.instants = list(instants or [])
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return self.instants.pop(0)


def make_request(
    url: str = "https://example.invalid/records",
    params: dict[str, str] | None = None,
) -> ApiRequest:
    request = ApiRequest(method="GET", url=url, timeout_seconds=30)
    if params:
        request = request.with_params(params)
    return request


def make_client(script: list[ResponseSpec | Exception]) -> tuple[ApiClient, FakeTransport]:
    transport = FakeTransport(script)
    return ApiClient(transport), transport


def build_strategy(family_under_test: FamilyUnderTest, *, sleeper=None):
    return family_under_test.strategy_cls(
        sleeper=sleeper or (lambda seconds: None),
        clock=lambda: 0.0,
    )


# ---------------------------------------------------------------------------
# Structured-log capture


_LOGGER_SEQUENCE = count(1)


def recording_logger() -> tuple[StructuredLogger, StringIO]:
    stream = StringIO()
    logger = build_structured_logger(
        f"janus.tests.http.characterization.{next(_LOGGER_SEQUENCE)}",
        stream=stream,
    )
    return logger, stream


def logged_events(stream: StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def warning_events(stream: StringIO) -> list[dict[str, Any]]:
    return [event for event in logged_events(stream) if event["level"] == "WARNING"]


# ---------------------------------------------------------------------------
# Minimal per-family plans


_FAMILY_CONFIG_SHAPE: dict[str, dict[str, str]] = {
    "api": {
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "spark_input_format": "json",
        "raw_format": "json",
    },
    "catalog": {
        "source_type": "catalog",
        "strategy": "catalog",
        "strategy_variant": "metadata_catalog",
        "spark_input_format": "jsonl",
        "raw_format": "json",
    },
    "file": {
        "source_type": "file",
        "strategy": "file",
        "strategy_variant": "static_file",
        "spark_input_format": "csv",
        "raw_format": "binary",
    },
}


def build_source_config(
    family_name: str,
    tmp_path: Path,
    *,
    source_id: str,
    access_format: str | None = None,
    url: str | None = None,
    base_url: str | None = "https://example.invalid",
    path: str | None = "/records",
    extraction_mode: str = "full_refresh",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
    lookback_days: int | None = None,
    retry_max_attempts: int = 3,
    retry_backoff_seconds: int = 2,
    retry_backoff_strategy: str = "fixed",
    rate_limit_backoff_seconds: int | None = 5,
    requests_per_minute: int | None = None,
) -> SourceConfig:
    shape = _FAMILY_CONFIG_SHAPE[family_name]
    if access_format is None:
        access_format = "csv" if family_name == "file" else "json"

    access: dict[str, Any] = {
        "method": "GET",
        "format": access_format,
        "timeout_seconds": 30,
        "auth": {"type": "none"},
        "pagination": (
            {"type": "none"}
            if family_name == "file"
            else {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 100,
            }
        ),
        "rate_limit": {
            "requests_per_minute": requests_per_minute,
            "concurrency": 1,
            "backoff_seconds": rate_limit_backoff_seconds,
        },
    }
    if family_name == "file":
        access["url"] = url or "https://example.invalid/data.csv"
    else:
        if url is not None:
            access["url"] = url
        if base_url is not None:
            access["base_url"] = base_url
        if path is not None:
            access["path"] = path

    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": shape["source_type"],
            "strategy": shape["strategy"],
            "strategy_variant": shape["strategy_variant"],
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": access,
            "extraction": {
                "mode": extraction_mode,
                "checkpoint_field": checkpoint_field,
                "checkpoint_strategy": checkpoint_strategy,
                "lookback_days": lookback_days,
                "retry": {
                    "max_attempts": retry_max_attempts,
                    "backoff_strategy": retry_backoff_strategy,
                    "backoff_seconds": retry_backoff_seconds,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {
                "input_format": shape["spark_input_format"],
                "write_mode": "append",
            },
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": shape["raw_format"]},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def build_plan(family_name: str, tmp_path: Path, *, source_id: str, **overrides) -> ExecutionPlan:
    source_config = build_source_config(family_name, tmp_path, source_id=source_id, **overrides)
    run_context = RunContext.create(
        run_id=f"run-{source_id}",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)
