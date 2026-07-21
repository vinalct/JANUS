from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import janus.strategies.http.transport as http_module
from janus.checkpoints import CheckpointStore, DeadLetterStore, ExtractionProgressStore
from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.planner import StrategyCatalog
from janus.strategies.api import ApiHook, ApiResponse, ApiStrategy
from janus.utils.logging import build_structured_logger
from janus.utils.storage import StorageLayout


@dataclass(frozen=True, slots=True)
class ResponseSpec:
    status_code: int
    payload: Any
    headers: dict[str, str] = field(default_factory=dict)
    format_name: str = "json"
    raw_body: bytes | None = None


class FakeTransport:
    def __init__(self, responses: list[ResponseSpec | Exception]) -> None:
        self._responses = list(responses)
        self.requests = []
        self.opened = False
        self.closed = False
        # Optional shared trail, so session lifecycle and requests can be ordered together.
        self.events: list[str] | None = None

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request):
        self.requests.append(request)
        if self.events is not None:
            self.events.append("request")
        if not self._responses:
            raise AssertionError("No fake responses remain for this transport")

        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response

        if response.raw_body is not None:
            body = response.raw_body
        elif response.format_name == "json":
            body = json.dumps(response.payload).encode("utf-8")
        elif response.format_name == "jsonl":
            body = "\n".join(json.dumps(item) for item in response.payload).encode("utf-8")
        elif response.format_name == "text":
            body = str(response.payload).encode("utf-8")
        elif response.format_name == "binary":
            body = bytes(response.payload)
        else:
            raise AssertionError(f"Unsupported fake format: {response.format_name}")

        return ApiResponse(
            request=request,
            status_code=response.status_code,
            body=body,
            headers=tuple(sorted(response.headers.items())),
        )


class WindowHook(ApiHook):
    def checkpoint_params(self, plan: ExecutionPlan, checkpoint_value: str) -> dict[str, str]:
        del plan
        return {"since": checkpoint_value}

    def extract_records(self, plan, request, response, payload):
        del plan
        del request
        del response
        return payload["payload"]["rows"]


@dataclass(slots=True)
class ConcurrentPageState:
    payloads: dict[int, list[dict[str, str]]]
    requests: list = field(default_factory=list)
    start_times: dict[int, float] = field(default_factory=dict)
    active_requests: int = 0
    max_active_requests: int = 0
    first_page_started: threading.Event = field(default_factory=threading.Event)
    second_page_started: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class ConcurrentPageTransport:
    def __init__(self, state: ConcurrentPageState) -> None:
        self.state = state

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def send(self, request):
        query = parse_qs(urlsplit(request.full_url()).query)
        page_number = int(query["page"][0])
        started_at = time.monotonic()
        with self.state.lock:
            self.state.requests.append(request)
            self.state.start_times[page_number] = started_at
            self.state.active_requests += 1
            self.state.max_active_requests = max(
                self.state.max_active_requests,
                self.state.active_requests,
            )

        try:
            if page_number == 1:
                self.state.first_page_started.set()
                assert self.state.second_page_started.wait(timeout=1)
                time.sleep(0.2)
            elif page_number == 2:
                self.state.second_page_started.set()
                time.sleep(0.15)
            else:
                time.sleep(0.02)

            payload = {"records": self.state.payloads.get(page_number, [])}
            return ApiResponse(
                request=request,
                status_code=200,
                body=json.dumps(payload).encode("utf-8"),
            )
        finally:
            with self.state.lock:
                self.state.active_requests -= 1


class FakeAliasedColumn:
    def __init__(self, source_name: str, alias_name: str) -> None:
        self.source_name = source_name
        self.alias_name = alias_name


class FakeColumn:
    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    def alias(self, alias_name: str) -> FakeAliasedColumn:
        return FakeAliasedColumn(self.source_name, alias_name)


class FakeRow:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = dict(values)

    def asDict(self, recursive: bool = False) -> dict[str, object]:
        del recursive
        return dict(self._values)


class FakeDataFrame:
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        columns: tuple[str, ...] | None = None,
    ) -> None:
        self._rows = [dict(row) for row in rows]
        if columns is not None:
            self.columns = columns
        elif rows:
            self.columns = tuple(rows[0])
        else:
            self.columns = ()

    def __getitem__(self, key: str) -> FakeColumn:
        return FakeColumn(key)

    def select(self, *selected_columns: FakeAliasedColumn) -> FakeDataFrame:
        projected_rows = []
        for row in self._rows:
            projected_rows.append(
                {selected.alias_name: row[selected.source_name] for selected in selected_columns}
            )
        projected_column_names = tuple(selected.alias_name for selected in selected_columns)
        return FakeDataFrame(projected_rows, columns=projected_column_names)

    def distinct(self) -> FakeDataFrame:
        seen = set()
        unique_rows = []
        for row in self._rows:
            key = tuple((column_name, row[column_name]) for column_name in self.columns)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        return FakeDataFrame(unique_rows, columns=self.columns)

    def sort(self, *column_names: str) -> FakeDataFrame:
        sorted_rows = sorted(
            self._rows,
            key=lambda row: tuple(row[column_name] for column_name in column_names),
        )
        return FakeDataFrame(sorted_rows, columns=self.columns)

    def collect(self) -> list[FakeRow]:
        return [FakeRow(row) for row in self._rows]


class FakeCatalog:
    def __init__(self, tables: dict[str, FakeDataFrame]) -> None:
        self._tables = dict(tables)

    def tableExists(self, table_identifier: str) -> bool:
        return table_identifier in self._tables


class FakeSparkSession:
    def __init__(self, tables: dict[str, FakeDataFrame]) -> None:
        self._tables = dict(tables)
        self.catalog = FakeCatalog(self._tables)

    def table(self, table_identifier: str) -> FakeDataFrame:
        return self._tables[table_identifier]


def test_urllib_transport_prefers_explicit_or_environment_ca_bundle(tmp_path, monkeypatch):
    explicit_bundle = tmp_path / "explicit-ca.pem"
    env_bundle = tmp_path / "env-ca.pem"
    explicit_bundle.write_text("explicit", encoding="utf-8")
    env_bundle.write_text("env", encoding="utf-8")

    monkeypatch.setenv("JANUS_CA_BUNDLE", str(env_bundle))
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/ignored-ssl-cert-file.pem")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/tmp/ignored-requests-ca-bundle.pem")

    assert http_module._resolve_ca_bundle(str(explicit_bundle)) == str(explicit_bundle)
    assert http_module._resolve_ca_bundle() == str(env_bundle)


def test_default_strategy_catalog_uses_api_strategy_for_api_variants(tmp_path):
    source_config = _build_source_config(tmp_path, source_id="catalog_api")

    binding = StrategyCatalog.with_defaults().resolve(source_config)

    assert isinstance(binding.strategy, ApiStrategy)
    assert binding.dispatch_path == "api.page_number_api"


def test_api_strategy_logs_page_progress(tmp_path):
    stream = StringIO()
    logger = build_structured_logger("janus.tests.api.progress", stream=stream)
    plan = _build_plan(tmp_path, source_id="logged_page_source", page_size=2)
    strategy, _transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"records": [{"id": "1"}, {"id": "2"}]}),
            ResponseSpec(200, {"records": [{"id": "3"}]}),
        ],
        logger=logger,
    )

    strategy.extract(plan)

    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    events = [payload["event"] for payload in payloads]
    finished_pages = [
        payload for payload in payloads if payload["event"] == "api_request_finished"
    ]

    assert events[0] == "api_extraction_started"
    assert events[-1] == "api_extraction_finished"
    assert [page["fields"]["page_number"] for page in finished_pages] == [1, 2]
    assert finished_pages[0]["fields"]["records_extracted"] == 2
    assert finished_pages[0]["fields"]["has_next_page"] is True
    assert finished_pages[0]["fields"]["next_page_number"] == 2
    assert finished_pages[1]["fields"]["records_extracted"] == 1
    assert finished_pages[1]["fields"]["has_next_page"] is False


def test_api_strategy_request_inputs_bind_date_windows_and_partition_raw_outputs(tmp_path):
    stream = StringIO()
    logger = build_structured_logger("janus.tests.api.request_inputs", stream=stream)
    plan = _build_plan(
        tmp_path,
        source_id="windowed_source",
        params={"situacao": "TODAS"},
        request_inputs={
            "type": "date_window",
            "start": date(2025, 1, 1),
            "end": date(2025, 2, 28),
            "step": "month",
        },
        parameter_bindings={
            "mesAno": {
                "from": "request_input.window_end",
                "format": "%Y%m",
            }
        },
        page_size=2,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"records": [{"id": "1"}]}),
            ResponseSpec(200, {"records": [{"id": "2"}]}),
        ],
        logger=logger,
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    queries = [parse_qs(urlsplit(request.full_url()).query) for request in transport.requests]
    assert [query["situacao"] for query in queries] == [["TODAS"], ["TODAS"]]
    assert [query["mesAno"] for query in queries] == [["202501"], ["202502"]]
    assert [query["page"] for query in queries] == [["1"], ["1"]]
    assert [query["page_size"] for query in queries] == [["2"], ["2"]]

    assert Path(result.artifacts[0].path) == (
        _raw_run_dir(tmp_path, "windowed_source")
        / "request-input-000001"
        / "page-0001.json"
    )
    assert Path(result.artifacts[1].path) == (
        _raw_run_dir(tmp_path, "windowed_source")
        / "request-input-000002"
        / "page-0001.json"
    )

    assert metadata["request_input_type"] == "date_window"
    assert metadata["request_input_count"] == "2"
    assert metadata["bound_parameter_names"] == "mesAno"

    emitted_metadata = strategy.emit_metadata(plan, result)
    assert emitted_metadata["request_input_type"] == "date_window"
    assert emitted_metadata["request_input_count"] == "2"

    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    finished_requests = [
        payload for payload in payloads if payload["event"] == "api_request_finished"
    ]
    assert [payload["fields"]["request_input_index"] for payload in finished_requests] == [1, 2]


def test_api_strategy_keeps_static_params_backward_compatible_when_request_inputs_default_to_none(
    tmp_path,
):
    plan = _build_plan(
        tmp_path,
        source_id="static_param_source",
        params={"situacao": "TODAS"},
        page_size=2,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {"records": [{"id": "1"}]})],
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    query = parse_qs(urlsplit(transport.requests[0].full_url()).query)
    assert query["situacao"] == ["TODAS"]
    assert query["page"] == ["1"]
    assert query["page_size"] == ["2"]

    assert result.records_extracted == 1
    assert metadata["request_input_type"] == "none"
    assert metadata["request_input_count"] == "1"
    assert "bound_parameter_names" not in metadata
    assert Path(result.artifacts[0].path) == (
        _raw_run_dir(tmp_path, "static_param_source") / "pages" / "page-0001.json"
    )


def test_api_strategy_page_number_extracts_raw_pages_and_tracks_checkpoint(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="page_source",
        extraction_mode="incremental",
        checkpoint_field="updated_at",
        checkpoint_strategy="max_value",
        lookback_days=1,
        page_size=2,
    )
    CheckpointStore().save(plan, "2026-04-09T12:00:00Z")

    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "records": [
                        {"id": "1", "updated_at": "2026-04-10T01:00:00Z"},
                        {"id": "2", "updated_at": "2026-04-10T03:00:00Z"},
                    ]
                },
            ),
            ResponseSpec(
                200,
                {
                    "records": [
                        {"id": "3", "updated_at": "2026-04-10T05:00:00Z"},
                    ]
                },
            ),
        ],
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert transport.opened is True
    assert transport.closed is True
    assert len(transport.requests) == 2

    first_query = parse_qs(urlsplit(transport.requests[0].full_url()).query)
    second_query = parse_qs(urlsplit(transport.requests[1].full_url()).query)
    assert first_query["page"] == ["1"]
    assert first_query["page_size"] == ["2"]
    assert first_query["updated_at"] == ["2026-04-08T12:00:00Z"]
    assert second_query["page"] == ["2"]
    assert second_query["page_size"] == ["2"]

    assert result.records_extracted == 3
    assert result.checkpoint_value == "2026-04-10T05:00:00Z"
    assert metadata["request_count"] == "2"
    assert metadata["retry_count"] == "0"
    assert metadata["checkpoint_loaded"] == "true"

    first_artifact = Path(result.artifacts[0].path)
    second_artifact = Path(result.artifacts[1].path)
    assert first_artifact == _raw_run_dir(tmp_path, "page_source") / "pages" / "page-0001.json"
    assert second_artifact == _raw_run_dir(tmp_path, "page_source") / "pages" / "page-0002.json"
    assert json.loads(first_artifact.read_text(encoding="utf-8"))["records"][0]["id"] == "1"

    emitted_metadata = strategy.emit_metadata(plan, result)
    assert emitted_metadata["artifact_count"] == 2
    assert emitted_metadata["pagination_type"] == "page_number"


def test_api_strategy_rerun_preserves_previous_raw_payloads(tmp_path):
    source_config = _build_source_config(
        tmp_path,
        source_id="preserved_rerun_source",
        pagination_type="none",
        requests_per_minute=None,
    )
    first_plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-preserved-rerun-001",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ),
    )
    second_plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-preserved-rerun-002",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 11, 12, 0, tzinfo=UTC),
        ),
    )

    first_strategy, _ = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {"records": [{"id": "old"}]})],
    )
    second_strategy, _ = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {"records": [{"id": "new"}]})],
    )

    first_result = first_strategy.extract(first_plan)
    second_result = second_strategy.extract(second_plan)

    first_path = Path(first_result.artifacts[0].path)
    second_path = Path(second_result.artifacts[0].path)
    assert first_path == (
        _raw_run_dir(
            tmp_path,
            "preserved_rerun_source",
            ingestion_date="2026-04-10",
            run_id="run-preserved-rerun-001",
        )
        / "pages"
        / "response-0001.json"
    )
    assert second_path == (
        _raw_run_dir(
            tmp_path,
            "preserved_rerun_source",
            ingestion_date="2026-04-11",
            run_id="run-preserved-rerun-002",
        )
        / "pages"
        / "response-0001.json"
    )
    assert json.loads(first_path.read_text(encoding="utf-8"))["records"][0]["id"] == "old"
    assert json.loads(second_path.read_text(encoding="utf-8"))["records"][0]["id"] == "new"


def test_api_strategy_offset_pagination_reuses_offset_parameters_from_config(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="offset_source",
        variant="offset_api",
        pagination_type="offset",
        page_size=2,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"records": [{"id": "1"}, {"id": "2"}]}),
            ResponseSpec(200, {"records": [{"id": "3"}]}),
        ],
    )

    result = strategy.extract(plan)

    first_query = parse_qs(urlsplit(transport.requests[0].full_url()).query)
    second_query = parse_qs(urlsplit(transport.requests[1].full_url()).query)
    assert first_query["offset"] == ["0"]
    assert first_query["limit"] == ["2"]
    assert second_query["offset"] == ["2"]
    assert second_query["limit"] == ["2"]

    assert result.records_extracted == 3
    assert Path(result.artifacts[0].path).name == "offset-00000000.json"
    assert Path(result.artifacts[1].path).name == "offset-00000002.json"


def test_api_strategy_page_number_supports_bounded_concurrency(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="concurrent_page_source",
        page_size=2,
        requests_per_minute=600,
        concurrency=2,
    )
    state = ConcurrentPageState(
        payloads={
            1: [{"id": "1"}, {"id": "2"}],
            2: [{"id": "3"}, {"id": "4"}],
            3: [{"id": "5"}],
            4: [],
        }
    )
    strategy = ApiStrategy(
        transport_factory=lambda: ConcurrentPageTransport(state),
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
    )

    result = strategy.extract(plan)

    assert state.max_active_requests >= 2
    assert state.start_times[2] > state.start_times[1]
    assert state.start_times[2] - state.start_times[1] >= 0.08
    assert len(result.artifacts) == 3
    assert [Path(artifact.path).name for artifact in result.artifacts] == [
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
    ]
    assert result.records_extracted == 5


def test_api_strategy_retries_transient_failures_with_bounded_backoff(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="retry_source",
        retry_max_attempts=3,
        retry_backoff_seconds=2,
        retry_backoff_strategy="exponential",
        rate_limit_backoff_seconds=3,
        requests_per_minute=None,
        page_size=2,
    )
    sleeps: list[float] = []
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(503, {"error": "temporarily unavailable"}),
            ResponseSpec(503, {"error": "temporarily unavailable"}),
            ResponseSpec(200, {"records": [{"id": "1"}]}),
        ],
        sleeper=sleeps.append,
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert sleeps == [2.0, 3.0]
    assert len(transport.requests) == 3
    assert metadata["attempt_count"] == "3"
    assert metadata["retry_count"] == "2"
    assert metadata["request_count"] == "1"


def test_api_strategy_retries_bad_payload_on_200_response(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="payload_retry_source",
        retry_max_attempts=3,
        retry_backoff_seconds=2,
        retry_backoff_strategy="fixed",
        requests_per_minute=None,
        page_size=2,
    )
    sleeps: list[float] = []
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, None, raw_body=b"<html>temporarily unavailable</html>"),
            ResponseSpec(200, None, raw_body=b"<html>temporarily unavailable</html>"),
            ResponseSpec(200, {"records": [{"id": "1"}]}),
        ],
        sleeper=sleeps.append,
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert sleeps == [2.0, 2.0]
    assert len(transport.requests) == 3
    assert metadata["attempt_count"] == "3"
    assert metadata["retry_count"] == "2"
    assert metadata["request_count"] == "1"


def test_api_strategy_raises_after_exhausting_retries_on_bad_payload(tmp_path):
    from janus.strategies.api.core import ApiPayloadError

    plan = _build_plan(
        tmp_path,
        source_id="payload_exhaust_source",
        retry_max_attempts=2,
        retry_backoff_seconds=1,
        retry_backoff_strategy="fixed",
        requests_per_minute=None,
        page_size=2,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, None, raw_body=b"<html>bad</html>"),
            ResponseSpec(200, None, raw_body=b"<html>bad</html>"),
        ],
    )

    import pytest

    with pytest.raises(ApiPayloadError):
        strategy.extract(plan)

    assert len(transport.requests) == 2


def test_api_strategy_dead_letters_failed_request_input_and_continues(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="dead_letter_source",
        pagination_type="none",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze_test",
            "table_name": "ids",
            "columns": {"entity_id": "id"},
        },
        parameter_bindings={"id": {"from": "request_input.entity_id"}},
        retry_max_attempts=2,
        requests_per_minute=None,
        dead_letter_max_items=1,
    )
    spark = FakeSparkSession(
        {
            "bronze_test.ids": FakeDataFrame(
                [
                    {"id": "alpha"},
                    {"id": "beta"},
                    {"id": "gamma"},
                ]
            )
        }
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"records": [{"id": "alpha"}]}),
            ResponseSpec(503, {"error": "missing"}),
            ResponseSpec(503, {"error": "missing"}),
            ResponseSpec(200, {"records": [{"id": "gamma"}]}),
        ],
    )

    result = strategy.extract(plan, spark=spark)
    metadata = result.metadata_as_dict()

    queries = [parse_qs(urlsplit(request.full_url()).query) for request in transport.requests]
    assert [query["id"] for query in queries] == [["alpha"], ["beta"], ["beta"], ["gamma"]]
    assert [Path(artifact.path).parent.name for artifact in result.artifacts] == [
        "request-input-000001",
        "request-input-000003",
    ]
    assert result.records_extracted == 2
    assert metadata["dead_letter_count"] == "1"
    assert metadata["dead_letter_skipped_count"] == "1"

    dead_letter_path = DeadLetterStore().path(plan)
    dead_letter_payload = json.loads(dead_letter_path.read_text(encoding="utf-8"))
    assert dead_letter_payload["entries"][0]["item_key"] == "entity_id=beta"
    assert dead_letter_payload["entries"][0]["item_type"] == "request_input"


def test_api_strategy_resume_skips_preexisting_dead_lettered_input(tmp_path):
    source_config = _build_source_config(
        tmp_path,
        source_id="resume_dead_letter_source",
        pagination_type="none",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze_test",
            "table_name": "ids",
            "columns": {"entity_id": "id"},
        },
        parameter_bindings={"id": {"from": "request_input.entity_id"}},
        requests_per_minute=None,
    )
    initial_plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-resume_dead_letter_source-initial",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
        ),
    )
    DeadLetterStore().record(
        initial_plan,
        item_key="entity_id=beta",
        item_type="request_input",
        error=RuntimeError("missing beta"),
        metadata={"request_url": "https://example.invalid/records?id=beta"},
    )
    ExtractionProgressStore().save(
        initial_plan,
        artifact_count=0,
        current_input_key="entity_id=beta",
        current_input_index=1,
        request_input_count=2,
    )
    plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-resume_dead_letter_source-resume",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 4, 10, 12, 5, tzinfo=UTC),
            attributes={"resume": "true"},
        ),
    )

    spark = FakeSparkSession(
        {
            "bronze_test.ids": FakeDataFrame(
                [
                    {"id": "beta"},
                    {"id": "gamma"},
                ]
            )
        }
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {"records": [{"id": "gamma"}]})],
    )

    result = strategy.extract(plan, spark=spark)
    metadata = result.metadata_as_dict()

    queries = [parse_qs(urlsplit(request.full_url()).query) for request in transport.requests]
    assert [query["id"] for query in queries] == [["gamma"]]
    assert result.records_extracted == 1
    assert metadata["dead_letter_count"] == "1"
    assert metadata["dead_letter_skipped_count"] == "1"


def test_api_strategy_request_inputs_bind_iceberg_rows_from_runtime_spark(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="iceberg_row_source",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze_transparencia",
            "table_name": "emendas_parlamentares__emendas",
            "columns": {"emenda_id": "id"},
        },
        parameter_bindings={"id": {"from": "request_input.emenda_id"}},
        page_size=2,
    )
    spark = FakeSparkSession(
        {
            "bronze_transparencia.emendas_parlamentares__emendas": FakeDataFrame(
                [
                    {"id": "2"},
                    {"id": "1"},
                ]
            )
        }
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"records": [{"id": "1"}]}),
            ResponseSpec(200, {"records": [{"id": "2"}]}),
        ],
    )

    result = strategy.extract(plan, spark=spark)
    metadata = result.metadata_as_dict()

    queries = [parse_qs(urlsplit(request.full_url()).query) for request in transport.requests]
    assert [query["id"] for query in queries] == [["1"], ["2"]]
    assert metadata["request_input_type"] == "iceberg_rows"
    assert metadata["request_input_count"] == "2"
    assert metadata["upstream_namespace"] == "bronze_transparencia"
    assert metadata["upstream_table_name"] == "emendas_parlamentares__emendas"
    assert metadata["upstream_column_names"] == "id"

    assert Path(result.artifacts[0].path) == (
        _raw_run_dir(tmp_path, "iceberg_row_source")
        / "request-input-000001"
        / "page-0001.json"
    )
    assert Path(result.artifacts[1].path) == (
        _raw_run_dir(tmp_path, "iceberg_row_source")
        / "request-input-000002"
        / "page-0001.json"
    )


def test_api_strategy_combined_request_inputs_cross_join_iceberg_and_date_window(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="combined_source",
        request_inputs={
            "type": "combined",
            "inputs": [
                {
                    "type": "iceberg_rows",
                    "namespace": "bronze_transparencia",
                    "table_name": "orgaos",
                    "columns": {"orgao_codigo": "codigo"},
                },
                {
                    "type": "date_window",
                    "start": date(2025, 1, 1),
                    "end": date(2025, 2, 28),
                    "step": "month",
                },
            ],
        },
        parameter_bindings={
            "codigoOrgao": {"from": "request_input.orgao_codigo"},
            "dataInicio": {"from": "request_input.window_start", "format": "%Y-%m-%d"},
            "dataFinal": {"from": "request_input.window_end", "format": "%Y-%m-%d"},
        },
        page_size=2,
    )
    spark = FakeSparkSession(
        {
            "bronze_transparencia.orgaos": FakeDataFrame(
                [
                    {"codigo": "26000"},
                    {"codigo": "52000"},
                ]
            )
        }
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"records": [{"id": "A"}]}),
            ResponseSpec(200, {"records": [{"id": "B"}]}),
            ResponseSpec(200, {"records": [{"id": "C"}]}),
            ResponseSpec(200, {"records": [{"id": "D"}]}),
        ],
    )

    result = strategy.extract(plan, spark=spark)
    metadata = result.metadata_as_dict()

    queries = [parse_qs(urlsplit(req.full_url()).query) for req in transport.requests]
    assert [q["codigoOrgao"] for q in queries] == [["26000"], ["26000"], ["52000"], ["52000"]]
    assert [q["dataInicio"] for q in queries] == [
        ["2025-01-01"],
        ["2025-02-01"],
        ["2025-01-01"],
        ["2025-02-01"],
    ]
    assert [q["dataFinal"] for q in queries] == [
        ["2025-01-31"],
        ["2025-02-28"],
        ["2025-01-31"],
        ["2025-02-28"],
    ]

    assert metadata["request_input_type"] == "combined"
    assert metadata["request_input_count"] == "4"
    assert metadata["upstream_namespace"] == "bronze_transparencia"
    assert metadata["upstream_table_name"] == "orgaos"
    assert metadata["upstream_column_names"] == "codigo"

    assert Path(result.artifacts[0].path) == (
        _raw_run_dir(tmp_path, "combined_source")
        / "request-input-000001"
        / "page-0001.json"
    )
    assert Path(result.artifacts[3].path) == (
        _raw_run_dir(tmp_path, "combined_source")
        / "request-input-000004"
        / "page-0001.json"
    )


def test_api_strategy_hook_can_override_checkpoint_params_and_record_extraction(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="hooked_source",
        variant="date_window_api",
        pagination_type="none",
        extraction_mode="incremental",
        checkpoint_field="updated_at",
        checkpoint_strategy="max_value",
        lookback_days=2,
    )
    CheckpointStore().save(plan, "2026-04-10T06:00:00Z")
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(
                200,
                {
                    "payload": {
                        "rows": [
                            {"id": "99", "updated_at": "2026-04-11T00:00:00Z"},
                        ]
                    }
                },
            )
        ],
    )

    result = strategy.extract(plan, hook=WindowHook())

    query = parse_qs(urlsplit(transport.requests[0].full_url()).query)
    assert query["since"] == ["2026-04-08T06:00:00Z"]
    assert "updated_at" not in query
    assert result.records_extracted == 1
    assert result.checkpoint_value == "2026-04-11T00:00:00Z"


def test_api_strategy_scopes_the_session_to_the_request_input_lookup(
    tmp_path,
    spy_spark_provider,
):
    events: list[str] = []
    plan = _build_plan(
        tmp_path,
        source_id="scoped_iceberg_source",
        pagination_type="none",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze_test",
            "table_name": "ids",
            "columns": {"entity_id": "id"},
        },
        parameter_bindings={"id": {"from": "request_input.entity_id"}},
        requests_per_minute=None,
    )
    provider = spy_spark_provider(
        {"bronze_test.ids": [{"id": "alpha"}, {"id": "beta"}]},
        events,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"records": [{"id": "alpha"}]}),
            ResponseSpec(200, {"records": [{"id": "beta"}]}),
        ],
    )
    transport.events = events

    result = strategy.extract(plan, spark=provider)

    queries = [parse_qs(urlsplit(request.full_url()).query) for request in transport.requests]
    assert [query["id"] for query in queries] == [["alpha"], ["beta"]]
    assert result.records_extracted == 2
    # The download is the long part of the run; the session must be gone before it starts.
    assert events == ["spark_start", "spark_stop", "request", "request"]
    assert provider.start_count == 1
    assert provider.stop_count == 1


def test_api_strategy_scopes_one_session_for_combined_request_inputs(
    tmp_path,
    spy_spark_provider,
):
    events: list[str] = []
    plan = _build_plan(
        tmp_path,
        source_id="scoped_combined_source",
        pagination_type="none",
        request_inputs={
            "type": "combined",
            "inputs": [
                {
                    "type": "date_window",
                    "start": date(2025, 1, 1),
                    "end": date(2025, 2, 28),
                    "step": "month",
                },
                {
                    "type": "iceberg_rows",
                    "namespace": "bronze_test",
                    "table_name": "ids",
                    "columns": {"entity_id": "id"},
                },
            ],
        },
        parameter_bindings={
            "id": {"from": "request_input.entity_id"},
            "mesAno": {"from": "request_input.window_end", "format": "%Y%m"},
        },
        requests_per_minute=None,
    )
    provider = spy_spark_provider(
        {"bronze_test.ids": [{"id": "alpha"}, {"id": "beta"}]},
        events,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, {"records": [{"id": "1"}]}) for _ in range(4)],
    )
    transport.events = events

    result = strategy.extract(plan, spark=provider)

    # Four contexts in the Cartesian product, but the whole product resolves inside a
    # single scoped acquire — sub-inputs never reopen a session of their own.
    assert len(transport.requests) == 4
    assert result.records_extracted == 4
    assert events[:2] == ["spark_start", "spark_stop"]
    assert events.count("spark_start") == 1
    assert provider.stop_count == 1


def test_api_strategy_never_touches_the_provider_for_date_window_inputs(
    tmp_path,
    spy_spark_provider,
):
    events: list[str] = []
    plan = _build_plan(
        tmp_path,
        source_id="session_free_source",
        pagination_type="none",
        request_inputs={
            "type": "date_window",
            "start": date(2025, 1, 1),
            "end": date(2025, 2, 28),
            "step": "month",
        },
        parameter_bindings={"mesAno": {"from": "request_input.window_end", "format": "%Y%m"}},
        requests_per_minute=None,
    )
    provider = spy_spark_provider({}, events)
    strategy, transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, {"records": [{"id": "1"}]}),
            ResponseSpec(200, {"records": [{"id": "2"}]}),
        ],
    )

    result = strategy.extract(plan, spark=provider)

    assert result.records_extracted == 2
    assert len(transport.requests) == 2
    assert events == []
    assert provider.was_started is False


def test_api_strategy_stops_the_scoped_session_when_the_lookup_fails(
    tmp_path,
    spy_spark_provider,
):
    import pytest

    from janus.strategies.api.request_inputs import ApiRequestInputLoadError

    events: list[str] = []
    plan = _build_plan(
        tmp_path,
        source_id="missing_table_source",
        pagination_type="none",
        request_inputs={
            "type": "iceberg_rows",
            "namespace": "bronze_test",
            "table_name": "absent",
            "columns": {"entity_id": "id"},
        },
        parameter_bindings={"id": {"from": "request_input.entity_id"}},
        requests_per_minute=None,
    )
    provider = spy_spark_provider({}, events)
    strategy, transport = _build_strategy(tmp_path, [])

    with pytest.raises(ApiRequestInputLoadError):
        strategy.extract(plan, spark=provider)

    assert transport.requests == []
    assert events == ["spark_start", "spark_stop"]
    assert provider.stop_count == 1


def _build_strategy(
    tmp_path: Path,
    responses: list[ResponseSpec | Exception],
    *,
    sleeper=None,
    logger=None,
):
    transport = FakeTransport(responses)
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
        sleeper=sleeper or (lambda seconds: None),
        clock=lambda: 0.0,
        logger=logger,
    )
    return strategy, transport


def _build_plan(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "page_number_api",
    pagination_type: str = "page_number",
    extraction_mode: str = "full_refresh",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
    lookback_days: int | None = None,
    page_size: int = 100,
    retry_max_attempts: int = 3,
    retry_backoff_seconds: int = 1,
    retry_backoff_strategy: str = "fixed",
    rate_limit_backoff_seconds: int | None = 5,
    requests_per_minute: int | None = 10,
    concurrency: int = 1,
    params: dict[str, str] | None = None,
    request_inputs: dict[str, Any] | None = None,
    parameter_bindings: dict[str, Any] | None = None,
    dead_letter_max_items: int = 0,
) -> ExecutionPlan:
    source_config = _build_source_config(
        tmp_path,
        source_id=source_id,
        variant=variant,
        pagination_type=pagination_type,
        extraction_mode=extraction_mode,
        checkpoint_field=checkpoint_field,
        checkpoint_strategy=checkpoint_strategy,
        lookback_days=lookback_days,
        page_size=page_size,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        retry_backoff_strategy=retry_backoff_strategy,
        rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        requests_per_minute=requests_per_minute,
        concurrency=concurrency,
        params=params,
        request_inputs=request_inputs,
        parameter_bindings=parameter_bindings,
        dead_letter_max_items=dead_letter_max_items,
    )
    run_context = RunContext.create(
        run_id=f"run-{source_id}",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _build_source_config(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "page_number_api",
    pagination_type: str = "page_number",
    extraction_mode: str = "full_refresh",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
    lookback_days: int | None = None,
    page_size: int = 100,
    retry_max_attempts: int = 3,
    retry_backoff_seconds: int = 1,
    retry_backoff_strategy: str = "fixed",
    rate_limit_backoff_seconds: int | None = 5,
    requests_per_minute: int | None = 10,
    concurrency: int = 1,
    params: dict[str, str] | None = None,
    request_inputs: dict[str, Any] | None = None,
    parameter_bindings: dict[str, Any] | None = None,
    dead_letter_max_items: int = 0,
) -> SourceConfig:
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
                "params": params,
                "request_inputs": request_inputs,
                "parameter_bindings": parameter_bindings,
                "pagination": _pagination_block(pagination_type, page_size),
                "rate_limit": {
                    "requests_per_minute": requests_per_minute,
                    "concurrency": concurrency,
                    "backoff_seconds": rate_limit_backoff_seconds,
                },
            },
            "extraction": {
                "mode": extraction_mode,
                "checkpoint_field": checkpoint_field,
                "checkpoint_strategy": checkpoint_strategy,
                "lookback_days": lookback_days,
                "dead_letter_max_items": dead_letter_max_items,
                "retry": {
                    "max_attempts": retry_max_attempts,
                    "backoff_strategy": retry_backoff_strategy,
                    "backoff_seconds": retry_backoff_seconds,
                },
            },
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
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _pagination_block(pagination_type: str, page_size: int) -> dict[str, Any]:
    if pagination_type == "page_number":
        return {
            "type": "page_number",
            "page_param": "page",
            "size_param": "page_size",
            "page_size": page_size,
        }
    if pagination_type == "offset":
        return {
            "type": "offset",
            "offset_param": "offset",
            "limit_param": "limit",
            "page_size": page_size,
        }
    if pagination_type == "cursor":
        return {
            "type": "cursor",
            "cursor_param": "cursor",
        }
    return {"type": "none"}


def _raw_run_dir(
    tmp_path: Path,
    source_id: str,
    *,
    ingestion_date: str = "2026-04-10",
    run_id: str | None = None,
) -> Path:
    resolved_run_id = run_id or f"run-{source_id}"
    return (
        tmp_path
        / "runtime"
        / "raw"
        / "example"
        / source_id
        / "runs"
        / f"ingestion_date={ingestion_date}"
        / f"run_id={resolved_run_id}"
    )


def _storage_layout(tmp_path: Path) -> StorageLayout:
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
