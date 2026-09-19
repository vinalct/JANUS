"""FR-2 / FR-4 / NFR-2: the three transports, and what emission may never cost a run."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker

from janus.lineage import PersistedArtifacts, RunObserver
from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    RunContext,
    WriteResult,
)
from janus.observability import (
    IcebergAppendOutcome,
    IcebergAppendResult,
    RunEmissionOutcome,
    build_run_event_emitter,
)
from janus.observability.openlineage import (
    DEFAULT_HTTP_ENDPOINT,
    NOT_CONFIGURED_REASON,
    PROFILE_ERROR_REASON,
    SUPPORTED_TRANSPORTS,
    DisabledOpenLineageTransport,
    FileOpenLineageTransport,
    HttpOpenLineageTransport,
    OpenLineageEmissionOutcome,
    OpenLineageProfileError,
    OpenLineageRunSink,
    OpenLineageTransportKind,
    build_openlineage_sink,
    build_openlineage_transport,
    resolve_openlineage_settings,
)
from janus.registry import load_registry
from janus.utils.environment import load_environment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = PROJECT_ROOT / "conf" / "environments"
OPENLINEAGE_SCHEMA = PROJECT_ROOT / "tests" / "fixtures" / "openlineage" / "OpenLineage-2-0-2.json"
STARTED_AT = datetime(2026, 9, 16, 12, tzinfo=UTC)
FINISHED_AT = datetime(2026, 9, 16, 12, 0, 5, tzinfo=UTC)
TOKEN = "s3cret-lineage-token"
ENV_EXPANSION = re.compile(r"^\$\{[A-Z0-9_]+(:-[^}]*)?\}$")


# ── logging spy ──────────────────────────────────────────────────────────────


class SpyLogger:
    """Captures exactly what an operator would see, so a leak is a failing assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.events.append(("info", event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        self.events.append(("warning", event, fields))

    @property
    def text(self) -> str:
        return json.dumps(self.events, default=str)

    def warnings(self, event: str) -> list[dict[str, Any]]:
        return [
            fields
            for level, name, fields in self.events
            if level == "warning" and name == event
        ]


# ── fake lineage receiver ────────────────────────────────────────────────────


class _Receiver(BaseHTTPRequestHandler):
    status = 200

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.server.received.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": self.rfile.read(length),
            }
        )
        self.send_response(self.server.status)  # type: ignore[attr-defined]
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args: Any) -> None:
        del args


@pytest.fixture
def receiver():
    """A local socket, never a real endpoint: no test in this module leaves the machine."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Receiver)
    server.received = []  # type: ignore[attr-defined]
    server.status = 200  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _receiver_url(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def _closed_port() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Receiver)
    port = server.server_address[1]
    server.server_close()
    return port


# ── profiles under test ──────────────────────────────────────────────────────


def _config(**openlineage: Any) -> dict[str, Any]:
    return {
        "spark": {"iceberg": {"catalog_name": "janus", "warehouse_dir": "/warehouse"}},
        "observability": {"openlineage": openlineage},
    }


def _paths(tmp_path: Path) -> dict[str, Any]:
    return {
        "metadata_dir": tmp_path / "data" / "metadata",
        "iceberg_warehouse_dir": "s3://janus-bronze/warehouse",
    }


def _validator(path: Path) -> Draft202012Validator:
    return Draft202012Validator(
        json.loads(path.read_text(encoding="utf-8")),
        format_checker=FormatChecker(),
    )


# ── selection ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({"transport": "file"}, FileOpenLineageTransport),
        ({"transport": "http", "url": "https://lineage.invalid"}, HttpOpenLineageTransport),
        ({"transport": "disabled"}, DisabledOpenLineageTransport),
    ],
)
def test_each_transport_is_selected_declaratively_from_the_profile(tmp_path, block, expected):
    transport = build_openlineage_transport(_config(**block), _paths(tmp_path))

    assert isinstance(transport, expected)
    assert transport.kind == block["transport"]


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"observability": {}},
        {"observability": {"runs_table": "audit.runs"}},
        {"observability": {"openlineage": {"transport": ""}}},
    ],
)
def test_an_absent_block_resolves_to_disabled_rather_than_to_an_error(tmp_path, config):
    """Not asking for lineage emission is a legitimate state, not a broken profile."""
    settings = resolve_openlineage_settings(config)
    transport = build_openlineage_transport(config, _paths(tmp_path))

    assert settings.kind is OpenLineageTransportKind.DISABLED
    assert not settings.enabled
    assert isinstance(transport, DisabledOpenLineageTransport)
    assert transport.reason == NOT_CONFIGURED_REASON
    assert not transport.reportable


def test_an_unrecognised_transport_is_refused_at_profile_read_time():
    """A typo must not mean silence forever; the message names what is accepted."""
    with pytest.raises(OpenLineageProfileError) as failure:
        resolve_openlineage_settings(_config(transport="htpp"))

    message = str(failure.value)
    assert "htpp" in message
    for accepted in SUPPORTED_TRANSPORTS:
        assert accepted in message


def test_a_profile_error_degrades_inside_a_run_instead_of_propagating(tmp_path):
    """FR-4 forbids raising during a run, so the same error becomes a logged warning."""
    logger = SpyLogger()

    transport = build_openlineage_transport(
        _config(transport="htpp"),
        _paths(tmp_path),
        logger=logger,
    )

    assert isinstance(transport, DisabledOpenLineageTransport)
    assert transport.reason == PROFILE_ERROR_REASON
    assert transport.reportable
    warning = logger.warnings("openlineage_transport_unavailable")[0]
    assert warning["exception_type"] == "OpenLineageProfileError"


@pytest.mark.parametrize(
    "block",
    [
        {"transport": "http"},
        {"transport": "http", "url": "lineage.invalid:5000"},
        {"transport": "http", "url": "https://lineage.invalid", "timeout_seconds": "0"},
        {"transport": "http", "url": "https://lineage.invalid", "timeout_seconds": "soon"},
        {"transport": "file", "directory": "events"},
        {"transport": "http", "url": "https://lineage.invalid", "auth": {"api_key": "x"}},
    ],
)
def test_an_unusable_block_fails_closed(block):
    with pytest.raises(OpenLineageProfileError):
        resolve_openlineage_settings(_config(**block))


# ── NFR-2 as a guardrail over the shipped configuration ──────────────────────


def _tracked_profiles() -> list[Path]:
    return sorted(ENVIRONMENTS.glob("*.yaml"))


@pytest.fixture
def shipped_profiles(monkeypatch) -> list[Path]:
    """Every tracked profile, read as the repository ships it rather than as this shell is."""
    for name in tuple(key for key in os.environ if key.startswith("JANUS_")):
        monkeypatch.delenv(name, raising=False)
    profiles = _tracked_profiles()
    assert profiles, "the profile sweep matched no files and would pass vacuously"
    return profiles


def test_no_tracked_profile_configures_an_http_endpoint(shipped_profiles):
    """Hermetic in CI is a property of what ships, not of a CI-only override."""
    for profile in shipped_profiles:
        config = load_environment_config(profile.stem, PROJECT_ROOT)
        settings = resolve_openlineage_settings(config)

        assert settings.kind is not OpenLineageTransportKind.HTTP, profile.name
        assert settings.http is None, profile.name


def test_no_tracked_profile_carries_a_credential_literal():
    """Order-13's rule, extended to this block: every credential is an env expansion."""
    profiles = _tracked_profiles()
    assert profiles, "the profile sweep matched no files and would pass vacuously"

    checked = 0
    for profile in profiles:
        document = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
        block = document.get("observability", {}).get("openlineage")
        if block is None:
            continue
        for key, value in dict(block.get("auth") or {}).items():
            assert ENV_EXPANSION.match(str(value)), f"{profile.name}:{key} is a literal"
            checked += 1
        assert ENV_EXPANSION.match(str(block.get("url", "${X:-}"))), profile.name

    assert checked, "no auth value was inspected; the sweep found no block to check"


def test_the_shipped_profiles_select_the_hermetic_file_transport(shipped_profiles):
    kinds = {
        profile.stem: resolve_openlineage_settings(
            load_environment_config(profile.stem, PROJECT_ROOT)
        ).kind
        for profile in shipped_profiles
    }

    assert set(kinds.values()) <= {OpenLineageTransportKind.FILE, OpenLineageTransportKind.DISABLED}
    assert OpenLineageTransportKind.FILE in kinds.values()


def test_an_unset_token_yields_no_authorization_header_rather_than_an_empty_one(
    tmp_path,
    receiver,
):
    """``token: ${JANUS_OPENLINEAGE_API_KEY:-}`` with nothing exported sends no header."""
    sink = build_openlineage_sink(
        _config(transport="http", url=_receiver_url(receiver), auth={"token": ""}),
        _paths(tmp_path),
    )

    assert sink.emit(_started_metadata(tmp_path), budget_seconds=5).emitted
    assert "Authorization" not in receiver.received[0]["headers"]


# ── the file transport ───────────────────────────────────────────────────────


def test_file_events_append_as_ndjson_that_validates_against_the_pinned_schema(tmp_path):
    sink = build_openlineage_sink(_config(transport="file"), _paths(tmp_path))
    persisted = _terminal_artifacts(tmp_path, status="succeeded")

    started = sink.emit(_started_metadata(tmp_path), budget_seconds=5)
    terminal = sink.emit(
        persisted.run_metadata,
        lineage_record=persisted.lineage_record,
        run_record=_project(persisted),
        budget_seconds=5,
    )

    assert (started.emitted, terminal.emitted) == (True, True)
    lines = Path(terminal.target).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    events = [json.loads(line) for line in lines]
    for event in events:
        _validator(OPENLINEAGE_SCHEMA).validate(event)
    assert [event["eventType"] for event in events] == ["START", "COMPLETE"]


def test_file_events_are_kept_one_file_per_day(tmp_path):
    transport = build_openlineage_transport(_config(transport="file"), _paths(tmp_path))

    first = transport.send({"eventTime": "2026-09-16T23:59:00+00:00"}, budget_seconds=1)
    second = transport.send({"eventTime": "2026-09-17T00:01:00+00:00"}, budget_seconds=1)

    assert Path(first.target).name == "events-2026-09-16.ndjson"
    assert Path(second.target).name == "events-2026-09-17.ndjson"


@pytest.mark.parametrize("configured", ["/etc/janus", "../../escape", "lineage/../../escape"])
def test_a_path_that_leaves_the_metadata_zone_is_refused(tmp_path, configured):
    logger = SpyLogger()

    transport = build_openlineage_transport(
        _config(transport="file", path=configured),
        _paths(tmp_path),
        logger=logger,
    )

    assert isinstance(transport, DisabledOpenLineageTransport)
    assert transport.reason == PROFILE_ERROR_REASON
    assert logger.warnings("openlineage_transport_unavailable")


def test_the_events_path_stays_inside_the_metadata_zone(tmp_path):
    transport = build_openlineage_transport(_config(transport="file"), _paths(tmp_path))

    assert transport.directory.is_relative_to(_paths(tmp_path)["metadata_dir"])


def test_concurrent_writers_do_not_interleave_a_line(tmp_path):
    """The batch runner and a parallel invocation append to the same day file at once."""
    transport = build_openlineage_transport(_config(transport="file"), _paths(tmp_path))
    event_time = "2026-09-16T12:00:00+00:00"
    barrier = threading.Barrier(8)

    def append(index: int) -> None:
        barrier.wait()
        transport.send(
            {"eventTime": event_time, "run": {"runId": f"run-{index}", "pad": "x" * 4096}},
            budget_seconds=5,
        )

    threads = [threading.Thread(target=append, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = Path(transport.path_for({"eventTime": event_time})).read_text().splitlines()
    assert len(lines) == 8
    assert {json.loads(line)["run"]["runId"] for line in lines} == {
        f"run-{index}" for index in range(8)
    }


def test_an_unwritable_events_directory_is_reported_not_raised(tmp_path):
    blocked = tmp_path / "data" / "metadata"
    blocked.parent.mkdir(parents=True)
    blocked.write_text("not a directory")
    transport = build_openlineage_transport(_config(transport="file"), _paths(tmp_path))

    result = transport.send({"eventTime": "2026-09-16T12:00:00+00:00"}, budget_seconds=5)

    assert result.outcome is OpenLineageEmissionOutcome.FAILED
    assert result.step == "directory_create"


# ── the HTTP transport ───────────────────────────────────────────────────────


def test_the_http_transport_posts_one_well_formed_event(tmp_path, receiver):
    sink = build_openlineage_sink(
        _config(transport="http", url=_receiver_url(receiver), auth={"token": TOKEN}),
        _paths(tmp_path),
    )

    result = sink.emit(_started_metadata(tmp_path), budget_seconds=5)

    assert result.emitted and result.status_code == 200
    assert len(receiver.received) == 1
    request = receiver.received[0]
    assert request["path"] == f"/{DEFAULT_HTTP_ENDPOINT}"
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(request["body"])["eventType"] == "START"


@pytest.mark.parametrize("status", [401, 500])
def test_a_rejected_post_is_logged_once_and_never_retried(tmp_path, receiver, status):
    receiver.status = status
    logger = SpyLogger()
    sink = build_openlineage_sink(
        _config(
            transport="http",
            url=f"{_receiver_url(receiver)}?api_key=leaky",
            auth={"token": TOKEN},
        ),
        _paths(tmp_path),
    )

    result = sink.emit(_started_metadata(tmp_path), budget_seconds=5, logger=logger)

    assert result.outcome is OpenLineageEmissionOutcome.FAILED
    assert result.status_code == status
    assert len(receiver.received) == 1, "a telemetry POST must be attempted exactly once"
    warning = logger.warnings("openlineage_emission_degraded")[0]
    assert warning["status_code"] == status
    assert "REDACTED" in warning["target"] and "leaky" not in warning["target"]
    assert TOKEN not in logger.text


def test_a_refused_connection_is_reported_without_reaching_the_run(tmp_path):
    logger = SpyLogger()
    sink = build_openlineage_sink(
        _config(transport="http", url=f"http://127.0.0.1:{_closed_port()}", auth={"token": TOKEN}),
        _paths(tmp_path),
    )

    result = sink.emit(_started_metadata(tmp_path), budget_seconds=5, logger=logger)

    assert result.outcome is OpenLineageEmissionOutcome.FAILED
    assert result.step == "request"
    assert result.exception_type == "ApiTransportError"
    assert TOKEN not in logger.text


def test_a_transport_that_raises_becomes_data(tmp_path, monkeypatch):
    """Whatever the shared transport does — a timeout, an SSL error — never escapes."""
    import janus.strategies.http.transport as http_transport

    class Exploding:
        def send(self, request):
            del request
            raise TimeoutError("timed out")

        def close(self) -> None:
            pass

    monkeypatch.setattr(http_transport, "UrllibApiTransport", Exploding)
    sink = build_openlineage_sink(
        _config(transport="http", url="https://lineage.invalid"),
        _paths(tmp_path),
    )

    result = sink.emit(_started_metadata(tmp_path), budget_seconds=5)

    assert result.outcome is OpenLineageEmissionOutcome.FAILED
    assert result.exception_type == "TimeoutError"


# ── what emission may never cost a run ───────────────────────────────────────


@pytest.mark.parametrize("transport_block", ["failing_http", "escaping_file", "unrecognised"])
def test_a_broken_transport_leaves_the_run_and_its_json_untouched(tmp_path, transport_block):
    blocks = {
        "failing_http": {"transport": "http", "url": f"http://127.0.0.1:{_closed_port()}"},
        "escaping_file": {"transport": "file", "path": "/etc/janus"},
        "unrecognised": {"transport": "htpp"},
    }
    plan = _plan(tmp_path, "task08-untouched")
    emitter = build_run_event_emitter(
        _config(**blocks[transport_block]),
        _paths(tmp_path),
        runs_table_sink=_emitting_sink,
    )
    observer = RunObserver(emitter=emitter)

    persisted = observer.record_success(
        plan,
        _extraction(plan),
        _writes(plan),
        finished_at=FINISHED_AT,
    )

    assert persisted.run_metadata.status == "succeeded"
    assert json.loads(persisted.run_metadata_path.read_text())["status"] == "succeeded"
    assert json.loads(persisted.lineage_path.read_text())["status"] == "succeeded"
    assert emitter.last_result.outcome is RunEmissionOutcome.EMITTED
    assert emitter.last_result.openlineage.outcome is not OpenLineageEmissionOutcome.EMITTED


def test_the_total_emission_budget_holds_with_the_slowest_transport(tmp_path):
    """An unbounded transport is the obvious way to break FR-4 by accident."""

    class Glacial:
        kind = "http"

        def send(self, event, *, budget_seconds):
            del event, budget_seconds
            time.sleep(30)
            raise AssertionError("the budget must expire long before this returns")

    sink = build_openlineage_sink(_config(transport="file"), _paths(tmp_path))
    emitter = build_run_event_emitter(
        _config(transport="file"),
        _paths(tmp_path),
        timeout_seconds=0.25,
        runs_table_sink=_emitting_sink,
        openlineage_sink=OpenLineageRunSink(
            transport=Glacial(),
            dataset_context=sink.dataset_context,
        ),
    )
    persisted = _terminal_artifacts(tmp_path, status="succeeded")
    started_at = time.monotonic()

    emitter.emit_succeeded(_plan(tmp_path, "unused"), persisted)

    assert time.monotonic() - started_at < 1.0
    assert emitter.last_result.outcome is RunEmissionOutcome.FAILED
    assert emitter.last_result.stage == "budget"


def test_start_emits_one_event_and_no_row(tmp_path):
    appends: list[Any] = []

    def sink(record, config, resolved_paths, *, logger, timeout_seconds):
        del config, resolved_paths, logger, timeout_seconds
        appends.append(record)
        return IcebergAppendResult(IcebergAppendOutcome.EMITTED, "metadata.runs")

    emitter = build_run_event_emitter(
        _config(transport="file"),
        _paths(tmp_path),
        runs_table_sink=sink,
    )
    observer = RunObserver(emitter=emitter)

    observer.start_run(_plan(tmp_path, "task08-start"))

    assert appends == [], "the runs table is a ledger of terminal runs"
    events = _emitted_events(tmp_path)
    assert [event["eventType"] for event in events] == ["START"]
    assert emitter.last_result is None
    assert emitter.last_started_result.openlineage.emitted


def test_a_disabled_profile_writes_nothing_and_still_appends_its_row(tmp_path):
    logger = SpyLogger()
    emitter = build_run_event_emitter({}, {}, logger=logger, runs_table_sink=_emitting_sink)
    observer = RunObserver(emitter=emitter)
    plan = _plan(tmp_path, "task08-disabled")

    observer.start_run(plan)
    observer.record_success(plan, _extraction(plan), _writes(plan), finished_at=FINISHED_AT)

    # A destination nobody configured is a state, not a degradation: no warning is logged.
    assert not logger.warnings("run_event_emission_finished")
    assert not logger.warnings("openlineage_emission_degraded")

    assert emitter.last_result.outcome is RunEmissionOutcome.EMITTED
    openlineage = emitter.last_result.openlineage
    assert openlineage.outcome is OpenLineageEmissionOutcome.SKIPPED
    assert openlineage.reason == NOT_CONFIGURED_REASON
    assert not list((tmp_path / "data" / "metadata" / "lineage" / "openlineage").glob("*"))


# ── fixtures ─────────────────────────────────────────────────────────────────


def _emitting_sink(*args: Any, **kwargs: Any) -> IcebergAppendResult:
    del args, kwargs
    return IcebergAppendResult(IcebergAppendOutcome.EMITTED, "metadata.runs")


def _emitted_events(tmp_path: Path) -> list[dict[str, Any]]:
    directory = tmp_path / "data" / "metadata" / "lineage" / "openlineage"
    return [
        json.loads(line)
        for path in sorted(directory.glob("*.ndjson"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _project(persisted: PersistedArtifacts):
    from janus.observability.emission import _project_run_record

    return _project_run_record(persisted)


def _started_metadata(tmp_path: Path):
    return RunObserver().start_run(_plan(tmp_path / "started", "task08-started")).run_metadata


def _terminal_artifacts(tmp_path: Path, *, status: str) -> PersistedArtifacts:
    plan = _plan(tmp_path / status, f"task08-{status}")
    observer = RunObserver()
    return observer.record_success(
        plan,
        _extraction(plan),
        _writes(plan),
        finished_at=FINISHED_AT,
    )


def _plan(root: Path, run_id: str) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id=run_id,
            environment="local",
            project_root=root,
            started_at=STARTED_AT,
        ),
    )


def _extraction(plan: ExecutionPlan) -> ExtractionResult:
    return ExtractionResult.from_plan(
        plan,
        artifacts=(
            ExtractedArtifact(path=f"{plan.raw_output.path}/page-0001.json", format="json"),
        ),
        records_extracted=3,
        checkpoint_value="2026-09-16T12:00:00Z",
    )


def _writes(plan: ExecutionPlan) -> tuple[WriteResult, ...]:
    return (
        WriteResult.from_plan(
            plan,
            "bronze",
            path="bronze.task",
            format_name="iceberg",
            mode="append",
            records_written=3,
        ),
    )
