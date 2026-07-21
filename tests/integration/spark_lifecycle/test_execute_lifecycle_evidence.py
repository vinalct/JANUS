"""End-to-end evidence that a run holds no compute while it downloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from janus.models import RunContext, SourceConfig
from janus.normalizers import NORMALIZATION_METADATA_COLUMNS
from janus.planner import PlannedRun
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.strategies.api import ApiHook, ApiResponse, ApiStrategy
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.logging import build_structured_logger
from janus.utils.storage import StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "lazy_spark_date_window_source"
ICEBERG_RUNTIME_JAR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ivy"
    / "jars"
    / "org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar"
)
ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}

# ── golden expectations ──────────────────────────────────────────────────────
# AC-2 is a claim about outputs being *unchanged*, so these are stated as literals
# derived from the fixed fixture payloads rather than read back out of the run under
# test — otherwise a defect that moved both would go unnoticed.

GOLDEN_WINDOWS = ("2026-01-01", "2026-01-02", "2026-01-03")
GOLDEN_RECORDS = (
    {"id": "rec-1", "record_window": "2026-01-01", "value": 10},
    {"id": "rec-2", "record_window": "2026-01-02", "value": 20},
    {"id": "rec-3", "record_window": "2026-01-03", "value": 30},
)
GOLDEN_RECORD_FIELDS = ("id", "record_window", "value")
# One raw document per request; the API family hands the response bodies to bronze
# untouched, so bronze holds one row per document.
GOLDEN_BRONZE_ROW_COUNT = len(GOLDEN_WINDOWS)
# Events the executor emits for the extraction phase; none may follow session startup.
EXTRACTION_PHASE_EVENTS = (
    "source_execution_started",
    "run_observation_started",
    "source_extraction_started",
    "source_extraction_finished",
    "raw_outputs_materialized",
    "normalization_handoff_prepared",
)


@dataclass(slots=True)
class FixtureTransport:
    """Serves one canned page per request and stamps each send onto a shared trail."""

    payloads: list[dict[str, Any]]
    events: list[str]
    requests: list = field(default_factory=list)
    opened: bool = False
    closed: bool = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request):
        self.requests.append(request)
        self.events.append("request")
        if not self.payloads:
            raise AssertionError("No fixture response remains for the lazy-spark transport")

        return ApiResponse(
            request=request,
            status_code=200,
            body=json.dumps(self.payloads.pop(0)).encode("utf-8"),
        )


class ActiveSessionProbeHook(ApiHook):
    """Test-only hook recording the in-process active session on every request.

    `prepare_request` runs inside the pagination loop, so this is a direct in-process
    check of the AC-1 claim rather than an inference from log ordering.
    """

    def __init__(self) -> None:
        self.observed_active_sessions: list[Any] = []

    def prepare_request(self, plan, request, *, checkpoint_state=None, pagination_state=None):
        from pyspark.sql import SparkSession

        self.observed_active_sessions.append(SparkSession.getActiveSession())
        return super().prepare_request(
            plan,
            request,
            checkpoint_state=checkpoint_state,
            pagination_state=pagination_state,
        )


@pytest.fixture
def session_factory(tmp_path):
    """Build a fresh Iceberg-enabled local session per call, over one warehouse."""

    pyspark_sql = pytest.importorskip("pyspark.sql")
    if not ICEBERG_RUNTIME_JAR.exists():
        pytest.skip("Iceberg runtime jar is not available in the local Ivy cache")
    warehouse_root = tmp_path / "warehouse"

    def build():
        session = (
            pyspark_sql.SparkSession.builder.appName("janus-spark-lifecycle-integration")
            .master("local[1]")
            .config("spark.jars", str(ICEBERG_RUNTIME_JAR))
            .config("spark.sql.extensions", ICEBERG_SESSION_EXTENSIONS)
            .config("spark.sql.defaultCatalog", "janus")
            .config("spark.sql.catalog.janus", ICEBERG_CATALOG_IMPL)
            .config("spark.sql.catalog.janus.type", "hadoop")
            .config("spark.sql.catalog.janus.warehouse", str(warehouse_root / "iceberg"))
            .config("spark.sql.catalog.janus.default-namespace", "bronze")
            .config("spark.sql.warehouse.dir", str(warehouse_root / "spark-warehouse"))
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("WARN")
        return session

    return build


# ── AC-1: the download could not have held a session ─────────────────────────


def test_every_extraction_event_precedes_the_single_session_startup(tmp_path, session_factory):
    """AC-1: the log stream orders the whole extraction phase before Spark starts."""

    stream = StringIO()
    logger = build_structured_logger("janus.tests.spark_lifecycle.ordering", stream=stream)
    executed, _ = _run_source(tmp_path, session_factory, logger=logger)

    assert executed.status == "succeeded", executed.failure_reason
    event_names = [event["event"] for event in _log_events(stream)]

    assert event_names.count("spark_session_starting") == 1
    session_start_index = event_names.index("spark_session_starting")
    for extraction_event in EXTRACTION_PHASE_EVENTS:
        assert extraction_event in event_names, f"{extraction_event} was never emitted"
        assert event_names.index(extraction_event) < session_start_index, (
            f"{extraction_event} was emitted after Spark had already started — "
            "the run held compute during I/O"
        )


def test_no_session_is_active_in_process_while_requests_are_prepared(tmp_path, session_factory):
    """AC-1: a direct in-process probe, not an inference from ordering.

    An extraction-phase hook reads `SparkSession.getActiveSession()` on every request.
    If anything had quietly started a session, this is where it would surface.
    """

    hook = ActiveSessionProbeHook()
    executed, _ = _run_source(tmp_path, session_factory, hook=hook)

    assert executed.status == "succeeded", executed.failure_reason
    assert len(hook.observed_active_sessions) == len(GOLDEN_WINDOWS)
    assert hook.observed_active_sessions == [None] * len(GOLDEN_WINDOWS)


# ── AC-2: one session lifetime, unchanged outputs ────────────────────────────


def test_exactly_one_session_lifetime_and_nothing_left_running(tmp_path, session_factory):
    """AC-2: started once for bronze, stopped after, no active session survives."""

    from pyspark.sql import SparkSession

    stream = StringIO()
    logger = build_structured_logger("janus.tests.spark_lifecycle.lifetime", stream=stream)
    executed, _ = _run_source(tmp_path, session_factory, logger=logger)

    assert executed.status == "succeeded", executed.failure_reason
    event_names = [event["event"] for event in _log_events(stream)]

    assert event_names.count("spark_session_started") == 1
    assert event_names.count("spark_session_stopped") == 1
    assert event_names.index("spark_session_started") < event_names.index("spark_session_stopped")
    assert SparkSession.getActiveSession() is None


def test_bronze_and_metadata_outputs_match_the_golden_expectations(tmp_path, session_factory):
    """AC-2: deferring the session changed when Spark runs, not what it produces."""

    executed, transport = _run_source(tmp_path, session_factory)

    assert executed.status == "succeeded", executed.failure_reason

    # One request per window, carrying the golden window bindings, in order.
    assert _requested_windows(transport) == list(GOLDEN_WINDOWS)

    bronze_results = [result for result in executed.write_results if result.zone == "bronze"]
    assert len(bronze_results) == 1

    verification_session = session_factory()
    try:
        dataframe = verification_session.table(bronze_results[0].path)
        assert dataframe.count() == GOLDEN_BRONZE_ROW_COUNT
        # Lineage columns survive untouched alongside the payload.
        assert set(NORMALIZATION_METADATA_COLUMNS).issubset(set(dataframe.columns))
        assert "records" in dataframe.columns
        assert _content_hash(dataframe) == _golden_content_hash()
    finally:
        verification_session.stop()

    assert executed.extraction_result is not None
    assert executed.extraction_result.records_extracted == len(GOLDEN_RECORDS)
    assert len(executed.extraction_result.artifacts) == len(GOLDEN_WINDOWS)
    assert executed.validation_report is not None
    assert executed.validation_report.report.is_successful is True

    run_metadata = json.loads(Path(executed.run_metadata_path).read_text(encoding="utf-8"))
    lineage = json.loads(Path(executed.lineage_path).read_text(encoding="utf-8"))
    assert run_metadata["status"] == "succeeded"
    assert lineage["status"] == "succeeded"
    assert lineage["strategy_variant"] == "page_number_api"


def test_the_run_summary_reports_the_session_it_actually_paid_for(tmp_path, session_factory):
    """AC-2: the CLI-facing summary is unchanged in shape and honest about the session."""

    provider_holder: list[SparkSessionProvider] = []
    executed, _ = _run_source(tmp_path, session_factory, provider_holder=provider_holder)

    assert executed.status == "succeeded", executed.failure_reason
    provider = provider_holder[0]
    assert provider.was_started is True
    assert provider.session_info is not None
    assert provider.session_info["master"] == "local[1]"

    summary = executed.to_summary()
    assert summary["status"] == "succeeded"
    assert summary["records_extracted"] == len(GOLDEN_RECORDS)
    assert summary["artifact_count"] == len(GOLDEN_WINDOWS)
    assert [output["zone"] for output in summary["materialized_outputs"]].count("bronze") == 1
    assert summary["validation"]["is_successful"] is True


# ── FR-4: the failure path still gives the compute back ──────────────────────


def test_a_quality_failure_still_releases_the_session(tmp_path, session_factory):
    """FR-4: a run failing `unique_fields` stops Spark and still persists metadata.

    `janus_source_id` is written identically onto every bronze row, so configuring it
    as a unique field is a genuine duplicate-key violation rather than a missing column.
    Session release must not depend on the run succeeding — a failed overnight job would
    otherwise leave a cluster allocated.
    """

    from pyspark.sql import SparkSession

    stream = StringIO()
    logger = build_structured_logger("janus.tests.spark_lifecycle.failure", stream=stream)
    executed, _ = _run_source(
        tmp_path,
        session_factory,
        logger=logger,
        unique_fields=["janus_source_id"],
    )

    assert executed.status == "failed"
    assert executed.validation_report is not None
    assert executed.validation_report.report.is_successful is False
    assert "data.unique_fields" in [
        f"{check.phase}.{check.name}"
        for check in executed.validation_report.report.failed_checks
    ]

    event_names = [event["event"] for event in _log_events(stream)]
    assert event_names.count("spark_session_started") == 1
    assert event_names.count("spark_session_stopped") == 1
    assert SparkSession.getActiveSession() is None

    # The failure is recorded, not swallowed, and the metadata zone still holds it.
    assert executed.run_metadata_path is not None
    run_metadata = json.loads(Path(executed.run_metadata_path).read_text(encoding="utf-8"))
    assert run_metadata["status"] == "failed"


# ── helpers ──────────────────────────────────────────────────────────────────


def _run_source(
    tmp_path: Path,
    session_factory,
    *,
    logger=None,
    hook=None,
    unique_fields: list[str] | None = None,
    provider_holder: list[SparkSessionProvider] | None = None,
):
    transport = FixtureTransport(
        payloads=[{"records": [record]} for record in GOLDEN_RECORDS],
        events=[],
    )
    storage_layout = StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path)
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    run_context = RunContext.create(
        run_id="run-spark-lifecycle-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 6, 11, 0, tzinfo=UTC),
    )
    source_config = _source_config(tmp_path, unique_fields=unique_fields)
    plan = strategy.plan(source_config, run_context, hook=hook)
    planned_run = PlannedRun(plan=plan, strategy=strategy, hook=hook)

    provider = SparkSessionProvider({}, {}, logger, session_factory=session_factory)
    if provider_holder is not None:
        provider_holder.append(provider)

    executed = SourceExecutor(logger=logger).execute(planned_run, provider, ENVIRONMENT_CONFIG)
    return executed, transport


def _log_events(stream: StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def _requested_windows(transport: FixtureTransport) -> list[str]:
    return [
        parse_qs(urlsplit(request.full_url()).query)["start_date"][0]
        for request in transport.requests
    ]


def _content_hash(dataframe) -> str:
    """Order-independent digest of the payload records that landed in bronze."""

    exploded = dataframe.selectExpr("explode(records) as record").select(
        *[f"record.{field_name}" for field_name in GOLDEN_RECORD_FIELDS]
    )
    rows = sorted(
        "|".join(str(row[field_name]) for field_name in GOLDEN_RECORD_FIELDS)
        for row in exploded.collect()
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _golden_content_hash() -> str:
    rows = sorted(
        "|".join(str(record[field_name]) for field_name in GOLDEN_RECORD_FIELDS)
        for record in GOLDEN_RECORDS
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _source_config(tmp_path: Path, *, unique_fields: list[str] | None = None) -> SourceConfig:
    payload = _source_config_payload(unique_fields=unique_fields)
    config_path = tmp_path / "conf" / "sources" / f"{SOURCE_ID}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return SourceConfig.from_mapping(payload, config_path)


def _source_config_payload(*, unique_fields: list[str] | None = None) -> dict[str, Any]:
    quality: dict[str, Any] = {"allow_schema_evolution": True}
    if unique_fields:
        quality["unique_fields"] = unique_fields

    return {
        "source_id": SOURCE_ID,
        "name": SOURCE_ID,
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
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
            "request_inputs": {
                "type": "date_window",
                "start": "2026-01-01",
                "end": "2026-01-03",
                "step": "day",
            },
            "parameter_bindings": {
                "start_date": {"from": "request_input.window_start"},
                "end_date": {"from": "request_input.window_end"},
            },
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 2,
            },
            "rate_limit": {
                "requests_per_minute": None,
                "concurrency": 1,
                "backoff_seconds": 5,
            },
        },
        "extraction": {
            "mode": "full_refresh",
            "checkpoint_strategy": "none",
            "dead_letter_max_items": 0,
            "retry": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "backoff_seconds": 1,
            },
        },
        "schema": {"mode": "infer"},
        "spark": {
            "input_format": "json",
            "write_mode": "append",
        },
        "outputs": {
            "raw": {"path": f"data/raw/example/{SOURCE_ID}", "format": "json"},
            "bronze": {"path": f"data/bronze/example/{SOURCE_ID}", "format": "iceberg"},
            "metadata": {"path": f"data/metadata/example/{SOURCE_ID}", "format": "json"},
        },
        "quality": quality,
    }
