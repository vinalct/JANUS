"""Same-JVM guard for the scoped `iceberg_rows` session.

An `iceberg_rows` run now starts a session for the upstream lookup, stops it, downloads
session-free, and starts a *second* session to materialize bronze — all in one process.
Restarting a `SparkSession` after `stop()` is supported but has sharp edges (stale
context singletons, metastore locks, Ivy re-resolution), so it is proven end to end here
against a real local Spark rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from janus.models import RunContext, SourceConfig
from janus.planner import PlannedRun
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "lazy_spark_iceberg_rows_source"
UPSTREAM_NAMESPACE = "bronze_test"
UPSTREAM_TABLE = "upstream_ids"
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
# The request inputs this run must resolve from the upstream table, in order.
GOLDEN_REQUEST_INPUT_IDS = ["alpha", "beta"]


@dataclass(slots=True)
class FixtureTransport:
    """Serves one canned page per request and marks each send on a shared trail."""

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


class LifecycleRecordingProvider(SparkSessionProvider):
    """Real provider that also records every session it hands out and releases."""

    def __init__(self, session_factory, events: list[str]) -> None:
        super().__init__({}, {}, session_factory=session_factory)
        self.events = events
        self.sessions: list[Any] = []
        self._live = False

    def get(self):
        session = super().get()
        if not self._live:
            self._live = True
            self.sessions.append(session)
            self.events.append("spark_start")
        return session

    def stop(self) -> None:
        super().stop()
        if self._live:
            self._live = False
            self.events.append("spark_stop")


@pytest.fixture
def session_factory(tmp_path):
    """Build a fresh Iceberg-enabled local session per call, over one warehouse."""

    pyspark_sql = pytest.importorskip("pyspark.sql")
    if not ICEBERG_RUNTIME_JAR.exists():
        pytest.skip("Iceberg runtime jar is not available in the local Ivy cache")
    warehouse_root = tmp_path / "warehouse"

    def build():
        session = (
            pyspark_sql.SparkSession.builder.appName("janus-lazy-spark-integration")
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


def test_iceberg_rows_run_uses_a_scoped_lookup_session_then_a_later_materialize_session(
    tmp_path,
    session_factory,
):
    _seed_upstream_table(session_factory)

    events: list[str] = []
    provider = LifecycleRecordingProvider(session_factory, events)
    transport = FixtureTransport(
        payloads=[
            {"records": [{"id": "alpha", "value": 1}]},
            {"records": [{"id": "beta", "value": 2}]},
        ],
        events=events,
    )
    storage_layout = StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path)
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    run_context = RunContext.create(
        run_id="run-lazy-spark-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 6, 11, 0, tzinfo=UTC),
    )
    plan = strategy.plan(_source_config(tmp_path), run_context)
    planned_run = PlannedRun(plan=plan, strategy=strategy)

    executed = SourceExecutor().execute(planned_run, provider, ENVIRONMENT_CONFIG)

    assert executed.status == "succeeded", executed.failure_reason

    # (a) The upstream lookup resolved exactly the expected request inputs.
    resolved_ids = [
        parse_qs(urlsplit(request.full_url()).query)["id"][0] for request in transport.requests
    ]
    assert resolved_ids == GOLDEN_REQUEST_INPUT_IDS

    # (b) Two sessions, in order: the scoped lookup one is stopped before the first
    #     request goes out, and the materialization one is started only afterwards.
    assert events == [
        "spark_start",
        "spark_stop",
        "request",
        "request",
        "spark_start",
        "spark_stop",
    ]
    assert len(provider.sessions) == 2
    assert provider.sessions[0] is not provider.sessions[1]

    # (c) The run still lands in bronze, and its summary/lineage are intact.
    bronze_results = [result for result in executed.write_results if result.zone == "bronze"]
    assert len(bronze_results) == 1
    assert executed.extraction_result is not None
    assert executed.extraction_result.records_extracted == 2

    summary = executed.to_summary()
    assert summary["status"] == "succeeded"
    assert summary["records_extracted"] == 2

    lineage_payload = json.loads(executed.lineage_path.read_text(encoding="utf-8"))
    assert lineage_payload["status"] == "succeeded"
    assert lineage_payload["strategy_variant"] == "page_number_api"

    verification_session = session_factory()
    try:
        assert verification_session.table(bronze_results[0].path).count() == 2
    finally:
        verification_session.stop()


def _seed_upstream_table(session_factory) -> None:
    """Write the upstream Iceberg table the request inputs read, then release Spark."""

    session = session_factory()
    try:
        session.sql(f"CREATE NAMESPACE IF NOT EXISTS {UPSTREAM_NAMESPACE}")
        rows = [(entity_id,) for entity_id in GOLDEN_REQUEST_INPUT_IDS]

        session.createDataFrame(rows, "entity_id string").writeTo(
            f"{UPSTREAM_NAMESPACE}.{UPSTREAM_TABLE}"
        ).using("iceberg").create()
    finally:
        session.stop()


def _source_config(tmp_path: Path) -> SourceConfig:
    payload = _source_config_payload()
   
    config_path = tmp_path / "conf" / "sources" / f"{SOURCE_ID}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return SourceConfig.from_mapping(payload, config_path)


def _source_config_payload() -> dict[str, Any]:
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
                "type": "iceberg_rows",
                "namespace": UPSTREAM_NAMESPACE,
                "table_name": UPSTREAM_TABLE,
                "columns": {"entity_id": "entity_id"},
            },
            "parameter_bindings": {"id": {"from": "request_input.entity_id"}},
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
        "quality": {"allow_schema_evolution": True},
    }
