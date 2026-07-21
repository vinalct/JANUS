"""AC-3 for the `combined(date_window, iceberg_rows)` shape — the FR-2 clause.

The sibling suite `tests/integration/lazy_spark/` proves a plain `iceberg_rows` source
takes a scoped lookup session and then a separate materialization session. This one
covers the harder shape: an `iceberg_rows` input nested inside a `combined` block, where
the Cartesian product is computed across sub-inputs.

Two things must hold, and only an end-to-end run can show them:

1. **One** scoped session serves the whole product — not one per sub-input, and not one
   held open across the download.
2. The product contexts the scoped read resolves are identical to the golden list the
   eager implementation produced, so deferring the session did not reorder or drop a
   combination.
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
SOURCE_ID = "lazy_spark_combined_source"
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

UPSTREAM_ENTITY_IDS = ("alpha", "beta")
WINDOW_START = "2026-01-01"
WINDOW_END = "2026-01-02"
# The Cartesian product the run must issue, in order: date windows are the outer loop,
# upstream rows the inner one. Golden, because reordering or dropping a combination is
# exactly the regression a scoped read could introduce.
GOLDEN_REQUEST_CONTEXTS = [
    ("2026-01-01", "alpha"),
    ("2026-01-01", "beta"),
    ("2026-01-02", "alpha"),
    ("2026-01-02", "beta"),
]


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
            raise AssertionError("No fixture response remains for the combined-input transport")

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
            pyspark_sql.SparkSession.builder.appName("janus-combined-lifecycle-integration")
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


def test_combined_with_iceberg_takes_one_scoped_session_for_the_whole_product(
    tmp_path,
    session_factory,
):
    """ one scoped lookup, a session-free download, then materialization."""

    _seed_upstream_table(session_factory)

    events: list[str] = []
    provider = LifecycleRecordingProvider(session_factory, events)
    transport = FixtureTransport(
        payloads=[
            {"records": [{"id": entity_id, "window": window}]}
            for window, entity_id in GOLDEN_REQUEST_CONTEXTS
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
        run_id="run-combined-lifecycle-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 6, 11, 0, tzinfo=UTC),
    )
    plan = strategy.plan(_source_config(tmp_path), run_context)
    planned_run = PlannedRun(plan=plan, strategy=strategy)

    executed = SourceExecutor().execute(planned_run, provider, ENVIRONMENT_CONFIG)

    assert executed.status == "succeeded", executed.failure_reason

    # (a) The product is exactly the golden one — no combination reordered or dropped.
    assert _requested_contexts(transport) == GOLDEN_REQUEST_CONTEXTS

    # (b) Two lifetimes, in order: one scoped pair covers the *whole* product before the
    #     first request goes out, and the materialization pair opens only afterwards.
    assert events == [
        "spark_start",
        "spark_stop",
        *["request"] * len(GOLDEN_REQUEST_CONTEXTS),
        "spark_start",
        "spark_stop",
    ]
    assert len(provider.sessions) == 2
    assert provider.sessions[0] is not provider.sessions[1]

    # (c) The run still lands in bronze with one document per product combination.
    bronze_results = [result for result in executed.write_results if result.zone == "bronze"]
    assert len(bronze_results) == 1
    assert executed.extraction_result is not None
    assert executed.extraction_result.records_extracted == len(GOLDEN_REQUEST_CONTEXTS)

    verification_session = session_factory()
    try:
        assert verification_session.table(bronze_results[0].path).count() == len(
            GOLDEN_REQUEST_CONTEXTS
        )
    finally:
        verification_session.stop()


def _requested_contexts(transport: FixtureTransport) -> list[tuple[str, str]]:
    contexts: list[tuple[str, str]] = []
    for request in transport.requests:
        query = parse_qs(urlsplit(request.full_url()).query)
        contexts.append((query["start_date"][0], query["id"][0]))
    return contexts


def _seed_upstream_table(session_factory) -> None:
    """Write the upstream Iceberg table the request inputs read, then release Spark."""

    session = session_factory()
    try:
        session.sql(f"CREATE NAMESPACE IF NOT EXISTS {UPSTREAM_NAMESPACE}")
        rows = [(entity_id,) for entity_id in UPSTREAM_ENTITY_IDS]

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
                "type": "combined",
                "inputs": [
                    {
                        "type": "date_window",
                        "start": WINDOW_START,
                        "end": WINDOW_END,
                        "step": "day",
                    },
                    {
                        "type": "iceberg_rows",
                        "namespace": UPSTREAM_NAMESPACE,
                        "table_name": UPSTREAM_TABLE,
                        "columns": {"entity_id": "entity_id"},
                    },
                ],
            },
            "parameter_bindings": {
                "start_date": {"from": "request_input.window_start"},
                "id": {"from": "request_input.entity_id"},
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
        "quality": {"allow_schema_evolution": True},
    }
