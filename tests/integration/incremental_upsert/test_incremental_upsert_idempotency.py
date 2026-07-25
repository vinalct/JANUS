"""Two-run duplicate evidence for incremental sources.

An incremental source re-fetches its boundary window on every run: the stored checkpoint
is sent back as an *inclusive* lower bound, shifted further back by ``lookback_days`` when
one is declared. The bronze write path answers that overlap with ``INSERT INTO`` and no
key awareness, so run *N+1* appends rows bronze already holds.

This suite records that bug as an executable artifact. It is expected to be **red** until
replaces the blind append with a key-based ``MERGE INTO``.
"""

from __future__ import annotations

import json
from collections import Counter
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
from janus.scripts import ingest_raw_to_bronze
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.utils.environment import ICEBERG_CATALOG_IMPL, ICEBERG_SESSION_EXTENSIONS
from janus.utils.storage import StorageLayout, bronze_table_identifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ID = "incremental_upsert_fixture"
BRONZE_NAMESPACE = "bronze_test"
BRONZE_TABLE_NAME = "incremental_upsert_fixture"
BRONZE_PATH = f"data/bronze/example/{SOURCE_ID}"
BRONZE_TABLE = bronze_table_identifier(
    BRONZE_PATH,
    fallback_name=SOURCE_ID,
    namespace=BRONZE_NAMESPACE,
    table_name=BRONZE_TABLE_NAME,
)
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

LOOKBACK_DAYS = 7
RUN_ONE_RUN_ID = "run-incremental-001"
RUN_ONE_STARTED_AT = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
RUN_TWO_STARTED_AT = datetime(2026, 7, 8, 0, 0, tzinfo=UTC)

RUN_ONE_RECORDS = [
    {"event_id": "e1", "event_date": "2026-07-01T00:00:00Z", "amount": 10},
    {"event_id": "e2", "event_date": "2026-07-02T00:00:00Z", "amount": 10},
    {"event_id": "e3", "event_date": "2026-07-03T00:00:00Z", "amount": 10},
]
RUN_TWO_RECORDS = [
    {"event_id": "e2", "event_date": "2026-07-02T00:00:00Z", "amount": 11},
    {"event_id": "e3", "event_date": "2026-07-03T00:00:00Z", "amount": 11},
    {"event_id": "e4", "event_date": "2026-07-04T00:00:00Z", "amount": 12},
]
RUN_ONE_CHECKPOINT_VALUE = "2026-07-03T00:00:00Z"

EXPECTED_LOOKBACK_REQUEST_VALUE = "2026-06-26T00:00:00Z"
EXPECTED_DISTINCT_KEYS = 4
NAIVE_APPEND_TOTAL = 6


@dataclass(slots=True)
class FixtureTransport:
    """Serves one canned page per request and keeps every request for inspection."""

    payloads: list[list[dict[str, Any]]]
    requests: list = field(default_factory=list)
    opened: bool = False
    closed: bool = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request):
        self.requests.append(request)
        if not self.payloads:
            raise AssertionError("No fixture response remains for the incremental transport")

        return ApiResponse(
            request=request,
            status_code=200,
            body=json.dumps(self.payloads.pop(0)).encode("utf-8"),
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
            pyspark_sql.SparkSession.builder.appName("janus-incremental-upsert-integration")
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


@pytest.mark.parametrize(
    "run_two_run_id",
    ["run-incremental-002", RUN_ONE_RUN_ID],
    ids=["new_run_id", "same_run_id"],
)
def test_reingesting_the_lookback_window_does_not_duplicate_bronze_rows(
    tmp_path,
    session_factory,
    run_two_run_id: str,
):
    """Re-ingesting the overlapping window must leave one bronze row per unique key."""

    run_one, transport_one = _run_once(
        tmp_path,
        session_factory,
        run_id=RUN_ONE_RUN_ID,
        started_at=RUN_ONE_STARTED_AT,
        records=RUN_ONE_RECORDS,
    )
    assert run_one.status == "succeeded", run_one.failure_reason
    assert run_one.extraction_result is not None
    assert run_one.extraction_result.records_extracted == len(RUN_ONE_RECORDS)

    # Run 1 has no stored checkpoint to send back yet.
    assert "event_date" not in _query_params(transport_one.requests[0])

    # The overlap must come from the real checkpoint path, not from hand-fed parameters:
    # run 1 persists max(event_date) and run 2 reads it back through the metadata zone.
    assert run_one.checkpoint_state_path is not None
    checkpoint_state = json.loads(run_one.checkpoint_state_path.read_text(encoding="utf-8"))
    assert checkpoint_state["checkpoint_field"] == "event_date"
    assert checkpoint_state["checkpoint_value"] == RUN_ONE_CHECKPOINT_VALUE

    run_two, transport_two = _run_once(
        tmp_path,
        session_factory,
        run_id=run_two_run_id,
        started_at=RUN_TWO_STARTED_AT,
        records=RUN_TWO_RECORDS,
    )
    assert run_two.status == "succeeded", run_two.failure_reason
    assert run_two.extraction_result is not None
    assert run_two.extraction_result.records_extracted == len(RUN_TWO_RECORDS)

    # The outgoing request carries the lookback-adjusted checkpoint, i.e. the re-fetched
    # window is genuine and produced by extraction, not staged by the test.
    assert _query_params(transport_two.requests[0])["event_date"] == [
        EXPECTED_LOOKBACK_REQUEST_VALUE
    ]

    bronze_rows = _read_bronze(session_factory)
    key_counts = Counter(row["event_id"] for row in bronze_rows)

    # AC-4 — the table holds the distinct-key count, not the additive count. On the
    # unfixed tree this is where the suite goes red: 6 rows for 4 keys.
    assert len(bronze_rows) == EXPECTED_DISTINCT_KEYS, (
        f"re-ingestion left {len(bronze_rows)} bronze rows for {len(key_counts)} "
        f"distinct keys (naive append total: {NAIVE_APPEND_TOTAL})"
    )
    assert len(key_counts) == EXPECTED_DISTINCT_KEYS

    # AC-1 — no duplicate rows on the declared unique_fields.
    duplicated_keys = {key: count for key, count in key_counts.items() if count > 1}
    assert not duplicated_keys, f"duplicate bronze keys after re-ingestion: {duplicated_keys}"

    # D3 — a re-fetched row is updated, not frozen at its first-seen version, and its
    # run metadata reflects the run that last observed it.
    rows_by_key = {row["event_id"]: row for row in bronze_rows}
    assert rows_by_key["e2"]["amount"] == 11
    assert rows_by_key["e3"]["amount"] == 11
    assert rows_by_key["e2"]["janus_run_id"] == run_two_run_id
    # e1 was not re-fetched, so it must survive untouched from run 1.
    assert rows_by_key["e1"]["amount"] == 10
    assert rows_by_key["e1"]["janus_run_id"] == RUN_ONE_RUN_ID
    assert rows_by_key["e4"]["amount"] == 12

    # AC-2 — the quality gate is green on both runs.
    for executed in (run_one, run_two):
        assert executed.validation_report is not None
        assert executed.validation_report.report.is_successful is True
        unique_check = _check(executed, phase="data", name="unique_fields")
        assert unique_check.outcome == "passed"

        bronze_check = _check(executed, phase="output", name="bronze_key_uniqueness")
        assert bronze_check.outcome == "passed"
        assert bronze_check.details_as_dict()["scan_scope"] == "run_keys"


def test_a_duplicate_injected_into_bronze_fails_the_next_run(tmp_path, session_factory):
    """The sabotage path: the regression net exists to hang.

    Run once, then inject a duplicate straight into the committed table for ``e2`` — a key
    the *next* run re-fetches. Before this task the run went green over the doubled row,
    because the quality gate only ever saw the last normalized batch, never the table. Now
    the output-phase ``bronze_key_uniqueness`` check reads the table, scoped to the keys the
    run wrote, and must fail the run. This is the test that would have caught the §5.4 bug.
    """

    run_one, _ = _run_once(
        tmp_path,
        session_factory,
        run_id=RUN_ONE_RUN_ID,
        started_at=RUN_ONE_STARTED_AT,
        records=RUN_ONE_RECORDS,
    )
    assert run_one.status == "succeeded", run_one.failure_reason

    # e2 is re-fetched by run two, so it lands in that run's key set.
    _inject_bronze_duplicate(session_factory, event_id="e2")

    run_two, _ = _run_once(
        tmp_path,
        session_factory,
        run_id="run-incremental-002",
        started_at=RUN_TWO_STARTED_AT,
        records=RUN_TWO_RECORDS,
    )

    # The run fails, and it fails on exactly the output-phase oracle this task added.
    assert run_two.status == "failed"
    assert run_two.validation_report is not None
    assert run_two.validation_report.report.is_successful is False
    failed_names = [
        f"{check.phase}.{check.name}"
        for check in run_two.validation_report.report.failed_checks
    ]
    assert "output.bronze_key_uniqueness" in failed_names

    bronze_check = _check(run_two, phase="output", name="bronze_key_uniqueness")
    assert bronze_check.outcome == "failed"
    assert int(bronze_check.details_as_dict()["duplicate_groups"]) >= 1
    assert "e2" in bronze_check.details_as_dict()["sample_duplicates"]

    # The failure is recorded by the observer, not merely returned in memory.
    assert run_two.failure_reason is not None
    assert "bronze_key_uniqueness" in run_two.failure_reason
    assert run_two.run_metadata_path is not None
    assert run_two.run_metadata_path.exists()


def test_a_preexisting_duplicate_on_an_untouched_key_does_not_fail_the_run(
    tmp_path, session_factory
):
    """§2.2's deliberate limit, asserted as behaviour: the scan is scoped to run keys.

    A duplicate on ``e1`` — which run two does *not* re-fetch — is historical damage. The
    check flags only keys this run wrote, so the run stays green and the pre-existing
    duplicate is left for the migration runbook rather than failing every future run.
    """

    run_one, _ = _run_once(
        tmp_path,
        session_factory,
        run_id=RUN_ONE_RUN_ID,
        started_at=RUN_ONE_STARTED_AT,
        records=RUN_ONE_RECORDS,
    )
    assert run_one.status == "succeeded", run_one.failure_reason

    # e1 is only in run one; run two's window is e2, e3, e4.
    _inject_bronze_duplicate(session_factory, event_id="e1")

    run_two, _ = _run_once(
        tmp_path,
        session_factory,
        run_id="run-incremental-002",
        started_at=RUN_TWO_STARTED_AT,
        records=RUN_TWO_RECORDS,
    )

    assert run_two.status == "succeeded", run_two.failure_reason
    bronze_check = _check(run_two, phase="output", name="bronze_key_uniqueness")
    assert bronze_check.outcome == "passed"

    # The untouched duplicate is deliberately still there — scope, not accident.
    key_counts = Counter(row["event_id"] for row in _read_bronze(session_factory))
    assert key_counts["e1"] == 2


def test_replay_over_the_same_raw_zone_is_idempotent_on_bronze(tmp_path, session_factory):
    """Replay is the pairing: the fix lives in the shared materializer.

    A live ``--execute`` writes bronze, then ``ingest_raw_to_bronze`` re-materializes the
    *same* raw zone into the *same* bronze table. Because the write intent is resolved once
    in the materializer both entry points share, the replay MERGEs the already-committed
    rows back onto their keys and changes nothing — same key set, same row count, green
    quality. This is the replay analogue of the bronze-materializer equivalence suite.
    """

    executed, _ = _run_once(
        tmp_path,
        session_factory,
        run_id=RUN_ONE_RUN_ID,
        started_at=RUN_ONE_STARTED_AT,
        records=RUN_ONE_RECORDS,
    )
    assert executed.status == "succeeded", executed.failure_reason

    live_rows = _read_bronze(session_factory)

    live_business = _business_projection(live_rows)
    assert len(live_business) == len(RUN_ONE_RECORDS)

    replay_planned_run = _replay_planned_run(tmp_path)
    session = session_factory()
    try:
        replay = ingest_raw_to_bronze(
            replay_planned_run,
            session,
            ENVIRONMENT_CONFIG,
            # Same table name + namespace => the replay lands on the live bronze table.
            bronze_table=BRONZE_TABLE_NAME,
        )
    finally:
        session.stop()
    assert replay.status == "succeeded", replay.failure_reason

    replay_rows = _read_bronze(session_factory)
    replay_business = _business_projection(replay_rows)

    # AC-1 / AC-4 — re-materializing the same raw zone leaves one row per key, unchanged.
    assert len(replay_rows) == len(live_rows)
    assert replay_business == live_business

    # AC-2 — the quality gate stays green on the replay.
    assert replay.validation_report is not None
    assert replay.validation_report.report.is_successful is True


def _run_once(
    tmp_path: Path,
    session_factory,
    *,
    run_id: str,
    started_at: datetime,
    records: list[dict[str, Any]],
):
    """Execute one full run against a private provider, as production does per run."""

    transport = FixtureTransport(payloads=[list(records)])
    storage_layout = StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path)
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    run_context = RunContext.create(
        run_id=run_id,
        environment="local",
        project_root=tmp_path,
        started_at=started_at,
    )
    plan = strategy.plan(_source_config(tmp_path), run_context)
    planned_run = PlannedRun(plan=plan, strategy=strategy)
    provider = SparkSessionProvider({}, {}, session_factory=session_factory)

    executed = SourceExecutor().execute(planned_run, provider, ENVIRONMENT_CONFIG)
    return executed, transport


def _replay_planned_run(tmp_path: Path) -> PlannedRun:
    """Build the replay's planned run; its strategy rehydrates the handoff from raw."""

    storage_layout = StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path)
    strategy = ApiStrategy(
        transport_factory=lambda: FixtureTransport(payloads=[]),
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    run_context = RunContext.create(
        run_id="run-incremental-replay-001",
        environment="local",
        project_root=tmp_path,
        started_at=RUN_TWO_STARTED_AT,
    )
    plan = strategy.plan(_source_config(tmp_path), run_context)
    return PlannedRun(plan=plan, strategy=strategy)


def _business_projection(rows: list[dict[str, Any]]) -> dict[str, tuple[Any, Any]]:
    return {row["event_id"]: (row["event_date"], row["amount"]) for row in rows}


def _read_bronze(session_factory) -> list[dict[str, Any]]:
    """Read the committed Iceberg table through a session the runs no longer hold.

    Opening a fresh session over the same warehouse proves the rows are committed rather
    than merely staged inside the writing session.
    """

    session = session_factory()
    try:
        return [row.asDict(recursive=True) for row in session.table(BRONZE_TABLE).collect()]
    finally:
        session.stop()


def _inject_bronze_duplicate(session_factory, *, event_id: str) -> None:
    """Duplicate one committed bronze row directly, through a session the runs don't hold.

    This forges the state a blind ``INSERT INTO`` used to leave behind, so the next run's
    output-phase check has something to catch.
    """

    session = session_factory()
    try:
        session.sql(
            f"INSERT INTO {BRONZE_TABLE} "
            f"SELECT * FROM {BRONZE_TABLE} WHERE event_id = '{event_id}'"
        )
    finally:
        session.stop()


def _query_params(request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(request.full_url()).query)


def _check(executed, *, phase: str, name: str):
    assert executed.validation_report is not None
    matches = [
        check
        for check in executed.validation_report.report.checks
        if check.phase == phase and check.name == name
    ]
    assert len(matches) == 1, f"expected exactly one {phase}.{name} check, got {len(matches)}"
    return matches[0]


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
            "path": "/events",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 10,
            },
            "rate_limit": {
                "requests_per_minute": None,
                "concurrency": 1,
                "backoff_seconds": 5,
            },
        },
        "extraction": {
            "mode": "incremental",
            "checkpoint_field": "event_date",
            "checkpoint_strategy": "max_value",
            # Widens the re-fetched window so the overlap is explicit. The boundary
            # window is re-fetched with or without this setting; nothing
            # here is gated on it.
            "lookback_days": LOOKBACK_DAYS,
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
            # Today's incremental idiom, and the carrier of the bug under test.
            "write_mode": "append",
            "repartition": 1,
            # Deliberately unpartitioned. Real sources partition by `ingestion_date`,
            # which is derived from the run timestamp, so the two runs would land in
            # different partitions and a green result could be misread as
            # "partition-overwrite fixed it".
            "partition_by": [],
        },
        "outputs": {
            "raw": {"path": f"data/raw/example/{SOURCE_ID}", "format": "json"},
            "bronze": {
                "path": BRONZE_PATH,
                "format": "iceberg",
                "namespace": BRONZE_NAMESPACE,
                "table_name": BRONZE_TABLE_NAME,
            },
            "metadata": {"path": f"data/metadata/example/{SOURCE_ID}", "format": "json"},
        },
        "quality": {
            "required_fields": ["event_id", "event_date"],
            # The idempotency key. Must stay a subset of required_fields, otherwise
            # `validate_quality_contract` fails the config check and the data check
            # never runs.
            "unique_fields": ["event_id"],
            "allow_schema_evolution": True,
        },
    }
