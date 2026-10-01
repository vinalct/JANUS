"""AC-5 on the real catalog: the preflight reads the table Spark wrote, session-free, in budget."""

from __future__ import annotations

import json
import multiprocessing
import time
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from janus.runtime import SparkSessionProvider
from janus.utils.logging import build_structured_logger
from janus.writers import quote_identifier
from tests.support.contract_enforcement import (
    ENVIRONMENT_CONFIG,
    BorrowedSession,
    EnforcementCase,
    FixtureTransport,
    execute_case_with_pages,
    plan_case,
    read_json,
    table_exists,
    write_case_project,
    write_frame,
)
from tests.support.spark_sessions import (
    CatalogTarget,
    catalog_acceptance_prerequisites_available,
    sqlite_catalog_target,
    start_session,
)

PREFLIGHT_REAL_CATALOG_UNAVAILABLE = (
    "PREFLIGHT_REAL_CATALOG_UNAVAILABLE: catalog engines or seeded jars are missing"
)
pytestmark = [
    pytest.mark.skipif(
        not catalog_acceptance_prerequisites_available(),
        reason=PREFLIGHT_REAL_CATALOG_UNAVAILABLE,
    ),
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
STARTED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
WHEN = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
BASE_DDL = "id string, label string, amount bigint, when timestamp"
INT_DDL = "id string, label string, amount int, when timestamp"
BASE_ROWS = [("r1", "one", 1, WHEN), ("r2", "two", 2, WHEN)]
PAGE = [{"id": "p1", "label": "one", "amount": 1, "when": "2026-09-01T10:00:00Z"}]
LOCK_BUDGET_SECONDS = 1.0
LOCK_READY_TIMEOUT_SECONDS = 15


# ── helpers ──────────────────────────────────────────────────────────────────


def _environment(catalog_target: CatalogTarget) -> dict[str, Any]:
    """The storage the harness writes to, plus the catalog both engines derive."""
    return {**ENVIRONMENT_CONFIG, **catalog_target.environment_config()}


def _case(source_id: str, contract: str, **options: Any) -> EnforcementCase:
    """One table per test: every catalog target gets its own root, so ids never collide."""
    return EnforcementCase.from_contract_file(source_id, HOSTILE / f"{contract}.yaml", **options)


def _plan(tmp_path: Path, case: EnforcementCase) -> Any:
    project = write_case_project(tmp_path / case.source_id, case)
    return plan_case(project, case, run_id=f"{case.source_id}-run", started_at=STARTED_AT).plan


def _forbidden_session() -> Any:
    raise AssertionError("the preflight must not start a Spark session")


def _session_free_provider(catalog_target: CatalogTarget) -> SparkSessionProvider:
    return SparkSessionProvider(
        _environment(catalog_target),
        catalog_target.resolved_paths,
        session_factory=_forbidden_session,
    )


def _create_table(
    spark: Any,
    tmp_path: Path,
    case: EnforcementCase,
    *,
    ddl: str = BASE_DDL,
    stamp: str | None = "1.0.0",
) -> None:
    write_frame(spark, _plan(tmp_path, case), BASE_ROWS, ddl)
    if stamp is not None:
        spark.sql(
            f"ALTER TABLE {quote_identifier(case.bronze_table)} "
            f"SET TBLPROPERTIES ('janus.contract_version' = '{stamp}')"
        )


def _preflight(
    catalog_target: CatalogTarget, tmp_path: Path, case: EnforcementCase, **options: Any
) -> Any:
    from janus.runtime.contract_preflight import run_contract_preflight

    provider = _session_free_provider(catalog_target)
    result = run_contract_preflight(
        _plan(tmp_path, case),
        _environment(catalog_target),
        provider.resolved_paths,
        identifier=case.bronze_table,
        logger=build_structured_logger("janus.tests.preflight.real", stream=StringIO()),
        **options,
    )
    assert provider.was_started is False
    return result


def _borrowing_provider(catalog_target: CatalogTarget, spark: Any) -> SparkSessionProvider:
    return SparkSessionProvider(
        _environment(catalog_target),
        catalog_target.resolved_paths,
        session_factory=lambda: BorrowedSession(spark),
    )


# ── the outcomes, on the catalog Spark wrote ─────────────────────────────────


def test_matching_table_is_ok(catalog_target, shared_catalog_session, tmp_path):
    case = _case("pf_ok", "base")
    _create_table(shared_catalog_session, tmp_path, case)

    assert _preflight(catalog_target, tmp_path, case).outcome == "ok"


def test_added_nullable_column_is_will_evolve(catalog_target, shared_catalog_session, tmp_path):
    live = _case("pf_evolve", "base")
    _create_table(shared_catalog_session, tmp_path, live)
    following = EnforcementCase.from_contract_file(
        live.source_id, HOSTILE / "base_plus_nullable.yaml"
    )

    result = _preflight(catalog_target, tmp_path, following)

    assert result.outcome == "will_evolve"
    assert "added:note" in result.reason


def test_undeclared_live_column_is_refused(catalog_target, shared_catalog_session, tmp_path):
    case = _case("pf_stray", "base")
    _create_table(shared_catalog_session, tmp_path, case)
    shared_catalog_session.sql(
        f"ALTER TABLE {quote_identifier(case.bronze_table)} ADD COLUMNS (stray string)"
    )

    result = _preflight(catalog_target, tmp_path, case)

    assert result.outcome == "refused"
    assert "stray" in result.reason


def test_missing_table_is_a_first_write(catalog_target, shared_catalog_session, tmp_path):
    case = _case("pf_missing", "base_lenient")

    assert _preflight(catalog_target, tmp_path, case).outcome == "table_missing"

    planned = plan_case(
        write_case_project(tmp_path / case.source_id, case),
        case,
        run_id=f"{case.source_id}-first",
        started_at=STARTED_AT,
    )
    run = execute_case_with_pages(
        planned,
        [PAGE],
        _forbidden_session,
        _environment(catalog_target),
        provider=_borrowing_provider(catalog_target, shared_catalog_session),
    )
    assert run.status == "succeeded", run.failure_reason
    assert read_json(run.run_metadata_path)["run_attributes"]["contract_preflight_outcome"] == (
        "table_missing"
    )
    assert table_exists(shared_catalog_session, case.bronze_table)


@pytest.mark.parametrize(
    ("write_mode", "outcome"), [("overwrite", "will_evolve"), ("append", "refused")]
)
def test_stamp_is_read_from_table_properties(
    catalog_target, shared_catalog_session, tmp_path, write_mode, outcome
):
    live = _case(f"pf_stamp_{write_mode}", "base")
    _create_table(shared_catalog_session, tmp_path, live, stamp="1.0.0")
    major_bump = EnforcementCase.from_contract_file(
        live.source_id, HOSTILE / "base_v2.yaml", write_mode=write_mode
    )

    result = _preflight(catalog_target, tmp_path, major_bump)

    assert result.outcome == outcome
    assert result.plan is not None
    assert (result.plan.recorded_major, result.plan.contract_major) == (1, 2)


# ── a strict refusal extracts nothing; a lenient one warns first ─────────────


def test_strict_refused_run_extracts_nothing(catalog_target, shared_catalog_session, tmp_path):
    case = _case("pf_strict", "base")  # additive: an int -> long promotion is refused
    _create_table(shared_catalog_session, tmp_path, case, ddl=INT_DDL)
    planned = plan_case(
        write_case_project(tmp_path / case.source_id, case),
        case,
        run_id=f"{case.source_id}-refused",
        started_at=STARTED_AT,
    )
    transport = FixtureTransport()
    provider = _session_free_provider(catalog_target)

    run = execute_case_with_pages(
        planned,
        [PAGE],
        _forbidden_session,
        _environment(catalog_target),
        transport=transport,
        provider=provider,
    )

    assert run.is_successful is False
    assert run.failure_stage == "contract_preflight"
    assert run.error_type == "ContractPreflightError"
    assert transport.requests == []
    assert provider.was_started is False
    assert run.extraction_result is None or run.extraction_result.artifacts == ()
    raw_root = tmp_path / case.source_id / "data" / "raw"
    assert not raw_root.exists() or not any(path.is_file() for path in raw_root.rglob("*"))
    metadata = read_json(run.run_metadata_path)
    assert metadata["run_attributes"]["contract_preflight_outcome"] == "refused"


def test_lenient_refused_run_warns_and_proceeds(
    catalog_target, shared_catalog_session, tmp_path
):
    case = _case("pf_lenient", "base_lenient")
    _create_table(shared_catalog_session, tmp_path, case, ddl=INT_DDL)
    planned = plan_case(
        write_case_project(tmp_path / case.source_id, case),
        case,
        run_id=f"{case.source_id}-lenient",
        started_at=STARTED_AT,
    )
    transport = FixtureTransport()
    stream = StringIO()

    run = execute_case_with_pages(
        planned,
        [PAGE],
        _forbidden_session,
        _environment(catalog_target),
        transport=transport,
        provider=_borrowing_provider(catalog_target, shared_catalog_session),
        logger=build_structured_logger("janus.tests.preflight.lenient", stream=stream),
    )

    events = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    warnings = [event["event"] for event in events if event["level"] == "WARNING"]
    assert warnings.count("contract_preflight_warning") == 1
    assert transport.requests, "a lenient refusal must not stop extraction"
    # The warning told the operator first; the write then refuses the same promotion.
    assert run.status == "failed"
    assert run.error_type == "SchemaEvolutionRefusedError"


# ── the budget bounds a locked catalog ───────────────────────────────────────


def _hold_exclusive_lock(database: str, seconds: float, ready: Any) -> None:
    import sqlite3

    connection = sqlite3.connect(database, timeout=5, isolation_level=None)
    try:
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        deadline = time.monotonic() + LOCK_READY_TIMEOUT_SECONDS - 1
        while True:
            try:
                connection.execute("BEGIN EXCLUSIVE")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        ready.set()
        time.sleep(seconds)
        connection.execute("ROLLBACK")
    finally:
        connection.close()


def _seed_locked_catalog(target: CatalogTarget, project_root: Path, case: EnforcementCase) -> None:
    spark = start_session("janus-preflight-locked-catalog", target.session_options())
    try:
        _create_table(spark, project_root, case)
    finally:
        spark.stop()


def test_budget_bounds_a_locked_catalog(tmp_path):
    target = sqlite_catalog_target(tmp_path / "locked_catalog")
    target.prepare()
    case = _case("pf_locked", "base")
    context = multiprocessing.get_context("spawn")
    seed = context.Process(target=_seed_locked_catalog, args=(target, tmp_path, case))
    seed.start()
    seed.join(timeout=60)
    if seed.is_alive():
        seed.terminate()
        seed.join(timeout=10)
    assert seed.exitcode == 0, "the Spark writer did not seed the locked catalog"

    ready = context.Event()
    holder = context.Process(
        target=_hold_exclusive_lock,
        args=(str(target.catalog_db), LOCK_BUDGET_SECONDS + 2, ready),
    )
    holder.start()
    try:
        assert ready.wait(LOCK_READY_TIMEOUT_SECONDS), "the lock holder never took the lock"
        started = time.monotonic()
        result = _preflight(target, tmp_path, case, budget_seconds=LOCK_BUDGET_SECONDS)
        elapsed = time.monotonic() - started
    finally:
        holder.join(timeout=LOCK_READY_TIMEOUT_SECONDS)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=10)

    assert holder.exitcode == 0
    assert result.outcome == "catalog_unavailable"
    assert elapsed <= LOCK_BUDGET_SECONDS + 0.5


def test_a_fresh_catalog_preflight_does_not_create_catalog_tables(tmp_path):
    """The read must leave bootstrap to the writer, including on repeated probes."""
    import sqlite3

    from tests.support.spark_sessions import sqlite_catalog_target

    target = sqlite_catalog_target(tmp_path / "fresh-catalog")
    target.prepare()
    case = _case("pf_fresh_read_only", "base")
    for _ in range(2):
        result = _preflight(target, tmp_path, case)
        assert result.outcome == "table_missing", result.reason
        with sqlite3.connect(target.catalog_db) as connection:
            assert connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall() == []
