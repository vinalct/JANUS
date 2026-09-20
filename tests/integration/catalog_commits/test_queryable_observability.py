"""AC-1/AC-2/FR-3 evidence against the catalog shared by Spark and PyIceberg."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from janus.lineage import RunObserver, compute_config_version
from janus.observability import (
    IcebergAppendOutcome,
    RunRecord,
    append_run_record,
    build_run_event_emitter,
)
from janus.planner import HookCatalog, Planner, PlanningRequest, StrategyBinding, StrategyCatalog
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.scripts.raw_to_bronze import RawToBronzeLoader
from janus.strategies.api import ApiStrategy
from janus.strategies.catalog import CatalogStrategy
from janus.utils.storage import StorageLayout
from tests.support import observability_baseline as baseline
from tests.support.spark_sessions import (
    CatalogTarget,
    catalog_acceptance_prerequisites_available,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUERY_DIRECTORY = PROJECT_ROOT / "docs" / "queries" / "observability"
STARTED_AT = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 7, 4, 12, 0, 5, tzinfo=UTC)

REAL_CATALOG_SKIP_REASON = (
    "AC1_REAL_CATALOG_SUITE_UNAVAILABLE: catalog engines or seeded jars are missing"
)
AC2_QUERY_SKIP_REASON = (
    "AC2_SEEDED_QUERY_SUITE_UNAVAILABLE: catalog engines or seeded jars are missing"
)
CATALOG_ACCEPTANCE_AVAILABLE = catalog_acceptance_prerequisites_available()


class _FixedObserver(RunObserver):
    """The production observer with deterministic terminal timestamps."""

    def record_success(self, *args, **kwargs):
        return super().record_success(*args, **{**kwargs, "finished_at": FINISHED_AT})

    def record_failure(self, *args, **kwargs):
        return super().record_failure(*args, **{**kwargs, "finished_at": FINISHED_AT})


@dataclass(frozen=True, slots=True)
class _RunEvidence:
    case: str
    run_id: str
    source_id: str
    config_path: Path
    status: str


def _environment(
    catalog_target: CatalogTarget,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = catalog_target.environment_config()
    config["storage"] = {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
    config["observability"] = {
        "openlineage": {
            "transport": "file",
            "path": "lineage/openlineage",
        }
    }
    paths = {
        **catalog_target.resolved_paths,
        "raw_dir": root / "data" / "raw",
        "bronze_dir": root / "data" / "bronze",
        "metadata_dir": root / "data" / "metadata",
    }
    return config, paths


def _planned_case(root: Path, case: str, source_id: str, config: dict[str, Any]):
    document = baseline.source_payload(case)
    document["source_id"] = source_id
    document["name"] = f"TASK-10 {case}"
    document["access"]["path"] = f"/{source_id}"
    if document["strategy"] == "catalog":
        document["quality"]["required_fields"] = ["entity_id"]
        document["quality"]["unique_fields"] = ["entity_id"]
    for zone in ("raw", "bronze", "metadata"):
        document["outputs"][zone]["path"] = f"data/{zone}/{source_id}"
    baseline.write_project(root, document)

    transport = baseline.OfflineTransport(case, [])
    strategy_type = CatalogStrategy if document["strategy"] == "catalog" else ApiStrategy
    strategy = strategy_type(
        transport_factory=lambda: transport,
        sleeper=lambda _: None,
        storage_layout_factory=lambda plan: StorageLayout.from_environment_config(config, root),
        dead_letter_store=baseline.FixedDeadLetters(),
    )
    planner = Planner(
        strategy_catalog=StrategyCatalog(
            (StrategyBinding(document["strategy"], document["strategy_variant"], strategy),)
        ),
        hook_catalog=HookCatalog((("order15.empty", baseline.EmptyHandoffHook()),)),
    )
    run_id = f"task10-{case}-{uuid4().hex}"
    planned = planner.plan(
        PlanningRequest.create(
            source_id=source_id,
            environment="local",
            project_root=root,
            run_id=run_id,
            started_at=STARTED_AT,
            attributes={"trigger": "task10"},
        )
    )
    return planned, strategy


def _execute_case(
    case: str,
    root: Path,
    catalog_target: CatalogTarget,
    session,
) -> _RunEvidence:
    config, paths = _environment(catalog_target, root)
    source_id = f"task10_{case}_{uuid4().hex[:10]}"
    planned, strategy = _planned_case(root, case, source_id, config)
    emitter = build_run_event_emitter(config, paths)
    observer = _FixedObserver(emitter=emitter)

    if case == "replay":
        strategy.extract(planned.plan)
        result = RawToBronzeLoader(observer=observer).ingest(
            planned,
            SparkSessionProvider.wrapping(session),
            config,
            bronze_table=f"bronze.{source_id}_replay",
        )
    else:
        result = SourceExecutor(observer=observer).execute(
            planned,
            SparkSessionProvider.wrapping(session),
            config,
        )

    assert result.run_event_emission is not None
    assert result.run_event_emission.outcome.value == "emitted"
    return _RunEvidence(
        case=case,
        run_id=planned.plan.run_context.run_id,
        source_id=source_id,
        config_path=planned.plan.source_config.config_path,
        status=result.status,
    )


@pytest.mark.skipif(
    not CATALOG_ACCEPTANCE_AVAILABLE,
    reason=REAL_CATALOG_SKIP_REASON,
)
def test_real_terminal_runs_land_field_by_field_and_spark_reads_across_sources(
    catalog_target: CatalogTarget,
    shared_catalog_session,
    tmp_path,
):
    """Run every terminal shape, then read PyIceberg commits through bronze's Spark."""
    cases = (
        "api_success",
        "catalog_success",
        "extraction_failure",
        "quality_failure",
        "empty_handoff",
        "replay",
    )
    evidence = [
        _execute_case(case, tmp_path / case, catalog_target, shared_catalog_session)
        for case in cases
    ]
    run_ids = [item.run_id for item in evidence]
    runs_table = shared_catalog_session.table("janus.metadata.runs")
    rows = {
        row["run_id"]: row.asDict(recursive=True)
        for row in runs_table.where(runs_table.run_id.isin(*run_ids)).collect()
    }

    assert set(rows) == set(run_ids)
    assert {row["source_id"] for row in rows.values()} == {item.source_id for item in evidence}
    expected_statuses = {item.run_id: item.status for item in evidence}
    assert {run_id: row["status"] for run_id, row in rows.items()} == expected_statuses

    for item in evidence:
        row = rows[item.run_id]
        assert row["started_at"] == STARTED_AT.replace(tzinfo=None)
        assert row["ended_at"] == FINISHED_AT.replace(tzinfo=None)
        assert row["duration_seconds"] == 5.0
        assert row["config_version"] == compute_config_version(item.config_path)
        assert row["source_config_path"] == str(item.config_path)

    by_case = {item.case: rows[item.run_id] for item in evidence}
    assert {case: row["records_written"] for case, row in by_case.items()} == {
        "api_success": 2,
        "catalog_success": 2,
        "extraction_failure": None,
        "quality_failure": 2,
        "empty_handoff": None,
        "replay": 2,
    }
    assert {case: row["checkpoint_decision"] for case, row in by_case.items()} == {
        "api_success": "advanced",
        "catalog_success": "advanced",
        "extraction_failure": None,
        "quality_failure": None,
        "empty_handoff": "advanced",
        "replay": "skipped",
    }
    assert {case: row["quality_outcome"] for case, row in by_case.items()} == {
        "api_success": "passed",
        "catalog_success": "passed",
        "extraction_failure": "not_run",
        "quality_failure": "failed",
        "empty_handoff": "passed",
        "replay": "passed",
    }
    assert by_case["quality_failure"]["quality_checks_failed"] > 0
    assert by_case["quality_failure"]["quality_failed_checks"]
    assert by_case["extraction_failure"]["records_extracted"] is None
    assert by_case["api_success"]["records_extracted"] == 2


def _seed_record(
    *,
    run_id: str,
    source_id: str,
    status: str,
    started_at: datetime,
    emitted_at: datetime,
    quality_outcome: str = "not_run",
    failed_checks: tuple[str, ...] | None = None,
) -> RunRecord:
    validation_ran = quality_outcome != "not_run"
    return RunRecord(
        run_id=run_id,
        source_id=source_id,
        source_name=f"Source {source_id}",
        environment="local",
        strategy_family="api",
        strategy_variant="page_number_api",
        extraction_mode="full_refresh",
        status=status,
        started_at=started_at,
        ended_at=started_at,
        emitted_at=emitted_at,
        duration_seconds=0.0,
        config_version="a" * 64,
        source_config_path=f"conf/sources/{source_id}.yaml",
        records_extracted=1,
        artifact_count=1,
        records_written=1,
        bronze_table_identifier=f"bronze.{source_id}",
        bronze_write_mode="append",
        quality_outcome=quality_outcome,
        quality_checks_passed=1 if validation_ran else None,
        quality_checks_failed=(len(failed_checks or ())) if validation_ran else None,
        quality_checks_skipped=0 if validation_ran else None,
        quality_failed_checks=failed_checks if validation_ran else None,
        failure_reason="scripted failure" if status == "failed" else None,
        failure_reason_truncated=False if status == "failed" else None,
        failure_reason_length=len("scripted failure") if status == "failed" else None,
        error_type="RuntimeError" if status == "failed" else None,
        validation_report_path=(
            f"data/metadata/{source_id}/validations/{run_id}.json" if validation_ran else None
        ),
    )


@pytest.mark.skipif(
    not CATALOG_ACCEPTANCE_AVAILABLE,
    reason=AC2_QUERY_SKIP_REASON,
)
def test_published_ac2_queries_execute_verbatim_with_retry_and_window_boundaries(
    catalog_target: CatalogTarget,
    shared_catalog_session,
):
    """The checked-in SQL, not a test rewrite, answers both promised questions."""
    token = f"task10_{uuid4().hex}"

    def instant(day):
        return datetime(2026, 9, day, 12, tzinfo=UTC)

    lower = datetime(2026, 9, 1, 0, tzinfo=UTC)
    upper = datetime(2026, 10, 1, 0, tzinfo=UTC)
    retry_id = f"{token}_retry"
    quality_source = f"{token}_quality"
    records = (
        _seed_record(
            run_id=f"{token}_failed_inside",
            source_id=f"{token}_source_a",
            status="failed",
            started_at=instant(15),
            emitted_at=instant(15),
        ),
        _seed_record(
            run_id=f"{token}_failed_lower",
            source_id=f"{token}_source_b",
            status="failed",
            started_at=lower,
            emitted_at=lower,
        ),
        _seed_record(
            run_id=f"{token}_failed_upper",
            source_id=f"{token}_source_c",
            status="failed",
            started_at=upper,
            emitted_at=upper,
        ),
        _seed_record(
            run_id=f"{token}_failed_before",
            source_id=f"{token}_source_d",
            status="failed",
            started_at=datetime(2026, 8, 31, 23, 59, 59, tzinfo=UTC),
            emitted_at=lower,
        ),
        _seed_record(
            run_id=retry_id,
            source_id=f"{token}_retry_source",
            status="failed",
            started_at=instant(10),
            emitted_at=instant(10),
        ),
        _seed_record(
            run_id=retry_id,
            source_id=f"{token}_retry_source",
            status="succeeded",
            started_at=instant(10),
            emitted_at=instant(11),
        ),
        _seed_record(
            run_id=f"{token}_no_validation",
            source_id=f"{token}_no_validation_source",
            status="succeeded",
            started_at=instant(12),
            emitted_at=instant(12),
        ),
        _seed_record(
            run_id=f"{token}_quality_one",
            source_id=quality_source,
            status="failed",
            started_at=instant(5),
            emitted_at=instant(5),
            quality_outcome="failed",
            failed_checks=("data.required_fields",),
        ),
        _seed_record(
            run_id=f"{token}_quality_two",
            source_id=quality_source,
            status="failed",
            started_at=instant(6),
            emitted_at=instant(6),
            quality_outcome="failed",
            failed_checks=("output.unique_fields",),
        ),
        _seed_record(
            run_id=f"{token}_quality_upper",
            source_id=quality_source,
            status="failed",
            started_at=upper,
            emitted_at=upper,
            quality_outcome="failed",
            failed_checks=("data.outside_window",),
        ),
    )
    config = catalog_target.environment_config()
    for record in records:
        result = append_run_record(record, config, catalog_target.resolved_paths)
        assert result.outcome is IcebergAppendOutcome.EMITTED, result

    failed_sql = (QUERY_DIRECTORY / "failed-runs-in-window.sql").read_text(encoding="utf-8")
    failed_rows = shared_catalog_session.sql(failed_sql).collect()
    failed_ids = {row["run_id"] for row in failed_rows if row["run_id"].startswith(token)}
    assert failed_ids == {
        f"{token}_failed_inside",
        f"{token}_failed_lower",
        f"{token}_quality_one",
        f"{token}_quality_two",
    }
    assert retry_id not in failed_ids

    quality_sql = (QUERY_DIRECTORY / "quality-breaches-by-source.sql").read_text(encoding="utf-8")
    quality_rows = [
        row.asDict(recursive=True)
        for row in shared_catalog_session.sql(quality_sql).collect()
        if row["source_id"] == quality_source
    ]
    assert quality_rows == [
        {
            "source_id": quality_source,
            "breached_runs": 2,
            "checks_passed": 2,
            "checks_failed": 2,
            "checks_skipped": 0,
            "failed_checks": ["data.required_fields", "output.unique_fields"],
            "validation_report_paths": [
                f"data/metadata/{quality_source}/validations/{token}_quality_one.json",
                f"data/metadata/{quality_source}/validations/{token}_quality_two.json",
            ],
        }
    ]
