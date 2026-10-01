"""Published observability SQL and its operator documentation stay one interface."""

from __future__ import annotations

import re
from pathlib import Path

from janus.observability.runs_table import RUNS_TABLE_SCHEMA
from janus.quality import ContractViolationError
from janus.quality.malformed_rows import MalformedRowsError
from janus.runtime.contract_preflight import ContractPreflightError
from janus.writers.errors import SchemaEvolutionRefusedError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPERATOR_GUIDE = PROJECT_ROOT / "docs" / "queryable-observability.md"
QUERY_DIRECTORY = PROJECT_ROOT / "docs" / "queries" / "observability"

PUBLISHED_QUERIES = frozenset(
    {
        "checkpoint-decisions-by-source.sql",
        "config-version-drift.sql",
        "failed-runs-in-window.sql",
        "pipeline-failures.sql",
        "quality-breaches-by-source.sql",
        "runs-by-source-over-time.sql",
        "schema-drift-by-source.sql",
    }
)
AC2_QUERIES = frozenset({"failed-runs-in-window.sql", "quality-breaches-by-source.sql"})
WINDOW_START = "TIMESTAMP '2026-09-01 00:00:00'"
WINDOW_END = "TIMESTAMP '2026-10-01 00:00:00'"


def _queries() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(QUERY_DIRECTORY.glob("*.sql"))
    }


def test_every_published_query_is_a_standalone_spark_statement_with_latest_run_logic():
    queries = _queries()

    assert set(queries) == PUBLISHED_QUERIES
    for name, sql in queries.items():
        assert sql.strip()
        assert sql.count(";") == 1, f"{name} must remain one executable statement"
        assert sql.rstrip().endswith(";")
        assert "FROM janus.metadata.runs" in sql
        assert "ROW_NUMBER() OVER (" in sql
        assert "PARTITION BY run_id" in sql
        assert "ORDER BY emitted_at DESC" in sql


def test_ac2_queries_pin_the_documented_half_open_window_and_partition_pruning():
    queries = _queries()

    for name in AC2_QUERIES:
        sql = queries[name]
        assert f"emitted_at >= {WINDOW_START}" in sql
        assert f"started_at >= {WINDOW_START}" in sql
        assert f"started_at < {WINDOW_END}" in sql
        assert "row_rank = 1" in sql

    failed_runs = queries["failed-runs-in-window.sql"]
    assert "status = 'failed'" in failed_runs
    assert {"failure_reason", "error_type", "run_metadata_path"} <= set(
        re.findall(r"\b[a-z][a-z0-9_]*\b", failed_runs)
    )

    quality = queries["quality-breaches-by-source.sql"]
    assert "quality_outcome = 'failed'" in quality
    assert "GROUP BY source_id" in quality
    assert "FLATTEN(COLLECT_LIST(quality_failed_checks))" in quality


def test_the_query_set_gives_every_declared_column_an_operational_use():
    query_text = "\n".join(_queries().values())
    used_identifiers = set(re.findall(r"\b[a-z][a-z0-9_]*\b", query_text))
    schema_columns = {column.name for column in RUNS_TABLE_SCHEMA}

    assert schema_columns
    assert schema_columns <= used_identifiers


def test_config_version_drift_query_includes_contract_identity_changes():
    sql = _queries()["config-version-drift.sql"]

    assert "LAG(schema_version)" in sql
    assert "previous_schema_version" in sql
    assert "contract_id" in sql
    assert "contract_version" in sql
    assert "NOT (previous_schema_version <=> schema_version)" in sql


def test_operator_column_register_matches_the_schema_in_both_directions():
    guide = OPERATOR_GUIDE.read_text(encoding="utf-8")
    column_register = guide.split("## Column register", 1)[1].split("## Published Spark SQL", 1)[0]
    documented_columns = set(
        re.findall(r"^\| `([a-z][a-z0-9_]*)` \|", column_register, flags=re.MULTILINE)
    )
    schema_columns = {column.name for column in RUNS_TABLE_SCHEMA}

    assert documented_columns
    assert documented_columns == schema_columns


def test_every_local_link_in_the_operator_guide_resolves():
    guides = (
        OPERATOR_GUIDE,
        PROJECT_ROOT / "docs" / "openlineage.md",
        PROJECT_ROOT / "docs" / "data-contracts.md",
    )
    for guide_path in guides:
        targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", guide_path.read_text(encoding="utf-8"))
        assert targets, guide_path
        for target in targets:
            path_text = target.split("#", 1)[0]
            if not path_text or "://" in path_text:
                continue
            assert (guide_path.parent / path_text).resolve().exists(), (guide_path, target)


# --------------------------------------------------------------------------------------
# the schema-drift query and the three enforcement columns (FR-8)
# --------------------------------------------------------------------------------------

SCHEMA_DRIFT_QUERY = "schema-drift-by-source.sql"
ENFORCEMENT_COLUMNS = ("contract_preflight_outcome", "schema_evolution", "malformed_rows")
ENFORCEMENT_ERRORS = (
    "ContractViolationError",
    "MalformedRowsError",
    "ContractPreflightError",
    "SchemaEvolutionRefusedError",
)


def test_the_drift_query_is_published_on_the_latest_row_of_each_run():
    sql = (QUERY_DIRECTORY / SCHEMA_DRIFT_QUERY).read_text(encoding="utf-8")

    assert sql.count(";") == 1 and sql.rstrip().endswith(";")
    assert "FROM janus.metadata.runs" in sql
    assert "ROW_NUMBER() OVER (" in sql
    assert "PARTITION BY run_id" in sql
    assert "ORDER BY emitted_at DESC" in sql


def test_the_drift_query_uses_every_enforcement_signal():
    sql = (QUERY_DIRECTORY / SCHEMA_DRIFT_QUERY).read_text(encoding="utf-8")
    identifiers = set(re.findall(r"\b[a-z][a-z0-9_]*\b", sql))

    assert set(ENFORCEMENT_COLUMNS) <= identifiers
    assert "LAG(schema_version)" in sql
    for outcome in ("will_evolve", "refused", "catalog_unavailable"):
        assert f"'{outcome}'" in sql
    for error_type in ENFORCEMENT_ERRORS:
        assert f"'{error_type}'" in sql


def test_the_error_types_the_drift_query_names_are_the_enforcement_exceptions():
    """``error_type`` is the exception's class name; a rename would silently blind the query."""
    raised = (
        ContractViolationError,
        MalformedRowsError,
        ContractPreflightError,
        SchemaEvolutionRefusedError,
    )

    assert tuple(error.__name__ for error in raised) == ENFORCEMENT_ERRORS


def test_the_operator_register_documents_the_three_columns():
    guide = OPERATOR_GUIDE.read_text(encoding="utf-8")
    register = guide.split("## Column register", 1)[1].split("## Published Spark SQL", 1)[0]

    for column in ENFORCEMENT_COLUMNS:
        assert f"| `{column}` |" in register
    assert SCHEMA_DRIFT_QUERY in guide
