"""The schema, identity, partitioning, and lifecycle contract for ``metadata.runs``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

import janus
import janus.observability.runs_table as runs_table
from janus.observability import (
    DEFAULT_RUNS_TABLE_IDENTIFIER,
    RUNS_TABLE,
    RUNS_TABLE_PARTITION_SPEC,
    RUNS_TABLE_SCHEMA,
    IcebergType,
    RunRecord,
    RunsTableContractError,
    RunsTablePartitionField,
    resolve_runs_table,
    resolve_runs_table_identifier,
)
from janus.observability.vocabulary import RUN_RECORD_SCHEMA_VERSION
from janus.utils.environment import load_environment_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REST_ENV_PATH = PROJECT_ROOT / "conf" / "environments" / "cluster-rest.env.example"
FORBIDDEN_IMPORT_ROOTS = ("pyspark", "pyiceberg", "pyarrow")
EXPECTED_COLUMN_COUNT = 45

EXPECTED_SCHEMA = (
    ("run_id", "string", False),
    ("source_id", "string", False),
    ("source_name", "string", False),
    ("environment", "string", False),
    ("strategy_family", "string", False),
    ("strategy_variant", "string", False),
    ("extraction_mode", "string", False),
    ("source_hook", "string", True),
    ("pipeline_run_id", "string", True),
    ("pipeline_attempt", "int", True),
    ("trigger", "string", True),
    ("status", "string", False),
    ("started_at", "timestamptz", False),
    ("ended_at", "timestamptz", True),
    ("emitted_at", "timestamptz", False),
    ("duration_seconds", "double", True),
    ("config_version", "string", False),
    ("source_config_path", "string", False),
    ("records_extracted", "long", True),
    ("artifact_count", "int", False),
    ("records_written", "long", True),
    ("bronze_table_identifier", "string", True),
    ("bronze_write_mode", "string", True),
    ("checkpoint_field", "string", True),
    ("checkpoint_strategy", "string", True),
    ("checkpoint_value", "string", True),
    ("checkpoint_decision", "string", True),
    ("checkpoint_advanced", "boolean", True),
    ("quality_outcome", "string", False),
    ("quality_checks_passed", "int", True),
    ("quality_checks_failed", "int", True),
    ("quality_checks_skipped", "int", True),
    ("quality_failed_checks", "list<string>", True),
    ("failure_reason", "string", True),
    ("failure_reason_truncated", "boolean", True),
    ("failure_reason_length", "int", True),
    ("error_type", "string", True),
    ("run_metadata_path", "string", True),
    ("lineage_path", "string", True),
    ("checkpoint_history_path", "string", True),
    ("validation_report_path", "string", True),
    ("record_schema_version", "int", False),
    ("schema_version", "string", True),
    ("contract_id", "string", True),
    ("contract_version", "string", True),
)


def _minimal_config(
    *,
    catalog_name: str = "janus",
    default_namespace: str = "bronze",
    observability: object | None = None,
) -> dict[str, object]:
    config: dict[str, object] = {
        "spark": {
            "iceberg": {
                "catalog_name": catalog_name,
                "default_namespace": default_namespace,
            }
        }
    }
    if observability is not None:
        config["observability"] = observability
    return config


def _env_file(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        entries[name] = value
    return entries


def _load_clean_profile(profile: str, overrides: dict[str, str] | None = None):
    clean_environment = {
        name: value for name, value in os.environ.items() if not name.startswith("JANUS_")
    }
    clean_environment.update(overrides or {})
    with mock.patch.dict(os.environ, clean_environment, clear=True):
        return load_environment_config(profile, PROJECT_ROOT)


# --------------------------------------------------------------------------------------
# Schema: the projection and table may never drift
# --------------------------------------------------------------------------------------


def test_schema_and_projection_cover_each_other_in_both_directions():
    record_fields = set(RunRecord.__dataclass_fields__)
    schema_fields = {column.name for column in RUNS_TABLE_SCHEMA}

    assert record_fields, "RunRecord unexpectedly has no fields; the drift check is vacuous"
    assert schema_fields, "the runs table unexpectedly has no columns; the drift check is vacuous"
    assert record_fields - schema_fields == set(), (
        f"RunRecord fields missing from the table schema: {sorted(record_fields - schema_fields)}"
    )
    assert schema_fields - record_fields == set(), (
        f"table columns missing from RunRecord: {sorted(schema_fields - record_fields)}"
    )


def test_schema_shape_types_and_nullability_are_pinned():
    actual = tuple(
        (column.name, column.iceberg_type, column.nullable) for column in RUNS_TABLE_SCHEMA
    )

    assert len(actual) == EXPECTED_COLUMN_COUNT
    assert actual == EXPECTED_SCHEMA


def test_field_ids_are_stable_unique_and_include_the_list_element():
    field_ids = [column.field_id for column in RUNS_TABLE_SCHEMA]
    failed_checks = next(
        column for column in RUNS_TABLE_SCHEMA if column.name == "quality_failed_checks"
    )
    all_ids = [*field_ids, failed_checks.element_id]

    assert field_ids == [*range(1, 43), 44, 45, 46]
    assert failed_checks.iceberg_type is IcebergType.STRING_LIST
    assert failed_checks.element_id == 43
    assert None not in all_ids
    assert len(all_ids) == len(set(all_ids))


def test_new_run_records_use_schema_version_two():
    assert RUN_RECORD_SCHEMA_VERSION == 2


def test_required_is_exactly_the_inverse_of_nullable():
    assert all(column.required is (not column.nullable) for column in RUNS_TABLE_SCHEMA)


# --------------------------------------------------------------------------------------
# Identifier: the catalog is shared, the metadata namespace is independent of bronze
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "environment"),
    (
        ("local", {}),
        ("cluster", {}),
        ("cluster", _env_file(REST_ENV_PATH)),
    ),
    ids=("local", "cluster", "cluster-rest"),
)
def test_shipped_profiles_resolve_the_default_runs_table(profile, environment):
    target = resolve_runs_table(_load_clean_profile(profile, environment))

    assert target.catalog_name == "janus"
    assert target.namespace == "metadata"
    assert target.table_name == "runs"
    assert target.identifier == DEFAULT_RUNS_TABLE_IDENTIFIER


@pytest.mark.parametrize("bronze_namespace", ("bronze", "metadata", "another_namespace"))
def test_runs_namespace_does_not_derive_from_bronze_default(bronze_namespace):
    target = resolve_runs_table(_minimal_config(default_namespace=bronze_namespace))

    assert target.namespace == "metadata"
    assert target.identifier == "metadata.runs"


def test_profile_override_is_honoured_and_sanitized():
    config = _minimal_config(
        catalog_name="shared_catalog",
        observability={"runs_table": " Shared Metadata . Run-History "},
    )

    target = resolve_runs_table(config)

    assert target.catalog_name == "shared_catalog"
    assert target.namespace == "shared_metadata"
    assert target.table_name == "run_history"
    assert resolve_runs_table_identifier(config) == "shared_metadata.run_history"


def test_override_uses_the_shared_identifier_sanitizer(monkeypatch):
    seen = []

    def recording_sanitizer(value):
        seen.append(value)
        return f"safe_{len(seen)}"

    monkeypatch.setattr(runs_table, "sanitize_identifier_segment", recording_sanitizer)

    target = resolve_runs_table(
        _minimal_config(observability={"runs_table": "Raw Namespace.Runs Table"})
    )

    assert seen == ["Raw Namespace", "Runs Table"]
    assert target.identifier == "safe_1.safe_2"


@pytest.mark.parametrize(
    "configured",
    (
        "",
        "   ",
        "runs",
        ".runs",
        "metadata.",
        "metadata.runs.extra",
        "!!!.runs",
        "metadata.!!!",
        None,
        42,
    ),
)
def test_invalid_profile_override_is_rejected(configured):
    config = _minimal_config(observability={"runs_table": configured})

    with pytest.raises(RunsTableContractError, match="observability.runs_table"):
        resolve_runs_table(config)


def test_non_mapping_observability_block_is_rejected():
    with pytest.raises(RunsTableContractError, match="observability must be a mapping"):
        resolve_runs_table(_minimal_config(observability=[]))


def test_missing_override_uses_the_documented_default():
    assert (
        resolve_runs_table_identifier(_minimal_config(observability={}))
        == DEFAULT_RUNS_TABLE_IDENTIFIER
    )


def test_catalog_name_comes_from_the_existing_derivation(monkeypatch):
    seen = []

    def recording_derivation(config):
        seen.append(config)
        return "derived_catalog"

    monkeypatch.setattr(runs_table, "derive_pyiceberg_catalog_name", recording_derivation)
    config = _minimal_config()

    assert resolve_runs_table(config).catalog_name == "derived_catalog"
    assert seen == [config]


# --------------------------------------------------------------------------------------
# Partitioning and lifecycle policy
# --------------------------------------------------------------------------------------


def test_partition_spec_is_only_day_of_emitted_at():
    assert (
        RunsTablePartitionField(
            source_column="emitted_at",
            transform="day",
            name="emitted_at_day",
        ),
    ) == RUNS_TABLE_PARTITION_SPEC
    assert RUNS_TABLE.partition_spec is RUNS_TABLE_PARTITION_SPEC


def test_table_is_append_only_with_no_retention():
    assert RUNS_TABLE.append_only is True
    assert RUNS_TABLE.retention_days is None
    assert RUNS_TABLE.schema is RUNS_TABLE_SCHEMA


# --------------------------------------------------------------------------------------
# Engine-free host-testable boundary
# --------------------------------------------------------------------------------------


def test_importing_the_declaration_pulls_in_no_engine():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(Path(janus.__file__).parents[1]), env.get("PYTHONPATH")) if part
    )
    program = (
        "import sys, janus.observability.runs_table\n"
        f"roots = {FORBIDDEN_IMPORT_ROOTS!r}\n"
        "print(sorted({m for m in sys.modules for r in roots "
        "if m == r or m.startswith(r + '.')}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"importing janus.observability.runs_table loaded {result.stdout.strip()}; "
        "the declaration must remain engine-free"
    )


def test_declaration_stays_well_under_the_module_size_ceiling():
    module_path = Path(runs_table.__file__)
    line_count = len(module_path.read_text(encoding="utf-8").splitlines())

    assert line_count <= 300, (
        f"{module_path.name} has {line_count} lines; the declaration should stay "
        "comfortably below the 600-line package ceiling"
    )
