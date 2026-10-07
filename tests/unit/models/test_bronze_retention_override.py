from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
import yaml

from janus.models import BronzeRetentionConfig, OutputTarget, SourceConfig
from janus.models.config.issues import ValidationIssue
from janus.models.config.outputs import _build_output_target
from janus.models.source_config import (
    BronzeRetentionConfig as ReexportedBronzeRetentionConfig,
)
from janus.models.source_config import SourceConfigValidationError

SOURCE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures/full_refresh_history/conf/sources/unpartitioned.yaml"
)


def test_iceberg_bronze_override_is_parsed():
    issues: list[ValidationIssue] = []
    target = _build_output_target(
        {
            "path": "bronze/table",
            "format": "iceberg",
            "retention": {"retain_last": 2, "older_than_days": 0},
        },
        "outputs.bronze",
        issues,
    )
    assert issues == []
    assert target.retention is not None
    assert target.retention.retain_last == 2
    assert target.retention.older_than_days == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("retain_last", 0),
        ("retain_last", -1),
        ("retain_last", True),
        ("retain_last", False),
        ("retain_last", "2"),
        ("retain_last", 2.0),
        ("retain_last", None),
        ("retain_last", []),
        ("retain_last", {}),
        ("older_than_days", -1),
        ("older_than_days", "3"),
        ("older_than_days", True),
        ("older_than_days", False),
        ("older_than_days", 0.0),
        ("older_than_days", None),
        ("older_than_days", []),
        ("older_than_days", {}),
    ],
)
def test_invalid_override_is_collected(key, value):
    issues: list[ValidationIssue] = []
    retention = {"retain_last": 2, "older_than_days": 0, key: value}
    target = _build_output_target(
        {"path": "bronze/table", "format": "iceberg", "retention": retention},
        "outputs.bronze",
        issues,
    )
    message = (
        "must be a positive integer" if key == "retain_last" else "must be a non-negative integer"
    )
    assert [issue.render() for issue in issues] == [f"outputs.bronze.retention.{key}: {message}"]
    assert target.retention is None


@pytest.mark.parametrize(
    ("zone", "format_name"),
    [
        ("bronze", "parquet"),
        ("raw", "json"),
        ("raw", "iceberg"),
        ("metadata", "json"),
        ("metadata", "iceberg"),
    ],
)
def test_override_rejected_outside_iceberg_bronze(zone, format_name):
    issues: list[ValidationIssue] = []
    target = _build_output_target(
        {
            "path": "data/target",
            "format": format_name,
            "retention": {"retain_last": 2, "older_than_days": 0},
        },
        f"outputs.{zone}",
        issues,
    )
    message = (
        "requires format='iceberg'" if zone == "bronze" else "is only supported for outputs.bronze"
    )
    assert [issue.render() for issue in issues] == [f"outputs.{zone}.retention: {message}"]
    assert target.retention is None


def test_override_issues_accumulate_with_existing_issues():
    issues: list[ValidationIssue] = []
    _build_output_target(
        {
            "format": "iceberg",
            "table": "retired",
            "retention": {"retain_last": 0, "older_than_days": -1},
        },
        "outputs.bronze",
        issues,
    )
    assert {
        "outputs.bronze.path",
        "outputs.bronze.table",
        "outputs.bronze.retention.retain_last",
        "outputs.bronze.retention.older_than_days",
    } <= {issue.path for issue in issues}


@pytest.mark.parametrize("zone", ["bronze", "raw", "metadata"])
def test_absent_override_has_no_default(zone):
    issues: list[ValidationIssue] = []
    target = _build_output_target(
        {"path": "data/target", "format": "iceberg"}, f"outputs.{zone}", issues
    )
    assert issues == []
    assert target.retention is None


@pytest.mark.parametrize("retention", [None, "", "invalid", 2, True, [], [2, 0]])
def test_non_mapping_override_is_rejected(retention):
    issues: list[ValidationIssue] = []
    target = _build_output_target(
        {"path": "bronze/table", "format": "iceberg", "retention": retention},
        "outputs.bronze",
        issues,
    )
    assert [issue.render() for issue in issues] == ["outputs.bronze.retention: must be a mapping"]
    assert target.retention is None


@pytest.mark.parametrize("key", ["retain_last", "older_than_days"])
def test_both_override_values_are_required_together(key):
    issues: list[ValidationIssue] = []
    retention = {"retain_last": 2, "older_than_days": 0}
    del retention[key]
    target = _build_output_target(
        {"path": "bronze/table", "format": "iceberg", "retention": retention},
        "outputs.bronze",
        issues,
    )
    message = (
        "must be a positive integer" if key == "retain_last" else "must be a non-negative integer"
    )
    assert [issue.render() for issue in issues] == [f"outputs.bronze.retention.{key}: {message}"]
    assert target.retention is None


def test_empty_mapping_reports_both_required_values():
    issues: list[ValidationIssue] = []
    target = _build_output_target(
        {"path": "bronze/table", "format": "iceberg", "retention": {}},
        "outputs.bronze",
        issues,
    )
    assert [issue.render() for issue in issues] == [
        "outputs.bronze.retention.retain_last: must be a positive integer",
        "outputs.bronze.retention.older_than_days: must be a non-negative integer",
    ]
    assert target.retention is None


@pytest.mark.parametrize("key", ["typo", "compact", 7])
def test_unknown_override_keys_fail_closed(key):
    issues: list[ValidationIssue] = []
    target = _build_output_target(
        {
            "path": "bronze/table",
            "format": "iceberg",
            "retention": {"retain_last": 2, "older_than_days": 0, key: "unused"},
        },
        "outputs.bronze",
        issues,
    )
    assert [issue.render() for issue in issues] == [
        f"outputs.bronze.retention.{key}: is not supported; "
        "supported keys: older_than_days, retain_last"
    ]
    assert target.retention is None


@pytest.mark.parametrize("retain_last, older_than_days", [(1, 0), (5, 30), (500, 3650)])
def test_complete_override_survives_unrelated_issues(retain_last, older_than_days):
    unrelated_issue = ValidationIssue("schema.contract", "is required")
    issues = [unrelated_issue]
    target = _build_output_target(
        {
            "path": "bronze/table",
            "format": "iceberg",
            "retention": {"retain_last": retain_last, "older_than_days": older_than_days},
        },
        "outputs.bronze",
        issues,
    )
    assert issues == [unrelated_issue]
    assert target.retention == BronzeRetentionConfig(retain_last, older_than_days)


def test_from_mapping_collects_retention_and_two_unrelated_issues():
    data = yaml.safe_load(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    data["schema"]["path"] = "retired.json"
    data["spark"]["repartition"] = 0
    data["outputs"]["bronze"]["retention"] = {"retain_last": 0, "older_than_days": 0}

    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(data, SOURCE_FIXTURE)

    assert {issue.path for issue in raised.value.issues} == {
        "schema.path",
        "spark.repartition",
        "outputs.bronze.retention.retain_last",
    }
    assert len(raised.value.issues) == 3
    message = str(raised.value)
    assert "schema.path: is no longer supported:" in message
    assert "spark.repartition: must be >= 1" in message
    assert "outputs.bronze.retention.retain_last: must be a positive integer" in message


def test_from_mapping_preserves_a_valid_override():
    data = yaml.safe_load(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    data["outputs"]["bronze"]["retention"] = {"retain_last": 5, "older_than_days": 30}
    config = SourceConfig.from_mapping(data, SOURCE_FIXTURE)
    assert config.outputs.bronze.retention == BronzeRetentionConfig(5, 30)


@pytest.mark.parametrize(
    ("retain_last", "older_than_days", "message"),
    [(0, 0, "retain_last must be at least 1"), (1, -1, "older_than_days must not be negative")],
)
def test_direct_construction_enforces_retention_bounds(retain_last, older_than_days, message):
    with pytest.raises(ValueError, match=message):
        BronzeRetentionConfig(retain_last, older_than_days)


def test_retention_type_is_frozen_slotted_and_reexported():
    assert ReexportedBronzeRetentionConfig is BronzeRetentionConfig
    retention = BronzeRetentionConfig(2, 0)
    assert not hasattr(retention, "__dict__")
    with pytest.raises(FrozenInstanceError):
        retention.retain_last = 3


def test_output_target_keeps_existing_positional_field_order():
    target = OutputTarget("bronze/table", "iceberg", "bronze", "table", ("peer",))
    assert target.shared_with == ("peer",)
    assert target.retention is None
    assert [field.name for field in fields(OutputTarget)][-1] == "retention"
