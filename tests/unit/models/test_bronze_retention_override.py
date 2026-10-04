import pytest

from janus.models.config.issues import ValidationIssue
from janus.models.config.outputs import _build_output_target

pytestmark = pytest.mark.xfail(strict=True, reason="retention key ignored")


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
    assert target.retention.retain_last == 2
    assert target.retention.older_than_days == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("retain_last", 0),
        ("retain_last", -1),
        ("retain_last", True),
        ("older_than_days", -1),
        ("older_than_days", "3"),
    ],
)
def test_invalid_override_is_collected(key, value):
    issues: list[ValidationIssue] = []
    retention = {"retain_last": 2, "older_than_days": 0, key: value}
    _build_output_target(
        {"path": "bronze/table", "format": "iceberg", "retention": retention},
        "outputs.bronze",
        issues,
    )
    assert f"outputs.bronze.retention.{key}" in {issue.path for issue in issues}


@pytest.mark.parametrize(
    ("zone", "format_name"),
    [
        ("bronze", "parquet"),
        ("raw", "json"),
        ("metadata", "json"),
    ],
)
def test_override_rejected_outside_iceberg_bronze(zone, format_name):
    issues: list[ValidationIssue] = []
    _build_output_target(
        {
            "path": "data/target",
            "format": format_name,
            "retention": {"retain_last": 2, "older_than_days": 0},
        },
        f"outputs.{zone}",
        issues,
    )
    assert f"outputs.{zone}.retention" in {issue.path for issue in issues}


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
