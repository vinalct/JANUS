"""FR-5: structural, collect-all validation of declared operational expectations."""

from pathlib import Path

import pytest

from janus.models.source_config import SourceConfig, SourceConfigValidationError
from tests.support.observability_baseline import source_payload

pytestmark = pytest.mark.xfail(
    strict=True, reason="operations contract builder and public types are absent"
)
CONFIG_PATH = Path("source.yaml")
AGE_MESSAGE = "must be a positive integer followed by one of: d, h (for example '24h')"


@pytest.mark.parametrize("declaration", [None, {}])
def test_absent_operations_is_undeclared_without_issues(declaration):
    from janus.models.config.operations import _build_operations_config

    issues = []
    operations = _build_operations_config(declaration, issues)
    config = SourceConfig.from_mapping(source_payload("api_success"), CONFIG_PATH)
    assert operations.max_duration_seconds is None
    assert operations.freshness is None
    assert config.operations == operations
    assert issues == []


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0, "must be >= 1"),
        (-1, "must be >= 1"),
        ("fast", "must be an integer"),
        (1.5, "must be an integer"),
        (True, "must be an integer"),
    ],
)
def test_duration_rejects_invalid_values_with_one_verbatim_issue(value, message):
    from janus.models.config.operations import _build_operations_config

    issues = []
    _build_operations_config({"max_duration_seconds": value}, issues)
    assert [(issue.path, issue.message) for issue in issues] == [
        ("operations.max_duration_seconds", message)
    ]


@pytest.mark.parametrize(("value", "seconds"), [("24h", 86400), ("7d", 604800), (" 24h ", 86400)])
def test_freshness_is_carried_as_integer_seconds(value, seconds):
    document = source_payload("api_success")
    document["operations"] = {"freshness": {"max_age": value}}
    operations = SourceConfig.from_mapping(document, CONFIG_PATH).operations
    assert operations.max_duration_seconds is None
    assert operations.freshness.max_age_seconds == seconds
    assert isinstance(operations.freshness.max_age_seconds, int)


@pytest.mark.parametrize("value", ["", 24, "24m", "1w", "-3h", "h", "0h", "1.5d", None])
def test_invalid_age_has_one_issue_and_no_placeholder(value):
    from janus.models.config.operations import _build_operations_config

    issues = []
    operations = _build_operations_config({"freshness": {"max_age": value}}, issues)
    assert [(issue.path, issue.message) for issue in issues] == [
        ("operations.freshness.max_age", AGE_MESSAGE)
    ]
    assert operations.freshness is None


@pytest.mark.parametrize(
    ("declaration", "path"),
    [
        ("fast", "operations"),
        ({"freshness": "24h"}, "operations.freshness"),
        ({"freshness": {}}, "operations.freshness.max_age"),
    ],
)
def test_malformed_blocks_collect_one_issue(declaration, path):
    from janus.models.config.operations import _build_operations_config

    issues = []
    _build_operations_config(declaration, issues)
    assert [issue.path for issue in issues] == [path]


def test_operations_and_access_issues_are_collected_in_block_order():
    document = source_payload("api_success")
    document["access"]["timeout_seconds"] = 0
    document["operations"] = {"max_duration_seconds": 0}
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(document, CONFIG_PATH)
    assert [(issue.path, issue.message) for issue in raised.value.issues] == [
        ("access.timeout_seconds", "must be >= 1"),
        ("operations.max_duration_seconds", "must be >= 1"),
    ]


@pytest.mark.parametrize(
    ("budget", "duration", "expected"),
    [(5, 5.0, "within"), (5, 6.0, "exceeded"), (None, 6.0, "undeclared"), (5, None, "undeclared")],
)
def test_duration_evaluation_includes_equality_and_missing_measurement(budget, duration, expected):
    from janus.models.source_config import OperationsConfig

    assert OperationsConfig(max_duration_seconds=budget).evaluate_duration(duration) == expected


def test_public_source_config_exports_the_operational_types():
    from janus.models.source_config import FreshnessConfig, OperationsConfig

    assert OperationsConfig().max_duration_seconds is None
    assert OperationsConfig(freshness=FreshnessConfig(86400)).freshness.max_age_seconds == 86400
