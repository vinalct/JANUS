"""FR-5: structural, collect-all validation of declared operational expectations."""

from pathlib import Path

import pytest

from janus.models.source_config import SourceConfig, SourceConfigValidationError
from tests.support.observability_baseline import source_payload

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


@pytest.mark.parametrize(
    "value",
    ["", 24, "24", "24m", "1w", "-3h", "h", "0h", "1.5d", "24 h", "24H", True, None],
)
def test_invalid_age_has_one_issue_and_no_placeholder(value):
    from janus.models.config.operations import _build_operations_config

    issues = []
    operations = _build_operations_config({"freshness": {"max_age": value}}, issues)
    assert [(issue.path, issue.message) for issue in issues] == [
        ("operations.freshness.max_age", AGE_MESSAGE)
    ]
    assert operations.freshness is None


@pytest.mark.parametrize(
    ("declaration", "path", "message"),
    [
        ("fast", "operations", "must be a mapping"),
        ([], "operations", "must be a mapping"),
        ({"freshness": "24h"}, "operations.freshness", "must be a mapping"),
        ({"freshness": None}, "operations.freshness", "must be a mapping"),
        ({"freshness": {}}, "operations.freshness.max_age", AGE_MESSAGE),
    ],
)
def test_malformed_blocks_collect_one_issue(declaration, path, message):
    from janus.models.config.operations import _build_operations_config

    issues = []
    operations = _build_operations_config(declaration, issues)
    assert [(issue.path, issue.message) for issue in issues] == [(path, message)]
    assert operations.freshness is None


@pytest.mark.parametrize(
    ("declaration", "budget", "age"),
    [
        ({"max_duration_seconds": 1}, 1, None),
        ({"max_duration_seconds": None}, None, None),
        ({"freshness": {"max_age": "1d"}}, None, 86400),
        ({"max_duration_seconds": 5, "freshness": {"max_age": "24h"}}, 5, 86400),
    ],
)
def test_declarations_are_optional_and_independent(declaration, budget, age):
    document = source_payload("api_success")
    document["operations"] = declaration
    operations = SourceConfig.from_mapping(document, CONFIG_PATH).operations
    assert operations.max_duration_seconds == budget
    assert operations.declares_duration_budget is (budget is not None)
    if age is None:
        assert operations.freshness is None
    else:
        assert operations.freshness.max_age_seconds == age


def test_unknown_operations_and_freshness_keys_are_ignored():
    document = source_payload("api_success")
    document["operations"] = {
        "future_setting": True,
        "freshness": {"max_age": "24h", "future_setting": True},
    }
    operations = SourceConfig.from_mapping(document, CONFIG_PATH).operations
    assert operations.max_duration_seconds is None
    assert operations.freshness.max_age_seconds == 86400


def test_freshness_parser_and_issue_message_share_the_canonical_units(monkeypatch):
    from janus.models.config.constants import FRESHNESS_UNIT_SECONDS
    from janus.models.config.operations import _build_operations_config

    monkeypatch.setitem(FRESHNESS_UNIT_SECONDS, "w", 604800)
    issues = []
    operations = _build_operations_config({"freshness": {"max_age": "2w"}}, issues)
    assert operations.freshness.max_age_seconds == 1209600
    assert issues == []
    _build_operations_config({"freshness": {"max_age": "2m"}}, issues)
    assert [issue.render() for issue in issues] == [
        "operations.freshness.max_age: must be a positive integer followed by "
        "one of: d, h, w (for example '24h')"
    ]


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


def test_operations_issues_follow_outputs_and_precede_quality():
    document = source_payload("api_success")
    document["outputs"]["metadata"]["format"] = "unsupported"
    document["operations"] = {"max_duration_seconds": 0, "freshness": {"max_age": "0h"}}
    document["quality"]["required_fields"] = "id"
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(document, CONFIG_PATH)
    assert [issue.path for issue in raised.value.issues] == [
        "outputs.metadata.format",
        "operations.max_duration_seconds",
        "operations.freshness.max_age",
        "quality.required_fields",
    ]


@pytest.mark.parametrize(
    ("budget", "duration", "expected"),
    [
        (5, 4.0, "within"),
        (5, 5.0, "within"),
        (5, 5.5, "exceeded"),
        (5, 6.0, "exceeded"),
        (None, 6.0, "undeclared"),
        (5, None, "undeclared"),
        (None, None, "undeclared"),
    ],
)
def test_duration_evaluation_includes_equality_and_missing_measurement(budget, duration, expected):
    from janus.models.source_config import OperationsConfig

    assert OperationsConfig(max_duration_seconds=budget).evaluate_duration(duration) == expected


@pytest.mark.parametrize("seconds", [0, -1])
def test_direct_freshness_construction_rejects_non_positive_windows(seconds):
    from janus.models.source_config import FreshnessConfig

    with pytest.raises(ValueError, match="max_age_seconds must be >= 1"):
        FreshnessConfig(max_age_seconds=seconds)


def test_public_source_config_exports_the_operational_types():
    from janus.models.source_config import FreshnessConfig, OperationsConfig

    assert OperationsConfig().max_duration_seconds is None
    assert OperationsConfig(freshness=FreshnessConfig(86400)).freshness.max_age_seconds == 86400
