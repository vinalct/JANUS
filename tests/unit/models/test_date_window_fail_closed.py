"""An invalid ``date_window`` must fail closed instead of planning a sentinel window.

Two layers are under test. Layer 1: ``_parse_request_input_entry`` returns ``None`` for an
entry it cannot build, so nothing carrying placeholder bounds ever exists. Layer 2:
``DateWindowRequestInputsConfig.__post_init__`` refuses the placeholders outright, so a
future refactor that tries to reinstate the old behaviour fails at construction.
"""

import inspect
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import janus.models.source_config as source_config_module
from janus.models.source_config import (
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    SourceConfig,
    SourceConfigValidationError,
    _parse_request_input_entry,
)

CONFIG_PATH = Path("conf/sources/example/date_window_source.yaml")

VALID_DATE_WINDOW = {
    "type": "date_window",
    "start": "2025-01-01",
    "end": "2025-03-31",
    "step": "month",
}
VALID_ICEBERG_ROWS = {
    "type": "iceberg_rows",
    "namespace": "bronze_transparencia",
    "table_name": "orgaos",
    "columns": {"orgao_codigo": "codigo"},
}


def test_missing_start_raises_with_the_field_path():
    with pytest.raises(SourceConfigValidationError) as exc_info:
        _load_request_inputs({"type": "date_window", "end": "2025-03-31", "step": "month"})

    assert "access.request_inputs.start: is required" in str(exc_info.value)


def test_unparseable_start_raises_and_still_collects_other_issues():
    with pytest.raises(SourceConfigValidationError) as exc_info:
        _load_request_inputs(
            {
                "type": "date_window",
                "start": "not-a-date",
                "end": "2025-03-31",
                "step": "year",
            }
        )

    message = str(exc_info.value)
    assert "access.request_inputs.start: must be a YYYY-MM-DD date" in message
    assert "access.request_inputs.step: must be one of: day, month" in message


def test_inverted_window_still_reports_end_before_start():
    with pytest.raises(SourceConfigValidationError) as exc_info:
        _load_request_inputs(
            {
                "type": "date_window",
                "start": "2025-03-31",
                "end": "2025-01-01",
                "step": "month",
            }
        )

    assert (
        "access.request_inputs.end: must be on or after access.request_inputs.start"
        in str(exc_info.value)
    )


def test_inverted_window_reports_both_its_issues():
    """The ordering guard: the range issue is recorded before the bail-out, not after.

    An inverted window with a missing step is the case that exercises both — the range
    check runs on two parseable dates, then the absent step forces the ``None`` return.
    """
    issues: list[Any] = []

    entry = _parse_request_input_entry(
        {"start": "2025-03-31", "end": "2025-01-01"},
        "date_window",
        "access.request_inputs",
        issues,
    )

    assert entry is None
    reported = {(issue.path, issue.message) for issue in issues}
    assert (
        "access.request_inputs.end",
        "must be on or after access.request_inputs.start",
    ) in reported
    assert ("access.request_inputs.step", "is required") in reported


def test_invalid_date_window_is_never_constructed(monkeypatch):
    constructed: list[DateWindowRequestInputsConfig] = []
    original_init = DateWindowRequestInputsConfig.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        constructed.append(self)

    monkeypatch.setattr(DateWindowRequestInputsConfig, "__init__", recording_init)

    issues: list[Any] = []
    entry = _parse_request_input_entry(
        {"start": "not-a-date", "end": "2025-03-31", "step": "month"},
        "date_window",
        "access.request_inputs",
        issues,
    )

    assert entry is None
    assert constructed == []
    assert issues


def test_unbuildable_iceberg_rows_entry_is_never_constructed():
    issues: list[Any] = []

    entry = _parse_request_input_entry(
        {"namespace": "bronze_transparencia", "columns": {"orgao_codigo": "codigo"}},
        "iceberg_rows",
        "access.request_inputs",
        issues,
    )

    assert entry is None
    assert any(issue.path == "access.request_inputs.table_name" for issue in issues)


def test_date_min_is_unrepresentable():
    with pytest.raises(ValueError, match="not a valid window bound"):
        DateWindowRequestInputsConfig(
            type="date_window",
            start=date.min,
            end=date(2026, 1, 1),
            step="day",
        )


def test_date_min_end_bound_is_unrepresentable():
    with pytest.raises(ValueError, match="not a valid window bound"):
        DateWindowRequestInputsConfig(
            type="date_window",
            start=date(2026, 1, 1),
            end=date.min,
            step="day",
        )


def test_empty_step_is_unrepresentable():
    with pytest.raises(ValueError, match="step must not be empty"):
        DateWindowRequestInputsConfig(
            type="date_window",
            start=date(2026, 1, 1),
            end=date(2026, 3, 31),
            step="",
        )


def test_valid_date_window_still_builds():
    request_inputs = _load_request_inputs(VALID_DATE_WINDOW)

    assert isinstance(request_inputs, DateWindowRequestInputsConfig)
    assert request_inputs.start == date(2025, 1, 1)
    assert request_inputs.end == date(2025, 3, 31)
    assert request_inputs.step == "month"


def test_valid_date_window_still_builds_from_yaml_native_dates():
    request_inputs = _load_request_inputs(
        {
            "type": "date_window",
            "start": date(2025, 1, 1),
            "end": date(2025, 3, 31),
            "step": "day",
        }
    )

    assert isinstance(request_inputs, DateWindowRequestInputsConfig)
    assert request_inputs.start == date(2025, 1, 1)
    assert request_inputs.end == date(2025, 3, 31)


def test_combined_skips_an_unbuildable_sub_input():
    combined = {
        "type": "combined",
        "inputs": [
            VALID_ICEBERG_ROWS,
            {"type": "date_window", "start": "not-a-date", "end": "2025-03-31", "step": "year"},
        ],
    }

    with pytest.raises(SourceConfigValidationError) as exc_info:
        _load_request_inputs(combined)

    message = str(exc_info.value)
    assert "access.request_inputs.inputs[1].start: must be a YYYY-MM-DD date" in message
    assert "access.request_inputs.inputs[1].step: must be one of: day, month" in message


def test_combined_never_holds_a_sentinel_window():
    """The failing build's intermediate config carries the sibling only, never a stub."""
    issues: list[Any] = []

    combined = source_config_module._build_combined_request_inputs_config(
        {
            "inputs": [
                VALID_ICEBERG_ROWS,
                {"type": "date_window", "start": "not-a-date", "end": "2025-03-31"},
            ]
        },
        issues,
    )

    assert isinstance(combined, CombinedRequestInputsConfig)
    assert issues
    assert [sub.type for sub in combined.inputs] == ["iceberg_rows"]


def _config_package_sources() -> dict[str, str]:
    """Source text of every module that participates in source-config parsing."""
    models_dir = Path(inspect.getfile(source_config_module)).parent
    paths = [models_dir / "source_config.py", *sorted((models_dir / "config").glob("*.py"))]
    return {
        str(path.relative_to(models_dir)): path.read_text(encoding="utf-8")
        for path in paths
        if path.exists()
    }


def test_no_date_min_sentinel_in_the_config_package():
    """AC-1 guardrail: the parse-failure sentinel must not reappear in the config modules."""
    sources = _config_package_sources()

    assert "source_config.py" in sources, (
        f"the sentinel sweep found {sorted(sources)} — it must always read source_config.py, "
        "or the assertion below is vacuous"
    )

    offenders = sorted(name for name, source in sources.items() if "date.min" in source)

    assert not offenders, (
        f"the date.min sentinel was reintroduced in janus.models.{'/'.join(offenders)}; "
        "an unbuildable request input must return None so the "
        "collected issues raise, never a config carrying placeholder bounds"
    )


def _load_request_inputs(request_inputs: dict[str, Any]):
    """Resolve request inputs through the real config load path, as sources do."""
    return SourceConfig.from_mapping(
        _source_mapping(request_inputs), CONFIG_PATH
    ).access.request_inputs


def _source_mapping(request_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": "date_window_source",
        "name": "date_window_source",
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "date_window_api",
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
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 100,
            },
            "rate_limit": {"requests_per_minute": 10, "concurrency": 1},
            "request_inputs": request_inputs,
        },
        "extraction": {
            "mode": "full_refresh",
            "retry": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "backoff_seconds": 1,
            },
        },
        "schema": {"mode": "infer"},
        "spark": {"input_format": "json", "write_mode": "append"},
        "outputs": {
            "raw": {"path": "data/raw/example/date_window_source", "format": "json"},
            "bronze": {
                "path": "data/bronze/example/date_window_source",
                "format": "iceberg",
            },
            "metadata": {
                "path": "data/metadata/example/date_window_source",
                "format": "json",
            },
        },
        "quality": {"allow_schema_evolution": True},
    }
