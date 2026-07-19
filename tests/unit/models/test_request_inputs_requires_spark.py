from datetime import date
from pathlib import Path
from typing import Any

from janus.models import (
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    RequestInputsConfig,
    SourceConfig,
)

CONFIG_PATH = Path("conf/sources/example/spark_need_source.yaml")

DATE_WINDOW_INPUT = {
    "type": "date_window",
    "start": "2025-01-01",
    "end": "2025-12-31",
    "step": "month",
}
ICEBERG_ROWS_INPUT = {
    "type": "iceberg_rows",
    "namespace": "bronze_transparencia",
    "table_name": "emendas_parlamentares__emendas",
    "columns": {"emenda_id": "id"},
    "distinct": True,
}


def test_absent_request_inputs_do_not_require_spark():
    request_inputs = _load_request_inputs(None)

    assert request_inputs.type == "none"
    assert request_inputs.requires_spark is False


def test_explicit_none_request_inputs_do_not_require_spark():
    request_inputs = _load_request_inputs({"type": "none"})

    assert request_inputs.requires_spark is False


def test_date_window_request_inputs_do_not_require_spark():
    request_inputs = _load_request_inputs(DATE_WINDOW_INPUT)

    assert request_inputs.type == "date_window"
    assert request_inputs.requires_spark is False


def test_iceberg_rows_request_inputs_require_spark():
    request_inputs = _load_request_inputs(ICEBERG_ROWS_INPUT)

    assert request_inputs.type == "iceberg_rows"
    assert request_inputs.requires_spark is True


def test_combined_request_inputs_require_spark_when_a_sub_input_does():
    request_inputs = _load_request_inputs(
        {"type": "combined", "inputs": [DATE_WINDOW_INPUT, ICEBERG_ROWS_INPUT]}
    )

    assert request_inputs.type == "combined"
    assert request_inputs.requires_spark is True


def test_combined_request_inputs_require_spark_regardless_of_sub_input_order():
    request_inputs = _load_request_inputs(
        {"type": "combined", "inputs": [ICEBERG_ROWS_INPUT, DATE_WINDOW_INPUT]}
    )

    assert request_inputs.requires_spark is True


def test_combined_request_inputs_of_spark_free_sub_inputs_do_not_require_spark():
    # Built by hand rather than through from_mapping: two date_window sub-inputs both
    # expose window_start/window_end, which the combined validator rejects as a
    # field-name conflict. The predicate still has to answer for the shape.
    request_inputs = CombinedRequestInputsConfig(
        type="combined",
        inputs=(
            _date_window_config(),
            _date_window_config(),
        ),
    )

    assert request_inputs.requires_spark is False


def test_combined_request_inputs_without_sub_inputs_do_not_require_spark():
    request_inputs = CombinedRequestInputsConfig(type="combined", inputs=())

    assert request_inputs.requires_spark is False


def _date_window_config() -> DateWindowRequestInputsConfig:
    return DateWindowRequestInputsConfig(
        type="date_window",
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        step="month",
    )


def _load_request_inputs(request_inputs: dict[str, Any] | None) -> RequestInputsConfig:
    """Resolve request inputs through the real config load path, as sources do."""
    return SourceConfig.from_mapping(
        _source_mapping(request_inputs), CONFIG_PATH
    ).access.request_inputs


def _source_mapping(request_inputs: dict[str, Any] | None) -> dict[str, Any]:
    access: dict[str, Any] = {
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
    }
    if request_inputs is not None:
        access["request_inputs"] = request_inputs

    return {
        "source_id": "spark_need_source",
        "name": "spark_need_source",
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": access,
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
            "raw": {"path": "data/raw/example/spark_need_source", "format": "json"},
            "bronze": {"path": "data/bronze/example/spark_need_source", "format": "iceberg"},
            "metadata": {"path": "data/metadata/example/spark_need_source", "format": "json"},
        },
        "quality": {"allow_schema_evolution": True},
    }
