from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import product as itertools_product
from typing import TYPE_CHECKING, Any

from janus.models import (
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    ParameterBinding,
    RequestInputsConfig,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

REQUEST_INPUT_BINDING_PREFIX = "request_input."
_SUPPORTED_STRFTIME_DIRECTIVES = frozenset(
    {
        "%",
        "A",
        "B",
        "C",
        "D",
        "F",
        "G",
        "H",
        "I",
        "M",
        "P",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "c",
        "d",
        "f",
        "g",
        "h",
        "j",
        "m",
        "p",
        "r",
        "u",
        "w",
        "x",
        "y",
        "z",
    }
)
_SUPPORTED_STRFTIME_MODIFIERS = frozenset({"#", "-", "0", "^", "_"})

_LAST_MONTH_OF_YEAR = 12


class ApiRequestInputError(RuntimeError):
    """Base error for request-input loading and parameter binding resolution."""


class ApiRequestInputLoadError(ApiRequestInputError):
    """Raised when a configured upstream request-input source cannot be loaded safely."""


class ApiParameterBindingError(ApiRequestInputError):
    """Raised when declarative request-parameter bindings cannot be resolved safely."""


def load_request_inputs(
    request_inputs: RequestInputsConfig,
    *,
    spark: SparkSession | None = None,
) -> tuple[dict[str, Any] | None, ...]:
    """Load or synthesize bounded request-input contexts for one API run."""
    if request_inputs.type == "none":
        return (None,)

    if request_inputs.type == "combined":
        if not isinstance(request_inputs, CombinedRequestInputsConfig):
            raise ApiRequestInputLoadError(
                "access.request_inputs: combined inputs must use CombinedRequestInputsConfig"
            )
        return _load_combined_request_inputs(request_inputs, spark=spark)

    return _load_single_request_input(request_inputs, spark=spark)


def _load_single_request_input(
    request_inputs: RequestInputsConfig,
    *,
    spark: SparkSession | None,
) -> tuple[dict[str, Any], ...]:
    """Load contexts for one atomic (non-combined) request-input type."""
    if request_inputs.type == "date_window":
        if not isinstance(request_inputs, DateWindowRequestInputsConfig):
            raise ApiRequestInputLoadError(
                "access.request_inputs: date_window inputs must use DateWindowRequestInputsConfig"
            )
        return generate_date_window_request_inputs(request_inputs)

    if request_inputs.type == "iceberg_rows":
        if not isinstance(request_inputs, IcebergRowsRequestInputsConfig):
            raise ApiRequestInputLoadError(
                "access.request_inputs: iceberg_rows inputs must use IcebergRowsRequestInputsConfig"
            )
        return load_iceberg_rows_request_inputs(request_inputs, spark=spark)

    raise ApiRequestInputLoadError(
        "access.request_inputs.type: unsupported runtime request-input "
        f"type {request_inputs.type!r}"
    )


def _load_combined_request_inputs(
    request_inputs: CombinedRequestInputsConfig,
    *,
    spark: SparkSession | None,
) -> tuple[dict[str, Any], ...]:
    """Compute the Cartesian product of all sub-input contexts."""
    all_contexts = [
        _load_single_request_input(sub, spark=spark) for sub in request_inputs.inputs
    ]
    combined: list[dict[str, Any]] = []
    for combo in itertools_product(*all_contexts):
        merged: dict[str, Any] = {}
        for ctx in combo:
            merged.update(ctx)
        combined.append(merged)
    return tuple(combined)


def generate_date_window_request_inputs(
    request_inputs: DateWindowRequestInputsConfig,
) -> tuple[dict[str, date], ...]:
    """Build deterministic day- or month-sized request windows."""
    if request_inputs.start > request_inputs.end:
        raise ApiRequestInputLoadError(
            "access.request_inputs.end: must be on or after access.request_inputs.start"
        )

    if request_inputs.step == "day":
        return _generate_daily_request_inputs(
            start=request_inputs.start,
            end=request_inputs.end,
        )
    if request_inputs.step == "month":
        return _generate_monthly_request_inputs(
            start=request_inputs.start,
            end=request_inputs.end,
        )

    raise ApiRequestInputLoadError("access.request_inputs.step: must be 'day' or 'month'")


def validate_iceberg_request_input_source(
    request_inputs: IcebergRowsRequestInputsConfig,
    *,
    available_columns: Sequence[str] | None,
) -> None:
    """Validate that a configured Iceberg request-input source exists and exposes columns."""
    table_identifier = _iceberg_table_identifier(request_inputs)
    if available_columns is None:
        raise ApiRequestInputLoadError(
            f"access.request_inputs: configured Iceberg table {table_identifier!r} does not exist"
        )

    normalized_columns = {
        normalized_column
        for column in available_columns
        if (normalized_column := str(column).strip())
    }
    for binding_name, source_column in sorted(request_inputs.columns.items()):
        # For nested struct references like "payload.id", check that the
        # root field exists in the top-level columns; Spark resolves the rest.
        root_field = source_column.split(".")[0]
        if root_field not in normalized_columns:
            message = (
                f"access.request_inputs.columns.{binding_name}: source column "
                f"{source_column!r} was not found in Iceberg table {table_identifier!r}"
            )
            raise ApiRequestInputLoadError(message)


def load_iceberg_rows_request_inputs(
    request_inputs: IcebergRowsRequestInputsConfig,
    *,
    spark: SparkSession | None,
) -> tuple[dict[str, Any], ...]:
    """Load projected request-input rows from one configured Iceberg table."""
    if spark is None:
        raise ApiRequestInputLoadError(
            "access.request_inputs: iceberg_rows inputs require an active SparkSession"
        )

    table_identifier = _iceberg_table_identifier(request_inputs)
    if not spark.catalog.tableExists(table_identifier):
        validate_iceberg_request_input_source(request_inputs, available_columns=None)

    dataframe = spark.table(table_identifier)
    validate_iceberg_request_input_source(
        request_inputs,
        available_columns=dataframe.columns,
    )
    projected_dataframe = _project_iceberg_request_input_columns(dataframe, request_inputs)
    if request_inputs.distinct:
        projected_dataframe = projected_dataframe.distinct()

    ordered_binding_names = tuple(request_inputs.columns)
    projected_rows = projected_dataframe.sort(*ordered_binding_names).collect()
    return tuple(row.asDict(recursive=False) for row in projected_rows)


def resolve_parameter_bindings(
    parameter_bindings: Mapping[str, ParameterBinding] | None,
    *,
    request_input: Mapping[str, Any] | None = None,
    checkpoint_value: str | None = None,
) -> dict[str, str]:
    """Resolve runtime request parameters from the current request-input context."""
    if not parameter_bindings:
        return {}

    resolved_bindings: dict[str, str] = {}
    for parameter_name, binding in parameter_bindings.items():
        binding_path = f"access.parameter_bindings.{parameter_name}"
        value = _resolve_binding_value(
            binding.from_,
            request_input=request_input,
            checkpoint_value=checkpoint_value,
            binding_path=binding_path,
        )
        resolved_bindings[parameter_name] = _render_bound_value(
            value,
            output_format=binding.format,
            binding_path=binding_path,
        )
    return resolved_bindings


def merge_request_params(
    static_params: Mapping[str, str] | None,
    bound_params: Mapping[str, str] | None,
) -> dict[str, str]:
    """Merge static and runtime-bound params while rejecting ambiguous overlaps."""
    static_mapping = dict(static_params or {})
    bound_mapping = dict(bound_params or {})
    duplicate_keys = sorted(set(static_mapping).intersection(bound_mapping))
    if duplicate_keys:
        duplicates = ", ".join(duplicate_keys)
        message = (
            "access.params and access.parameter_bindings must not define the same "
            f"request parameter(s): {duplicates}"
        )
        raise ApiParameterBindingError(message)

    merged = dict(static_mapping)
    merged.update(bound_mapping)
    return merged


def _resolve_binding_value(
    binding_source: str,
    *,
    request_input: Mapping[str, Any] | None,
    checkpoint_value: str | None,
    binding_path: str,
) -> Any:
    if binding_source == "checkpoint_value":
        if checkpoint_value is None or not checkpoint_value.strip():
            raise ApiParameterBindingError(
                f"{binding_path}.from: checkpoint_value is not available for this run"
            )
        return checkpoint_value

    field_name = binding_source.removeprefix(REQUEST_INPUT_BINDING_PREFIX)
    if request_input is None:
        raise ApiParameterBindingError(
            f"{binding_path}.from: request_input context is not available for this binding"
        )
    if field_name not in request_input:
        raise ApiParameterBindingError(
            f"{binding_path}.from: {binding_source} is not available in the current request input"
        )
    return request_input[field_name]


def _render_bound_value(
    value: Any,
    *,
    output_format: str | None,
    binding_path: str,
) -> str:
    if output_format is not None:
        _validate_strftime_format(output_format, field_path=f"{binding_path}.format")
        if not isinstance(value, date | datetime):
            message = (
                f"{binding_path}.format: requires a date or datetime value, got "
                f"{type(value).__name__}"
            )
            raise ApiParameterBindingError(message)
        rendered_value = value.strftime(output_format)
        if not rendered_value:
            raise ApiParameterBindingError(f"{binding_path}.format: resolved to an empty string")
        return rendered_value

    if value is None:
        raise ApiParameterBindingError(f"{binding_path}.from: resolved to null")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Decimal | float | int):
        return str(value)
    if isinstance(value, str):
        normalized_value = value.strip()
        if not normalized_value:
            raise ApiParameterBindingError(f"{binding_path}.from: resolved to an empty string")
        return normalized_value

    message = (
        f"{binding_path}.from: resolved to unsupported type "
        f"{type(value).__name__}; expected a scalar, date, or datetime"
    )
    raise ApiParameterBindingError(message)


def _validate_strftime_format(format_string: str, *, field_path: str) -> None:
    index = 0
    while index < len(format_string):
        if format_string[index] != "%":
            index += 1
            continue

        index += 1
        if index >= len(format_string):
            raise ApiParameterBindingError(
                f"{field_path}: contains an incomplete strftime directive"
            )

        while index < len(format_string) and format_string[index] in _SUPPORTED_STRFTIME_MODIFIERS:
            index += 1
        while index < len(format_string) and format_string[index].isdigit():
            index += 1

        if index >= len(format_string):
            raise ApiParameterBindingError(
                f"{field_path}: contains an incomplete strftime directive"
            )

        directive = format_string[index]
        if directive not in _SUPPORTED_STRFTIME_DIRECTIVES:
            raise ApiParameterBindingError(
                f"{field_path}: contains unsupported strftime directive '%{directive}'"
            )
        index += 1


def _iceberg_table_identifier(request_inputs: IcebergRowsRequestInputsConfig) -> str:
    return f"{request_inputs.namespace}.{request_inputs.table_name}"


def _project_iceberg_request_input_columns(
    dataframe: DataFrame,
    request_inputs: IcebergRowsRequestInputsConfig,
) -> DataFrame:
    projected_columns = tuple(
        dataframe[source_column].alias(binding_name)
        for binding_name, source_column in request_inputs.columns.items()
    )
    return dataframe.select(*projected_columns)


def _generate_daily_request_inputs(
    *,
    start: date,
    end: date,
) -> tuple[dict[str, date], ...]:
    request_windows: list[dict[str, date]] = []
    current_day = start
    while current_day <= end:
        request_windows.append(
            {
                "window_start": current_day,
                "window_end": current_day,
            }
        )
        current_day += timedelta(days=1)
    return tuple(request_windows)


def _generate_monthly_request_inputs(
    *,
    start: date,
    end: date,
) -> tuple[dict[str, date], ...]:
    request_windows: list[dict[str, date]] = []
    current_start = start
    while current_start <= end:
        current_end = min(_month_window_end(current_start), end)
        request_windows.append(
            {
                "window_start": current_start,
                "window_end": current_end,
            }
        )
        current_start = current_end + timedelta(days=1)
    return tuple(request_windows)


def _month_window_end(value: date) -> date:
    if value.month == _LAST_MONTH_OF_YEAR:
        next_month_start = date(value.year + 1, 1, 1)
    else:
        next_month_start = date(value.year, value.month + 1, 1)
    return next_month_start - timedelta(days=1)
