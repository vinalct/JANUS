"""Builders for ``access.request_inputs`` — the bounded outer contexts a run iterates.

Parsing here **fails closed**: an entry that cannot be built yields no config
object at all rather than one carrying placeholder bounds. Every ``None`` return is
paired with at least one recorded issue, so ``from_mapping`` still raises with the full,
path-prefixed list.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from janus.models.config.coercion import (
    _optional_bool,
    _require_date,
    _require_enum,
    _require_mapping,
    _require_non_empty_string_mapping,
    _require_string,
)
from janus.models.config.constants import (
    _SUPPORTED_SUB_REQUEST_INPUT_TYPES,
    SUPPORTED_REQUEST_INPUT_STEPS,
    SUPPORTED_REQUEST_INPUT_TYPES,
)
from janus.models.config.issues import ValidationIssue
from janus.models.config.types import (
    CombinedRequestInputsConfig,
    DateWindowRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    RequestInputsConfig,
)

_MINIMUM_COMBINED_SUB_INPUTS = 2


def _build_request_inputs_config(
    raw_value: Any,
    source_type: str,
    issues: list[ValidationIssue],
) -> RequestInputsConfig:
    """Validate and normalize API request-input configuration with a safe default."""
    if raw_value is None:
        return RequestInputsConfig(type="none")

    if source_type not in ("api", "catalog"):
        issues.append(
            ValidationIssue(
                "access.request_inputs",
                "is only supported for api and catalog sources",
            )
        )
        return RequestInputsConfig(type="none")

    data = _require_mapping(raw_value, "access.request_inputs", issues)
    request_input_type = _require_enum(
        data,
        "type",
        SUPPORTED_REQUEST_INPUT_TYPES,
        issues,
        "access.request_inputs",
    )

    if request_input_type == "combined":
        return _build_combined_request_inputs_config(data, issues)

    entry = _parse_request_input_entry(
        data, request_input_type, "access.request_inputs", issues
    )
    return entry if entry is not None else RequestInputsConfig(type="none")


def _parse_request_input_entry(
    data: Mapping[str, Any],
    input_type: str,
    prefix: str,
    issues: list[ValidationIssue],
) -> RequestInputsConfig | None:
    """Parse one atomic request-input config, or ``None`` when it cannot be built.

    Returning ``None`` — rather than a config carrying placeholder values — keeps the
    invalid state unrepresentable. Every ``None`` return is paired with at least one
    recorded issue, so ``from_mapping`` still raises with the full, path-prefixed list.
    """
    if input_type == "date_window":
        start = _require_date(data, "start", issues, prefix)
        end = _require_date(data, "end", issues, prefix)
        step = _require_enum(data, "step", SUPPORTED_REQUEST_INPUT_STEPS, issues, prefix)

        if start is not None and end is not None and start > end:
            issues.append(
                ValidationIssue(
                    f"{prefix}.end",
                    f"must be on or after {prefix}.start",
                )
            )

        if start is None or end is None or not step:
            return None

        return DateWindowRequestInputsConfig(
            type=input_type,
            start=start,
            end=end,
            step=step,
        )

    if input_type == "iceberg_rows":
        namespace = _require_string(data, "namespace", issues, prefix)
        table_name = _require_string(data, "table_name", issues, prefix)
        columns_value = data.get("columns")
        columns = _require_non_empty_string_mapping(
            columns_value,
            f"{prefix}.columns",
            issues,
        )
        if isinstance(columns_value, Mapping) and not columns_value:
            issues.append(
                ValidationIssue(
                    f"{prefix}.columns",
                    "must not be empty",
                )
            )
        distinct = _optional_bool(data, "distinct", issues, prefix, default=False)

        if not namespace or not table_name:
            return None

        return IcebergRowsRequestInputsConfig(
            type=input_type,
            namespace=namespace,
            table_name=table_name,
            columns=columns,
            distinct=distinct,
        )

    return RequestInputsConfig(type="none")


def _build_combined_request_inputs_config(
    data: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> RequestInputsConfig:
    """Validate and build a combined request-input config from a list of sub-inputs."""
    inputs_raw = data.get("inputs")
    if not isinstance(inputs_raw, list):
        issues.append(
            ValidationIssue(
                "access.request_inputs.inputs",
                "is required and must be a list when type is 'combined'",
            )
        )
        return CombinedRequestInputsConfig(type="combined", inputs=())

    if len(inputs_raw) < _MINIMUM_COMBINED_SUB_INPUTS:
        issues.append(
            ValidationIssue(
                "access.request_inputs.inputs",
                "must contain at least 2 entries when type is 'combined'",
            )
        )
        return CombinedRequestInputsConfig(type="combined", inputs=())

    sub_configs: list[RequestInputsConfig] = []
    seen_fields: set[str] = set()

    for idx, sub_raw in enumerate(inputs_raw):
        sub_prefix = f"access.request_inputs.inputs[{idx}]"
        if not isinstance(sub_raw, Mapping):
            issues.append(ValidationIssue(sub_prefix, "must be a mapping"))
            continue

        sub_type = _require_enum(
            sub_raw,
            "type",
            _SUPPORTED_SUB_REQUEST_INPUT_TYPES,
            issues,
            sub_prefix,
        )
        if not sub_type:
            continue

        sub_config = _parse_request_input_entry(sub_raw, sub_type, sub_prefix, issues)
        if sub_config is None:
            continue

        sub_fields = _request_input_field_names_for(sub_config)
        conflicts = seen_fields.intersection(sub_fields)
        if conflicts:
            conflicting = ", ".join(sorted(conflicts))
            issues.append(
                ValidationIssue(
                    sub_prefix,
                    f"field name(s) {conflicting!r} conflict with another input "
                    "in this combined config",
                )
            )
        seen_fields.update(sub_fields)
        sub_configs.append(sub_config)

    return CombinedRequestInputsConfig(type="combined", inputs=tuple(sub_configs))


def _request_input_field_names_for(config: RequestInputsConfig) -> frozenset[str]:
    """Return the set of field names that a request-input config exposes at runtime."""
    if config.type == "date_window":
        return frozenset({"window_start", "window_end"})
    if config.type == "iceberg_rows" and isinstance(config, IcebergRowsRequestInputsConfig):
        return frozenset(config.columns.keys())
    return frozenset()
