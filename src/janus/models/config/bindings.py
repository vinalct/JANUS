"""Builders for ``access.parameter_bindings`` — runtime values bound into a request.

Depends on ``request_inputs`` because a binding is only valid against the fields the
declared request input actually exposes; that is the one direction this edge ever runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from janus.models.config.coercion import _optional_string, _require_mapping, _require_string
from janus.models.config.constants import (
    REQUEST_INPUT_BINDING_PREFIX,
    SUPPORTED_PARAMETER_BINDING_WINDOW_SOURCES,
)
from janus.models.config.issues import ValidationIssue
from janus.models.config.request_inputs import _request_input_field_names_for
from janus.models.config.types import (
    CombinedRequestInputsConfig,
    IcebergRowsRequestInputsConfig,
    ParameterBinding,
    RequestInputsConfig,
)


def _build_parameter_bindings_config(
    raw_value: Any,
    source_type: str,
    request_inputs: RequestInputsConfig,
    issues: list[ValidationIssue],
) -> dict[str, ParameterBinding] | None:
    """Validate declarative runtime request-parameter bindings for API sources."""
    if raw_value is None:
        return None

    if source_type not in ("api", "catalog"):
        issues.append(
            ValidationIssue(
                "access.parameter_bindings",
                "is only supported for api and catalog sources",
            )
        )
        return None

    data = _require_mapping(raw_value, "access.parameter_bindings", issues)
    bindings: dict[str, ParameterBinding] = {}

    for key, item in data.items():
        binding_path = f"access.parameter_bindings.{key}"
        if not isinstance(key, str):
            issues.append(ValidationIssue(binding_path, "keys must be strings"))
            continue

        parameter_name = key.strip()
        if not parameter_name:
            issues.append(ValidationIssue(binding_path, "must not be empty"))
            continue
        if not isinstance(item, Mapping):
            issues.append(ValidationIssue(binding_path, "must be a mapping"))
            continue

        from_source = _require_string(item, "from", issues, binding_path)
        output_format = _optional_string(item, "format", issues, binding_path)
        if from_source:
            _validate_parameter_binding_source(
                from_source,
                request_inputs,
                issues,
                f"{binding_path}.from",
            )

        bindings[parameter_name] = ParameterBinding(
            from_=from_source,
            format=output_format,
        )

    return bindings


def _validate_parameter_binding_source(
    from_source: str,
    request_inputs: RequestInputsConfig,
    issues: list[ValidationIssue],
    field_path: str,
) -> None:
    """Validate the limited phase-1 binding sources supported by the API contract."""
    if from_source == "checkpoint_value":
        return

    if not from_source.startswith(REQUEST_INPUT_BINDING_PREFIX):
        issues.append(
            ValidationIssue(
                field_path,
                (
                    "must be 'checkpoint_value', 'request_input.window_start', "
                    "'request_input.window_end', or 'request_input.<field>'"
                ),
            )
        )
        return

    request_input_field = from_source.removeprefix(REQUEST_INPUT_BINDING_PREFIX).strip()
    if not request_input_field:
        issues.append(
            ValidationIssue(
                field_path,
                "request_input bindings must reference a field name",
            )
        )
        return

    if request_inputs.type == "none":
        issues.append(
            ValidationIssue(
                field_path,
                "requires access.request_inputs to declare a non-'none' type",
            )
        )
        return

    if request_inputs.type == "date_window":
        if from_source not in SUPPORTED_PARAMETER_BINDING_WINDOW_SOURCES:
            issues.append(
                ValidationIssue(
                    field_path,
                    (
                        "must be 'request_input.window_start' or "
                        "'request_input.window_end' when "
                        "access.request_inputs.type is 'date_window'"
                    ),
                )
            )
        return

    if request_inputs.type == "iceberg_rows":
        if not isinstance(request_inputs, IcebergRowsRequestInputsConfig):
            return

        if request_input_field in {"window_start", "window_end"}:
            issues.append(
                ValidationIssue(
                    field_path,
                    "must reference one of access.request_inputs.columns when "
                    "access.request_inputs.type is 'iceberg_rows'",
                )
            )
            return

        if request_input_field not in request_inputs.columns:
            allowed_fields = ", ".join(sorted(request_inputs.columns))
            issues.append(
                ValidationIssue(
                    field_path,
                    f"must reference one of access.request_inputs.columns: {allowed_fields}",
                )
            )

    if request_inputs.type == "combined":
        if not isinstance(request_inputs, CombinedRequestInputsConfig):
            return

        all_fields = frozenset(
            field
            for sub in request_inputs.inputs
            for field in _request_input_field_names_for(sub)
        )
        if request_input_field not in all_fields:
            allowed_fields = ", ".join(sorted(all_fields))
            issues.append(
                ValidationIssue(
                    field_path,
                    f"must reference one of the combined input fields: {allowed_fields}",
                )
            )
