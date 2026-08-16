"""Primitive readers that turn raw YAML scalars into typed values or recorded issues.

Every helper here follows the same contract: return a usable fallback and append a
``ValidationIssue`` rather than raising, so a malformed file still reports all of its
problems at once. ``_field_path`` composes the dotted prefixes those messages carry —
changing how it composes changes every error message a user sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, overload

from janus.models.config.issues import ValidationIssue


def _require_mapping(
    value: Any, field_path: str, issues: list[ValidationIssue]
) -> Mapping[str, Any]:
    """Return a mapping value or record a validation issue when the field is malformed."""
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return {}
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(field_path, "must be a mapping"))
        return {}
    return value


def _require_string(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> str:
    """Read a required non-empty string field and register a clear error otherwise."""
    value = data.get(field_name)
    field_path = _field_path(field_name, prefix)
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return ""
    if not isinstance(value, str):
        issues.append(ValidationIssue(field_path, "must be a string"))
        return ""
    value = value.strip()
    if not value:
        issues.append(ValidationIssue(field_path, "must not be empty"))
        return ""
    return value


def _optional_string(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> str | None:
    """Read an optional string field while reusing the required-string validation rules."""
    if field_name not in data or data[field_name] is None:
        return None
    return _require_string(data, field_name, issues, prefix)


def _require_bool(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> bool:
    """Read a required boolean field and register an issue when the type is wrong."""
    value = data.get(field_name)
    field_path = _field_path(field_name, prefix)
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return False
    if not isinstance(value, bool):
        issues.append(ValidationIssue(field_path, "must be a boolean"))
        return False
    return value


def _optional_bool(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
    default: bool = False,
) -> bool:
    """Read an optional boolean field and fall back to the provided default."""
    if field_name not in data or data[field_name] is None:
        return default
    return _require_bool(data, field_name, issues, prefix)


def _require_enum(
    data: Mapping[str, Any],
    field_name: str,
    allowed_values: frozenset[str],
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> str:
    """Read a required string field and ensure it belongs to the allowed value set."""
    value = _require_string(data, field_name, issues, prefix)
    if value and value not in allowed_values:
        issues.append(
            ValidationIssue(
                _field_path(field_name, prefix),
                f"must be one of: {', '.join(sorted(allowed_values))}",
            )
        )
    return value


def _optional_enum(
    data: Mapping[str, Any],
    field_name: str,
    allowed_values: frozenset[str],
    issues: list[ValidationIssue],
    prefix: str | None = None,
    default: str | None = None,
) -> str:
    """Read an optional enum field and return the configured default when absent."""
    if field_name not in data or data[field_name] is None:
        return default or ""
    return _require_enum(data, field_name, allowed_values, issues, prefix)


@overload
def _optional_int(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = ...,
    *,
    default: int,
    minimum: int | None = ...,
) -> int: ...


@overload
def _optional_int(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = ...,
    default: None = ...,
    minimum: int | None = ...,
) -> int | None: ...


def _optional_int(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
    default: int | None = None,
    minimum: int | None = None,
) -> int | None:
    """Read an optional integer field and enforce a minimum when one is provided.

    A non-``None`` ``default`` guarantees a non-``None`` result, which the overloads
    above express so callers assigning to a required ``int`` field type-check cleanly.
    """
    if field_name not in data or data[field_name] is None:
        return default

    value = data[field_name]
    field_path = _field_path(field_name, prefix)
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(ValidationIssue(field_path, "must be an integer"))
        return default
    if minimum is not None and value < minimum:
        issues.append(ValidationIssue(field_path, f"must be >= {minimum}"))
    return value


def _require_date(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> date | None:
    """Read a required ISO date field while accepting YAML-native date scalars."""
    value = data.get(field_name)
    field_path = _field_path(field_name, prefix)
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            issues.append(ValidationIssue(field_path, "must not be empty"))
            return None
        try:
            return date.fromisoformat(stripped_value)
        except ValueError:
            issues.append(ValidationIssue(field_path, "must be a YYYY-MM-DD date"))
            return None

    issues.append(ValidationIssue(field_path, "must be a YYYY-MM-DD date"))
    return None


def _optional_string_mapping(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> dict[str, str] | None:
    """Read an optional mapping whose keys and values must both be strings."""
    if field_name not in data or data[field_name] is None:
        return None

    value = data[field_name]
    field_path = _field_path(field_name, prefix)
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(field_path, "must be a mapping"))
        return None

    result: dict[str, str] = {}
    for key, item in value.items():
        child_path = f"{field_path}.{key}"
        if not isinstance(key, str):
            issues.append(ValidationIssue(child_path, "keys must be strings"))
            continue
        if not isinstance(item, str):
            issues.append(ValidationIssue(child_path, "values must be strings"))
            continue
        result[key] = item
    return result


def _require_non_empty_string_mapping(
    value: Any,
    field_path: str,
    issues: list[ValidationIssue],
) -> dict[str, str]:
    """Read a required mapping whose keys and values must be non-empty strings."""
    data = _require_mapping(value, field_path, issues)
    result: dict[str, str] = {}

    for key, item in data.items():
        child_path = f"{field_path}.{key}"
        if not isinstance(key, str):
            issues.append(ValidationIssue(child_path, "keys must be strings"))
            continue
        normalized_key = key.strip()
        if not normalized_key:
            issues.append(ValidationIssue(child_path, "keys must not be empty"))
            continue
        if not isinstance(item, str):
            issues.append(ValidationIssue(child_path, "values must be strings"))
            continue
        normalized_item = item.strip()
        if not normalized_item:
            issues.append(ValidationIssue(child_path, "values must not be empty"))
            continue
        result[normalized_key] = normalized_item

    return result


def _optional_string_list(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> list[str]:
    """Read an optional list of non-empty strings and report invalid entries inline."""
    if field_name not in data or data[field_name] is None:
        return []

    value = data[field_name]
    field_path = _field_path(field_name, prefix)
    if not isinstance(value, list):
        issues.append(ValidationIssue(field_path, "must be a list"))
        return []

    result: list[str] = []
    for index, item in enumerate(value):
        child_path = f"{field_path}[{index}]"
        if not isinstance(item, str):
            issues.append(ValidationIssue(child_path, "must be a string"))
            continue
        stripped_item = item.strip()
        if not stripped_item:
            issues.append(ValidationIssue(child_path, "must not be empty"))
            continue
        result.append(stripped_item)
    return result


def _optional_int_list(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> list[int] | None:
    """Read an optional list of integers; return None when the key is absent or null."""
    if field_name not in data or data[field_name] is None:
        return None

    value = data[field_name]
    field_path = _field_path(field_name, prefix)
    if not isinstance(value, list):
        issues.append(ValidationIssue(field_path, "must be a list"))
        return None

    result: list[int] = []
    for index, item in enumerate(value):
        child_path = f"{field_path}[{index}]"
        # bool is a subclass of int in Python; `true` in YAML is not a status code.
        if not isinstance(item, int) or isinstance(item, bool):
            issues.append(ValidationIssue(child_path, "must be an integer"))
            continue
        result.append(item)
    return result


def _field_path(field_name: str, prefix: str | None) -> str:
    """Compose the dotted path used in nested validation messages."""
    if prefix:
        return f"{prefix}.{field_name}"
    return field_name
