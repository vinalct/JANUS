"""The one resolution of "what columns should this run have written"."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from janus.models import ExecutionPlan


@dataclass(frozen=True, slots=True)
class SchemaExpectation:
    """Resolved schema contract used by schema-related validations."""

    fields: tuple[str, ...] = ()
    source: str | None = None
    error: str | None = None


def resolve_schema_expectation(
    plan: ExecutionPlan,
    *,
    expected_fields: Sequence[str] | None = None,
) -> SchemaExpectation:
    """Name the columns this plan promised, and where that promise was written."""
    if expected_fields is not None:
        return SchemaExpectation(
            fields=_normalize_field_names(expected_fields),
            source="provided",
        )

    contract = plan.data_contract
    if contract is None:
        return SchemaExpectation()

    return SchemaExpectation(
        fields=contract.column_names,
        source=str(contract.contract_path),
    )


def _normalize_field_names(fields: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_field in fields:
        field = str(raw_field).strip()
        if field and field not in seen:
            normalized.append(field)
            seen.add(field)
    return tuple(normalized)


__all__ = [
    "SchemaExpectation",
    "resolve_schema_expectation",
]
