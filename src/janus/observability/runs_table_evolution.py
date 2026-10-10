"""Engine-free field identity checks and additive plans for the runs-table sink."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from janus.observability.runs_table import RUNS_TABLE_SCHEMA, RunsTableColumn

SchemaFieldSignature = tuple[int, str, str, bool, int | None, bool | None]
_SCHEMA_FIELD_WITHOUT_NESTED_IDS_LENGTH = 4
_SCHEMA_FIELD_WITH_ELEMENT_ID_LENGTH = 5
_SCHEMA_FIELD_SIGNATURE_LENGTH = 6
SchemaFieldInput = (
    tuple[int, str, str, bool] | tuple[int, str, str, bool, int | None] | SchemaFieldSignature
)


@dataclass(frozen=True, slots=True)
class AdditivePlan:
    """Nullable declared columns that may be added to a live schema."""

    columns: tuple[SchemaFieldSignature, ...]


def _creation_columns(
    columns: Sequence[RunsTableColumn] = RUNS_TABLE_SCHEMA,
) -> tuple[RunsTableColumn, ...]:
    """The longest declared prefix a fresh ``create_table`` numbers exactly as declared."""
    for size in range(len(columns), 0, -1):
        prefix = tuple(columns[:size])
        element_ids = [column.element_id for column in prefix if column.element_id is not None]
        if all(column.field_id == index for index, column in enumerate(prefix, 1)) and (
            element_ids == list(range(size + 1, size + 1 + len(element_ids)))
        ):
            return prefix
    return tuple(columns)


def _schema_field_signatures(schema: Any) -> tuple[SchemaFieldSignature, ...]:
    """Return the field identity needed to decide whether a change is additive."""
    return tuple(
        (
            field.field_id,
            field.name,
            str(field.field_type),
            field.required,
            getattr(field.field_type, "element_id", None),
            getattr(field.field_type, "element_required", None),
        )
        for field in schema.fields
    )


def _plan_additive_evolution(
    live: Sequence[SchemaFieldInput],
    declared: Sequence[SchemaFieldInput],
) -> AdditivePlan | None:
    """Plan only newer nullable IDs from 4/5/6-item host tuples; reject conflicts."""
    live_fields = tuple(_normalize_schema_field(field) for field in live)
    declared_fields = tuple(_normalize_schema_field(field) for field in declared)
    live_by_id = {field[0]: field for field in live_fields}
    declared_by_id = {field[0]: field for field in declared_fields}
    live_by_name = {field[1]: field for field in live_fields}
    declared_by_name = {field[1]: field for field in declared_fields}
    if (
        len(live_by_id) != len(live)
        or len(declared_by_id) != len(declared)
        or len(live_by_name) != len(live)
        or len(declared_by_name) != len(declared)
    ):
        return None

    for field in live_fields:
        expected = declared_by_id.get(field[0])
        same_name = declared_by_name.get(field[1])
        if expected != field or same_name != field:
            return None

    missing = tuple(field for field in declared_fields if field[0] not in live_by_id)
    highest_live_id = max(
        (max(field[0], field[4] or 0) for field in live_fields),
        default=0,
    )
    if any(field[3] or field[0] <= highest_live_id for field in missing):
        return None
    return AdditivePlan(columns=missing)


def _normalize_schema_field(field: SchemaFieldInput) -> SchemaFieldSignature:
    values = cast(tuple[Any, ...], field)
    if len(values) not in {
        _SCHEMA_FIELD_WITHOUT_NESTED_IDS_LENGTH,
        _SCHEMA_FIELD_WITH_ELEMENT_ID_LENGTH,
        _SCHEMA_FIELD_SIGNATURE_LENGTH,
    }:
        raise ValueError("schema field signatures must contain four, five, or six items")
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4] if len(values) > _SCHEMA_FIELD_WITHOUT_NESTED_IDS_LENGTH else None,
        values[5] if len(values) > _SCHEMA_FIELD_WITH_ELEMENT_ID_LENGTH else None,
    )


def _conflicting_field_ids(
    live: Sequence[SchemaFieldSignature],
    declared: Sequence[SchemaFieldSignature],
) -> tuple[int, ...]:
    """Name conflicts without rendering live names, types or catalog error text."""
    live_by_id = {field[0]: field for field in live}
    declared_by_id = {field[0]: field for field in declared}
    id_counts = Counter(field[0] for field in live)
    name_counts = Counter(field[1] for field in live)
    highest_live_id = max((max(field[0], field[4] or 0) for field in live), default=0)
    conflicts = {
        field[0]
        for field in live
        if declared_by_id.get(field[0]) != field
        or id_counts[field[0]] > 1
        or name_counts[field[1]] > 1
    }
    missing = tuple(field for field in declared if field[0] not in live_by_id)
    conflicts.update(field[0] for field in missing if field[3] or field[0] <= highest_live_id)
    # After evolution even an otherwise valid missing suffix is a mismatch.
    return tuple(sorted(conflicts or {field[0] for field in missing}))
