"""Schema profiling and ODCS document helpers for contract drafts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.models import ContractProperty, ExtractionResult, SourceConfig
from janus.schema_contracts import contract_properties_from_spark_schema

_DRAFT_REQUEST_ATTRIBUTE = "janus.contract_draft"


@dataclass(frozen=True, slots=True)
class _DraftInput:
    mode: str
    output_path: Path
    raw_run_id: str | None = None
    fixture_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _DataProfile:
    schema: Any
    row_count: int
    null_counts: tuple[int, ...]


def _profile_dataframe(dataframe: Any) -> _DataProfile:
    """Infer all null counts in one aggregate action, then count the rows once."""
    from pyspark.sql.functions import col, count, lit, when

    schema = dataframe.schema
    names = tuple(field.name for field in schema.fields)
    if names:
        null_expressions = tuple(
            count(
                when(col(f"`{name.replace('`', '``')}`").isNull(), lit(1))
            ).alias(f"__janus_null_{index}")
            for index, name in enumerate(names)
        )
        null_row = dataframe.select(*null_expressions).first()
        null_counts = tuple(int(null_row[index] or 0) for index in range(len(names)))
    else:
        null_counts = ()
    return _DataProfile(schema=schema, row_count=int(dataframe.count()), null_counts=null_counts)


def _draft_properties(
    profile: _DataProfile, source: SourceConfig
) -> tuple[ContractProperty, ...]:
    """Infer the columns; the source's quality keys, when declared, set required and primaryKey.

    The registry loader refuses quality keys that disagree with the contract, so a draft for a
    source that still declares them must reproduce them exactly. Without ``required_fields``,
    a column is required when the profiled rows hold no null in it.
    """
    inferred = contract_properties_from_spark_schema(profile.schema)
    required_fields = frozenset(source.quality.required_fields)
    unique_fields = frozenset(source.quality.unique_fields)
    return tuple(
        _decorate_property(
            prop,
            source_format=source.spark.input_format,
            required=(
                prop.name in required_fields
                if required_fields
                else profile.row_count > 0 and profile.null_counts[index] == 0
            ),
            unique=prop.name in unique_fields,
        )
        for index, prop in enumerate(inferred)
    )


def _decorate_property(
    prop: ContractProperty,
    *,
    source_format: str,
    required: bool | None = None,
    unique: bool = False,
) -> ContractProperty:
    nested = tuple(
        _decorate_property(child, source_format=source_format)
        for child in prop.properties
    )
    items = (
        _decorate_property(prop.items, source_format=source_format)
        if prop.items is not None
        else None
    )
    keys = (
        _decorate_property(prop.keys, source_format=source_format)
        if prop.keys is not None
        else None
    )
    values = (
        _decorate_property(prop.values, source_format=source_format)
        if prop.values is not None
        else None
    )
    return replace(
        prop,
        business_name="TODO",
        description="TODO",
        required=prop.required if required is None else required,
        unique=unique,
        primary_key=unique,
        classification="TODO",
        source_field=prop.name,
        source_format=source_format,
        properties=nested,
        items=items,
        keys=keys,
        values=values,
    )


def _contract_document(
    source: SourceConfig,
    table_name: str,
    properties: tuple[ContractProperty, ...],
    drafted_from: str,
) -> dict[str, Any]:
    purpose = source.description or (
        f"Draft contract for {source.name}, generated from inferred data."
    )
    return {
        "apiVersion": "v3.2.0",
        "kind": "DataContract",
        "id": f"{source.domain}.{table_name}",
        "name": source.name,
        "version": "0.1.0",
        "status": "draft",
        "domain": source.domain,
        "description": {"purpose": purpose},
        "team": [{"username": source.owner, "role": "owner"}],
        "tags": list(source.tags),
        "schema": [
            {
                "name": table_name,
                "physicalType": "table",
                "properties": [_property_document(prop) for prop in properties],
            }
        ],
        "customProperties": [
            {"property": "janus.compatibility", "value": "additive"},
            {"property": "janus.enforcement", "value": "lenient"},
            {"property": "janus.draftedFrom", "value": drafted_from},
        ],
    }


def _property_document(prop: ContractProperty) -> dict[str, Any]:
    document: dict[str, Any] = {
        "name": prop.name,
        "businessName": "TODO",
        "description": "TODO",
        "logicalType": prop.logical_type,
        "physicalType": prop.physical_type,
        "required": prop.required,
        "unique": prop.unique,
        "primaryKey": prop.primary_key,
        "classification": "TODO",
        "customProperties": [
            {"property": "sourceField", "value": prop.source_field or prop.name},
            {"property": "sourceFormat", "value": prop.source_format or "unknown"},
            {
                "property": "sourceNullable",
                "value": str(bool(prop.source_nullable)).lower(),
            },
        ],
    }
    if prop.physical_type == "struct":
        document["properties"] = [_property_document(child) for child in prop.properties]
    elif prop.physical_type == "array":
        if prop.items is None:
            raise ValueError(f"Array property {prop.name!r} has no item declaration")
        document["items"] = _property_document(prop.items)
    elif prop.physical_type == "map":
        if prop.keys is None or prop.values is None:
            raise ValueError(f"Map property {prop.name!r} has no key or value declaration")
        document["map"] = {
            "key": _property_document(prop.keys),
            "value": _property_document(prop.values),
        }
    return document


def _drafted_from(
    request: _DraftInput,
    extraction_result: ExtractionResult,
    row_count: int,
) -> str:
    if request.mode == "raw":
        assert request.raw_run_id is not None
        artifact_count = extraction_result.metadata_as_dict().get(
            "rediscovered_raw_artifact_count",
            str(len(extraction_result.artifacts)),
        )
        try:
            count = int(artifact_count)
        except ValueError:
            count = len(extraction_result.artifacts)
        date = datetime.now(tz=UTC).date().isoformat()
        return f"raw run {request.raw_run_id} on {date}, {row_count} rows, {count} artifacts"
    paths = ", ".join(path.as_posix() for path in request.fixture_paths)
    return f"fixture {paths}, {row_count} rows"


def _read_draft_input(attributes: Mapping[str, str]) -> _DraftInput:
    encoded = attributes.get(_DRAFT_REQUEST_ATTRIBUTE)
    if not encoded:
        raise ValueError("Draft input was not included in the planning request")
    try:
        values = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("Draft input in the planning request is malformed") from exc
    if not isinstance(values, Mapping):
        raise ValueError("Draft input in the planning request must be a mapping")

    mode = values.get("mode")
    output = values.get("output_path")
    if mode not in {"raw", "fixture"} or not isinstance(output, str) or not output:
        raise ValueError("Draft input must carry a mode and resolved output path")
    raw_run_id = values.get("raw_run_id")
    fixture_values = values.get("fixture_paths", [])
    if not isinstance(fixture_values, list) or not all(
        isinstance(value, str) and value for value in fixture_values
    ):
        raise ValueError("Draft fixture paths must be a list of non-empty paths")
    return _DraftInput(
        mode=mode,
        output_path=Path(output),
        raw_run_id=raw_run_id if isinstance(raw_run_id, str) else None,
        fixture_paths=tuple(Path(value) for value in fixture_values),
    )


