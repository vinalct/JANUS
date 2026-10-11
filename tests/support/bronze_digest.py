"""Comparable bronze goldens, independent of Spark partitions and file layout."""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def bronze_schema_digest(dataframe: DataFrame) -> str:
    """Hash (name, simpleString type, nullable) triples in schema order."""
    fields = [
        (field.name, field.dataType.simpleString(), field.nullable)
        for field in dataframe.schema.fields
    ]
    return sha256(json.dumps(fields, ensure_ascii=False).encode("utf-8")).hexdigest()


def bronze_row_digest(
    dataframe: DataFrame, *, order_by: Sequence[str] | None = None
) -> str:
    """Hash newline-joined canonical JSON rows, including all provenance columns.

    Sort by the supplied keys, or every column in schema order. JSON breaks ties
    deterministically, including maps when explicit sortable keys are supplied.
    Stream the sorted result to the driver rather than collecting Row reprs.
    """
    from pyspark.sql import functions as sf
    from pyspark.sql.types import StringType

    columns = list(dataframe.columns if order_by is None else order_by)
    if not columns:
        raise ValueError("bronze_row_digest requires at least one ordering column")
    unknown = set(columns) - set(dataframe.columns)
    if unknown:
        raise ValueError(f"unknown ordering columns: {sorted(unknown)}")
    unsortable = [
        field.name
        for field in dataframe.schema.fields
        if field.name in columns and "map<" in field.dataType.simpleString()
    ]
    if unsortable:
        raise ValueError(f"map columns cannot be ordered; supply sortable order_by: {unsortable}")

    def render(row):
        return json.dumps(
            row.asDict(recursive=True), sort_keys=True, default=str, ensure_ascii=False
        )

    def column(name):
        return sf.col(f"`{name.replace('`', '``')}`")

    rendered_name = "_bronze_digest_json"
    while rendered_name in dataframe.columns:
        rendered_name += "_"
    rendered = dataframe.withColumn(
        rendered_name,
        sf.udf(render, StringType())(sf.struct(*(column(name) for name in dataframe.columns))),
    )
    ordered = rendered.orderBy(*(column(name) for name in columns), column(rendered_name))
    digest = sha256()
    separator = b""
    for row in ordered.select(column(rendered_name)).toLocalIterator():
        digest.update(separator)
        digest.update(row[0].encode("utf-8"))
        separator = b"\n"
    return digest.hexdigest()


def bronze_golden(
    dataframe: DataFrame, *, identifier: str, order_by: Sequence[str] | None = None
) -> dict[str, str]:
    """Return the identifier, schema digest, row digest and string row count."""
    return {
        "identifier": identifier,
        "schema_digest": bronze_schema_digest(dataframe),
        "row_digest": bronze_row_digest(dataframe, order_by=order_by),
        "row_count": str(dataframe.count()),
    }
