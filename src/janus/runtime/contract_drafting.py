"""Compute-only dataframe profiling for contract authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DataProfile:
    schema: Any
    row_count: int
    null_counts: tuple[int, ...]


def profile_dataframe(dataframe: Any) -> DataProfile:
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
    return DataProfile(schema=schema, row_count=int(dataframe.count()), null_counts=null_counts)
