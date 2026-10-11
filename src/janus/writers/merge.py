"""SQL rendering and batch preparation for keyed bronze merges."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from janus.writers.identifiers import quote_identifier

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


_MERGE_SEQUENCE_COLUMN = "_janus_merge_seq"
_MERGE_RANK_COLUMN = "_janus_merge_rank"


def build_merge_sql(
    *,
    table_identifier: str,
    source_view: str,
    merge_keys: Sequence[str],
) -> str:
    """Render the idempotent ``MERGE INTO`` for a bronze upsert."""
    if not merge_keys:
        raise ValueError("merge_on_keys requires at least one merge key")

    quoted_table = quote_identifier(table_identifier)
    quoted_view = quote_identifier(source_view)
    conditions = "\n   AND ".join(
        f"janus_target.{quote_identifier(key)} <=> janus_source.{quote_identifier(key)}"
        for key in merge_keys
    )
    return (
        f"MERGE INTO {quoted_table} AS janus_target\n"
        f"USING {quoted_view} AS janus_source\n"
        f"ON {conditions}\n"
        "WHEN MATCHED THEN UPDATE SET *\n"
        "WHEN NOT MATCHED THEN INSERT *"
    )


def _dedupe_for_merge(
    frame: DataFrame,
    merge_keys: tuple[str, ...],
    *,
    count: bool,
) -> tuple[DataFrame, int | None, int | None]:
    """Keep exactly one row per key so MERGE never sees a many-to-one match."""
    from pyspark.sql.functions import col, monotonically_increasing_id, row_number
    from pyspark.sql.window import Window

    ordering = [
        col("ingestion_timestamp").desc_nulls_last(),
        col(_MERGE_SEQUENCE_COLUMN).asc(),
    ]
    window = Window.partitionBy(*[col(key) for key in merge_keys]).orderBy(*ordering)
    deduped = (
        frame.withColumn(_MERGE_SEQUENCE_COLUMN, monotonically_increasing_id())
        .withColumn(_MERGE_RANK_COLUMN, row_number().over(window))
        .where(col(_MERGE_RANK_COLUMN) == 1)
        .drop(_MERGE_RANK_COLUMN, _MERGE_SEQUENCE_COLUMN)
    )

    if not count:
        return deduped, None, None

    before = frame.count()
    after = deduped.count()
    return deduped, before - after, after


def _reject_complex_merge_keys(frame: DataFrame, merge_keys: tuple[str, ...]) -> None:
    """Reject nested/complex-typed key columns up front with a clear message."""
    from pyspark.sql.types import ArrayType, MapType, StructType

    field_types = {field.name: field.dataType for field in frame.schema.fields}
    offending = [
        key
        for key in merge_keys
        if isinstance(field_types.get(key), ArrayType | MapType | StructType)
    ]
    if offending:
        raise ValueError(
            "merge keys must be scalar columns; these have nested/complex types and cannot "
            f"be used as idempotency keys: {', '.join(offending)}"
        )
