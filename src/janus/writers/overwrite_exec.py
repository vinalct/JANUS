"""Execute a planned full refresh against an existing Iceberg table."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from janus.writers.evolution import EvolutionPlan
from janus.writers.overwrite import (
    FullRefreshOverwritePlan,
    build_insert_overwrite_sql,
    build_replace_table_as_select_sql,
    plan_full_refresh_overwrite,
)
from janus.writers.table_state import read_target_columns, read_target_partitions

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def execute_full_refresh_overwrite(
    spark: Any,
    table_identifier: str,
    source_view: str,
    frame: DataFrame,
    partition_columns: tuple[str, ...],
    evolution: EvolutionPlan | None,
) -> FullRefreshOverwritePlan:
    """Plan from the post-ALTER table state, then execute exactly one bronze statement."""
    target_columns = read_target_columns(spark, table_identifier)
    target_partitions = read_target_partitions(spark, table_identifier)
    overwrite_plan = plan_full_refresh_overwrite(
        source_columns=tuple(
            (field.name, field.dataType.simpleString()) for field in frame.schema.fields
        ),
        target_columns=target_columns,
        configured_partitions=partition_columns,
        target_partitions=target_partitions,
        evolution=evolution,
    )
    if overwrite_plan.mechanism == "replace_table":
        spark.sql(
            build_replace_table_as_select_sql(
                table_identifier=table_identifier,
                source_view=source_view,
                partition_columns=partition_columns,
            )
        )
    else:
        with static_partition_overwrite(spark):
            spark.sql(
                build_insert_overwrite_sql(
                    table_identifier=table_identifier,
                    source_view=source_view,
                    projection=overwrite_plan.projection,
                )
            )
    return overwrite_plan


@contextmanager
def static_partition_overwrite(spark: Any) -> Iterator[None]:
    """Force a full partition replacement and restore the caller's session setting."""
    key = "spark.sql.sources.partitionOverwriteMode"
    previous = spark.conf.get(key, None)
    spark.conf.set(key, "static")
    try:
        yield
    finally:
        if previous is None:
            spark.conf.unset(key)
        else:
            spark.conf.set(key, previous)
