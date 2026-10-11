"""Pure write-partitioning decisions and a DataFrame executor for those decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

    from janus.models.config.types import SparkWriteConfig


@dataclass(frozen=True, slots=True)
class WritePartitioningPlan:
    """What to do before writing a frame, with a reason suitable for run metadata."""

    action: str  # "none" | "coalesce" | "repartition"
    target_partitions: int | None
    reason: str

    @property
    def changes_partitioning(self) -> bool:
        return self.action != "none"


def plan_write_partitioning(
    *,
    current_partitions: int,
    estimated_input_bytes: int | None,
    declaration: SparkWriteConfig,
) -> WritePartitioningPlan:
    """Decide a bronze frame's partition count without Spark or I/O.

    An exact declaration takes precedence over size. With no estimate, retain the
    current count; an unknown size must never be treated as a small input.
    """
    if current_partitions <= 0:
        return WritePartitioningPlan("none", None, "frame reports no partitions")

    exact = declaration.partitions
    if exact is not None:
        if exact == current_partitions:
            return WritePartitioningPlan("none", None, "already at the declared partition count")
        return WritePartitioningPlan(
            "coalesce" if exact < current_partitions else "repartition",
            exact,
            "declared spark.write.partitions",
        )

    if estimated_input_bytes is None:
        return WritePartitioningPlan("none", None, "no size estimate available")

    target_bytes = declaration.target_file_size_bytes
    # Integer ceiling keeps one-byte boundaries exact even beyond float precision.
    ideal = max(1, (estimated_input_bytes + target_bytes - 1) // target_bytes)
    reason = (
        f"estimated {estimated_input_bytes} B at {declaration.target_file_size_mb} MiB per file"
    )
    if ideal == current_partitions:
        return WritePartitioningPlan("none", None, reason)
    return WritePartitioningPlan(
        "coalesce" if ideal < current_partitions else "repartition", ideal, reason
    )


def apply_write_partitioning(dataframe: DataFrame, plan: WritePartitioningPlan) -> DataFrame:
    """Execute the action; the planner supplies a target for every changing plan."""
    if plan.action == "coalesce":
        return dataframe.coalesce(cast(int, plan.target_partitions))
    if plan.action == "repartition":
        return dataframe.repartition(cast(int, plan.target_partitions))
    return dataframe
