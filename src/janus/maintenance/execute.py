"""The execution seam."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from janus.maintenance.errors import MaintenanceExecutionUnavailable
from janus.maintenance.planning import RetentionPlan
from janus.maintenance.records import ItemOutcome
from janus.maintenance.settings import MaintenancePolicy

if TYPE_CHECKING:
    from janus.runtime.spark_lifecycle import SparkSessionProvider


def execute_retention(
    plan: RetentionPlan,
    *,
    policy: MaintenancePolicy,
    provider_factory: Callable[[], SparkSessionProvider],
) -> tuple[ItemOutcome, ...]:
    """Preserve skips and refuse actions until their zone executor is available.

    The empty plan succeeds without acquiring compute. Each unsupported action is a
    failed item, so an accidentally populated inventory cannot simulate a deletion.
    """
    outcomes = []
    for item in plan.items:
        if item.skipped_reason is not None:
            outcomes.append(ItemOutcome.from_planned_item(item))
            continue
        failure = MaintenanceExecutionUnavailable(
            f"No maintenance executor is available for {item.zone}: {item.action}"
        )
        outcomes.append(
            ItemOutcome(
                zone=item.zone,
                target=item.target,
                action=item.action,
                status="failed",
                detail=item.detail,
                removed_count=0,
                removed_bytes=0,
                failure_type=type(failure).__name__,
                failure_message=str(failure),
            )
        )
    return tuple(outcomes)
