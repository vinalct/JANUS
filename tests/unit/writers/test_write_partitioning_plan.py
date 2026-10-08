import pytest

MIB = 1024 * 1024


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize(
    ("current", "size", "exact", "action", "target"),
    [
        (200, None, 1, "coalesce", 1),
        (1, None, 4, "repartition", 4),
        (4, None, 4, "none", None),
        (200, 50 * MIB, None, "coalesce", 1),
        (1, 2048 * MIB, None, "repartition", 16),
        (8, 1024 * MIB, None, "none", None),
        (4, None, None, "none", None),
        (0, 2048 * MIB, 4, "none", None),
        (0, None, None, "none", None),
        (1, 128 * MIB - 1, None, "none", None),
        (1, 128 * MIB, None, "none", None),
        (1, 128 * MIB + 1, None, "repartition", 2),
        (2, 128 * MIB - 1, None, "coalesce", 1),
        (2, 128 * MIB, None, "coalesce", 1),
        (2, 128 * MIB + 1, None, "none", None),
        (200, 0, None, "coalesce", 1),
        (200, 2048 * MIB, 1, "coalesce", 1),
        (1, 50 * MIB, 4, "repartition", 4),
    ],
)
def test_partitioning_decision(current, size, exact, action, target):
    from janus.models.config.types import SparkWriteConfig
    from janus.writers.partitioning import plan_write_partitioning

    plan = plan_write_partitioning(
        current_partitions=current,
        estimated_input_bytes=size,
        declaration=SparkWriteConfig(target_file_size_mb=128, partitions=exact),
    )
    assert plan.action == action
    assert plan.target_partitions == target
    assert plan.changes_partitioning == (action != "none")
    assert isinstance(plan.reason, str) and plan.reason.strip()
