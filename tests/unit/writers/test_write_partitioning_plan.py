import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from janus.models.config.types import SparkWriteConfig
from janus.writers.partitioning import (
    WritePartitioningPlan,
    apply_write_partitioning,
    plan_write_partitioning,
)

MIB = 1024 * 1024
DECISION_CASES = [
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
    (4, 2048 * MIB, 4, "none", None),
    (-1, 2048 * MIB, 4, "none", None),
    (-1, None, None, "none", None),
    (4, 3 * 128 * MIB, None, "coalesce", 3),
    (4, 4 * 128 * MIB, None, "none", None),
    (4, 4 * 128 * MIB + 1, None, "repartition", 5),
    (1, 0, None, "none", None),
    (2**26, 2**53 - 1, None, "none", None),
    (2**26, 2**53, None, "none", None),
    (2**26, 2**53 + 1, None, "repartition", 2**26 + 1),
]


@pytest.mark.parametrize(
    ("current", "size", "exact", "action", "target"), DECISION_CASES,
)
def test_partitioning_decision(current, size, exact, action, target):
    plan = plan_write_partitioning(
        current_partitions=current,
        estimated_input_bytes=size,
        declaration=SparkWriteConfig(target_file_size_mb=128, partitions=exact),
    )
    assert plan.action == action
    assert plan.target_partitions == target
    assert plan.changes_partitioning == (action != "none")
    assert isinstance(plan.reason, str) and plan.reason.strip()


@pytest.mark.parametrize(
    ("current", "size", "exact", "reason"),
    [
        (0, 2048 * MIB, 4, "frame reports no partitions"),
        (4, 2048 * MIB, 4, "already at the declared partition count"),
        (200, None, 1, "declared spark.write.partitions"),
        (1, None, 4, "declared spark.write.partitions"),
        (4, None, None, "no size estimate available"),
        (200, 50 * MIB, None, "estimated 52428800 B at 128 MiB per file"),
        (1, 2048 * MIB, None, "estimated 2147483648 B at 128 MiB per file"),
        (8, 1024 * MIB, None, "estimated 1073741824 B at 128 MiB per file"),
    ],
)
def test_plan_explains_the_decision(current, size, exact, reason):
    plan = plan_write_partitioning(
        current_partitions=current,
        estimated_input_bytes=size,
        declaration=SparkWriteConfig(partitions=exact),
    )
    assert plan.reason == reason


@pytest.mark.parametrize(
    ("target_mb", "action", "target"),
    [(32, "repartition", 32), (128, "none", None), (256, "coalesce", 4)],
)
def test_size_based_count_uses_the_declared_file_size(target_mb, action, target):
    plan = plan_write_partitioning(
        current_partitions=8,
        estimated_input_bytes=1024 * MIB,
        declaration=SparkWriteConfig(target_file_size_mb=target_mb),
    )
    assert (plan.action, plan.target_partitions) == (action, target)
    assert plan.reason == f"estimated 1073741824 B at {target_mb} MiB per file"


@pytest.mark.parametrize("mode", ["none", "hash", "range"])
def test_table_distribution_does_not_change_the_task_count(mode):
    plan = plan_write_partitioning(
        current_partitions=200,
        estimated_input_bytes=None,
        declaration=SparkWriteConfig(partitions=1, distribution_mode=mode),
    )
    assert (plan.action, plan.target_partitions) == ("coalesce", 1)


@pytest.mark.parametrize(
    ("current", "exact", "action"),
    [(200, 1, "coalesce"), (1, 4, "repartition"), (4, 4, "none")],
)
def test_executor_dispatches_the_plan_without_inspecting_the_frame(current, exact, action):
    class FakeFrame:
        # No RDD, size estimate or count method: execution may only dispatch the action.
        def __init__(self):
            self.calls = []
            self.output = object()

        def coalesce(self, target):
            self.calls.append(("coalesce", target))
            return self.output

        def repartition(self, target):
            self.calls.append(("repartition", target))
            return self.output

    frame = FakeFrame()
    plan = plan_write_partitioning(
        current_partitions=current,
        estimated_input_bytes=None,
        declaration=SparkWriteConfig(partitions=exact),
    )
    result = apply_write_partitioning(frame, plan)
    if action == "none":
        assert result is frame
        assert frame.calls == []
    else:
        assert result is frame.output
        assert frame.calls == [(action, exact)]


def test_plan_is_an_immutable_value():
    plan = WritePartitioningPlan("coalesce", 1, "declared spark.write.partitions")
    assert not hasattr(plan, "__dict__")
    with pytest.raises(FrozenInstanceError):
        plan.action = "none"


def test_partitioning_transitive_runtime_imports_need_no_pyspark():
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, env.get("PYTHONPATH")) if part
    )
    program = """
import importlib.abc
import sys

class RefuseSpark(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyspark" or fullname.startswith("pyspark."):
            raise AssertionError("runtime import attempted: " + fullname)

sys.meta_path.insert(0, RefuseSpark())
from janus.models.config.types import SparkWriteConfig
from janus.writers.partitioning import plan_write_partitioning
plan = plan_write_partitioning(
    current_partitions=200, estimated_input_bytes=None,
    declaration=SparkWriteConfig(partitions=1))
assert (plan.action, plan.target_partitions) == ("coalesce", 1)
print(sorted(m for m in sys.modules if m == "pyspark" or m.startswith("pyspark.")))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout
