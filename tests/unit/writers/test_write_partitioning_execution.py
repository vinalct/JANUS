"""Execute declared actions without adding a Spark scan or making a second decision."""

from dataclasses import replace

import pytest
from test_bronze_append_by_name import BASE_TABLE, TypedFakeDataFrame, contract, contract_plan
from test_bronze_overwrite_writer import ENVIRONMENT_CONFIG, FakeSparkSession

from janus.models import BronzeWriteIntent
from janus.models.config.types import SparkWriteConfig
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter


class PartitionedFrame(TypedFakeDataFrame):
    def __init__(self, session):
        super().__init__(session, BASE_TABLE)
        self.actions = []

    def coalesce(self, count):
        self.actions.append(("coalesce", count))
        return self

    def repartition(self, count):
        self.actions.append(("repartition", count))
        return self

    def count(self):
        raise AssertionError("partition sizing must not scan the frame")


@pytest.mark.parametrize("strategy", ["replace_table", "insert", "merge_on_keys"])
@pytest.mark.parametrize(
    ("exact", "estimate", "enabled", "action", "target"),
    [
        (1, None, True, "coalesce", 1),
        (8, None, True, "repartition", 8),
        (4, None, True, "none", None),
        (None, None, True, "none", None),
        (None, 129 * 1024**2, True, "coalesce", 2),
        (1, 0, False, "none", None),
    ],
)
def test_write_executes_action_and_records_metadata(
    tmp_path, monkeypatch, strategy, exact, estimate, enabled, action, target
):
    session = FakeSparkSession(table_exists=False)
    frame = PartitionedFrame(session)
    plan = contract_plan(tmp_path, contract("base"))
    plan = replace(
        plan,
        source_config=replace(
            plan.source_config,
            spark=replace(plan.source_config.spark, write=SparkWriteConfig(partitions=exact)),
        ),
    )
    events = []
    if strategy == "merge_on_keys":
        from janus.writers import spark as writer_module

        def dedupe(prepared, keys, *, count):
            assert prepared.actions == ([] if action == "none" else [(action, target)])
            events.append("dedupe")
            return prepared, None, None

        monkeypatch.setattr(writer_module, "_reject_complex_merge_keys", lambda *_args: None)
        monkeypatch.setattr(writer_module, "_dedupe_for_merge", dedupe)
    writer = SparkDatasetWriter(StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path))
    result = writer.write(
        frame, plan, "bronze",
        intent=BronzeWriteIntent(strategy, "overwrite", merge_keys=("id",)),
        estimated_input_bytes=estimate, apply_repartition=enabled,
        metadata={"caller": "preserved"},
    )
    metadata = result.metadata_as_dict()
    assert frame.actions == ([] if action == "none" else [(action, target)])
    assert metadata["caller"] == "preserved"
    assert metadata["write_partitioning"] == action
    assert metadata["write_input_partitions"] == "4"
    if target is None:
        assert "write_partitioning_target" not in metadata
        assert "write_partitioning_reason" not in metadata
    else:
        assert metadata["write_partitioning_target"] == str(target)
        assert metadata["write_partitioning_reason"]
    assert events == (["dedupe"] if strategy == "merge_on_keys" else [])
