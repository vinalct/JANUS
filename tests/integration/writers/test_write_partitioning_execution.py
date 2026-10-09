from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from time import monotonic

import pytest

from janus.models import BronzeWriteIntent, ExtractedArtifact, ExtractionResult
from janus.models.config.types import SparkWriteConfig
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.readers import SparkDatasetReader
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.runtime.materialize import BronzeMaterializer
from janus.strategies.api import ApiStrategy
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter
from tests.fixtures.handoff.generate_pages import synthetic_plan
from tests.support.bronze_digest import bronze_golden
from tests.support.contract_baseline import _CASE_BY_NAME, _ENVIRONMENT_CONFIG, BRONZE_CASES
from tests.support.handoff_matrix import identifier_for, plan_cell, prepare_fixture
from tests.support.spark_sessions import sqlite_catalog_target, start_session
from tests.support.writer_contracts import contract_for_columns

pytestmark = pytest.mark.spark
GIB = 1024**3
SIZE_TOLERANCE = 1.5


@pytest.fixture(scope="module")
def catalog_target(tmp_path_factory):
    target = sqlite_catalog_target(tmp_path_factory.mktemp("write-partitioning-catalog"))
    target.prepare()
    return target


@pytest.fixture(scope="module")
def spark(catalog_target):
    options = {
        **catalog_target.session_options(),
        "spark.driver.memory": "2g",
        "spark.memory.fraction": "0.25",
        "spark.sql.inMemoryColumnarStorage.batchSize": "512",
        "spark.sql.inMemoryColumnarStorage.compressed": "false",
        "spark.sql.files.maxPartitionBytes": str(64 * 1024**2),
    }
    session = start_session(
        "janus-write-partitioning", options, master="local[2]"
    )
    yield session
    session.stop()


class RecordingWriter(SparkDatasetWriter):
    def __init__(self, layout, *, enabled=True):
        super().__init__(layout)
        self.enabled = enabled
        self.results = []

    def write(self, *args, **kwargs):
        result = super().write(*args, **kwargs, apply_repartition=self.enabled)
        self.results.append(result)
        return result


@pytest.mark.parametrize("case_name", BRONZE_CASES)
def test_small_fixture_file_count_does_not_increase(
    spark, catalog_target, tmp_path, monkeypatch, case_name
):
    mapping, config_path, strategy_factory = prepare_fixture(case_name, tmp_path)
    token = _CASE_BY_NAME[case_name].token
    if token is not None:
        monkeypatch.setenv(*token)
    canonical_identifier = identifier_for(mapping)
    captures = {}
    for label, enabled in (("before", False), ("after", True)):
        planned = plan_cell(
            mapping, config_path, strategy_factory(), mode="pages", case_name=case_name
        )
        plan = replace(
            planned.plan,
            bronze_output=replace(
                planned.plan.bronze_output,
                table_name=planned.plan.bronze_output.table_name + f"__ac7_{label}",
            ),
        )
        writers = []

        def factory(layout, enabled=enabled, writers=writers):
            writer = RecordingWriter(layout, enabled=enabled)
            writers.append(writer)
            return writer

        executed = SourceExecutor(writer_factory=factory).execute(
            replace(planned, plan=plan),
            SparkSessionProvider.wrapping(spark, resolved_paths=catalog_target.resolved_paths),
            {**_ENVIRONMENT_CONFIG, **catalog_target.environment_config()},
        )
        assert executed.status == "succeeded", executed.failure_reason
        writes = [result for writer in writers for result in writer.results]
        table = writes[-1].path
        captures[label] = {
            "files": spark.table(f"{table}.files").count(),
            "golden": bronze_golden(spark.table(table), identifier=canonical_identifier),
            "write_metadata": [write.metadata_as_dict() for write in writes],
        }
    print("AC7_FIXTURE=" + json.dumps({"fixture": case_name, **captures}), flush=True)
    assert captures["after"]["files"] <= captures["before"]["files"]
    assert captures["after"]["golden"] == captures["before"]["golden"]


def write_plan(tmp_path, table_name, declaration):
    plan = synthetic_plan(tmp_path / "raw")
    config = replace(
        plan.source_config,
        spark=replace(plan.source_config.spark, write=declaration, partition_by=()),
    )
    return replace(
        plan,
        source_config=config,
        bronze_output=replace(plan.bronze_output, table_name=table_name),
    ).with_data_contract(
        contract_for_columns(
            (("id", "long"), ("value", "string")), source_id=config.source_id, primary_key=("id",)
        )
    )


@pytest.mark.parametrize("strategy", ["replace_table", "insert", "merge_on_keys"])
@pytest.mark.parametrize(("exact", "action"), [(1, "coalesce"), (8, "repartition"), (4, "none")])
def test_metadata_on_each_strategy(spark, tmp_path, strategy, exact, action):
    plan = write_plan(
        tmp_path, f"partitioning_{strategy}_{exact}", SparkWriteConfig(partitions=exact)
    )
    frame = BaseNormalizer().normalize(
        spark.createDataFrame([(1, "first"), (2, "second")], "id long, value string"), plan
    ).repartition(4)
    writer = SparkDatasetWriter(
        StorageLayout.from_environment_config(_ENVIRONMENT_CONFIG, tmp_path)
    )
    intent = BronzeWriteIntent(strategy, "overwrite", merge_keys=("id",))
    writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    result = writer.write(frame, plan, "bronze", intent=intent)
    metadata = result.metadata_as_dict()
    print("FR5_METADATA=" + json.dumps({"strategy": strategy, **metadata}), flush=True)
    assert metadata["write_partitioning"] == action
    assert metadata["write_input_partitions"] == "4"
    if action != "none":
        assert metadata["write_partitioning_target"] == str(exact)
        assert metadata["write_partitioning_reason"] == "declared spark.write.partitions"
    else:
        assert "write_partitioning_target" not in metadata
        assert "write_partitioning_reason" not in metadata
    assert spark.table(result.path).count() == (4 if strategy == "insert" else 2)


@pytest.mark.parametrize("zone", ["bronze", "raw"])
@pytest.mark.parametrize("enabled", [True, False])
def test_path_write_preserves_non_bronze_and_disabled_partitioning(
    spark, tmp_path, zone, enabled
):
    plan = write_plan(tmp_path, "path_write", SparkWriteConfig(partitions=1))
    frame = spark.range(20).repartition(4)
    writer = SparkDatasetWriter(
        StorageLayout.from_environment_config(_ENVIRONMENT_CONFIG, tmp_path)
    )
    result = writer.write(
        frame, plan, zone, format_name="parquet", mode="overwrite", partition_by=(),
        apply_repartition=enabled,
    )
    files = list(Path(result.path).glob("part-*.parquet"))
    assert len(files) == (1 if zone == "bronze" and enabled else 4)
    assert spark.read.parquet(result.path).count() == 20
    if zone == "bronze":
        assert result.metadata_as_dict()["write_partitioning"] == (
            "coalesce" if enabled else "none"
        )
        assert result.metadata_as_dict()["write_input_partitions"] == "4"


def _write_large_csv(path, *, seed=22):
    """Stream slightly over 2 GiB; seeded random payloads avoid trivial compression."""
    rng = random.Random(seed)
    rows = 2 * GIB // 8192
    with path.open("w", encoding="ascii", buffering=1024**2) as handle:
        handle.write("id,value\n")
        for index in range(rows):
            handle.write(f"{index},{rng.randbytes(4096).hex()}\n")
    return rows


@pytest.mark.large_handoff
def test_size_based_write_of_two_gib_csv(spark, catalog_target, tmp_path):
    # Iceberg's file target is advisory: row groups and compression cannot be split
    # exactly at that boundary. Allow 50% overshoot to detect oversized output while
    # tolerating those boundaries; this is a ceiling, not an expected file size.
    declaration = SparkWriteConfig(target_file_size_mb=128, distribution_mode="none")
    assert declaration.partitions is None  # Exercise size planning, never an exact override.
    plan = write_plan(tmp_path, "partitioning_two_gib_csv", declaration)
    plan = replace(
        plan,
        source_config=replace(
            plan.source_config,
            spark=replace(
                plan.source_config.spark, input_format="csv", read_options={"header": "true"}
            ),
        ),
    )
    path = tmp_path / "two-gib.csv"
    started = monotonic()
    rows = _write_large_csv(path)
    input_bytes = path.stat().st_size
    assert input_bytes >= 2 * GIB
    handoff = ExtractionResult.from_plan(plan, (ExtractedArtifact(str(path), "csv"),))
    materializer = BronzeMaterializer(SparkDatasetReader(), BaseNormalizer(), SparkDatasetWriter)
    results, _frame, _keys, _evidence = materializer.materialize(
        PlannedRun(plan=plan, strategy=ApiStrategy()), plan, spark, handoff,
        StorageLayout.from_environment_config(_ENVIRONMENT_CONFIG, tmp_path), None,
    )
    result = results[0]
    snapshots = spark.table(f"{result.path}.snapshots").collect()
    added_files = int(snapshots[-1].summary["added-data-files"])
    sizes = sorted(row.file_size_in_bytes for row in spark.table(f"{result.path}.files").collect())
    print("AC7_LARGE=" + json.dumps({
        "input_bytes": input_bytes, "rows": rows, "seed": 22,
        "added_data_files": added_files, "file_sizes": sizes,
        "target_bytes": declaration.target_file_size_bytes, "tolerance": SIZE_TOLERANCE,
        "write_metadata": result.metadata_as_dict(), "seconds": monotonic() - started,
    }), flush=True)
    assert added_files > 1
    assert len(sizes) == added_files
    assert max(sizes) <= declaration.target_file_size_bytes * SIZE_TOLERANCE
    assert result.metadata_as_dict()["write_partitioning_target"] == str(
        (input_bytes + declaration.target_file_size_bytes - 1) // declaration.target_file_size_bytes
    )
    assert spark.table(result.path).count() == rows
