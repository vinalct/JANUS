"""AC-4: one 10,000-page measurement; partitions are gated, wall time is recorded."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import pytest

from janus.models import ExtractionResult
from janus.models.data_contracts import load_data_contract
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.readers import SparkDatasetReader
from janus.readers import spark as spark_reader
from janus.runtime.materialize import BronzeMaterializer
from janus.strategies.api import ApiStrategy
from janus.strategies.common import _raw_run_path_prefix
from janus.utils.environment import load_environment_config
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter
from tests.fixtures.handoff.cases import ENVIRONMENT_CONFIG
from tests.fixtures.handoff.generate_pages import FIXTURE_DIR, generate_pages, synthetic_plan
from tests.support.spark_sessions import PROJECT_ROOT, sqlite_catalog_target, start_session

PAGE_COUNT = 10_000
RUN_COUNT = 3
MAX_PARTITION_RATIO = 0.05
MAX_ARTIFACT_RATIO = 0.01


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    target = sqlite_catalog_target(tmp_path_factory.mktemp("ac4-handoff-catalog"))
    target.prepare()
    profile = load_environment_config("local", PROJECT_ROOT)
    session = start_session(
        "janus-ac4-read-partitions",
        {**target.session_options(), **profile["spark"]["config"]},
        master="local[4]",
    )
    yield session
    session.stop()


@dataclass
class MeasurementReader(SparkDatasetReader):
    """Observe the materializer's actual read before contract checks or normalization."""

    reads: list[dict] = field(default_factory=list)

    def read_extraction_result(self, spark, extraction_result, **kwargs):
        resolved_options = {}
        resolve_options = spark_reader._resolved_read_options

        def capture_options(*args, **options_kwargs):
            options = resolve_options(*args, **options_kwargs)
            resolved_options.update(options)
            return options

        with patch.object(
            spark_reader, "_resolved_read_options", side_effect=capture_options
        ) as options_spy:
            dataframe = super().read_extraction_result(spark, extraction_result, **kwargs)
        assert options_spy.call_count == 1, "the actual reader options must be observed"
        self.reads.append(
            {
                "read_partitions": dataframe.rdd.getNumPartitions(),
                "read_format": extraction_result.single_artifact_format(),
                "read_options": resolved_options,
            }
        )
        return dataframe


def _build_handoffs(work_root):
    raw_root = work_root / "raw"
    plan = synthetic_plan(raw_root).with_data_contract(
        load_data_contract(FIXTURE_DIR / "envelope_contract.yaml")
    )
    # Reuse M0's bytes in the production run layout the handoff builder expects.
    prefix = _raw_run_path_prefix(plan)
    assert prefix is not None
    artifacts = generate_pages(raw_root / prefix, page_count=PAGE_COUNT)
    extraction = ExtractionResult.from_plan(plan, artifacts, records_extracted=PAGE_COUNT * 15)
    layout = StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, work_root)
    strategy = ApiStrategy(storage_layout_factory=lambda _plan: layout)
    handoffs = {}
    for mode in ("pages", "compact"):
        mode_plan = replace(
            plan,
            source_config=replace(
                plan.source_config,
                spark=replace(
                    plan.source_config.spark,
                    handoff=replace(plan.source_config.spark.handoff, mode=mode),
                ),
            ),
        )
        handoff = strategy.build_normalization_handoff(mode_plan, extraction)
        assert handoff.metadata_as_dict()["handoff_compaction"] == (
            "pages" if mode == "pages" else "jsonl"
        )
        handoffs[mode] = (PlannedRun(mode_plan, strategy), handoff)
    return layout, handoffs


def _measure_once(spark, layout, planned, handoff, *, mode, run):
    plan = replace(
        planned.plan,
        bronze_output=replace(
            planned.plan.bronze_output, table_name=f"synthetic_ac4_{mode}_run_{run}"
        ),
    )
    reader = MeasurementReader()
    materializer = BronzeMaterializer(reader, BaseNormalizer(), SparkDatasetWriter)
    started = monotonic()
    results, _frame, _keys, _evidence = materializer.materialize(
        replace(planned, plan=plan), plan, spark, handoff, layout, None
    )
    seconds = monotonic() - started
    assert len(reader.reads) == 1
    assert len(results) == 1
    table = results[0].path
    return {
        **reader.reads[0],
        "run": run,
        "input_file_count": len(handoff.artifacts),
        "input_bytes": sum(Path(artifact.path).stat().st_size for artifact in handoff.artifacts),
        "materialize_seconds": seconds,
        "bronze_data_files": spark.table(f"{table}.files").count(),
        "bronze_rows": spark.table(table).count(),
    }


def _report(spark, measurements):
    medians = {
        mode: statistics.median(run["materialize_seconds"] for run in runs)
        for mode, runs in measurements.items()
    }
    max_partition_bytes = int(spark.conf.get("spark.sql.files.maxPartitionBytes"))
    evidence = {
        "page_count": PAGE_COUNT,
        "records_per_page": 15,
        "seed": 22,
        "spark_version": spark.version,
        "spark_master": spark.sparkContext.master,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "spark_config": {
            key: spark.conf.get(key)
            for key in (
                "spark.sql.files.maxPartitionBytes",
                "spark.sql.files.openCostInBytes",
                "spark.sql.shuffle.partitions",
                "spark.sql.adaptive.enabled",
                "spark.sql.session.timeZone",
            )
        },
        "arms": measurements,
        "materialize_median_seconds": medians,
        "partition_ratio": (
            measurements["compact"][0]["read_partitions"]
            / measurements["pages"][0]["read_partitions"]
        ),
        "max_partition_bytes_floor": max(
            1, math.ceil(measurements["compact"][0]["input_bytes"] / max_partition_bytes)
        ),
    }
    print("arm | run | read partitions | artifacts | input bytes | materialize s | bronze files")
    for mode, runs in measurements.items():
        for run in runs:
            print(
                f"{mode} | {run['run']} | {run['read_partitions']} | "
                f"{run['input_file_count']} | {run['input_bytes']} | "
                f"{run['materialize_seconds']:.6f} | {run['bronze_data_files']}"
            )
        print(f"{mode} median: {medians[mode]:.6f} s")
    print("AC4_MEASUREMENT=" + json.dumps(evidence, sort_keys=True), flush=True)


def test_read_partition_reduction_on_10000_pages(spark, tmp_path, capsys):
    layout, handoffs = _build_handoffs(tmp_path)
    measurements = {mode: [] for mode in handoffs}
    # Alternate arms so all three compact samples do not follow all three page samples.
    for run in range(1, RUN_COUNT + 1):
        for mode, (planned, handoff) in handoffs.items():
            measurements[mode].append(
                _measure_once(spark, layout, planned, handoff, mode=mode, run=run)
            )
    _report(spark, measurements)
    # Successful tests normally hide captured stdout; keep the measurement in CI logs.
    captured = capsys.readouterr()
    with capsys.disabled():
        print(captured.out, end="", flush=True)

    for runs in measurements.values():
        assert len(runs) == RUN_COUNT
        assert len({run["read_partitions"] for run in runs}) == 1, measurements
        assert all(run["bronze_rows"] == PAGE_COUNT for run in runs), measurements
        assert all(run["bronze_data_files"] >= 1 for run in runs), measurements
    pages_partitions = measurements["pages"][0]["read_partitions"]
    compact_partitions = measurements["compact"][0]["read_partitions"]
    assert compact_partitions >= 1, measurements
    assert compact_partitions <= math.ceil(pages_partitions * MAX_PARTITION_RATIO), (
        f"AC-4: compacted read produced {compact_partitions} partitions, "
        f"more than 5% of the {pages_partitions} the page read produced"
    )
    assert all(run["input_file_count"] == PAGE_COUNT for run in measurements["pages"])
    for run in measurements["pages"]:
        assert run["read_format"] == "json", run
        assert run["read_options"]["multiLine"] == "true", run
    for run in measurements["compact"]:
        assert run["read_format"] == "jsonl", run
        assert "multiline" not in {key.lower() for key in run["read_options"]}, run
        assert 1 <= run["input_file_count"] <= PAGE_COUNT * MAX_ARTIFACT_RATIO, run
