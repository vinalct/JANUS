"""Wall-clock of ``BronzeMaterializer.materialize`` under the shipped ``local`` profile."""

from __future__ import annotations

import shutil
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.models import ExtractedArtifact, ExtractionResult, RunContext
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.runtime import BronzeMaterializer
from janus.strategies.files import FileStrategy
from janus.utils.environment import load_environment_config
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CNPJ_SOURCE_ID = "receita_federal__cnpj__estabelecimentos_full_refresh"
STARTED_AT = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
TIMED_RUNS = 3
STORAGE_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}


def cost_inep(out_dir: Path) -> dict[str, Any]:
    """The INEP fixture archive, through the INEP integration suite's own helpers."""
    from tests.integration.inep import test_inep_integration as inep
    from tests.support.contracts import with_registry_contract

    root = _fresh(out_dir, "cost-inep")
    project = root / "project"
    project.mkdir()
    archive = inep._build_fixture_archive(project)
    source_config = inep._cloned_source_config(project, archive_path=archive)
    storage_layout = inep._storage_layout(project)
    strategy = _file_strategy(storage_layout)
    run_context = RunContext.create(
        run_id="m0-cost-inep", environment="local", project_root=project, started_at=STARTED_AT
    )
    plan = with_registry_contract(strategy.plan(source_config, run_context))
    handoff = strategy.build_normalization_handoff(plan, strategy.extract(plan))
    planned_run = PlannedRun(plan=plan, strategy=strategy)
    return time_materialize(root, planned_run, handoff, storage_layout)


def cost_cnpj(out_dir: Path, csv_path: Path) -> dict[str, Any]:
    """A generated ``Estabelecimentos`` part file as the one artifact of a file handoff."""
    root = _fresh(out_dir, "cost-cnpj")
    project = root / "project"
    project.mkdir()
    registry = load_registry(PROJECT_ROOT)
    source = registry.get_source(CNPJ_SOURCE_ID, include_disabled=True)
    storage_layout = StorageLayout.from_environment_config(STORAGE_CONFIG, project)
    strategy = _file_strategy(storage_layout)
    run_context = RunContext.create(
        run_id="m0-cost-cnpj", environment="local", project_root=project, started_at=STARTED_AT
    )
    plan = strategy.plan(source, run_context).with_data_contract(
        registry.contract_for(CNPJ_SOURCE_ID)
    )
    handoff = ExtractionResult.from_plan(
        plan, (ExtractedArtifact(path=str(csv_path.resolve()), format="csv"),)
    )
    planned_run = PlannedRun(plan=plan, strategy=strategy)
    measured = time_materialize(root, planned_run, handoff, storage_layout)
    measured["csv"] = {"path": str(csv_path), "bytes": csv_path.stat().st_size}
    return measured


def time_materialize(
    root: Path,
    planned_run: PlannedRun,
    handoff: ExtractionResult,
    storage_layout: StorageLayout,
    *,
    timed_runs: int = TIMED_RUNS,
) -> dict[str, Any]:
    """One untimed warm-up, then ``timed_runs`` timed calls; median, jobs and peak RSS."""
    spark, profile = shipped_profile_session(root / "warehouse")
    try:
        materializer = BronzeMaterializer(
            reader=SparkDatasetReader(),
            normalizer=BaseNormalizer(),
            writer_factory=SparkDatasetWriter,
        )
        plan = planned_run.plan

        def once(group: str) -> tuple[float, int, Any]:
            spark.sparkContext.setJobGroup(group, group)
            started = time.perf_counter()
            results, _, _, _ = materializer.materialize(
                planned_run, plan, spark, handoff, storage_layout, None
            )
            elapsed = time.perf_counter() - started
            jobs = len(spark.sparkContext.statusTracker().getJobIdsForGroup(group))
            return elapsed, jobs, results

        warm_seconds, warm_jobs, _ = once("cost-warmup")
        timed = [once(f"cost-run-{index}") for index in range(1, timed_runs + 1)]
        last_results = timed[-1][2]
        rows = spark.table(last_results[0].path).count()
        seconds = [round(elapsed, 3) for elapsed, _, _ in timed]
        median = statistics.median(seconds)
        contract = plan.data_contract
        return {
            "source_id": plan.source.source_id,
            "contract": contract.id if contract else None,
            "enforcement": contract.janus.enforcement if contract else None,
            "bronze_write_metadata": [result.metadata_as_dict() for result in last_results],
            "rows": rows,
            "warmup": {"seconds": round(warm_seconds, 3), "jobs": warm_jobs},
            "timed_seconds": seconds,
            "timed_jobs": [jobs for _, jobs, _ in timed],
            "median_seconds": median,
            "rows_per_second": round(rows / median, 1) if median else None,
            "driver_peak_rss_kib": jvm_peak_rss_kib(spark),
            **profile,
        }
    finally:
        spark.stop()


def shipped_profile_session(warehouse: Path) -> tuple[Any, dict[str, Any]]:
    """A session over a suite catalog, with the ``local`` profile's master and Spark config."""
    from tests.support.spark_sessions import sqlite_catalog_target, start_session

    profile = load_environment_config("local", PROJECT_ROOT)
    target = sqlite_catalog_target(warehouse)
    target.prepare()
    options = target.session_options()
    options.update({key: str(value) for key, value in profile["spark"]["config"].items()})
    master = str(profile["spark"]["master"])
    spark = start_session("janus-contract-enforcement-cost", options, master=master)
    jvm_max_heap = spark.sparkContext._jvm.java.lang.Runtime.getRuntime().maxMemory()
    return spark, {
        "master": master,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "spark.driver.memory": spark.sparkContext.getConf().get("spark.driver.memory"),
        "spark.sql.shuffle.partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "jvm_max_heap_mib": round(jvm_max_heap / 1024 / 1024),
    }


def jvm_peak_rss_kib(spark: Any) -> int | None:
    """``VmHWM`` of the driver JVM — its peak resident set since it started."""
    pid = spark.sparkContext._jvm.java.lang.ProcessHandle.current().pid()
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return None


def _file_strategy(storage_layout: StorageLayout) -> FileStrategy:
    return FileStrategy(
        storage_layout_factory=lambda plan: storage_layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )


def _fresh(out_dir: Path, name: str) -> Path:
    path = out_dir / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


__all__ = [
    "CNPJ_SOURCE_ID",
    "cost_cnpj",
    "cost_inep",
    "jvm_peak_rss_kib",
    "shipped_profile_session",
    "time_materialize",
]
