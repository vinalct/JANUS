from __future__ import annotations

import argparse
import json
import platform
import statistics
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from janus.models import ExtractionResult
from janus.models.data_contracts import load_data_contract
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.readers import SparkDatasetReader
from janus.runtime.materialize import BronzeMaterializer
from janus.schema_contracts import spark_schema_from_contract
from janus.strategies.api import ApiStrategy
from janus.utils.environment import load_environment_config
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter
from tests.fixtures.handoff.generate_pages import FIXTURE_DIR, generate_pages, synthetic_plan
from tests.support.bronze_digest import bronze_golden
from tests.support.contract_baseline import BRONZE_CASES
from tests.support.contract_baseline import bronze_golden as run_fixture
from tests.support.spark_sessions import sqlite_catalog_target, start_session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_ROOT = Path("/tmp/janus-baseline")
REGENERATION_COMMAND = (
    "python -m tests.support.handoff_baseline --output-dir /tmp/baseline-output"
)


def capture_goldens(spark, target, work_root: Path) -> tuple[dict, dict]:
    """Run every existing offline bronze fixture through the real executor."""
    goldens, file_counts = {}, {}
    for case in BRONZE_CASES:
        captured = run_fixture(
            case,
            PROJECT_ROOT,
            spark=spark,
            catalog_target=target,
            work_root=work_root / "cases" / case,
        )
        identifier = captured["table_identifier"]
        goldens[case] = bronze_golden(spark.table(identifier), identifier=identifier)
        file_counts[case] = spark.table(f"{identifier}.files").count()
        print(f"captured {case}: {goldens[case]['row_count']} rows", flush=True)
    return goldens, file_counts


def measure_pages(spark, work_root: Path, *, page_count: int = 10_000, runs: int = 3) -> dict:
    """Time materialize only; each run creates a fresh, equivalent bronze table."""
    if page_count < 1 or runs < 1:
        raise ValueError("page_count and runs must be positive")
    raw_root = work_root / "raw"
    started = monotonic()
    artifacts = generate_pages(raw_root, page_count=page_count)
    generation_seconds = monotonic() - started
    plan = synthetic_plan(raw_root).with_data_contract(
        load_data_contract(FIXTURE_DIR / "envelope_contract.yaml")
    )
    handoff = ExtractionResult.from_plan(plan, artifacts, records_extracted=page_count * 15)
    reader = SparkDatasetReader()
    dataframe = reader.read_extraction_result(
        spark, handoff, schema=spark_schema_from_contract(plan.data_contract)
    )
    read_partitions = dataframe.rdd.getNumPartitions()
    # Infer over every page as well: the deliberately late field must be discovered.
    inferred = reader.read_extraction_result(spark, handoff)
    late_type = (
        inferred.schema["data"].dataType.elementType["observacaoTardia"].dataType.simpleString()
    )
    assert late_type == "string", late_type
    layout = StorageLayout.from_environment_config(
        {"storage": {key: "data" for key in ("root_dir", "raw_dir", "bronze_dir", "metadata_dir")}},
        raw_root,
    )
    materializer = BronzeMaterializer(reader, BaseNormalizer(), SparkDatasetWriter)
    seconds, file_counts, row_counts = [], [], []
    for run in range(1, runs + 1):
        run_plan = replace(
            plan, bronze_output=replace(plan.bronze_output, table_name=f"synthetic_pages_run_{run}")
        )
        planned = PlannedRun(plan=run_plan, strategy=ApiStrategy())
        started = monotonic()
        results, _frame, _keys, _evidence = materializer.materialize(
            planned, run_plan, spark, handoff, layout, None
        )
        seconds.append(monotonic() - started)
        table = results[0].path
        file_counts.append(spark.table(f"{table}.files").count())
        row_counts.append(spark.table(table).count())
        assert row_counts[-1] == page_count
        print(f"materialize run {run}: {seconds[-1]:.3f}s, {file_counts[-1]} files", flush=True)
    payload_bytes = sum(Path(artifact.path).stat().st_size for artifact in artifacts)
    sidecar_bytes = sum(
        Path(artifact.path + ".sha256").stat().st_size for artifact in artifacts
    )
    return {
        "page_count": page_count,
        "records_per_page": 15,
        "seed": 22,
        "generation_seconds": generation_seconds,
        "payload_bytes": payload_bytes,
        "sidecar_bytes": sidecar_bytes,
        "total_raw_bytes": payload_bytes + sidecar_bytes,
        "baseline_read_partitions": read_partitions,
        "baseline_materialize_seconds": seconds,
        "baseline_materialize_spread_seconds": max(seconds) - min(seconds),
        "baseline_materialize_median_seconds": statistics.median(seconds),
        "baseline_bronze_file_count": file_counts,
        "baseline_bronze_row_count": row_counts,
        "baseline_synthetic_golden": bronze_golden(spark.table(table), identifier=table),
        "inferred_late_field_type": late_type,
        "spark_version": spark.version,
        "spark_master": spark.sparkContext.master,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--page-count", type=int, default=10_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--goldens-only", action="store_true")
    args = parser.parse_args()
    # Refuse existing work rather than accidentally measuring an overwrite/reusing old raw.
    args.work_root.mkdir(parents=True, exist_ok=False)
    target = sqlite_catalog_target(args.work_root / "catalog")
    target.prepare()
    profile = load_environment_config("local", PROJECT_ROOT)
    options = {**target.session_options(), **profile["spark"]["config"]}
    spark = start_session("janus-baseline", options, master="local[4]")
    try:
        goldens, fixture_file_counts = capture_goldens(spark, target, args.work_root)
        measurements = (
            {} if args.goldens_only else measure_pages(
                spark, args.work_root, page_count=args.page_count, runs=args.runs
            )
        )
    finally:
        spark.stop()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": f"Regenerate in a fresh container scratch zone: {REGENERATION_COMMAND}",
        "metadata": {
            "run_date": datetime.now(UTC).date().isoformat(),
            "python_version": platform.python_version(),
            "work_root": str(args.work_root),
            "provenance": "All columns hashed verbatim; no path or timestamp exclusions.",
        },
        "baseline_goldens": goldens,
        "baseline_fixture_file_count": fixture_file_counts,
    }
    (args.output_dir / "baseline_goldens.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.goldens_only:
        (args.output_dir / "baseline_measurements.json").write_text(
            json.dumps(measurements, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
