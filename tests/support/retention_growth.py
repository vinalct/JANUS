from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

from janus.models import ExtractedArtifact, ExtractionResult, RunContext
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.runtime import BronzeMaterializer
from janus.strategies.files import FileStrategy
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter
from tests.integration.full_refresh_history.conftest import (
    ENVIRONMENT_CONFIG,
    FIXTURE_PROJECT_ROOT,
    FullRefreshHarness,
)
from tests.integration.inep import test_inep_integration as inep
from tests.support.cnpj_csv_generator import write_csv
from tests.support.contracts import with_registry_contract
from tests.support.retention_baseline import PROJECT_ROOT, filesystem_state
from tests.support.spark_sessions import build_iceberg_session

STORAGE_CONFIG = {
    "storage": {"root_dir": "data", "raw_dir": "data/raw",
                "bronze_dir": "data/bronze", "metadata_dir": "data/metadata"},
}
CNPJ_SOURCE = "receita_federal__cnpj__estabelecimentos_full_refresh"


def table_measurement(spark, identifier: str) -> dict[str, object]:
    """Distinguish current .files from all retained data files on disk."""
    location = next(
        row.data_type for row in spark.sql(f"DESCRIBE TABLE EXTENDED {identifier}").collect()
        if row.col_name == "Location"
    )
    parsed = urlsplit(location)
    if parsed.scheme not in {"", "file"}:
        raise ValueError("Growth measurement requires a local fixture warehouse")
    root = Path(unquote(parsed.path))
    state = filesystem_state(root)
    parquet = list(root.rglob("*.parquet"))
    snapshots = spark.table(f"{identifier}.snapshots").orderBy("committed_at", "snapshot_id")
    return {
        "table": identifier,
        "location": str(root),
        "snapshots": snapshots.count(),
        "snapshot_ids": [row.snapshot_id for row in snapshots.select("snapshot_id").collect()],
        "history": spark.table(f"{identifier}.history").count(),
        "current_data_files": spark.table(f"{identifier}.files").count(),
        "current_data_bytes": sum(row.file_size_in_bytes for row in
                                  spark.table(f"{identifier}.files").collect()),
        "retained_data_files": len(parquet),
        "retained_data_bytes": sum(path.stat().st_size for path in parquet),
        **state,
    }


def _history_growth(spark, root: Path, partitioned: bool) -> list[dict[str, object]]:
    name = "partitioned" if partitioned else "unpartitioned"
    source_id = f"full_refresh_history_{name}"
    harness = FullRefreshHarness(
        spark=spark, registry=load_registry(FIXTURE_PROJECT_ROOT), project_root=root / name,
        table_name=f"m0_{name}",
        writer=SparkDatasetWriter(StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, root)),
    )
    measurements = []
    for index in range(1, 4):
        plan = harness.plan(source_id, run_id=f"m0-{name}-{index}")
        rows = [(f"run{index}-a", "alpha"), (f"run{index}-b", "beta")]
        schema = "id string, value string"
        if partitioned:
            rows = [(*row, key) for row, key in zip(rows, ("north", "south"), strict=True)]
            schema += ", partition_key string"
        result = harness.write_rows(rows, schema, plan)
        measurements.append({"run": index, **table_measurement(spark, result.path)})
    return measurements


def _file_growth(spark, root: Path, kind: str) -> list[dict[str, object]]:
    project = root / kind
    project.mkdir()
    registry = load_registry(PROJECT_ROOT)
    layout = StorageLayout.from_environment_config(STORAGE_CONFIG, project)
    strategy = FileStrategy(storage_layout_factory=lambda _plan: layout,
                            sleeper=lambda _seconds: None, clock=lambda: 0.0)
    if kind == "inep":
        archive = inep._build_fixture_archive(project)
        source = inep._cloned_source_config(project, archive_path=archive)
    else:
        source = registry.get_source(CNPJ_SOURCE, include_disabled=True)
        csv_path = project / "fixtures" / "estabelecimentos.csv"
        write_csv(csv_path, rows=100, seed=19)
    materializer = BronzeMaterializer(SparkDatasetReader(), BaseNormalizer(), SparkDatasetWriter)
    measurements = []
    for index in range(1, 4):
        context = RunContext.create(run_id=f"m0-{kind}-{index}", environment="local",
                                    project_root=project,
                                    started_at=datetime(2026, 10, 5, 12, 0, tzinfo=UTC))
        plan = strategy.plan(source, context)
        if kind == "inep":
            plan = with_registry_contract(plan)
            handoff = strategy.build_normalization_handoff(plan, strategy.extract(plan))
        else:
            plan = plan.with_data_contract(registry.contract_for(CNPJ_SOURCE))
            handoff = ExtractionResult.from_plan(
                plan, (ExtractedArtifact(path=str(csv_path), format="csv"),),
            )
        results, _, _, _ = materializer.materialize(
            PlannedRun(plan=plan, strategy=strategy), plan, spark, handoff, layout, None,
        )
        identifier = results[-1].path
        measurements.append({"run": index, "rows": spark.table(identifier).count(),
                             **table_measurement(spark, identifier)})
    return measurements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    spark = build_iceberg_session("janus-order21-growth", root / "catalog")
    try:
        measurements = {
            "unpartitioned": _history_growth(spark, root, False),
            "partitioned": _history_growth(spark, root, True),
            "inep": _file_growth(spark, root, "inep"),
            "cnpj": _file_growth(spark, root, "cnpj"),
        }
    finally:
        spark.stop()
    # Catalog is outside iceberg/, so this digest includes no open SQLite WAL.
    result = {"measurements": measurements,
              "warehouse": filesystem_state(root / "catalog" / "iceberg")}
    (root / "measurement.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
