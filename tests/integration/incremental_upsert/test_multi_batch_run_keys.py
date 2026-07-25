"""run keys accumulate across every batch, not only the last one.

A multi-batch file handoff normalizes and writes each batch independently, and the old
quality gate only ever validated the final batch — a duplicate sitting in batch 1 was
invisible. ``BronzeMaterializer.materialize`` now returns the union of the per-batch
distinct-key frames, so the bronze uniqueness oracle scans keys from *every* batch. This
proves that accumulation runs over real Spark frames, with a key that appears only in the
first batch surviving into the returned key set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    RunContext,
    SourceConfig,
    WriteResult,
)
from janus.planner import PlannedRun
from janus.runtime.materialize import BronzeMaterializer
from janus.utils.storage import StorageLayout

ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}

# One key lives only in batch 1, one only in batch 2, one only in batch 3 (batches of 5,
# 5, 1 over the file-handoff limit). Only "b1only" would be missed if run keys came from
# the last batch alone.
BATCH_KEY_ROWS = [
    [{"event_id": "b1only", "event_date": "2026-07-01T00:00:00Z"}],
    [{"event_id": "b2", "event_date": "2026-07-02T00:00:00Z"}],
    [{"event_id": "b3", "event_date": "2026-07-03T00:00:00Z"}],
]


@pytest.fixture(scope="module")
def spark():
    pyspark_sql = pytest.importorskip("pyspark.sql")
    session = (
        pyspark_sql.SparkSession.builder.appName("janus-multi-batch-run-keys")
        .master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


@dataclass(slots=True)
class FakeReader:
    def read_extraction_result(
        self, spark, extraction_result, format_name=None, schema=None, options=None
    ):
        del spark, extraction_result, format_name, schema, options
        return object()


@dataclass(slots=True)
class QueueNormalizer:
    """Returns one prepared per-batch frame per ``normalize`` call, in order."""

    frames: list[Any]

    def normalize(self, dataframe, plan):
        del dataframe, plan
        return self.frames.pop(0)


@dataclass(slots=True)
class NoopWriter:
    bronze_path: str

    def write(self, dataframe, plan, zone, *, intent=None, count_records=False, **kwargs):
        del dataframe, count_records, kwargs
        return WriteResult.from_plan(
            plan,
            zone,
            path=self.bronze_path,
            format_name="iceberg",
            mode=intent.reported_mode if intent is not None else "append",
            records_written=None,
            partition_by=intent.partition_columns if intent is not None else (),
        )


def test_run_keys_span_every_batch(spark, tmp_path):
    plan, planned_run = _plan(tmp_path)
    handoff = _handoff(plan, artifact_count=11)
    frames = [spark.createDataFrame(rows) for rows in BATCH_KEY_ROWS]

    materializer = BronzeMaterializer(
        reader=FakeReader(),
        normalizer=QueueNormalizer(frames=frames),
        writer_factory=lambda storage_layout: NoopWriter(plan.bronze_output.path),
    )
    _, _, run_keys = materializer.materialize(
        planned_run,
        plan,
        spark,
        handoff,
        StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path),
        None,
    )

    assert run_keys is not None
    collected = {row["event_id"] for row in run_keys.collect()}
    # The batch-1-only key is present, so a duplicate in an earlier batch is scannable.
    assert collected == {"b1only", "b2", "b3"}


def _handoff(plan: ExecutionPlan, *, artifact_count: int) -> ExtractionResult:
    raw_root = Path(plan.run_context.project_root) / "data" / "raw"
    artifacts = tuple(
        ExtractedArtifact(
            path=str(raw_root / f"page-{index:04d}.json"),
            format="json",
            checksum=f"sum{index}",
        )
        for index in range(artifact_count)
    )
    return ExtractionResult.from_plan(plan, artifacts=artifacts, records_extracted=artifact_count)


def _plan(tmp_path: Path) -> tuple[ExecutionPlan, PlannedRun]:
    source_config = _source_config(tmp_path)
    run_context = RunContext.create(
        run_id="run-multi-batch-run-keys-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 8, 12, 0, tzinfo=UTC),
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)
    planned_run = PlannedRun(
        plan=plan,
        strategy=SimpleNamespace(strategy_family="file"),
        hook=None,
    )
    return plan, planned_run


def _source_config(tmp_path: Path) -> SourceConfig:
    source_id = "multi_batch_run_keys_fixture"
    payload: dict[str, Any] = {
        "source_id": source_id,
        "name": source_id,
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": {
            "base_url": "https://example.invalid",
            "path": "/events",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 10,
            },
            "rate_limit": {"requests_per_minute": None, "concurrency": 1, "backoff_seconds": 5},
        },
        "extraction": {
            "mode": "incremental",
            "checkpoint_field": "event_date",
            "checkpoint_strategy": "max_value",
            "dead_letter_max_items": 0,
            "retry": {"max_attempts": 3, "backoff_strategy": "fixed", "backoff_seconds": 1},
        },
        "schema": {"mode": "infer"},
        "spark": {
            "input_format": "json",
            "write_mode": "append",
            "repartition": 1,
            "partition_by": [],
        },
        "outputs": {
            "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
            "bronze": {
                "path": f"data/bronze/example/{source_id}",
                "format": "iceberg",
                "namespace": "bronze_test",
                "table_name": source_id,
            },
            "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
        },
        "quality": {
            "required_fields": ["event_id", "event_date"],
            "unique_fields": ["event_id"],
            "allow_schema_evolution": True,
        },
    }
    config_path = tmp_path / "conf" / "sources" / f"{source_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return SourceConfig.from_mapping(payload, config_path)
