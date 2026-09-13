"""Shared real-Iceberg harness for full-refresh history integration coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from janus.models import (
    BronzeWriteIntent,
    ExecutionPlan,
    RunContext,
    WriteResult,
    resolve_bronze_write_intent,
)
from janus.registry import SourceRegistry, load_registry
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter
from tests.support.spark_sessions import build_iceberg_session

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PROJECT_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "full_refresh_history"

ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "runtime",
        "raw_dir": "runtime/raw",
        "bronze_dir": "runtime/bronze",
        "metadata_dir": "runtime/metadata",
    }
}


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    """Provide one local Iceberg session and warehouse per test module."""
    session = build_iceberg_session(
        "janus-full-refresh-history-tests",
        tmp_path_factory.mktemp("janus-full-refresh-history-iceberg"),
    )
    yield session
    session.stop()


@pytest.fixture(scope="session")
def full_refresh_registry() -> SourceRegistry:
    """Load the two full-refresh fixture sources through the production registry."""
    return load_registry(FIXTURE_PROJECT_ROOT)


@dataclass(frozen=True)
class FullRefreshHarness:
    spark: Any
    registry: SourceRegistry
    project_root: Path
    table_name: str
    writer: SparkDatasetWriter

    def plan(
        self,
        source_id: str,
        *,
        run_id: str,
        partition_by: tuple[str, ...] | None = None,
    ) -> ExecutionPlan:
        source_config = self.registry.get_source(source_id)
        source_config = replace(
            source_config,
            outputs=replace(
                source_config.outputs,
                bronze=replace(
                    source_config.outputs.bronze,
                    namespace="bronze_full_refresh_history",
                    table_name=self.table_name,
                ),
            ),
        )
        if partition_by is not None:
            source_config = replace(
                source_config,
                spark=replace(source_config.spark, partition_by=partition_by),
            )
        run_context = RunContext.create(
            run_id=run_id,
            environment="local",
            project_root=self.project_root,
            started_at=datetime(2026, 7, 9, 12, 0, tzinfo=UTC),
        )
        return ExecutionPlan.from_source_config(source_config, run_context)

    def write_rows(
        self,
        rows: list[tuple[Any, ...]],
        schema: str,
        plan: ExecutionPlan,
        *,
        intent: BronzeWriteIntent | None = None,
    ) -> WriteResult:
        dataframe = self.spark.createDataFrame(rows, schema)
        return self.writer.write(
            dataframe,
            plan,
            "bronze",
            intent=intent or resolve_bronze_write_intent(plan),
            apply_repartition=False,
        )


@pytest.fixture
def full_refresh_harness(
    spark,
    full_refresh_registry: SourceRegistry,
    tmp_path: Path,
    request,
) -> FullRefreshHarness:
    module_name = request.node.module.__name__.rsplit(".", 1)[-1]
    raw_table_name = f"{module_name}_{request.node.name}"
    table_name = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_table_name).strip("_").lower()
    return FullRefreshHarness(
        spark=spark,
        registry=full_refresh_registry,
        project_root=tmp_path,
        table_name=table_name,
        writer=SparkDatasetWriter(
            StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path)
        ),
    )
