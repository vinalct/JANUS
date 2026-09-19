"""The runs sink commits through PyIceberg into the catalog Spark reads."""

from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime
from queue import Empty
from typing import Any
from uuid import uuid4

from janus.observability import IcebergAppendOutcome, RunRecord, append_run_record
from janus.utils.catalog_properties import (
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
)
from tests.support.spark_sessions import CatalogTarget, require_pyiceberg

PROCESS_RESULT_TIMEOUT_SECONDS = 15
PROCESS_EXIT_TIMEOUT_SECONDS = 20


def _record(run_id: str) -> RunRecord:
    instant = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    return RunRecord(
        run_id=run_id,
        source_id="runs_sink_source",
        source_name="Runs sink source",
        environment="local",
        strategy_family="api",
        strategy_variant="paginated",
        extraction_mode="incremental",
        status="succeeded",
        started_at=instant,
        ended_at=instant,
        emitted_at=instant,
        duration_seconds=0.0,
        config_version="sha256:runs-sink",
        source_config_path="conf/sources/runs_sink.yaml",
        records_extracted=1,
        artifact_count=1,
        records_written=1,
        bronze_table_identifier="bronze.runs_sink_source",
        bronze_write_mode="append",
        quality_outcome="not_run",
    )


def _isolated_config(catalog_target: CatalogTarget) -> tuple[dict[str, Any], str]:
    identifier = f"metadata.runs_sink_{uuid4().hex}"
    config = catalog_target.environment_config()
    config["observability"] = {"runs_table": identifier}
    return config, identifier


def _append_in_process(
    run_id: str,
    config: dict[str, Any],
    resolved_paths: dict[str, Any],
    result_queue: Any,
) -> None:
    result = append_run_record(_record(run_id), config, resolved_paths)
    result_queue.put((result.outcome, result.reason, result.step, result.exception_type))


def _run_process_append(
    run_id: str,
    config: dict[str, Any],
    resolved_paths: dict[str, Any],
) -> tuple[str, str | None, str | None, str | None]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_append_in_process,
        args=(run_id, config, resolved_paths, result_queue),
    )
    process.start()
    try:
        result = result_queue.get(timeout=PROCESS_RESULT_TIMEOUT_SECONDS)
    except Empty:
        process.terminate()
        process.join(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
        raise AssertionError("runs sink subprocess produced no result") from None
    process.join(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
    assert process.exitcode == 0
    return result


def _load_table(config: dict[str, Any], catalog_target: CatalogTarget):
    catalog_module = require_pyiceberg()
    catalog = catalog_module.load_catalog(
        derive_pyiceberg_catalog_name(config),
        **derive_pyiceberg_catalog_properties(config, catalog_target.resolved_paths),
    )
    return catalog.load_table(config["observability"]["runs_table"])


def test_two_calls_bootstrap_once_append_twice_and_spark_reads_both(
    catalog_target: CatalogTarget,
    shared_catalog_session,
):
    config, identifier = _isolated_config(catalog_target)

    first = append_run_record(
        _record("same-process-001"), config, catalog_target.resolved_paths
    )
    second = append_run_record(
        _record("same-process-002"), config, catalog_target.resolved_paths
    )

    assert first.outcome is IcebergAppendOutcome.EMITTED
    assert second.outcome is IcebergAppendOutcome.EMITTED
    table = _load_table(config, catalog_target)
    assert len(table.metadata.snapshots) == 2
    assert [field.name for field in table.spec().fields] == ["emitted_at_day"]

    rows = shared_catalog_session.newSession().table(identifier).orderBy("run_id").collect()
    assert [row["run_id"] for row in rows] == ["same-process-001", "same-process-002"]


def test_bootstrap_and_append_are_idempotent_across_processes(
    catalog_target: CatalogTarget,
):
    config, _identifier = _isolated_config(catalog_target)

    first = _run_process_append("process-001", config, catalog_target.resolved_paths)
    second = _run_process_append("process-002", config, catalog_target.resolved_paths)

    assert first == (IcebergAppendOutcome.EMITTED, None, None, None)
    assert second == (IcebergAppendOutcome.EMITTED, None, None, None)
    table = _load_table(config, catalog_target)
    rows = sorted(row["run_id"] for row in table.scan().to_arrow().to_pylist())
    assert rows == ["process-001", "process-002"]
    assert len(table.metadata.snapshots) == 2
