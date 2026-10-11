from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from janus.lineage import LineageRecord, RunMetadata
from janus.models import ExecutionPlan, ExtractionResult, RunContext
from janus.observability import RunRecord
from janus.registry import load_registry
from tests.support.observability_baseline import source_payload, write_project

NOW = datetime(2026, 10, 9, 12, tzinfo=UTC)
NEW_FIELDS = {
    "duration_budget_seconds": 50,
    "duration_budget_outcome": 51,
    "freshness_max_age_seconds": 52,
    "request_count": 53,
    "retry_count": 54,
    "attempt_count": 55,
    "dead_letter_count": 56,
}


class WarningSpy:
    def __init__(self):
        self.warnings = []

    def warning(self, event, **fields):
        self.warnings.append((event, fields))

    def info(self, event, **fields):
        pass


def record(root: Path, **overrides) -> RunRecord:
    return replace(
        RunRecord(
            run_id="run",
            source_id="fixture.api",
            source_name="Metrics fixture",
            environment="local",
            strategy_family="api",
            strategy_variant="page_number_api",
            extraction_mode="full_refresh",
            status="succeeded",
            started_at=NOW - timedelta(seconds=5),
            ended_at=NOW,
            emitted_at=NOW,
            duration_seconds=5.0,
            config_version="fixture-version",
            source_config_path=str(root / "source.yaml"),
            records_extracted=3,
            artifact_count=1,
            records_written=3,
            quality_outcome="not_run",
        ),
        **overrides,
    )


def plan(root: Path, *, operations=None, family="api"):
    document = source_payload("api_success")
    if operations is not None:
        document["operations"] = operations
    if family == "file":
        document.update(source_type="file", strategy="file", strategy_variant="static_file")
        document["access"] = {
            "url": "https://fixtures.invalid/records.json",
            "method": "GET",
            "format": "json",
            "auth": {"type": "none"},
            "pagination": {"type": "none"},
            "rate_limit": {"concurrency": 1},
        }
        document["extraction"] = {
            "mode": "snapshot",
            "checkpoint_strategy": "none",
            "retry": {"max_attempts": 1, "backoff_seconds": 1},
        }
    write_project(root, document)
    config = load_registry(root).get_source(document["source_id"])
    return ExecutionPlan.from_source_config(
        config,
        RunContext.create(
            run_id="run", environment="local", project_root=root, started_at=NOW
        ),
    )


def project(plan, *, metadata=None):
    extraction = ExtractionResult.from_plan(plan, (), metadata=metadata or {})
    run = RunMetadata.succeeded(plan, extraction, (), finished_at=NOW)
    lineage = LineageRecord.from_runtime(
        plan, status="succeeded", extraction_result=extraction, write_results=(), emitted_at=NOW
    )
    return RunRecord.from_run(run, lineage, emitted_at=NOW)


def sink(root: Path, kind: str, *, logger=None, **settings):
    from janus.observability.metrics.sink import build_metrics_sink

    return build_metrics_sink(
        {"observability": {"metrics": {"transport": kind, **settings}}},
        {"metadata_dir": root},
        logger=logger,
    )


def samples(root: Path):
    from janus.observability.metrics.sink import MetricsRunSink
    from janus.observability.metrics.transport import MetricsEmissionOutcome, MetricsEmissionResult

    captured = []

    class CaptureTransport:
        kind = "file"

        def send(self, values, *, budget_seconds):
            assert budget_seconds > 0
            captured.extend(values)
            return MetricsEmissionResult(MetricsEmissionOutcome.EMITTED, "file")

    result = MetricsRunSink(CaptureTransport()).emit(record(root), budget_seconds=1.0)
    assert result.outcome is MetricsEmissionOutcome.EMITTED
    assert captured, "a sample sweep must exercise at least one metric"
    return tuple(captured)


def write_profile(root: Path, metrics=None):
    directory = root / "conf" / "environments"
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "name": "local",
        "runtime": {"log_level": "INFO"},
        "spark": {
            "app_name": "tests",
            "master": "local[1]",
            "warehouse_dir": "data/metadata/spark-warehouse",
            "config": {},
            "iceberg": {
                "catalog_name": "janus",
                "catalog_type": "jdbc",
                "uri": f"jdbc:sqlite:{root / 'data/metadata/catalog/iceberg.db'}",
                "warehouse_dir": "data/bronze/iceberg",
            },
        },
        "storage": {
            "root_dir": "data",
            **{f"{zone}_dir": f"data/{zone}" for zone in ("raw", "bronze", "metadata")},
        },
    }
    if metrics is not None:
        config["observability"] = {"metrics": metrics}
    (directory / "local.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def exit_code(call):
    try:
        return call()
    except SystemExit as exc:
        return exc.code
