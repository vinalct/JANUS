"""Offline captures through the existing planner, executor, and replay loader.

HTTP and compute are explicit test doubles. Raw bytes, checksums, handoffs, run
metadata and lineage files are real; checkpoints are disabled in these fixtures.
This is not Spark/Iceberg evidence.
Run as ``PYTHONPATH=src python -m tests.support.orchestration_capture OUTPUT_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.checkpoints import DeadLetterStore
from janus.lineage import RunObserver
from janus.main import record_spark_session
from janus.models import WriteResult
from janus.planner import HookCatalog, Planner, PlanningRequest, StrategyBinding, StrategyCatalog
from janus.quality import PersistedValidationReport, QualityGate, ValidationReportStore
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.scripts.raw_to_bronze import RawToBronzeLoader
from janus.strategies.api import ApiHook, ApiResponse, ApiStrategy
from janus.strategies.catalog import CatalogStrategy
from janus.utils.storage import StorageLayout, bronze_table_identifier
from tests.support.orchestration import GraphCase, SourceSpec, source_documents, write_project
from tests.unit.strategies.conftest import FakeDataFrame

STARTED_AT = datetime(2026, 9, 13, 12, tzinfo=UTC)
FINISHED_AT = datetime(2026, 9, 13, 12, 0, 5, tzinfo=UTC)
CAPTURE_CASES = ("independent", "api_consumer", "catalog_consumer", "failed", "empty", "replay")


class FixedObserver(RunObserver):
    def record_success(self, *args, **kwargs):
        return super().record_success(*args, **{**kwargs, "finished_at": FINISHED_AT})

    def record_failure(self, *args, **kwargs):
        return super().record_failure(*args, **{**kwargs, "finished_at": FINISHED_AT})


class FixedDeadLetters(DeadLetterStore):
    def record(self, *args, **kwargs):
        return super().record(*args, **{**kwargs, "recorded_at": FINISHED_AT})


class EmptyHandoffHook(ApiHook):
    def on_normalization_handoff(self, plan, extraction_result):
        return replace(extraction_result, artifacts=())


class OfflineTransport:
    def __init__(self, case: str, events: list[str]):
        self.case = case
        self.events = events
        self.requests: list[str] = []

    def open(self):
        pass

    def close(self):
        pass

    def send(self, request):
        if self.requests:
            raise AssertionError("script exhausted: each capture has exactly one response")
        self.events.append("request")
        self.requests.append(request.full_url())
        if self.case == "failed":
            raise RuntimeError("scripted extraction failure")
        rows = [] if self.case == "empty" else [{"id": "1", "title": "Fixture"}]
        return ApiResponse(
            request=request,
            status_code=200,
            body=json.dumps(rows).encode(),
            received_at=STARTED_AT,
        )


class MemorySession:
    def __init__(self, tables: dict[str, Any], events: list[str]):
        self.tables = tables
        self.events = events
        self.catalog = self
        self.events.append("session_start")

    def tableExists(self, identifier):
        return identifier in self.tables

    def table(self, identifier):
        return FakeDataFrame(self.tables[identifier])

    def stop(self):
        self.events.append("session_stop")


class JsonReader:
    def read_extraction_result(self, spark, extraction_result, **kwargs):
        rows = []
        for artifact in extraction_result.artifacts:
            content = Path(artifact.path).read_text(encoding="utf-8")
            rows.extend(
                [json.loads(line) for line in content.splitlines() if line]
                if artifact.format == "jsonl"
                else json.loads(content)
            )
        return rows


class IdentityNormalizer:
    def normalize(self, dataframe, plan):
        return dataframe


class MemoryWriter:
    def __init__(self, tables: dict[str, Any]):
        self.tables = tables

    def write(self, dataframe, plan, zone, **kwargs):
        output = plan.bronze_output
        identifier = bronze_table_identifier(
            output.path,
            fallback_name=plan.source.source_id,
            namespace=output.namespace,
            table_name=output.table_name,
        )
        self.tables[identifier] = dataframe
        return WriteResult.from_plan(
            plan,
            zone,
            path=identifier,
            format_name="iceberg",
            mode="overwrite",
            records_written=len(dataframe),
            partition_by=plan.source_config.spark.partition_by,
            metadata={"writer": "order14-compute-double"},
        )


class BoundaryQualityGate:
    """Exercise real output/config checks; explicitly skip Spark data checks."""

    def validate_and_store(self, plan, *, write_results=(), **kwargs):
        report = QualityGate().validate(plan, write_results=write_results)
        report = replace(report, emitted_at=FINISHED_AT)
        return PersistedValidationReport(report, ValidationReportStore().write(plan, report))


def capture_case(root: Path, case: str) -> dict[str, Any]:
    family = "catalog" if case == "catalog_consumer" else "api"
    upstreams = ("A",) if case.endswith("consumer") else ()
    spec = SourceSpec("B" if upstreams else "C", upstreams, family=family)
    graph = GraphCase((SourceSpec("A"), spec)) if upstreams else GraphCase((spec,))
    documents = source_documents(graph, declare_upstreams=False)
    if case == "empty":
        documents[0]["source_hook"] = "order14.empty"
    write_project(root, documents)
    environment = {
        "storage": {f"{zone}_dir": f"data/{zone}" for zone in ("raw", "bronze", "metadata")}
    }
    environment["storage"]["root_dir"] = "data"
    events: list[str] = []
    transport = OfflineTransport(case, events)
    strategy_type = CatalogStrategy if family == "catalog" else ApiStrategy
    strategy = strategy_type(
        transport_factory=lambda: transport,
        sleeper=lambda _: None,
        storage_layout_factory=lambda plan: StorageLayout.from_environment_config(
            environment, root
        ),
        dead_letter_store=FixedDeadLetters(),
    )
    variant = "metadata_catalog" if family == "catalog" else "page_number_api"
    planner = Planner(
        strategy_catalog=StrategyCatalog((StrategyBinding(family, variant, strategy),)),
        hook_catalog=HookCatalog((("order14.empty", EmptyHandoffHook()),)),
    )
    planned = planner.plan(
        PlanningRequest.create(
            source_id=spec.source_id,
            environment="local",
            project_root=root,
            run_id=f"order14-{case}",
            started_at=STARTED_AT,
            attributes={"trigger": "baseline"},
        )
    )
    tables = {"bronze.a": [{"id": "1"}]} if upstreams else {}
    provider = SparkSessionProvider({}, {}, session_factory=lambda: MemorySession(tables, events))
    collaborators = {
        "reader": JsonReader(),
        "normalizer": IdentityNormalizer(),
        "writer_factory": lambda _: MemoryWriter(tables),
        "quality_gate": BoundaryQualityGate(),
        "observer": FixedObserver(),
    }
    if case == "replay":
        # Prepare real raw artifacts with the existing extraction path, then forbid HTTP.
        strategy.extract(planned.plan)
        transport.case = "failed"
        events.clear()
        transport.requests.clear()
        executed = RawToBronzeLoader(**collaborators).ingest(
            planned, provider, environment, bronze_table="bronze.replayed"
        )
    else:
        executed = SourceExecutor(**collaborators).execute(planned, provider, environment)
    summary = {"planned_run": planned.to_summary(), "executed_run": executed.to_summary()}
    record_spark_session(summary, provider)
    persisted = {
        str(path.relative_to(root)): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "data" / "metadata").rglob("*.json"))
    }
    captured = _portable(
        {
            "evidence": (
                "offline boundary capture; HTTP and compute doubles; no Spark or Iceberg commit"
            ),
            "summary": summary,
            "persisted": persisted,
            "requests": transport.requests,
            "events": events,
            "compute_double_tables": tables,
        },
        root,
    )

    # Catalog handoffs embed their absolute raw path. Verify real sidecars first,
    # then replace only those content hashes with hashes of root-normalized bytes.
    serialized = json.dumps(captured, sort_keys=True)
    for sidecar in sorted((root / "data" / "raw").rglob("*.sha256")):
        artifact = Path(str(sidecar).removesuffix(".sha256"))
        content = artifact.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        assert sidecar.read_text().strip() == actual
        portable = content.replace(str(root).encode(), b"<PROJECT>")
        serialized = serialized.replace(actual, hashlib.sha256(portable).hexdigest())
    return json.loads(serialized)


def _portable(value: Any, root: Path) -> Any:
    """Normalize only the isolated root; retain every field and actual checksum."""
    if isinstance(value, str):
        return value.replace(str(root), "<PROJECT>")
    if isinstance(value, dict):
        return {key: _portable(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item, root) for item in value]
    return value


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    # A fresh root is required: replay and checkpoints must never consume a prior capture.
    output.mkdir(parents=True, exist_ok=False)
    for case in CAPTURE_CASES:
        capture = capture_case(output / case, case)
        (output / f"{case}.json").write_text(
            json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
