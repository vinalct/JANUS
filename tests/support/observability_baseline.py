"""The metadata zone exactly as it is *before* emission."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from janus.checkpoints import DeadLetterStore
from janus.cli.run import record_spark_session
from janus.lineage import RunObserver
from janus.models import WriteResult
from janus.planner import HookCatalog, Planner, PlanningRequest, StrategyBinding, StrategyCatalog
from janus.quality import PersistedValidationReport, QualityGate, ValidationReportStore
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.scripts.raw_to_bronze import RawToBronzeLoader
from janus.strategies.api import ApiHook, ApiResponse, ApiStrategy
from janus.strategies.catalog import CatalogStrategy
from janus.utils.storage import StorageLayout, bronze_table_identifier
from tests.support.contract_frames import SchemaRows
from tests.support.contracts import KEYED_CONTRACT_PATH, keyed_contract_yaml

# Fixed instants: every timestamp in a golden is derived from one of these two.
STARTED_AT = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 7, 4, 12, 0, 5, tzinfo=UTC)

PROJECT_PLACEHOLDER = "<PROJECT>"

BASELINE_CASES = (
    "api_success",
    "catalog_success",
    "extraction_failure",
    "quality_failure",
    "empty_handoff",
    "replay",
)

RECORDS = (
    {"id": "1", "title": "Fixture one", "updated_at": "2026-07-01"},
    {"id": "2", "title": "Fixture two", "updated_at": "2026-07-02"},
)


class FixedObserver(RunObserver):
    """The shipped observer with only its clock pinned."""

    def record_success(self, *args, **kwargs):
        return super().record_success(*args, **{**kwargs, "finished_at": FINISHED_AT})

    def record_failure(self, *args, **kwargs):
        return super().record_failure(*args, **{**kwargs, "finished_at": FINISHED_AT})


class FixedDeadLetters(DeadLetterStore):
    def record(self, *args, **kwargs):
        return super().record(*args, **{**kwargs, "recorded_at": FINISHED_AT})


class EmptyHandoffHook(ApiHook):
    """Drop every artifact from the handoff, so no Spark session is ever acquired."""

    def on_normalization_handoff(self, plan, extraction_result):
        return replace(extraction_result, artifacts=())


class OfflineTransport:
    """One scripted response per capture; a second request is a scripting error."""

    def __init__(self, case: str, events: list[str]) -> None:
        self.case = case
        self.events = events
        self.requests: list[str] = []

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def send(self, request):
        if self.requests:
            raise AssertionError("script exhausted: each capture has exactly one response")
        self.events.append("request")
        self.requests.append(request.full_url())
        if self.case == "extraction_failure":
            raise RuntimeError("scripted extraction failure")
        return ApiResponse(
            request=request,
            status_code=200,
            body=json.dumps(list(RECORDS)).encode(),
            received_at=STARTED_AT,
        )


class MemorySession:
    """A compute double that records its own lifetime, so ordering is observable."""

    def __init__(self, tables: dict[str, Any], events: list[str]) -> None:
        self.tables = tables
        self.events = events
        self.catalog = self
        self.events.append("session_start")

    def tableExists(self, identifier):  # the SparkSession spelling, matched deliberately
        return identifier in self.tables

    def table(self, identifier):
        return self.tables[identifier]

    def stop(self) -> None:
        self.events.append("session_stop")


class JsonReader:
    def read_extraction_result(self, spark, extraction_result, **kwargs):
        rows: list[dict[str, Any]] = []
        for artifact in extraction_result.artifacts:
            content = Path(artifact.path).read_text(encoding="utf-8")
            rows.extend(
                [json.loads(line) for line in content.splitlines() if line]
                if artifact.format == "jsonl"
                else json.loads(content)
            )
        # The rows keep their shape; the schema is the one the read was handed, as Spark applies.
        return SchemaRows(rows, schema=kwargs.get("schema"))


class IdentityNormalizer:
    def normalize(self, dataframe, plan):
        return dataframe


class MemoryWriter:
    def __init__(self, tables: dict[str, Any]) -> None:
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
            metadata={"writer": "order15-baseline-compute-double"},
        )


class BaselineQualityGate(QualityGate):
    """The shipped gate, with the Spark-bound data checks left to skip."""

    def validate_and_store(self, plan, *, write_results=(), **kwargs):
        report = replace(
            self.validate(plan, write_results=write_results), emitted_at=FINISHED_AT
        )
        return PersistedValidationReport(report, ValidationReportStore().write(plan, report))


def source_payload(case: str) -> dict[str, Any]:
    """One valid source: no credentials, no delays, no live endpoint."""

    family = "catalog" if case == "catalog_success" else "api"
    source_id = "baseline_catalog" if family == "catalog" else "baseline_api"
    payload: dict[str, Any] = {
        "source_id": source_id,
        "name": f"Baseline {family} source",
        "owner": "janus-tests",
        "enabled": True,
        "source_type": family,
        "strategy": family,
        "strategy_variant": "metadata_catalog" if family == "catalog" else "page_number_api",
        "federation_level": "federal",
        "domain": "reference",
        "public_access": True,
        "tags": ["baseline"],
        "access": {
            "base_url": "https://fixtures.invalid",
            "path": f"/{source_id}",
            "method": "GET",
            "format": "json",
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "size",
                "page_size": 10,
            },
            "rate_limit": {"concurrency": 1, "backoff_seconds": 1},
        },
        "extraction": {
            "mode": "full_refresh",
            "checkpoint_field": "updated_at",
            "checkpoint_strategy": "max_value",
            "retry": {"max_attempts": 1, "backoff_seconds": 1},
        },
        "schema": {"contract": KEYED_CONTRACT_PATH},
        "spark": {
            "input_format": "jsonl" if family == "catalog" else "json",
            "write_mode": "overwrite",
            "partition_by": ["ingestion_date"],
        },
        "outputs": {
            zone: {
                "path": f"data/{zone}/{source_id}",
                "format": "iceberg" if zone == "bronze" else "json",
            }
            for zone in ("raw", "bronze", "metadata")
        },
        "quality": {
            "required_fields": ["id"],
            "unique_fields": ["id"],
        },
    }

    if case == "quality_failure":
        payload["quality"]["required_fields"] = []
    if case == "empty_handoff":
        payload["source_hook"] = "order15.empty"
    return payload


def contract_yaml(case: str) -> str:
    """``id`` is the key; quality_failure leaves it unrequired, which the config check refuses."""
    return keyed_contract_yaml(required=() if case == "quality_failure" else None)


def write_project(root: Path, document: dict[str, Any], *, contract: str | None = None) -> None:
    sources_dir = root / "conf" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (root / "conf" / "app.yaml").write_text(
        "registry:\n  sources_dir: conf/sources\n  file_pattern: '*.yaml'\n", encoding="utf-8"
    )
    contract_path = root / KEYED_CONTRACT_PATH
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(contract or keyed_contract_yaml(), encoding="utf-8")
    (sources_dir / "sources.yaml").write_text(
        yaml.safe_dump({"sources": [document]}, sort_keys=False), encoding="utf-8"
    )
    for zone in ("raw", "bronze", "metadata"):
        (root / "data" / zone).mkdir(parents=True, exist_ok=True)


def capture_case(root: Path, case: str) -> dict[str, Any]:
    """Run one case end to end and return its manifest; JSON lands in the metadata zone."""

    document = source_payload(case)
    write_project(root, document, contract=contract_yaml(case))

    environment: dict[str, Any] = {
        "storage": {
            "root_dir": "data",
            **{f"{zone}_dir": f"data/{zone}" for zone in ("raw", "bronze", "metadata")},
        }
    }
    events: list[str] = []
    transport = OfflineTransport(case, events)
    family = document["strategy"]
    strategy_type = CatalogStrategy if family == "catalog" else ApiStrategy
    strategy = strategy_type(
        transport_factory=lambda: transport,
        sleeper=lambda _: None,
        storage_layout_factory=lambda plan: StorageLayout.from_environment_config(
            environment, root
        ),
        dead_letter_store=FixedDeadLetters(),
    )
    planner = Planner(
        strategy_catalog=StrategyCatalog(
            (StrategyBinding(family, document["strategy_variant"], strategy),)
        ),
        hook_catalog=HookCatalog((("order15.empty", EmptyHandoffHook()),)),
    )
    planned = planner.plan(
        PlanningRequest.create(
            source_id=document["source_id"],
            environment="local",
            project_root=root,
            run_id=f"order15-{case}",
            started_at=STARTED_AT,
            attributes={"trigger": "baseline"},
        )
    )

    tables: dict[str, Any] = {}
    provider = SparkSessionProvider({}, {}, session_factory=lambda: MemorySession(tables, events))
    collaborators = {
        "reader": JsonReader(),
        "normalizer": IdentityNormalizer(),
        "writer_factory": lambda _: MemoryWriter(tables),
        "quality_gate": BaselineQualityGate(),
        "observer": FixedObserver(),
    }

    if case == "replay":
        strategy.extract(planned.plan)
        transport.case = "extraction_failure"
        events.clear()
        transport.requests.clear()
        executed = RawToBronzeLoader(**collaborators).ingest(
            planned, provider, environment, bronze_table="bronze.baseline_replay"
        )
    else:
        executed = SourceExecutor(**collaborators).execute(planned, provider, environment)

    summary: dict[str, Any] = {
        "planned_run": planned.to_summary(),
        "executed_run": executed.to_summary(),
    }
    record_spark_session(summary, provider)
    return {
        "case": case,
        "run_id": planned.plan.run_context.run_id,
        "source_id": document["source_id"],
        "strategy_family": family,
        "status": executed.status,
        "entry_point": "ingest_raw_to_bronze" if case == "replay" else "execute",
        "events": list(events),
        "requests": list(transport.requests),
        "spark_session_started": provider.was_started,
        "summary": summary,
    }


def _rewrite(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(root), PROJECT_PLACEHOLDER)
    if isinstance(value, dict):
        return {key: _rewrite(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, root) for item in value]
    return value


def _checksum_substitutions(root: Path) -> dict[str, str]:
    """Verify every real sidecar, then map its digest to the root-rewritten one."""

    substitutions: dict[str, str] = {}
    for sidecar in sorted((root / "data" / "raw").rglob("*.sha256")):
        artifact = Path(str(sidecar).removesuffix(".sha256"))
        content = artifact.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        assert sidecar.read_text(encoding="utf-8").strip() == actual, (
            f"sidecar does not match its artifact: {sidecar}"
        )
        rewritten = content.replace(str(root).encode(), PROJECT_PLACEHOLDER.encode())
        if rewritten != content:
            substitutions[actual] = hashlib.sha256(rewritten).hexdigest()
    return substitutions


def _publish(root: Path, destination: Path, manifest: dict[str, Any]) -> list[str]:
    """Copy the metadata zone out as byte goldens, root-normalized, nothing else."""

    substitutions = _checksum_substitutions(root)
    metadata_root = root / "data" / "metadata"
    written: list[str] = []
    for path in sorted(metadata_root.rglob("*.json")):
        text = path.read_text(encoding="utf-8").replace(str(root), PROJECT_PLACEHOLDER)
        for actual, rewritten in substitutions.items():
            text = text.replace(actual, rewritten)
        relative = path.relative_to(metadata_root)
        target = destination / "metadata" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written.append(str(Path("metadata") / relative))

    manifest = _rewrite(manifest, root)
    for actual, rewritten in substitutions.items():
        manifest = json.loads(json.dumps(manifest).replace(actual, rewritten))
    manifest["captured_files"] = written
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return written


def capture_all(destination: Path) -> dict[str, list[str]]:
    """Regenerate every golden. Existing captures are replaced, never merged."""

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    captured: dict[str, list[str]] = {}
    for case in BASELINE_CASES:
        # A fresh root per case: replay and checkpoints must never read a prior capture.
        with TemporaryDirectory(prefix=f"order15-{case}-") as temporary:
            root = Path(temporary).resolve()
            manifest = capture_case(root, case)
            captured[case] = _publish(root, destination / case, manifest)
    return captured


def main() -> None:
    destination = Path(sys.argv[1]).resolve()
    for case, files in capture_all(destination).items():
        print(f"{case}: {len(files)} file(s)")
        for name in files:
            print(f"    {name}")


if __name__ == "__main__":
    main()
