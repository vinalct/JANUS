"""Real Spark/Iceberg evidence for dependency ordering and lifecycle compatibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from janus.lineage import RunObserver
from janus.normalizers import NORMALIZATION_METADATA_COLUMNS
from janus.orchestration import BatchPlanner, BatchPlanRequest
from janus.planner import HookCatalog, Planner, StrategyBinding, StrategyCatalog
from janus.quality import QualityGate, ValidationReportStore
from janus.registry import load_registry
from janus.runtime import BatchExecutor, SourceExecutionService, SourceExecutor
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.strategies.catalog import CatalogStrategy
from janus.utils.storage import StorageLayout
from tests.support.contracts import DECLARED_CONTRACT_PATH
from tests.support.orchestration import GraphCase, SourceSpec, source_documents, write_project
from tests.support.orchestration_capture import EmptyHandoffHook
from tests.support.spark_sessions import (
    CatalogTarget,
    build_iceberg_session,
    catalog_target_params,
    require_iceberg_runtime,
    start_session,
)

PLANNED_AT = datetime(2026, 9, 15, 12, tzinfo=UTC)
A_ROWS = ({"id": "a-2", "value": 20}, {"id": "a-1", "value": 10})
C_ROWS = ({"id": "c-1", "value": 30},)


@dataclass(slots=True)
class _FixtureTransport:
    respond: Callable[[str, Mapping[str, list[str]]], Any]
    events: list[str]
    active_sources: set[str]
    requests: list[tuple[str, dict[str, list[str]]]] = field(default_factory=list)

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def send(self, request):
        assert self.active_sources == set(), (
            f"HTTP request {request.full_url()} ran while compute was active for "
            f"{sorted(self.active_sources)}"
        )
        parsed = urlsplit(request.full_url())
        source_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        params = parse_qs(parsed.query)
        self.requests.append((source_id, params))
        self.events.append(f"http:{source_id}")
        payload = self.respond(source_id, params)
        return ApiResponse(
            request=request,
            status_code=200,
            body=json.dumps(payload).encode("utf-8"),
            received_at=PLANNED_AT,
        )


class _TrackingProvider(SparkSessionProvider):
    def __init__(
        self,
        source_id: str,
        session_factory,
        events: list[str],
        active_sources: set[str],
        provider_config: Mapping[str, Any],
        provider_paths: Mapping[str, Any],
    ) -> None:
        super().__init__(provider_config, provider_paths, session_factory=session_factory)
        self.source_id = source_id
        self.events = events
        self.active_sources = active_sources
        self.session_number = 0
        self.live = False

    def get(self):
        session = super().get()
        if not self.live:
            self.live = True
            self.session_number += 1
            self.active_sources.add(self.source_id)
            self.events.append(f"session_start:{self.source_id}:{self.session_number}")
        return session

    def stop(self) -> None:
        was_live = self.live
        super().stop()
        if was_live:
            self.live = False
            self.active_sources.discard(self.source_id)
            self.events.append(f"session_stop:{self.source_id}:{self.session_number}")


@dataclass(slots=True)
class _LifecycleQualityGate:
    active_sources: set[str]
    events: list[str]
    delegate: QualityGate = field(default_factory=lambda: QualityGate(ValidationReportStore()))

    def validate_and_store(self, plan, *, write_results=(), **kwargs):
        source_id = plan.source.source_id
        if any(result.zone == "bronze" for result in write_results):
            assert source_id in self.active_sources
            self.events.append(f"quality_with_compute:{source_id}")
        else:
            self.events.append(f"quality_without_compute:{source_id}")
        return self.delegate.validate_and_store(plan, write_results=write_results, **kwargs)


@dataclass(slots=True)
class _LifecycleObserver(RunObserver):
    active_sources: set[str] = field(default_factory=set)
    events: list[str] = field(default_factory=list)

    def record_success(self, plan, *args, **kwargs):
        assert self.active_sources == set()
        self.events.append(f"observer_finalized:{plan.source.source_id}:succeeded")
        return RunObserver.record_success(self, plan, *args, **kwargs)

    def record_failure(self, plan, *args, **kwargs):
        assert self.active_sources == set()
        self.events.append(f"observer_finalized:{plan.source.source_id}:failed")
        return RunObserver.record_failure(self, plan, *args, **kwargs)


@dataclass(slots=True)
class _EvidenceExecution:
    executor: SourceExecutor
    session_factory: Callable[[], Any]
    events: list[str]
    active_sources: set[str]
    provider_config: Mapping[str, Any] = field(default_factory=dict)
    provider_paths: Mapping[str, Any] = field(default_factory=dict)
    raise_sources: set[str] = field(default_factory=set)
    provider_acquisitions: list[str] = field(default_factory=list)

    def execute(self, planned_run, environment_config, resolved_paths):
        source_id = planned_run.plan.source.source_id
        if source_id in self.raise_sources:
            raise RuntimeError(f"raised source failure for {source_id}")

        provider = _TrackingProvider(
            source_id,
            self.session_factory,
            self.events,
            self.active_sources,
            self.provider_config,
            self.provider_paths,
        )
        self.provider_acquisitions.append(source_id)
        service = SourceExecutionService(
            executor=self.executor,
            provider_factory=lambda *_args: provider,
        )
        executed = service.execute(planned_run, environment_config, resolved_paths)
        if any(result.zone == "bronze" for result in executed.write_results):
            self.events.append(f"bronze_committed:{source_id}")
        return executed


@dataclass(frozen=True, slots=True)
class _HarnessResult:
    outcome: Any
    plan: Any
    registry: Any
    execution: _EvidenceExecution
    transport: _FixtureTransport
    events: list[str]
    tables: Mapping[str, str]


@pytest.mark.parametrize("target_factory", catalog_target_params())
@pytest.mark.parametrize("consumer_family", ("api", "catalog"))
def test_fresh_upstream_commit_is_read_by_api_and_catalog_consumers(
    tmp_path,
    target_factory,
    consumer_family,
):
    """A's real commit precedes B's lookup, HTTP calls, and bronze write."""

    root = tmp_path / f"fresh-{consumer_family}"
    target = target_factory(root / "catalog")
    target.prepare()

    def session_factory():
        return start_session(
            f"janus-orchestration-{consumer_family}-{target.id}",
            target.session_options(),
        )

    suffix = _table_suffix(root)
    specs = (
        SourceSpec("B", ("A",), family=consumer_family, table_name=f"b_{suffix}"),
        SourceSpec("A", table_name=f"a_{suffix}"),
    )

    def respond(source_id, params):
        if source_id == "A":
            return list(A_ROWS)
        if source_id == "B":
            upstream_id = params["a_id"][0]
            if consumer_family == "catalog":
                return [{"id": f"dataset-{upstream_id}", "title": f"Dataset {upstream_id}"}]
            return [{"upstream_id": upstream_id, "consumer": consumer_family}]
        raise AssertionError(f"unexpected source request {source_id}")

    result = _execute_graph(root, specs, session_factory, respond)
    sources = _sources(result.outcome)

    assert result.plan.source_ids == ("A", "B")
    assert result.outcome.is_successful
    assert {source_id: source.status for source_id, source in sources.items()} == {
        "A": "succeeded",
        "B": "succeeded",
    }
    assert _request_values(result.transport, "B", "a_id") == ["a-1", "a-2"]
    assert _table_rows(session_factory, result.tables["A"], "id", "value") == [
        ("a-1", 10),
        ("a-2", 20),
    ]
    schema = _table_schema(session_factory, result.tables["A"])
    assert schema["id"] == "string"
    assert schema["value"] == "bigint"
    assert set(NORMALIZATION_METADATA_COLUMNS).issubset(schema)
    assert _partition_fields(session_factory, result.tables["A"]) == {"ingestion_date"}
    a_evidence = sources["A"].attempts[0].to_summary()["evidence"]
    a_bronze = next(
        output for output in a_evidence["materialized_outputs"] if output["zone"] == "bronze"
    )
    assert {
        "path": a_bronze["path"],
        "format": a_bronze["format"],
        "mode": a_bronze["mode"],
        "records_written": a_bronze["records_written"],
        "partition_by": a_bronze["partition_by"],
    } == {
        "path": result.tables["A"],
        "format": "iceberg",
        "mode": "overwrite",
        "records_written": len(A_ROWS),
        "partition_by": ["ingestion_date"],
    }
    assert a_evidence["validation"]["is_successful"] is True
    assert a_evidence["checkpoint_value"] is None
    metadata_outputs = a_evidence["metadata_outputs"]
    assert metadata_outputs["checkpoint_state_path"] is None
    assert metadata_outputs["checkpoint_history_path"] is None
    for key in ("run_metadata_path", "lineage_path", "validation_report_path"):
        assert Path(metadata_outputs[key]).is_file()
    lineage = json.loads(Path(metadata_outputs["lineage_path"]).read_text(encoding="utf-8"))
    assert lineage["config_version"] == sources["A"].config_version
    assert lineage["status"] == "succeeded"
    assert a_evidence["strategy_metadata"]["strategy_family"] == "api"
    assert _table_count(session_factory, result.tables["B"]) == 2
    if consumer_family == "api":
        assert _table_rows(session_factory, result.tables["B"], "upstream_id", "consumer") == [
            ("a-1", "api"),
            ("a-2", "api"),
        ]

    commit = result.events.index("bronze_committed:A")
    lookup = result.events.index("session_start:B:1")
    lookup_stop = result.events.index("session_stop:B:1")
    first_request = result.events.index("http:B")
    materialize = result.events.index("session_start:B:2")
    assert commit < lookup < lookup_stop < first_request < materialize
    _assert_lifecycle(result, attempted=("A", "B"))


def test_combined_fan_in_waits_for_both_commits_and_ignores_date_windows_in_graph(tmp_path):
    require_iceberg_runtime()
    root = tmp_path / "fan-in"
    suffix = _table_suffix(root)
    specs = (
        SourceSpec("B", ("A", "C"), table_name=f"b_{suffix}"),
        SourceSpec("C", table_name=f"c_{suffix}"),
        SourceSpec("A", table_name=f"a_{suffix}"),
    )
    documents = source_documents(GraphCase(specs))
    consumer = next(document for document in documents if document["source_id"] == "B")
    iceberg_inputs = consumer["access"]["request_inputs"]["inputs"]
    consumer["access"]["request_inputs"] = {
        "type": "combined",
        "inputs": [
            {
                "type": "date_window",
                "start": "2026-09-14",
                "end": "2026-09-15",
                "step": "day",
            },
            *iceberg_inputs,
        ],
    }
    consumer["access"]["parameter_bindings"]["day"] = {"from": "request_input.window_start"}

    def session_factory():
        return build_iceberg_session(
            "janus-orchestration-fan-in",
            root / "catalog",
        )

    def respond(source_id, params):
        if source_id == "A":
            return [A_ROWS[0]]
        if source_id == "C":
            return list(C_ROWS)
        if source_id == "B":
            return [
                {
                    "a_id": params["a_id"][0],
                    "c_id": params["c_id"][0],
                    "day": params["day"][0],
                }
            ]
        raise AssertionError(f"unexpected source request {source_id}")

    result = _execute_graph(root, specs, session_factory, respond, documents=documents)

    assert result.plan.source_ids == ("A", "C", "B")
    assert {(edge.producer_id, edge.consumer_id) for edge in result.plan.edges} == {
        ("A", "B"),
        ("C", "B"),
    }
    b_requests = [params for source_id, params in result.transport.requests if source_id == "B"]
    assert [params["day"][0] for params in b_requests] == ["2026-09-14", "2026-09-15"]
    assert all(params["a_id"] == ["a-2"] for params in b_requests)
    assert all(params["c_id"] == ["c-1"] for params in b_requests)
    lookup = result.events.index("session_start:B:1")
    assert result.events.index("bronze_committed:A") < lookup
    assert result.events.index("bronze_committed:C") < lookup
    assert _table_rows(session_factory, result.tables["B"], "a_id", "c_id", "day") == [
        ("a-2", "c-1", "2026-09-14"),
        ("a-2", "c-1", "2026-09-15"),
    ]


@pytest.mark.parametrize("failure_mode", ("returned", "quality", "raised"))
def test_all_failure_forms_block_descendants_while_independent_source_commits(
    tmp_path,
    failure_mode,
):
    require_iceberg_runtime()
    root = tmp_path / f"failure-{failure_mode}"
    suffix = _table_suffix(root)
    specs = (
        SourceSpec("D", ("B",), table_name=f"d_{suffix}"),
        SourceSpec("C", table_name=f"c_{suffix}"),
        SourceSpec("B", ("A",), table_name=f"b_{suffix}"),
        SourceSpec("A", table_name=f"a_{suffix}"),
    )
    documents = source_documents(GraphCase(specs))
    if failure_mode == "quality":
        producer = next(document for document in documents if document["source_id"] == "A")
        producer["quality"] = {
            "required_fields": ["janus_source_id"],
            "unique_fields": ["janus_source_id"],
        }

    def session_factory():
        return build_iceberg_session(
            f"janus-orchestration-failure-{failure_mode}",
            root / "catalog",
        )

    def respond(source_id, _params):
        if source_id == "A" and failure_mode == "returned":
            raise RuntimeError("returned extraction failure for A")
        if source_id == "A":
            return list(A_ROWS)
        if source_id == "C":
            return list(C_ROWS)
        raise AssertionError(f"blocked source {source_id} reached HTTP")

    result = _execute_graph(
        root,
        specs,
        session_factory,
        respond,
        documents=documents,
        raise_sources={"A"} if failure_mode == "raised" else set(),
    )
    sources = _sources(result.outcome)

    assert {source_id: source.status for source_id, source in sources.items()} == {
        "A": "failed",
        "B": "skipped",
        "C": "succeeded",
        "D": "skipped",
    }
    assert result.execution.provider_acquisitions == (
        ["C"] if failure_mode == "raised" else ["A", "C"]
    )
    assert sources["B"].skip.direct_blocking_upstream_ids == ("A",)
    assert sources["D"].skip.root_failed_source_ids == ("A",)
    assert _table_rows(session_factory, result.tables["C"], "id", "value") == [("c-1", 30)]
    if failure_mode == "quality":
        assert sources["A"].attempts[0].evidence["validation"]["is_successful"] is False
        assert _table_count(session_factory, result.tables["A"]) == len(A_ROWS)
    assert not any(source_id in {"B", "D"} for source_id, _params in result.transport.requests)


@pytest.mark.parametrize("target_factory", catalog_target_params())
def test_refused_contract_preflight_skips_dependent_without_extraction(
    tmp_path, target_factory
):
    root = tmp_path / "preflight-refusal"
    target = target_factory(root / "catalog")
    target.prepare()
    suffix = _table_suffix(root)
    specs = (
        SourceSpec("B", ("A",), table_name=f"b_{suffix}"),
        SourceSpec("A", table_name=f"a_{suffix}"),
    )
    tables = _table_identifiers(specs)

    def session_factory():
        return start_session(
            f"janus-orchestration-preflight-{target.id}", target.session_options()
        )

    _seed_table(
        session_factory,
        tables["A"],
        [("seed", 1, None, None, None, None, None, None)],
        "id string, value bigint, upstream_id string, consumer string, "
        "title string, a_id string, c_id string, day string",
    )
    def add_stray(spark):
        spark.sql(
            f"ALTER TABLE {tables['A']} "
            "SET TBLPROPERTIES ('janus.contract_version' = '1.0.0')"
        )
        spark.sql(f"ALTER TABLE {tables['A']} ADD COLUMNS (stray string)")

    _with_session(session_factory, add_stray)

    def reject_request(source_id, _params):
        raise AssertionError(f"refused batch reached HTTP for {source_id}")

    result = _execute_graph(
        root,
        specs,
        session_factory,
        reject_request,
        catalog_target=target,
        contract_enforcement="strict",
    )
    summary = result.outcome.to_summary()
    sources = {source["source_id"]: source for source in summary["sources"]}
    a, b = sources["A"], sources["B"]

    assert a["status"] == "failed"
    assert a["failure"]["error_type"] == "ContractPreflightError"
    assert b["status"] == "skipped"
    assert b["skip"]["reason_code"] == "upstream_failed"
    assert b["skip"]["direct_blocking_upstream_ids"] == ["A"]
    assert result.transport.requests == []
    raw_root = root / "data" / "raw" / "A"
    assert not raw_root.exists() or not any(path.is_file() for path in raw_root.rglob("*"))
    assert not any(event.startswith("session_start:A") for event in result.events)
    assert result.execution.provider_acquisitions == ["A"]
    evidence = a["attempts"][0]["evidence"]
    metadata_path = Path(evidence["metadata_outputs"]["run_metadata_path"])
    assert metadata_path.is_file()
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["run_attributes"][
        "contract_preflight_outcome"
    ] == "refused"


def test_failed_current_upstream_blocks_consumer_even_when_stale_table_exists(tmp_path):
    require_iceberg_runtime()
    root = tmp_path / "stale"
    suffix = _table_suffix(root)
    specs = (
        SourceSpec("B", ("A",), table_name=f"b_{suffix}"),
        SourceSpec("C", table_name=f"c_{suffix}"),
        SourceSpec("A", table_name=f"a_{suffix}"),
    )
    tables = _table_identifiers(specs)

    def session_factory():
        return build_iceberg_session(
            "janus-orchestration-stale",
            root / "catalog",
        )

    _seed_table(session_factory, tables["A"], [("stale-a",)], "id string")

    def respond(source_id, _params):
        if source_id == "A":
            raise RuntimeError("current A extraction failed")
        if source_id == "C":
            return list(C_ROWS)
        raise AssertionError("B must not read a stale upstream table")

    result = _execute_graph(root, specs, session_factory, respond)
    sources = _sources(result.outcome)

    assert sources["A"].status == "failed"
    assert sources["B"].status == "skipped"
    assert sources["B"].skip.root_failed_source_ids == ("A",)
    assert sources["C"].status == "succeeded"
    assert _table_rows(session_factory, tables["A"], "id") == [("stale-a",)]
    assert not any(source_id == "B" for source_id, _params in result.transport.requests)
    assert result.execution.provider_acquisitions == ["A", "C"]


def test_empty_success_does_not_fabricate_commit_and_missing_consumer_table_is_attributable(
    tmp_path,
):
    require_iceberg_runtime()
    root = tmp_path / "empty"
    suffix = _table_suffix(root)
    specs = (
        SourceSpec("B", ("A",), table_name=f"b_{suffix}"),
        SourceSpec("A", table_name=f"a_{suffix}"),
    )
    documents = source_documents(GraphCase(specs))
    producer = next(document for document in documents if document["source_id"] == "A")
    producer["source_hook"] = "task10.empty"

    def session_factory():
        return build_iceberg_session(
            "janus-orchestration-empty",
            root / "catalog",
        )

    def respond(source_id, _params):
        if source_id == "A":
            return [A_ROWS[0]]
        raise AssertionError("B must fail its lookup before HTTP extraction")

    result = _execute_graph(
        root,
        specs,
        session_factory,
        respond,
        documents=documents,
        hooks=(("task10.empty", EmptyHandoffHook()),),
    )
    sources = _sources(result.outcome)
    a_evidence = sources["A"].attempts[0].evidence

    assert sources["A"].status == "succeeded"
    assert not any(item["zone"] == "bronze" for item in a_evidence["materialized_outputs"])
    assert sources["B"].status == "failed"
    assert "does not exist" in sources["B"].failure.reason
    assert sources["B"].failure.error_type == "ApiRequestInputLoadError"
    assert result.execution.provider_acquisitions == ["A", "B"]
    assert not _table_exists(session_factory, result.tables["A"])
    assert not any(source_id == "B" for source_id, _params in result.transport.requests)


def _execute_graph(
    root: Path,
    specs: tuple[SourceSpec, ...],
    session_factory,
    respond,
    *,
    documents: list[dict[str, Any]] | None = None,
    hooks=(),
    raise_sources: set[str] | None = None,
    catalog_target: CatalogTarget | None = None,
    contract_enforcement: str | None = None,
) -> _HarnessResult:
    write_project(root, documents or source_documents(GraphCase(specs)))
    contract_path = root / DECLARED_CONTRACT_PATH
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    for name, physical_type, logical_type in (
        ("value", "long", "integer"),
        ("upstream_id", "string", "string"),
        ("consumer", "string", "string"),
        ("title", "string", "string"),
        ("a_id", "string", "string"),
        ("c_id", "string", "string"),
        ("day", "string", "string"),
    ):
        contract["schema"][0]["properties"].append(
            {"name": name, "logicalType": logical_type, "physicalType": physical_type}
        )
    if contract_enforcement is not None:
        next(
            item for item in contract["customProperties"]
            if item["property"] == "janus.enforcement"
        )["value"] = contract_enforcement
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    environment_config = {
        "name": "local",
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        },
    }
    if catalog_target is not None:
        environment_config.update(catalog_target.environment_config())
    resolved_paths = catalog_target.resolved_paths if catalog_target is not None else {}
    events: list[str] = []
    active_sources: set[str] = set()
    transport = _FixtureTransport(respond, events, active_sources)
    layout = StorageLayout.from_environment_config(environment_config, root)
    bindings = (
        StrategyBinding(
            "api",
            "page_number_api",
            ApiStrategy(
                transport_factory=lambda: transport,
                storage_layout_factory=lambda _plan: layout,
                sleeper=lambda _seconds: None,
                clock=lambda: 0.0,
            ),
        ),
        StrategyBinding(
            "catalog",
            "metadata_catalog",
            CatalogStrategy(
                transport_factory=lambda: transport,
                storage_layout_factory=lambda _plan: layout,
                sleeper=lambda _seconds: None,
                clock=lambda: 0.0,
            ),
        ),
    )
    planner = Planner(
        strategy_catalog=StrategyCatalog(bindings),
        hook_catalog=HookCatalog(tuple(hooks)),
    )
    registry = load_registry(root)
    plan = BatchPlanner(planner=planner).plan(
        BatchPlanRequest.create(
            environment="local",
            project_root=root,
            pipeline_run_id=f"task10-{_table_suffix(root)}",
            planned_at=PLANNED_AT,
        ),
        registry=registry,
    )
    observer = _LifecycleObserver(active_sources=active_sources, events=events)
    quality = _LifecycleQualityGate(active_sources, events)
    executor = SourceExecutor(observer=observer, quality_gate=quality)
    execution = _EvidenceExecution(
        executor,
        session_factory,
        events,
        active_sources,
        environment_config if catalog_target is not None else {},
        resolved_paths,
        raise_sources or set(),
    )
    outcome = BatchExecutor(source_execution=execution).execute(
        plan,
        registry,
        environment_config,
        resolved_paths,
    )
    return _HarnessResult(
        outcome=outcome,
        plan=plan,
        registry=registry,
        execution=execution,
        transport=transport,
        events=events,
        tables=_table_identifiers(specs),
    )


def _table_suffix(root: Path) -> str:
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]


def _table_identifiers(specs: tuple[SourceSpec, ...]) -> dict[str, str]:
    return {
        spec.source_id: f"bronze.{spec.table_name}" for spec in specs if spec.table_name is not None
    }


def _sources(outcome) -> dict[str, Any]:
    return {source.source_id: source for source in outcome.sources}


def _request_values(transport: _FixtureTransport, source_id: str, parameter: str) -> list[str]:
    return [
        params[parameter][0]
        for requested_source, params in transport.requests
        if requested_source == source_id
    ]


def _assert_lifecycle(result: _HarnessResult, *, attempted: tuple[str, ...]) -> None:
    assert result.execution.provider_acquisitions == list(attempted)
    assert result.execution.active_sources == set()
    for source_id in attempted:
        quality = result.events.index(f"quality_with_compute:{source_id}")
        stop = max(
            index
            for index, event in enumerate(result.events)
            if event.startswith(f"session_stop:{source_id}:")
        )
        observer = result.events.index(f"observer_finalized:{source_id}:succeeded")
        assert quality < stop < observer


def _with_session(session_factory, operation):
    session = session_factory()
    try:
        return operation(session)
    finally:
        session.stop()


def _table_rows(session_factory, table: str, *columns: str) -> list[tuple[Any, ...]]:
    return _with_session(
        session_factory,
        lambda session: sorted(
            tuple(row[column] for column in columns)
            for row in session.table(table).select(*columns).collect()
        ),
    )


def _table_schema(session_factory, table: str) -> dict[str, str]:
    return _with_session(
        session_factory,
        lambda session: {
            field.name: field.dataType.simpleString()
            for field in session.table(table).schema.fields
        },
    )


def _partition_fields(session_factory, table: str) -> set[str]:
    def inspect(session):
        fields = session.table(f"{table}.partitions").schema.fields
        partition = next(field for field in fields if field.name == "partition")
        return {field.name for field in partition.dataType.fields}

    return _with_session(session_factory, inspect)


def _table_count(session_factory, table: str) -> int:
    return _with_session(session_factory, lambda session: session.table(table).count())


def _table_exists(session_factory, table: str) -> bool:
    return _with_session(session_factory, lambda session: session.catalog.tableExists(table))


def _seed_table(session_factory, table: str, rows: list[tuple], schema: str) -> None:
    def seed(session):
        namespace = table.rsplit(".", 1)[0]
        session.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")
        session.createDataFrame(rows, schema).writeTo(table).using("iceberg").create()

    _with_session(session_factory, seed)
