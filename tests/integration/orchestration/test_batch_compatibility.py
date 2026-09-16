"""Offline integration of batch planning/execution with frozen single-source evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from janus.orchestration import BatchPlanner, BatchPlanRequest
from janus.planner import Planner, StrategyBinding, StrategyCatalog
from janus.registry import load_registry
from janus.runtime import BatchExecutor, SourceExecutionService, SourceExecutor
from janus.runtime.spark_lifecycle import SparkSessionProvider
from janus.strategies.api import ApiStrategy
from janus.strategies.catalog import CatalogStrategy
from janus.utils.storage import StorageLayout
from tests.support.orchestration import (
    GRAPH_CASES,
    GraphCase,
    SourceSpec,
    build_graph_project,
    source_documents,
    write_project,
)
from tests.support.orchestration_capture import (
    BoundaryQualityGate,
    FixedObserver,
    IdentityNormalizer,
    JsonReader,
    MemorySession,
    MemoryWriter,
    OfflineTransport,
)

BASELINE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "orchestration" / "baseline"
PLANNED_AT = datetime(2026, 9, 13, 12, tzinfo=UTC)
ENVIRONMENT_CONFIG = {
    "name": "local",
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    },
}


class _FakeExecutedRun:
    status = "succeeded"
    failure_reason = None
    error_type = None

    @property
    def is_successful(self) -> bool:
        return True

    def to_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "strategy_metadata": {},
            "materialized_outputs": [],
            "metadata_outputs": {},
        }


class _NoopProvider:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1

    def take_cleanup_failures(self) -> tuple[Exception, ...]:
        return ()


class _ProfileRecordingExecutor:
    logger = None

    def __init__(self) -> None:
        self.environment_config: Mapping[str, Any] | None = None
        self.provider: Any = None

    def execute(self, planned_run, provider, environment_config):
        self.environment_config = environment_config
        self.provider = provider
        return _FakeExecutedRun()


@pytest.mark.parametrize("case", ("independent", "api_consumer", "catalog_consumer"))
def test_batch_sources_retain_the_frozen_single_source_contract(tmp_path, case):
    root = tmp_path / case
    outcome, tables = _capture_batch(root, case)
    baseline = json.loads((BASELINE_DIR / f"{case}.json").read_text(encoding="utf-8"))
    source_id = "B" if case.endswith("consumer") else "C"
    source = next(item for item in outcome.sources if item.source_id == source_id)
    actual = source.attempts[-1].to_summary()["evidence"]
    expected = baseline["summary"]["executed_run"]

    assert source.status == expected["status"] == "succeeded"
    assert _bronze_output(actual) == _bronze_output(expected)
    actual_rows = tables[_bronze_output(actual)["path"]]
    expected_rows = baseline["compute_double_tables"][_bronze_output(expected)["path"]]

    # Catalog provenance carries the attempt-specific raw path; root and run id are
    # already documented exclusions, while every other row value remains comparable.
    def stable_rows(rows):
        return [
            {key: value for key, value in row.items() if key != "catalog_raw_artifact_path"}
            for row in rows
        ]

    assert stable_rows(actual_rows) == stable_rows(expected_rows)
    assert actual["validation"] == expected["validation"]
    assert actual["checkpoint_value"] == expected["checkpoint_value"]
    assert _metadata_presence(actual) == _metadata_presence(expected)
    assert _normalized_strategy_metadata(actual, source.run_id) == _normalized_strategy_metadata(
        expected,
        baseline["summary"]["planned_run"]["run"]["run_id"],
    )

    expected_lineage = next(
        value for path, value in baseline["persisted"].items() if "/lineage/" in path
    )
    actual_lineage = json.loads(
        Path(actual["metadata_outputs"]["lineage_path"]).read_text(encoding="utf-8")
    )
    assert dict(outcome.config_versions)[source_id] == expected_lineage["config_version"]
    assert {
        key: actual_lineage[key]
        for key in (
            "source_id",
            "status",
            "strategy_family",
            "strategy_variant",
            "extraction_mode",
            "checkpoint_strategy",
            "records_extracted",
            "config_version",
        )
    } == {
        key: expected_lineage[key]
        for key in (
            "source_id",
            "status",
            "strategy_family",
            "strategy_variant",
            "extraction_mode",
            "checkpoint_strategy",
            "records_extracted",
            "config_version",
        )
    }


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cycle", "cycle"),
        ("self_cycle", "cycle"),
        ("missing_producer", "missing from the registry"),
        ("disabled_producer", "disabled"),
        ("duplicate_producer", "ambiguous producer target"),
        ("wrong_declared_table", "produces 'bronze.a'"),
    ),
)
def test_invalid_registry_prevents_source_network_and_compute_activity(
    tmp_path,
    monkeypatch,
    case,
    message,
):
    root = _invalid_project(tmp_path, case)
    activity: list[str] = []

    def reject(label):
        def fail(*_args, **_kwargs):
            activity.append(label)
            raise AssertionError(f"invalid-registry preflight reached {label}")

        return fail

    monkeypatch.setattr("socket.socket.connect", reject("network"))
    monkeypatch.setattr(SparkSessionProvider, "get", reject("compute"))
    monkeypatch.setattr(SourceExecutor, "execute", reject("source execution"))

    with pytest.raises(ValueError, match=message):
        load_registry(root)

    assert activity == []
    assert not any(path.is_file() for path in (root / "data").rglob("*"))


def test_execution_provider_seam_passes_cluster_profile_through_unchanged(tmp_path):
    """This proves injection, not another real engine."""

    root = build_graph_project(tmp_path, "independent")
    registry = load_registry(root)
    plan = BatchPlanner().plan(
        BatchPlanRequest.create(
            environment="cluster",
            project_root=root,
            pipeline_run_id="profile-passthrough",
            planned_at=PLANNED_AT,
        ),
        registry=registry,
    )
    profile = {
        "name": "cluster",
        "spark": {
            "master": "spark://cluster.invalid:7077",
            "iceberg": {
                "catalog_type": "jdbc",
                "uri": "jdbc:postgresql://catalog.invalid/janus",
                "warehouse_dir": "s3a://janus/warehouse",
                "object_store": {"endpoint": "https://objects.invalid"},
            },
        },
    }
    resolved_paths = {
        "warehouse_dir": Path("/tmp/janus-profile-scratch"),
        "iceberg_warehouse_dir": "s3a://janus/warehouse",
    }
    executor = _ProfileRecordingExecutor()
    provider = _NoopProvider()
    provider_arguments: list[tuple[Any, Any, Any]] = []

    def provider_factory(environment_config, paths, logger):
        provider_arguments.append((environment_config, paths, logger))
        return provider

    service = SourceExecutionService(executor=executor, provider_factory=provider_factory)
    result = service.execute(plan.source("A").require_planned_run(), profile, resolved_paths)

    assert result.is_successful
    assert provider_arguments == [(profile, resolved_paths, None)]
    assert executor.environment_config is profile
    assert executor.provider is provider
    assert provider.stop_calls == 1


def _capture_batch(root: Path, case: str):
    family = "catalog" if case == "catalog_consumer" else "api"
    upstreams = ("A",) if case.endswith("consumer") else ()
    selected = SourceSpec("B" if upstreams else "C", upstreams, family=family)
    graph = GraphCase((SourceSpec("A"), selected)) if upstreams else GraphCase((selected,))
    write_project(root, source_documents(graph))

    events: list[str] = []
    tables: dict[str, Any] = {}
    api_transports = iter(
        [OfflineTransport("independent", events), OfflineTransport(case, events)]
        if upstreams
        else [OfflineTransport(case, events)]
    )
    bindings = [
        StrategyBinding(
            "api",
            "page_number_api",
            ApiStrategy(
                transport_factory=lambda: next(api_transports),
                storage_layout_factory=lambda _plan: StorageLayout.from_environment_config(
                    ENVIRONMENT_CONFIG,
                    root,
                ),
                sleeper=lambda _seconds: None,
            ),
        )
    ]
    if family == "catalog":
        catalog_transport = OfflineTransport(case, events)
        bindings.append(
            StrategyBinding(
                "catalog",
                "metadata_catalog",
                CatalogStrategy(
                    transport_factory=lambda: catalog_transport,
                    storage_layout_factory=lambda _plan: StorageLayout.from_environment_config(
                        ENVIRONMENT_CONFIG,
                        root,
                    ),
                    sleeper=lambda _seconds: None,
                ),
            )
        )

    planner = Planner(strategy_catalog=StrategyCatalog(tuple(bindings)))
    registry = load_registry(root)
    plan = BatchPlanner(planner=planner).plan(
        BatchPlanRequest.create(
            environment="local",
            project_root=root,
            pipeline_run_id=f"compat-{case}",
            planned_at=PLANNED_AT,
        ),
        registry=registry,
    )
    executor = SourceExecutor(
        reader=JsonReader(),
        normalizer=IdentityNormalizer(),
        writer_factory=lambda _layout: MemoryWriter(tables),
        quality_gate=BoundaryQualityGate(),
        observer=FixedObserver(),
    )
    service = SourceExecutionService(
        executor=executor,
        provider_factory=lambda *_args: SparkSessionProvider(
            {},
            {},
            session_factory=lambda: MemorySession(tables, events),
        ),
    )
    outcome = BatchExecutor(source_execution=service).execute(
        plan,
        registry,
        ENVIRONMENT_CONFIG,
        {},
    )
    return outcome, tables


def _bronze_output(summary: Mapping[str, Any]) -> dict[str, Any]:
    outputs = [item for item in summary["materialized_outputs"] if item["zone"] == "bronze"]
    assert len(outputs) == 1
    return outputs[0]


def _metadata_presence(summary: Mapping[str, Any]) -> dict[str, bool]:
    return {key: value is not None for key, value in summary["metadata_outputs"].items()}


def _normalized_strategy_metadata(summary: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    metadata = dict(summary["strategy_metadata"])
    prefix = metadata.get("raw_path_prefix")
    if isinstance(prefix, str):
        metadata["raw_path_prefix"] = prefix.replace(run_id, "<RUN_ID>")
    return metadata


def _invalid_project(tmp_path: Path, case: str) -> Path:
    if case != "wrong_declared_table":
        return build_graph_project(tmp_path, case)

    root = tmp_path / case
    documents = source_documents(GRAPH_CASES["chain"])
    consumer = next(document for document in documents if document["source_id"] == "B")
    consumer["access"]["request_inputs"]["table_name"] = "not_a"
    return write_project(root, documents)
