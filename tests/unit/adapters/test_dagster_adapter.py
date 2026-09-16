"""Installed-extra coverage for the thin Dagster adapter."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

dagster = pytest.importorskip("dagster")

from dagster import DagsterInstance, Definitions, RetryPolicy  # noqa: E402

from janus.adapters.dagster import (  # noqa: E402
    DagsterAdapterServices,
    build_dagster_adapter,
    source_op_name,
)
from janus.planner import Planner  # noqa: E402
from janus.runtime import SourceExecutor  # noqa: E402
from tests.support.orchestration import (  # noqa: E402
    SourceSpec,
    build_graph_project,
    source_payload,
    write_project,
)


def _environment_config() -> dict[str, Any]:
    return {
        "name": "local",
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        },
        "spark": {"warehouse_dir": "data/warehouse"},
        "runtime": {"log_level": "ERROR"},
    }


class FakeExecutedRun:
    def __init__(self, planned_run: Any, *, succeeded: bool = True) -> None:
        self.planned_run = planned_run
        self.status = "succeeded" if succeeded else "failed"
        self.failure_reason = None if succeeded else "fixture source failure"
        self.error_type = None if succeeded else "FixtureSourceFailure"

    @property
    def is_successful(self) -> bool:
        return self.status == "succeeded"

    def to_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_reason": self.failure_reason,
            "strategy_metadata": {
                "source": self.planned_run.plan.source.source_id,
            },
            "materialized_outputs": [],
            "metadata_outputs": {},
        }


class RecordingExecution:
    def __init__(
        self,
        *,
        fail_once: set[str] | None = None,
        fail_always: set[str] | None = None,
        raise_for: set[str] | None = None,
    ) -> None:
        self.fail_once = fail_once or set()
        self.fail_always = fail_always or set()
        self.raise_for = raise_for or set()
        self.calls: list[tuple[str, int, str]] = []

    def execute(
        self,
        planned_run: Any,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, Any],
    ) -> FakeExecutedRun:
        source_id = planned_run.plan.source.source_id
        attributes = planned_run.plan.run_context.attributes_as_dict()
        attempt = int(attributes["pipeline_attempt"])
        run_id = planned_run.plan.run_context.run_id
        self.calls.append((source_id, attempt, run_id))
        if source_id in self.raise_for:
            raise RuntimeError(f"raised source failure for {source_id}")
        succeeded = source_id not in self.fail_always and not (
            source_id in self.fail_once and attempt == 1
        )
        return FakeExecutedRun(planned_run, succeeded=succeeded)


def _adapter(root: Path, execution: RecordingExecution, **kwargs: Any):
    services = DagsterAdapterServices(
        source_execution_factory=lambda _logger: execution,
    )
    return build_dagster_adapter(
        root,
        environment_config=_environment_config(),
        services=services,
        **kwargs,
    )


def _sources(outcome) -> dict[str, Any]:
    return {source.source_id: source for source in outcome.sources}


def _definition_edges(adapter) -> set[tuple[str, str]]:
    structure = adapter.job.graph.dependency_structure
    by_op = {op_name: source_id for source_id, op_name in adapter.source_op_names.items()}
    edges: set[tuple[str, str]] = set()
    for consumer_id, consumer_op in adapter.source_op_names.items():
        upstreams = structure.input_to_upstream_outputs_for_node(consumer_op)
        edges.update(
            (by_op[output.node_name], consumer_id)
            for outputs in upstreams.values()
            for output in outputs
        )
    return edges


def test_definitions_have_one_metadata_rich_op_per_expanded_source_and_exact_edges(tmp_path):
    root = build_graph_project(tmp_path, "diamond")

    adapter = _adapter(root, RecordingExecution())
    Definitions.validate_loadable(adapter.definitions)

    assert set(adapter.source_op_names) == {"A", "B", "C", "D"}
    assert {node.name for node in adapter.job.graph.node_defs} == set(
        adapter.source_op_names.values()
    )
    assert _definition_edges(adapter) == {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")}
    assert {
        (edge.producer_id, edge.consumer_id) for edge in adapter.manifest.edges
    } == _definition_edges(adapter)

    nodes = {node.name: node for node in adapter.job.graph.node_defs}
    consumer = nodes[adapter.source_op_names["D"]]
    assert consumer.tags["janus/source_id"] == "D"
    assert consumer.tags["janus/domain"] == "reference"
    assert json.loads(consumer.tags["janus/source_tags"]) == ["reference"]
    assert consumer.tags["janus/bronze_table"] == "bronze.d"
    assert {
        item["producer_id"]
        for item in json.loads(consumer.tags["janus/dependency_provenance"])
    } == {"B", "C"}


def test_op_names_are_stable_valid_and_distinguish_colliding_stems(tmp_path):
    specs = (
        SourceSpec("a-b", table_name="dash"),
        SourceSpec("a b", table_name="space"),
        SourceSpec("1.leading", table_name="leading"),
        SourceSpec("punct!?", table_name="punctuation"),
    )
    root = write_project(
        tmp_path / "identifiers",
        [source_payload(spec) for spec in specs],
    )

    adapter = _adapter(root, RecordingExecution())
    names = adapter.source_op_names

    assert len(set(names.values())) == len(specs)
    assert names["a-b"] != names["a b"]
    assert names["1.leading"].startswith("janus_source_1_leading_")
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", name) for name in names.values())
    assert names == {spec.source_id: source_op_name(spec.source_id) for spec in specs}


def test_definition_loading_performs_no_runtime_or_external_work(tmp_path, monkeypatch):
    root = build_graph_project(tmp_path, "chain")

    def reject(*args: object, **kwargs: object) -> None:
        raise AssertionError("definition loading crossed into runtime work")

    monkeypatch.setattr("socket.socket.connect", reject)
    monkeypatch.setattr("janus.adapters.dagster.runtime.prepare_runtime", reject)
    monkeypatch.setattr("janus.runtime.spark_lifecycle.SparkSessionProvider.get", reject)
    monkeypatch.setattr("janus.runtime.executor.SourceExecutor.execute", reject)
    monkeypatch.setattr("janus.orchestration.PipelineSummaryStore.persist", reject)

    adapter = build_dagster_adapter(
        root,
        environment_config=_environment_config(),
    )
    Definitions.validate_loadable(adapter.definitions)

    assert not (root / "data" / "warehouse").exists()
    assert not (root / "data" / "metadata" / "pipelines").exists()


def test_execution_delegates_once_per_source_to_the_shared_planner_and_executor(
    tmp_path,
    monkeypatch,
):
    root = build_graph_project(tmp_path, "diamond")
    planned: list[str] = []
    executed: list[str] = []
    original_plan = Planner.plan

    def recording_plan(self, request, *, registry=None):
        planned.append(request.source_id)
        return original_plan(self, request, registry=registry)

    def recording_execute(self, planned_run, provider, environment_config):
        executed.append(planned_run.plan.source.source_id)
        return FakeExecutedRun(planned_run)

    monkeypatch.setattr(Planner, "plan", recording_plan)
    monkeypatch.setattr(SourceExecutor, "execute", recording_execute)
    adapter = build_dagster_adapter(root, environment_config=_environment_config())

    assert planned == []
    result, outcome = adapter.execute_in_process()

    assert result.success
    assert outcome.is_successful
    assert planned == list(adapter.manifest.source_order)
    assert sorted(executed) == sorted(adapter.manifest.source_order)
    assert len(executed) == len(adapter.manifest.source_order)


@pytest.mark.parametrize("failure_mode", ("returned", "raised"))
def test_failure_blocks_descendants_but_independent_peer_finishes_and_is_collected(
    tmp_path,
    failure_mode,
):
    root = build_graph_project(tmp_path, "chain_and_peer")
    execution = RecordingExecution(
        fail_always={"A"} if failure_mode == "returned" else None,
        raise_for={"A"} if failure_mode == "raised" else None,
    )
    adapter = _adapter(root, execution)
    instance = DagsterInstance.ephemeral()

    result, outcome = adapter.execute_in_process(instance=instance)
    sources = _sources(outcome)

    assert not result.success
    assert outcome.status == "failed"
    assert [call[:2] for call in execution.calls] == [("A", 1), ("C", 1)]
    assert sources["A"].status == "failed"
    assert sources["A"].attempted
    assert sources["B"].status == "skipped"
    assert sources["B"].skip.direct_blocking_upstream_ids == ("A",)
    assert sources["D"].status == "skipped"
    assert sources["D"].skip.root_failed_source_ids == ("A",)
    assert sources["C"].status == "succeeded"
    assert outcome.summary_persistence.path.is_file()

    repeated = adapter.collect(instance, result.run_id)
    assert repeated.to_summary() == outcome.to_summary()
    stored = json.loads(outcome.summary_persistence.path.read_text(encoding="utf-8"))
    assert stored == outcome.to_summary()


def test_retries_are_opt_in_and_keep_distinct_attempt_ids_in_one_pipeline(tmp_path):
    root = build_graph_project(tmp_path, "chain")
    execution = RecordingExecution(fail_once={"A"})
    adapter = _adapter(root, execution, retry_policy=RetryPolicy(max_retries=1))

    result, outcome = adapter.execute_in_process()
    source_a = _sources(outcome)["A"]

    assert result.success
    assert outcome.pipeline_run_id == result.run_id
    assert [call[:2] for call in execution.calls] == [("A", 1), ("A", 2), ("B", 1)]
    assert [attempt.status for attempt in source_a.attempts] == ["failed", "succeeded"]
    assert [attempt.attempt for attempt in source_a.attempts] == [1, 2]
    assert source_a.attempts[0].run_id != source_a.attempts[1].run_id
    assert all(attempt.run_id.startswith(result.run_id) for attempt in source_a.attempts)


def test_default_policy_does_not_retry_and_whole_runs_get_new_pipeline_ids(tmp_path):
    root = build_graph_project(tmp_path, "independent")
    execution = RecordingExecution()
    adapter = _adapter(root, execution)

    first_result, first = adapter.execute_in_process()
    second_result, second = adapter.execute_in_process()

    assert first_result.run_id != second_result.run_id
    assert first.pipeline_run_id != second.pipeline_run_id
    assert all(len(source.attempts) == 1 for source in (*first.sources, *second.sources))
    assert all(source.attempt == 1 for source in (*first.sources, *second.sources))


def test_source_config_drift_is_rejected_before_executor_calls(tmp_path):
    root = build_graph_project(tmp_path, "chain")
    execution = RecordingExecution()
    adapter = _adapter(root, execution)
    source_path = next((root / "conf" / "sources").glob("*.yaml"))
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    result, outcome = adapter.execute_in_process()

    assert not result.success
    assert execution.calls == []
    assert outcome.status == "failed"
    assert _sources(outcome)["A"].failure.phase == "orchestration"
    assert _sources(outcome)["B"].status == "skipped"


def test_terminal_status_sensors_are_enabled_for_success_failure_and_cancel(tmp_path):
    root = build_graph_project(tmp_path, "independent")

    adapter = _adapter(root, RecordingExecution())

    assert {sensor.name for sensor in adapter.definitions.sensors} == {
        "janus_sources_success_summary",
        "janus_sources_failure_summary",
        "janus_sources_canceled_summary",
    }
    assert all(
        sensor.default_status == dagster.DefaultSensorStatus.RUNNING
        for sensor in adapter.definitions.sensors
    )
