"""Installed-Dagster integration against the same batch contract used by the CLI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

dagster = pytest.importorskip("dagster")

from dagster import DagsterInstance, RetryPolicy  # noqa: E402

import janus.cli.run_all as run_all_module  # noqa: E402
from janus.adapters.dagster import (  # noqa: E402
    DagsterAdapterServices,
    build_dagster_adapter,
)
from janus.main import main  # noqa: E402
from janus.orchestration import BatchSelection  # noqa: E402
from janus.runtime import BatchExecutor  # noqa: E402
from tests.support.orchestration import build_graph_project  # noqa: E402

PLANNED_AT = "2026-09-15T12:00:00Z"


class _FakeExecutedRun:
    def __init__(self, planned_run: Any, *, succeeded: bool) -> None:
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


class _ScenarioExecution:
    def __init__(
        self,
        *,
        fail_once: set[str] | None = None,
        fail_always: set[str] | None = None,
    ) -> None:
        self.fail_once = fail_once or set()
        self.fail_always = fail_always or set()
        self.calls: list[tuple[str, int, str]] = []

    def execute(
        self,
        planned_run: Any,
        environment_config: Mapping[str, Any],
        resolved_paths: Mapping[str, Any],
    ) -> _FakeExecutedRun:
        source_id = planned_run.plan.source.source_id
        attempt = int(planned_run.plan.run_context.attributes_as_dict()["pipeline_attempt"])
        self.calls.append((source_id, attempt, planned_run.plan.run_context.run_id))
        failed = source_id in self.fail_always or (source_id in self.fail_once and attempt == 1)
        return _FakeExecutedRun(planned_run, succeeded=not failed)


def test_cli_and_dagster_have_identical_graph_selection_and_terminal_contract(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = _project(tmp_path, "chain_and_peer")
    cli_execution = _ScenarioExecution(fail_always={"A"})
    monkeypatch.setattr(
        run_all_module,
        "_build_batch_executor",
        lambda _logger: BatchExecutor(source_execution=cli_execution),
    )

    cli_status = main(
        [
            "run-all",
            "--project-root",
            str(root),
            "--tag",
            "terminal",
            "--tag",
            "independent",
            "--pipeline-run-id",
            "task10-cli-parity",
            "--started-at",
            PLANNED_AT,
        ]
    )
    cli_summary = json.loads(capsys.readouterr().out)

    dagster_execution = _ScenarioExecution(fail_always={"A"})
    adapter = _adapter(
        root,
        dagster_execution,
        selection=BatchSelection.create(tags=("terminal", "independent")),
    )
    native, dagster_outcome = adapter.execute_in_process(instance=DagsterInstance.ephemeral())
    dagster_summary = dagster_outcome.to_summary()

    assert cli_status == 1
    assert not native.success
    assert _normalized_terminal_contract(cli_summary) == _normalized_terminal_contract(
        dagster_summary
    )
    assert [call[:2] for call in cli_execution.calls] == [("A", 1), ("C", 1)]
    assert [call[:2] for call in dagster_execution.calls] == [("A", 1), ("C", 1)]


def test_bounded_native_retry_is_collected_once_and_releases_the_consumer(tmp_path):
    root = _project(tmp_path, "chain")
    execution = _ScenarioExecution(fail_once={"A"})
    adapter = _adapter(root, execution, retry_policy=RetryPolicy(max_retries=1))
    instance = DagsterInstance.ephemeral()

    native, outcome = adapter.execute_in_process(instance=instance)
    repeated = adapter.collect(instance, native.run_id)
    sources = {source.source_id: source for source in outcome.sources}

    assert native.success
    assert outcome.is_successful
    assert [call[:2] for call in execution.calls] == [("A", 1), ("A", 2), ("B", 1)]
    assert [attempt.status for attempt in sources["A"].attempts] == ["failed", "succeeded"]
    assert [attempt.attempt for attempt in sources["A"].attempts] == [1, 2]
    assert len({attempt.run_id for attempt in sources["A"].attempts}) == 2
    assert sources["B"].status == "succeeded"
    assert repeated.to_summary() == outcome.to_summary()


def _project(tmp_path: Path, graph: str) -> Path:
    root = build_graph_project(tmp_path, graph)
    environments = root / "conf" / "environments"
    environments.mkdir(parents=True)
    (environments / "local.yaml").write_text(
        """
name: local
runtime:
  log_level: ERROR
spark:
  app_name: janus-task10-dagster
  master: local[1]
  warehouse_dir: data/metadata/spark-warehouse
  config: {}
storage:
  root_dir: data
  raw_dir: data/raw
  bronze_dir: data/bronze
  metadata_dir: data/metadata
""".lstrip(),
        encoding="utf-8",
    )
    return root


def _adapter(root: Path, execution: _ScenarioExecution, **kwargs: Any):
    return build_dagster_adapter(
        root,
        environment_config=_environment_config(),
        services=DagsterAdapterServices(
            source_execution_factory=lambda _logger: execution,
        ),
        **kwargs,
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
        "spark": {"warehouse_dir": "data/metadata/spark-warehouse"},
        "runtime": {"log_level": "ERROR"},
    }


def _normalized_terminal_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude only runtime identity/timing/trigger and persistence locations.

    Dagster owns its run id, trigger, wall-clock timestamps, and retry attempt ids; the
    CLI owns a caller-supplied pipeline id and its own timing. Attempt numbers/statuses,
    graph selection, config versions, failures, and skip causes remain comparable.
    """

    return {
        "schema_version": summary["schema_version"],
        "environment": summary["pipeline"]["environment"],
        "selection": summary["selection"],
        "graph": summary["graph"],
        "config_versions": summary["config_versions"],
        "sources": [
            {
                "source_id": source["source_id"],
                "selected_directly": source["selected_directly"],
                "upstream_ids": source["upstream_ids"],
                "config_version": source["config_version"],
                "status": source["status"],
                "attempted": source["attempted"],
                "attempts": [
                    {
                        "attempt": attempt["attempt"],
                        "status": attempt["status"],
                        "failure": attempt.get("failure"),
                    }
                    for attempt in source["attempts"]
                ],
                "failure": source.get("failure"),
                "skip": source.get("skip"),
            }
            for source in summary["sources"]
        ],
        "totals": {
            key: value for key, value in summary["totals"].items() if key != "duration_seconds"
        },
    }
