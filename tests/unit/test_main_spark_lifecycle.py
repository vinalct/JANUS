"""CLI-level lifecycle evidence: what `main()` does with the Spark session (AC-1, AC-2).

`main()` is where the eager session used to live, so the claim "extraction runs with no
Spark session" has to hold at this level too — not only inside the executor. These tests
drive the real `main()` with the planning and execution stages stubbed out, so what is
under test is exactly the lifecycle wiring:

- the CLI hands the executor a **provider**, never a started session;
- the `spark_session` summary block reports what actually happened, not what was intended;
- the provider is released on the success path, the failed-run path, and the raised-exception
  path (FR-4);
- `--execute` and `--ingest-raw-to-bronze` converge on the same shape (FR-3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import janus.main as main_module
import janus.runtime.spark_lifecycle as spark_lifecycle
from janus.main import main
from janus.runtime import SparkSessionProvider

SOURCE_ID = "federal_open_data_example"
ENVIRONMENT_CONFIG: dict[str, Any] = {
    "name": "local",
    "spark": {"app_name": "janus-cli", "master": "local[*]", "config": {}},
    "runtime": {"log_level": "INFO"},
}


@dataclass(slots=True)
class StubSession:
    """Minimal stand-in for a built SparkSession."""

    stop_calls: int = 0

    @property
    def sparkContext(self) -> SimpleNamespace:  # matches the PySpark attribute name
        return SimpleNamespace(appName="janus-cli", master="local[*]")

    def stop(self) -> None:
        self.stop_calls += 1


@dataclass(slots=True)
class RecordingRun:
    """Whatever the stubbed execution stage returns to `main()`."""

    status: str = "succeeded"

    @property
    def is_successful(self) -> bool:
        return self.status == "succeeded"

    def to_summary(self) -> dict[str, Any]:
        return {"status": self.status}


@dataclass(slots=True)
class ExecutionSpy:
    """Captures the provider `main()` passes in and what it had done by then."""

    sessions: list[StubSession] = field(default_factory=list)
    seen_provider: Any = None
    was_started_on_entry: bool | None = None
    starts_a_session: bool = False
    raises: Exception | None = None
    status: str = "succeeded"

    def run(self, provider: SparkSessionProvider) -> RecordingRun:
        self.seen_provider = provider
        self.was_started_on_entry = provider.was_started
        if self.starts_a_session:
            self.sessions.append(provider.get())
        if self.raises is not None:
            raise self.raises
        return RecordingRun(self.status)


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Drive the real `main()` with planning and execution stubbed to a spy."""

    spy = ExecutionSpy()

    monkeypatch.setattr(main_module, "load_environment_config", lambda *a, **k: ENVIRONMENT_CONFIG)
    monkeypatch.setattr(main_module, "prepare_runtime", lambda *a, **k: {"root_dir": tmp_path})
    monkeypatch.setattr(
        main_module,
        "Planner",
        lambda: SimpleNamespace(plan=lambda request: SimpleNamespace(to_summary=lambda: {})),
    )
    monkeypatch.setattr(
        spark_lifecycle,
        "build_spark_session",
        lambda config, paths: StubSession(),
    )

    class StubExecutor:
        def __init__(self, logger=None):
            del logger

        def execute(self, planned_run, spark_provider, environment_config):
            del planned_run
            del environment_config
            return spy.run(spark_provider)

    monkeypatch.setattr(main_module, "SourceExecutor", StubExecutor)
    monkeypatch.setattr(
        main_module,
        "ingest_raw_to_bronze",
        lambda planned_run, spark_provider, config, **kwargs: spy.run(spark_provider),
    )
    return spy


def _summary(capsys) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


# ── AC-1 at the CLI level ────────────────────────────────────────────────────


def test_execute_hands_the_executor_a_provider_that_has_not_started_anything(cli, capsys):
    """AC-1: no session exists when the pipeline begins — the CLI builds none up front."""

    exit_code = main(["--execute", "--source-id", SOURCE_ID])

    assert exit_code == 0
    assert isinstance(cli.seen_provider, SparkSessionProvider)
    assert cli.was_started_on_entry is False
    _summary(capsys)


def test_a_run_that_never_materializes_reports_no_spark_session_block(cli, capsys):
    """The summary reports what happened: no session started, no `spark_session` key.

    A source whose handoff is empty pays for no compute at all, and the CLI output must
    say so rather than echoing the configured session back.
    """

    cli.starts_a_session = False

    main(["--execute", "--source-id", SOURCE_ID])

    assert "spark_session" not in _summary(capsys)


def test_a_materializing_run_reports_the_session_it_actually_started(cli, capsys):
    """AC-2: the `spark_session` block describes the live session, not the config."""

    cli.starts_a_session = True

    main(["--execute", "--source-id", SOURCE_ID])

    assert _summary(capsys)["spark_session"] == {
        "app_name": "janus-cli",
        "master": "local[*]",
    }


# ── FR-4: the session is released on every path ──────────────────────────────


def test_the_session_is_released_when_the_run_succeeds(cli, capsys):
    cli.starts_a_session = True

    exit_code = main(["--execute", "--source-id", SOURCE_ID])

    assert exit_code == 0
    assert [session.stop_calls for session in cli.sessions] == [1]
    _summary(capsys)


def test_the_session_is_released_when_the_run_fails_validation(cli, capsys):
    """A failed run still exits non-zero *and* still gives the compute back."""

    cli.starts_a_session = True
    cli.status = "failed"

    exit_code = main(["--execute", "--source-id", SOURCE_ID])

    assert exit_code == 1
    assert [session.stop_calls for session in cli.sessions] == [1]
    assert _summary(capsys)["executed_run"]["status"] == "failed"


def test_the_session_is_released_when_the_run_raises(cli):
    """FR-4: the CLI's `finally` is the backstop for anything the executor missed."""

    cli.starts_a_session = True
    cli.raises = RuntimeError("materialize boom")

    with pytest.raises(RuntimeError, match="materialize boom"):
        main(["--execute", "--source-id", SOURCE_ID])

    assert [session.stop_calls for session in cli.sessions] == [1]


# ── FR-3: both entry points converge on the same lifecycle ───────────────────


def test_raw_to_bronze_uses_the_same_deferred_provider_shape(cli, capsys):
    """FR-3: the replay entry point also receives a provider, not a started session."""

    exit_code = main(
        [
            "--ingest-raw-to-bronze",
            "--source-id",
            SOURCE_ID,
            "--bronze-table",
            "bronze_example.federal_open_data_example",
        ]
    )

    assert exit_code == 0
    assert isinstance(cli.seen_provider, SparkSessionProvider)
    assert cli.was_started_on_entry is False
    _summary(capsys)


def test_raw_to_bronze_reports_and_releases_its_session(cli, capsys):
    cli.starts_a_session = True

    main(
        [
            "--ingest-raw-to-bronze",
            "--source-id",
            SOURCE_ID,
            "--bronze-table",
            "bronze_example.federal_open_data_example",
        ]
    )

    summary = _summary(capsys)
    assert summary["spark_session"] == {"app_name": "janus-cli", "master": "local[*]"}
    assert [session.stop_calls for session in cli.sessions] == [1]


# ── the config-only path stays session-free ──────────────────────────────────


def test_validating_configuration_alone_never_builds_a_session(monkeypatch, tmp_path, capsys):
    """Without `--with-spark`, config validation must not reach for compute at all."""

    monkeypatch.setattr(main_module, "load_environment_config", lambda *a, **k: ENVIRONMENT_CONFIG)
    monkeypatch.setattr(main_module, "prepare_runtime", lambda *a, **k: {"root_dir": tmp_path})

    def fail_if_built(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("session requested while only validating configuration")

    monkeypatch.setattr(main_module, "build_spark_session", fail_if_built)
    monkeypatch.setattr(spark_lifecycle, "build_spark_session", fail_if_built)

    exit_code = main(["--environment", "local", "--project-root", str(tmp_path)])

    assert exit_code == 0
    assert "spark_session" not in _summary(capsys)


def test_project_root_is_resolved_before_planning(cli, capsys, tmp_path):
    """Guards the CLI contract the lifecycle tests above rely on."""

    main(["--execute", "--source-id", SOURCE_ID, "--project-root", str(tmp_path)])

    assert _summary(capsys)["project_root"] == str(Path(tmp_path).resolve())
