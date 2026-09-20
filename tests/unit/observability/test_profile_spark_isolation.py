"""Emission stays outside Spark's lifetime for every shipped environment profile."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import janus.observability.emission as emission
from janus.lineage import RunObserver
from janus.observability import IcebergAppendOutcome, IcebergAppendResult
from janus.runtime.executor import _prepare_execution_observer
from janus.utils.environment import load_environment_config, materialize_runtime_paths
from tests.unit.observability.test_guarded_emission import _extraction, _plan, _writes

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILES = tuple(
    path.stem for path in sorted((PROJECT_ROOT / "conf" / "environments").glob("*.yaml"))
)


class _RejectingProvider:
    def __init__(self, resolved_paths) -> None:
        self.resolved_paths = resolved_paths
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        raise AssertionError("observability attempted to acquire Spark")

    def stop(self) -> None:
        pass


@pytest.mark.parametrize("profile", PROFILES)
def test_started_success_and_failure_emission_never_acquire_spark(
    profile,
    tmp_path,
    monkeypatch,
):
    """Use the runtime's real wiring seam with a provider whose ``get`` always fails."""
    for name in tuple(key for key in os.environ if key.startswith("JANUS_")):
        monkeypatch.delenv(name, raising=False)
    config = load_environment_config(profile, PROJECT_ROOT)
    paths = materialize_runtime_paths(config, tmp_path)
    provider = _RejectingProvider(paths)
    monkeypatch.setattr(
        emission,
        "append_run_record",
        lambda *args, **kwargs: IcebergAppendResult(
            IcebergAppendOutcome.EMITTED,
            "metadata.runs",
        ),
    )

    observer = _prepare_execution_observer(
        RunObserver(),
        SimpleNamespace(logger=None),
        config,
        provider,
        None,
    )
    success = _plan(tmp_path / "success", f"task10-{profile}-success")
    observer.start_run(success)
    observer.record_success(success, _extraction(success), _writes(success))

    failure = _plan(tmp_path / "failure", f"task10-{profile}-failure")
    observer.start_run(failure)
    observer.record_failure(
        failure,
        RuntimeError("scripted run failure"),
        _extraction(failure),
        _writes(failure),
    )

    assert provider.get_calls == 0
