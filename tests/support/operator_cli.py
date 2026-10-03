"""Run operator verbs in process, and prove they never reach for Spark."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import pytest

OPERATOR_ENV = "JANUS_OPERATOR"


@dataclass(frozen=True, slots=True)
class CliResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def run_janus(argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> CliResult:
    from janus.cli.dispatch import main

    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        _hermetic_environment(env or {}),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        try:
            code = main(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return CliResult(int(code), stdout.getvalue(), stderr.getvalue())


def arm_spark_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test at every point a session could be built."""
    import janus.runtime.spark_lifecycle as spark_lifecycle
    import janus.utils.environment as environment

    def tripped(*args: object, **kwargs: object) -> None:
        pytest.fail("a session-free verb asked for a Spark session")

    builder = environment.build_spark_session
    monkeypatch.setattr(spark_lifecycle.SparkSessionProvider, "get", tripped)
    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] != "janus" or module is None:
            continue
        if vars(module).get("build_spark_session") is builder:
            monkeypatch.setattr(module, "build_spark_session", tripped)


def engine_modules_loaded() -> frozenset[str]:
    return frozenset(name for name in ("pyspark", "pyiceberg", "pyarrow") if name in sys.modules)


@contextlib.contextmanager
def _hermetic_environment(overrides: Mapping[str, str]) -> Iterator[None]:
    saved = dict(os.environ)
    try:
        for key in [key for key in os.environ if key.startswith("JANUS_")]:
            del os.environ[key]
        os.environ.update(overrides)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
