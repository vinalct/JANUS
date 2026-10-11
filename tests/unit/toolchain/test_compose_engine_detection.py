"""Compose selection checks daemon access and honors the engine selected by CI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROVIDERS = ("podman compose", "podman-compose", "docker compose", "docker-compose")


@pytest.fixture
def run_setup(tmp_path):
    make = shutil.which("make")
    if make is None:
        pytest.skip("GNU make is required for container-engine selection checks")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "compose.jsonl"
    script = (
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "binary = Path(sys.argv[0]).name\n"
        "args = sys.argv[1:]\n"
        "provider = binary + ' compose' if args[:1] == ['compose'] else binary\n"
        f"with Path({str(log)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps({'provider': provider, 'args': args}) + '\\n')\n"
        "if args[-1:] == ['version']:\n"
        "    sys.exit(0)\n"
        "available = json.loads(os.environ['JANUS_TEST_REACHABLE_PROVIDERS'])\n"
        "if provider not in available:\n"
        "    print('Cannot connect to the container daemon', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    for binary in ("podman", "podman-compose", "docker", "docker-compose"):
        path = binaries / binary
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key not in {
        "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "JANUS_CONTAINER_ENGINE", "DETECT_COMPOSE",
    }}
    env["PATH"] = str(binaries) + os.pathsep + env.get("PATH", "")

    def run(engine, available):
        result = subprocess.run(
            [make, "--no-print-directory", "-f", str(PROJECT_ROOT / "Makefile"),
             "ensure-up", "IVY_JAR_NAMES="],
            cwd=tmp_path,
            env={**env, "JANUS_CONTAINER_ENGINE": engine,
                 "JANUS_TEST_REACHABLE_PROVIDERS": json.dumps(available)},
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize(("engine", "available", "expected"), [
    ("auto", ["docker compose"], "docker compose"),
    ("auto", ["podman compose", "docker compose"], "podman compose"),
    ("auto", ["podman-compose", "docker compose"], "podman-compose"),
    ("auto", ["docker-compose"], "docker-compose"),
    ("docker", list(PROVIDERS), "docker compose"),
    ("docker", ["podman compose", "docker-compose"], "docker-compose"),
    ("podman", list(PROVIDERS), "podman compose"),
    ("podman", ["podman-compose", "docker compose"], "podman-compose"),
])
def test_setup_uses_a_reachable_provider_on_the_selected_engine(
    run_setup, engine, available, expected,
):
    result, calls = run_setup(engine, available)
    assert result.returncode == 0, result.stderr
    assert f"Using {expected}\n" in result.stdout
    assert all(call["args"][-1:] != ["version"] for call in calls)
    started = [call for call in calls if "up" in call["args"]]
    assert len(started) == 1 and started[0]["provider"] == expected
    uses_podman_overlay = "docker/docker-compose.podman.yml" in started[0]["args"]
    assert uses_podman_overlay == expected.startswith("podman")
    if engine != "auto":
        assert all(call["provider"].startswith(engine) for call in calls)


@pytest.mark.parametrize(("engine", "available"), [
    ("auto", []),
    ("docker", ["podman compose", "podman-compose"]),
    ("podman", ["docker compose", "docker-compose"]),
])
def test_unreachable_selected_engine_fails_before_container_setup(
    run_setup, tmp_path, engine, available,
):
    result, calls = run_setup(engine, available)
    assert result.returncode != 0
    assert "No reachable Compose provider found" in result.stderr
    assert all(call["args"][-1:] == ["ps"] for call in calls)
    assert not (tmp_path / "data").exists()


def test_unknown_engine_is_rejected_without_contacting_any_provider(run_setup):
    result, calls = run_setup("other", list(PROVIDERS))
    assert result.returncode != 0
    assert "Invalid JANUS_CONTAINER_ENGINE: other" in result.stderr
    assert calls == []


def test_spark_job_checks_its_engine_before_building_the_image():
    workflow = yaml.safe_load((PROJECT_ROOT / ".github/workflows/ci.yml").read_text())
    spark = workflow["jobs"]["spark"]
    assert spark["env"]["JANUS_CONTAINER_ENGINE"] == "docker"
    steps = spark["steps"]
    check = next(
        index for index, step in enumerate(steps) if step.get("run") == "make check-compose"
    )
    build = next(index for index, step in enumerate(steps) if "load" in step.get("with", {}))
    assert steps[build]["with"]["load"] is True
    assert check < build
