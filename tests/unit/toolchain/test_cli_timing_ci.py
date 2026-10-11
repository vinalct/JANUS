"""CLI timing remains required while coverage measures the other unit cases."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_NAME = "test_the_verb_completes_within_the_budget"
TEST_NODE = f"tests/unit/toolchain/test_cli_surface.py::{TEST_NAME}"


def test_hosted_gate_requires_all_five_timing_cases_before_coverage():
    workflow = yaml.safe_load((PROJECT_ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["spark"]["steps"]
    timing = next(
        index for index, step in enumerate(steps)
        if step.get("name") == "CLI response-time budget without coverage"
    )
    coverage = next(
        index for index, step in enumerate(steps)
        if step.get("name") == "Full test suite + coverage in isolated Spark processes"
    )
    assert timing < coverage
    command = steps[timing]["run"]
    assert f"{TEST_NODE} \\\n" in command
    assert "--no-cov" in command and "--cov=janus" not in command
    assert "--class-name tests.unit.toolchain.test_cli_surface" in command
    assert f"--test-name {TEST_NAME} --minimum-passed 5" in command
    assert f"--deselect={TEST_NODE}" in steps[coverage]["run"]
    assert "--cov=janus" in steps[coverage]["run"]


def test_local_ci_runs_the_same_timing_gate_before_coverage():
    makefile = (PROJECT_ROOT / "Makefile").read_text()
    recipe = makefile.split("\nci: ensure-up\n", 1)[1].split("\n\n", 1)[0]
    assert recipe.index("test-cli-timing") < recipe.index("--cov=janus")
    assert "$(SPARK_COVERAGE_UNIT_TEST_ARGS) --cov=janus" in recipe
    assert "$(SPARK_UNIT_TEST_ARGS) --deselect=$(CLI_TIMING_TEST)" in makefile
    if shutil.which("make") is None:
        pytest.skip("GNU make is required to expand the local timing recipe")
    env = {key: value for key, value in os.environ.items() if key not in {
        "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES",
    }}
    result = subprocess.run(
        ["make", "--no-print-directory", "-n", "-o", "ensure-up", "test-cli-timing"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert f"--no-cov {TEST_NODE}" in result.stdout
    assert "--class-name tests.unit.toolchain.test_cli_surface" in result.stdout
    assert f"--test-name {TEST_NAME} --minimum-passed 5" in result.stdout
