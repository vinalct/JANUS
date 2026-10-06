"""Maintenance refuses implicit profiles before setup, including parallel make."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.support.maintenance_smoke import MANIFEST, SOURCE_ID, verify_transcript
from tests.support.retention_baseline import filesystem_state

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAKEFILE = PROJECT_ROOT / "Makefile"


@pytest.fixture
def run_make(tmp_path):
    if shutil.which("make") is None or not MAKEFILE.is_file():
        pytest.skip("the maintenance target checks need GNU make and the checkout's Makefile")
    log = tmp_path / "compose.jsonl"
    compose = tmp_path / "fixture-compose"
    compose.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"with open({str(log)!r}, 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    compose.chmod(0o755)
    env = {name: value for name, value in os.environ.items() if name != "ENVIRONMENT"}
    env.pop("MAKEFLAGS", None)
    env.pop("MFLAGS", None)
    env.pop("MAKEOVERRIDES", None)

    def run(target, *args, environment=None):
        selected = env if environment is None else {**env, "ENVIRONMENT": environment}
        result = subprocess.run(
            [
                "make", "--no-print-directory", "-j4", "-f", str(MAKEFILE),
                target, f"DETECT_COMPOSE=printf '%s' '{compose}'", "IVY_JAR_NAMES=", *args,
            ],
            cwd=tmp_path,
            env=selected,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize("target", ["maintain-dry-run", "maintain-apply"])
@pytest.mark.parametrize("explicit", [None, "", "   "])
def test_missing_or_empty_environment_refuses_before_any_setup(
    run_make, tmp_path, target, explicit
):
    args = () if explicit is None else (f"ENVIRONMENT={explicit}",)
    result, calls = run_make(target, *args)
    assert result.returncode == 2
    assert "ENVIRONMENT must be set explicitly" in result.stderr
    assert calls == [] and not (tmp_path / "data").exists()


@pytest.mark.parametrize("target", ["maintain-dry-run", "maintain-apply"])
@pytest.mark.parametrize("origin", ["command line", "environment"])
def test_explicit_environment_and_selection_reach_the_container(run_make, target, origin):
    args = ["MAINTAIN_ARGS=--zone metadata --source-id fixture"]
    environment = "local" if origin == "environment" else None
    if origin == "command line":
        args.append("ENVIRONMENT=local")
    result, calls = run_make(target, *args, environment=environment)
    assert result.returncode == 0, result.stderr
    assert any("up" in call for call in calls)
    command = calls[-1]
    verb = command.index("maintain")
    mode = "--dry-run" if target == "maintain-dry-run" else "--apply"
    assert command[verb:] == [
        "maintain", "--environment", "local", mode,
        "--zone", "metadata", "--source-id", "fixture",
    ]


def test_a_makefile_default_other_than_local_still_requires_an_operator_choice(run_make, tmp_path):
    override = tmp_path / "defaults.mk"
    override.write_text("ENVIRONMENT := cluster\n", encoding="utf-8")
    result, calls = run_make("maintain-apply", "-f", str(override))
    assert result.returncode == 2 and calls == []


@pytest.mark.parametrize("target", ["maintain-dry-run", "maintain-apply", "require-environment"])
def test_same_named_files_cannot_bypass_the_guard(run_make, tmp_path, target):
    (tmp_path / target).touch()
    result, calls = run_make(target)
    assert result.returncode == 2 and calls == []


def test_workflow_and_make_ci_run_the_same_maintenance_steps_in_order():
    if not MAKEFILE.is_file():
        pytest.skip("the workflow parity check needs the checkout's Makefile")
    workflow = yaml.safe_load((PROJECT_ROOT / ".github/workflows/ci.yml").read_text())
    steps = [step.get("run", "") for step in workflow["jobs"]["spark"]["steps"]]
    full_suite = next(index for index, step in enumerate(steps) if "--cov=janus" in step)
    maintenance = steps[full_suite + 1 : full_suite + 4]
    assert "test_no_deletion_outside_maintenance --minimum-passed 63" in maintenance[0]
    assert maintenance[1:] == ["make test-maintenance", "make test-maintenance-smoke"]
    makefile = MAKEFILE.read_text()
    recipe = makefile.split("\nci: ensure-up\n", 1)[1].split("\n\n", 1)[0]
    assert recipe.index("--cov=janus") < recipe.index("$(MAINTENANCE_SWEEP_CLASS)")
    assert recipe.index("$(MAINTENANCE_SWEEP_CLASS)") < recipe.index("test-maintenance\n")
    assert recipe.index("test-maintenance\n") < recipe.index("test-maintenance-smoke\n")
    assert "maintain-apply" not in recipe
    fast = makefile.split("\ntest-fast:\n", 1)[1].split("\n\n", 1)[0]
    assert '"$(FAST_TEST_REPORT)" --class-name $(MAINTENANCE_SWEEP_CLASS)' in fast
    assert "--minimum-passed 63" in fast


def _transcript_fixture(tmp_path):
    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    (warehouse / "data.parquet").write_bytes(b"fixture")
    table = "bronze_full_refresh_history.unpartitioned_fixture"
    manifest = {
        "table": table, "snapshot_ids": [1, 2], "current_snapshot_id": 2,
        "warehouse": str(warehouse), "warehouse_before": filesystem_state(warehouse),
    }
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    record = {
        "dry_run": True, "environment": "local", "zones": ["bronze"],
        "source_ids": [SOURCE_ID], "failures": [], "maintenance_run_id": "fixture",
        "items": [{
            "zone": "bronze", "target": table, "action": "expire_snapshots",
            "status": "planned", "expired_snapshot_ids": [1],
        }],
    }
    return warehouse, record


@pytest.mark.parametrize("invalid", ["empty", "absent", "current", "mutated", "apply"])
def test_smoke_refuses_vacuous_or_destructive_results(tmp_path, invalid):
    warehouse, record = _transcript_fixture(tmp_path)
    if invalid == "empty":
        record["items"] = []
    elif invalid == "absent":
        record["items"][0]["status"] = "skipped"
    elif invalid == "current":
        record["items"][0]["expired_snapshot_ids"] = [2]
    elif invalid == "mutated":
        (warehouse / "data.parquet").write_bytes(b"changed")
    else:
        record["dry_run"] = False
    transcript = tmp_path / "dry-run.json"
    transcript.write_text(json.dumps(record))
    with pytest.raises((AssertionError, ValueError)):
        verify_transcript(tmp_path, transcript)


def test_smoke_accepts_a_planned_table_with_unchanged_warehouse_and_persisted_evidence(tmp_path):
    _warehouse, record = _transcript_fixture(tmp_path)
    transcript = tmp_path / "dry-run.json"
    transcript.write_text(json.dumps(record))
    evidence = tmp_path / "data/metadata/maintenance"
    evidence.mkdir(parents=True)
    (evidence / "fixture.json").write_text(json.dumps(record))
    verify_transcript(tmp_path, transcript)
