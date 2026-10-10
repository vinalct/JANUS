"""AC-7: the status command evaluates real per-source YAML and run histories."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from janus.main import main
from janus.registry import load_registry
from tests.support.metadata_zone_fixture import build_metadata_zone
from tests.support.metrics_order23 import NOW, exit_code, write_profile

pytestmark = pytest.mark.xfail(
    strict=True, reason="status verb and runs-table freshness reader are absent"
)
TOP_KEYS = {"schema_version", "environment", "evaluated_at", "source", "totals", "sources"}
ROW_KEYS = {
    "source_id",
    "enabled",
    "outcome",
    "last_success_at",
    "age_seconds",
    "max_age_seconds",
    "runs_scanned",
    "unreadable_run_files",
    "metadata_path",
}


@pytest.fixture
def zone(tmp_path):
    fixture = build_metadata_zone(tmp_path / "project", now=NOW, seed=23)
    write_profile(fixture.root)
    registry = load_registry(fixture.root)
    assert registry.sources
    assert {source.source_id for source in registry.sources} == set(fixture.expected_outcomes)
    return fixture


def _arguments(zone, *extra):
    return ["status", "--project-root", str(zone.root), "--now", zone.now.isoformat(), *extra]


def _invoke(zone, *extra):
    # Missing verbs must xfail in the body, not fail collection or silently skip.
    import janus.cli.status  # noqa: F401

    return exit_code(lambda: main(_arguments(zone, *extra)))


def test_all_six_sources_have_the_fixture_outcomes_and_stable_json_keys(zone, capsys):
    assert _invoke(zone, "--format", "json") == 1
    output = json.loads(capsys.readouterr().out)
    assert set(output) == TOP_KEYS
    assert output["schema_version"] == 1
    assert output["environment"] == "local"
    assert output["source"] == "metadata_zone"
    assert output["evaluated_at"] == zone.now.isoformat()
    rows = output["sources"]
    assert rows
    assert {row["source_id"]: row["outcome"] for row in rows} == zone.expected_outcomes
    assert all(set(row) == ROW_KEYS for row in rows)
    assert [row["source_id"] for row in rows] == sorted(zone.expected_outcomes)
    assert output["totals"] == {
        "fresh": 2,
        "stale": 1,
        "never_ran": 2,
        "undeclared": 1,
        "breaching": 3,
    }
    by_id = {row["source_id"]: row for row in rows}
    assert by_id["fixture.fresh"]["age_seconds"] == 7200
    assert by_id["fixture.stale"]["age_seconds"] == 180000
    assert by_id["fixture.stale"]["max_age_seconds"] == 86400
    assert by_id["fixture.unreadable"]["unreadable_run_files"] == 1
    assert by_id["fixture.never"]["last_success_at"] is None
    assert by_id["fixture.undeclared"]["max_age_seconds"] is None
    for source in zone.sources:
        assert Path(by_id[source.source_id]["metadata_path"]) == source.metadata_dir


def test_text_output_is_stable_and_final_line_names_every_breach(zone, capsys):
    assert _invoke(zone, "--format", "text") == 1
    first = capsys.readouterr().out
    assert _invoke(zone, "--format", "text") == 1
    assert capsys.readouterr().out == first
    assert "SOURCE" in first and "OUTCOME" in first and "MAX AGE" in first
    breaches = {
        source
        for source, outcome in zone.expected_outcomes.items()
        if outcome in {"stale", "never_ran"}
    }
    assert breaches
    assert all(source in first.splitlines()[-1] for source in breaches)
    assert all(source in first for source in zone.expected_outcomes)


def test_fresh_and_undeclared_selection_exits_zero_and_lists_undeclared(zone, capsys):
    assert (
        _invoke(
            zone,
            "--source-id",
            "fixture.fresh",
            "--source-id",
            "fixture.undeclared",
            "--format",
            "json",
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["breaching"] == 0
    assert {row["source_id"]: row["outcome"] for row in payload["sources"]} == {
        "fixture.fresh": "fresh",
        "fixture.undeclared": "undeclared",
    }


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("fixture.fresh", 0),
        ("fixture.stale", 1),
        ("fixture.never", 1),
        ("fixture.empty", 1),
        ("fixture.undeclared", 0),
        ("fixture.unreadable", 0),
    ],
)
def test_source_filter_limits_both_rows_and_exit_code(source, code, zone, capsys):
    assert _invoke(zone, "--source-id", source, "--format", "json") == code
    rows = json.loads(capsys.readouterr().out)["sources"]
    assert len(rows) == 1
    assert rows[0]["source_id"] == source
    assert rows[0]["outcome"] == zone.expected_outcomes[source]


def test_unknown_source_is_a_named_argument_error(zone, capsys):
    assert _invoke(zone, "--source-id", "fixture.missing") == 2
    assert "fixture.missing" in capsys.readouterr().err


def test_runs_table_reader_matches_zone_using_one_fake_catalog_scan(zone, monkeypatch, capsys):
    import pyiceberg.catalog

    import janus.observability.freshness  # noqa: F401

    rows = [
        {
            "source_id": source.source_id,
            "ended_at": run.ended_at,
            "status": run.status,
            "freshness_max_age_seconds": 1,
        }
        for source in zone.sources
        for run in source.runs
        if run.status == "succeeded"
    ]
    scans = []

    def scan(**kwargs):
        scans.append(kwargs)
        return SimpleNamespace(to_arrow=lambda: SimpleNamespace(to_pylist=lambda: rows))

    loaded = []

    def load_table(identifier):
        loaded.append(identifier)
        return SimpleNamespace(scan=scan)

    monkeypatch.setattr(
        pyiceberg.catalog,
        "load_catalog",
        lambda *args, **kwargs: SimpleNamespace(load_table=load_table),
    )
    assert _invoke(zone, "--format", "json") == 1
    metadata = json.loads(capsys.readouterr().out)
    spark_before = {name for name in sys.modules if name.startswith("pyspark")}
    assert _invoke(zone, "--from-runs-table", "--format", "json") == 1
    table = json.loads(capsys.readouterr().out)
    assert table["source"] == "runs_table"
    expected = {
        row["source_id"]: (row["outcome"], row["last_success_at"], row["max_age_seconds"])
        for row in metadata["sources"]
    }
    assert expected
    assert {
        row["source_id"]: (row["outcome"], row["last_success_at"], row["max_age_seconds"])
        for row in table["sources"]
    } == expected
    assert len(scans) == len(loaded) == 1
    assert {name for name in sys.modules if name.startswith("pyspark")} == spark_before


@pytest.mark.parametrize("from_table", [False, True])
def test_status_paths_import_no_spark_in_a_fresh_process(from_table, zone):
    import janus.cli.status  # noqa: F401

    script = """
import json, sys, types
rows = json.loads(sys.argv[2])
from datetime import datetime
for row in rows:
    row['ended_at'] = datetime.fromisoformat(row['ended_at'])
class Table:
    def scan(self, **kwargs): return self
    def to_arrow(self): return self
    def to_pylist(self): return rows
catalog = types.ModuleType('pyiceberg.catalog')
catalog.load_catalog = lambda *args, **kwargs: types.SimpleNamespace(load_table=lambda _: Table())
package = types.ModuleType('pyiceberg')
package.__path__ = []
package.catalog = catalog
sys.modules['pyiceberg'] = package
sys.modules['pyiceberg.catalog'] = catalog
from janus.main import main
code = main(json.loads(sys.argv[1]))
assert code == 1, code
assert not [name for name in sys.modules if name == 'pyspark' or name.startswith('pyspark.')]
"""
    rows = [
        {
            "source_id": source.source_id,
            "ended_at": run.ended_at.isoformat(),
            "freshness_max_age_seconds": 1,
            "status": "succeeded",
        }
        for source in zone.sources
        for run in source.runs
        if run.status == "succeeded"
    ]
    argv = _arguments(zone, "--format", "json", *(["--from-runs-table"] if from_table else []))
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(argv), json.dumps(rows)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unreadable_only_history_reports_never_ran_with_a_count(zone, capsys):
    source = next(source for source in zone.sources if source.source_id == "fixture.unreadable")
    for run in source.runs:
        (source.metadata_dir / "runs" / f"{run.run_id}.json").unlink()
    assert _invoke(zone, "--source-id", source.source_id, "--format", "json") == 1
    row = json.loads(capsys.readouterr().out)["sources"][0]
    assert row["outcome"] == "never_ran"
    assert row["unreadable_run_files"] == 1


def test_disabled_sources_are_still_evaluated(zone, capsys):
    source = next(source for source in zone.sources if source.source_id == "fixture.stale")
    mapping = yaml.safe_load(source.config_path.read_text())
    mapping["enabled"] = False
    source.config_path.write_text(yaml.safe_dump(mapping))
    assert _invoke(zone, "--source-id", source.source_id, "--format", "text") == 1
    output = capsys.readouterr().out
    assert source.source_id in output
    assert "disabled" in output
