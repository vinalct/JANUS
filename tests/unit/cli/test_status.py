"""AC-7: the status command evaluates real per-source YAML and run histories."""

import json
import os
import subprocess
import sys
from datetime import UTC, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from janus.main import main
from janus.registry import load_registry
from tests.support.metadata_zone_fixture import build_metadata_zone
from tests.support.metrics_order23 import NOW, exit_code, write_profile

GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "observability" / "status.json"
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
    assert scans[0]["selected_fields"] == ("source_id", "ended_at", "freshness_max_age_seconds")
    assert loaded == ["metadata.runs"]
    by_id = {row["source_id"]: row for row in table["sources"]}
    assert by_id["fixture.fresh"]["runs_scanned"] == 2
    assert by_id["fixture.empty"]["runs_scanned"] == 0
    assert all(row["unreadable_run_files"] == 0 for row in table["sources"])
    # Rewriting every historical declaration must not move the answer.
    for row in rows:
        row["freshness_max_age_seconds"] = 10**9
    assert _invoke(zone, "--from-runs-table", "--format", "json") == 1
    assert json.loads(capsys.readouterr().out) == table
    assert {name for name in sys.modules if name.startswith("pyspark")} == spark_before


@pytest.mark.parametrize("from_table", [False, True])
def test_status_paths_import_no_spark_in_a_fresh_process(from_table, zone, record_property):
    script = """
import builtins, contextlib, io, json, sys, types
engine_roots = ('pyspark', 'pyiceberg', 'pyarrow')
def loaded_engines():
    return sorted({name.split('.', 1)[0] for name in sys.modules
                   if name.split('.', 1)[0] in engine_roots})
assert loaded_engines() == [], loaded_engines()
rows = json.loads(sys.argv[2])
from datetime import datetime
for row in rows:
    row['ended_at'] = datetime.fromisoformat(row['ended_at'])
class Table:
    def scan(self, **kwargs): return self
    def to_arrow(self): return self
    def to_pylist(self): return rows
import janus.cli.status
after_import = loaded_engines()
assert after_import == [], after_import
original_import = builtins.__import__
def offline_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    if name == 'pyiceberg.catalog':
        module.load_catalog = lambda *a, **k: types.SimpleNamespace(load_table=lambda _: Table())
    return module
builtins.__import__ = offline_import
from janus.main import main
argv = json.loads(sys.argv[1])
with contextlib.redirect_stdout(io.StringIO()):
    code = main(argv)
assert code == 1, code
after_status = loaded_engines()
if '--from-runs-table' in argv:
    assert 'pyiceberg' in after_status, after_status
    assert 'pyspark' not in after_status, after_status
else:
    assert after_status == [], after_status
print(json.dumps({'after_import': after_import, 'after_status': after_status, 'exit_code': code}))
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
    transcript = json.loads(result.stdout)
    record_property("engine_modules", transcript)
    assert transcript["after_import"] == []
    assert transcript["exit_code"] == 1
    if from_table:
        assert "pyiceberg" in transcript["after_status"]
        assert "pyspark" not in transcript["after_status"]
    else:
        assert transcript["after_status"] == []


def test_unreadable_only_history_reports_never_ran_with_a_count(zone, capsys):
    source = next(source for source in zone.sources if source.source_id == "fixture.unreadable")
    for run in source.runs:
        (source.metadata_dir / "runs" / f"{run.run_id}.json").unlink()
    assert _invoke(zone, "--source-id", source.source_id, "--format", "json") == 1
    row = json.loads(capsys.readouterr().out)["sources"][0]
    assert row["outcome"] == "never_ran"
    assert row["unreadable_run_files"] == 1
    assert _invoke(zone, "--source-id", source.source_id) == 1
    text = capsys.readouterr().out
    assert "never_ran (1 unreadable)" in text
    assert "Unreadable run files: 1" in text


def test_disabled_sources_are_still_evaluated(zone, capsys):
    source = next(source for source in zone.sources if source.source_id == "fixture.stale")
    mapping = yaml.safe_load(source.config_path.read_text())
    mapping["enabled"] = False
    source.config_path.write_text(yaml.safe_dump(mapping))
    assert _invoke(zone, "--source-id", source.source_id, "--format", "text") == 1
    output = capsys.readouterr().out
    assert source.source_id in output
    assert "disabled" in output
    assert _invoke(zone, "--source-id", source.source_id, "--format", "json") == 1
    assert json.loads(capsys.readouterr().out)["sources"][0]["enabled"] is False


def test_json_output_matches_reviewed_golden(zone, capsys):
    assert _invoke(zone, "--format", "json") == 1
    # Only the temporary project root differs between machines; every report field is pinned.
    output = capsys.readouterr().out.replace(str(zone.root), "<PROJECT_ROOT>")
    assert output == GOLDEN.read_text(encoding="utf-8")


@pytest.mark.parametrize("from_table", [False, True])
def test_unloadable_registry_exits_two_without_a_report(zone, from_table, capsys):
    (zone.root / "conf" / "app.yaml").unlink()
    assert _invoke(zone, *(["--from-runs-table"] if from_table else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "app.yaml" in captured.err


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "spark: [broken",
        "{}",
        "spark: []",
        "spark: null",
        "spark: {}\nstorage: {}\nobservability: {metrics: {transport: statds}}",
    ],
)
def test_unusable_profile_exits_two_without_a_report(zone, contents, capsys):
    (zone.root / "conf" / "environments" / "local.yaml").write_text(contents)
    assert _invoke(zone) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err


@pytest.mark.parametrize("block", ["metrics", "openlineage"])
def test_bad_transport_in_profile_is_a_named_configuration_error(zone, block, capsys):
    path = zone.root / "conf" / "environments" / "local.yaml"
    document = yaml.safe_load(path.read_text())
    document["observability"] = {block: {"transport": "invalid"}}
    path.write_text(yaml.safe_dump(document))
    assert _invoke(zone) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert block in captured.err and "invalid" in captured.err


@pytest.mark.parametrize("value", ["2026-10-09T12:00:00", "not-a-timestamp"])
def test_invalid_now_is_an_argument_error_before_evaluation(zone, value, capsys):
    assert _invoke(zone, "--now", value) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--now" in captured.err
    assert "timezone" in captured.err if "T" in value else "ISO-8601" in captured.err


@pytest.mark.parametrize("stage", ["catalog", "table", "scan"])
def test_unreachable_runs_table_exits_two_with_the_reason(zone, stage, monkeypatch, capsys):
    import pyiceberg.catalog

    def refuse(*args, **kwargs):
        raise ConnectionError(f"offline {stage}")

    table = SimpleNamespace(scan=refuse)
    catalog = SimpleNamespace(load_table=refuse if stage == "table" else lambda _: table)
    monkeypatch.setattr(
        pyiceberg.catalog, "load_catalog", refuse if stage == "catalog" else lambda *a, **k: catalog
    )
    assert _invoke(zone, "--from-runs-table") == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Could not read runs table: offline {stage}" in captured.err


def test_shared_metadata_path_is_noted_without_printing_the_path(zone, capsys):
    stale = next(source for source in zone.sources if source.source_id == "fixture.stale")
    document = yaml.safe_load(stale.config_path.read_text())
    document["outputs"]["metadata"]["path"] = "data/metadata/fixture/fresh"
    document["operations"]["freshness"]["max_age"] = "1h"
    stale.config_path.write_text(yaml.safe_dump(document))
    assert _invoke(zone, "--source-id", "fixture.fresh", "--source-id", "fixture.stale") == 1
    text = capsys.readouterr().out
    assert "Note: fixture.fresh, fixture.stale share a metadata path." in text
    assert str(zone.root) not in text
    assert "data/metadata" not in text
    assert "fixture.stale" in text.splitlines()[-1]


def test_now_with_an_offset_is_rendered_in_utc(zone, capsys):
    assert _invoke(zone, "--now", "2026-10-09T09:00:00-03:00", "--format", "json") == 1
    output = json.loads(capsys.readouterr().out)
    assert output["evaluated_at"] == zone.now.isoformat()
    assert output["sources"][1]["age_seconds"] == 7200.0


def test_default_now_reads_one_utc_clock_at_the_cli_boundary(zone, monkeypatch, capsys):
    import janus.cli.status as status

    calls = []

    class FixedClock:
        @staticmethod
        def now(tz):
            calls.append(tz)
            return zone.now

    monkeypatch.setattr(status, "datetime", FixedClock)
    argv = ["status", "--project-root", str(zone.root), "--format", "json"]
    assert main(argv) == 1
    assert json.loads(capsys.readouterr().out)["evaluated_at"] == zone.now.isoformat()
    assert calls == [UTC]


def test_future_success_keeps_a_negative_age_visible(zone, capsys):
    assert (
        _invoke(
            zone, "--source-id", "fixture.fresh", "--now", (NOW - timedelta(hours=3)).isoformat()
        )
        == 0
    )
    assert "-1h" in capsys.readouterr().out


def test_text_ages_show_hours_and_minutes_without_calendar_arithmetic(zone, capsys):
    assert _invoke(zone, "--now", (NOW + timedelta(minutes=35)).isoformat()) == 1
    text = capsys.readouterr().out
    assert "50h 35m" in text
    assert "2h 35m" in text
    assert "2026-10-07T10:00:00Z" in text


def test_status_does_not_create_runtime_or_metrics_output(zone, capsys):
    write_profile(zone.root, {"transport": "textfile", "path": "metrics"})
    before = sorted(path.relative_to(zone.root) for path in zone.root.rglob("*"))
    assert _invoke(zone) == 1
    capsys.readouterr()
    assert sorted(path.relative_to(zone.root) for path in zone.root.rglob("*")) == before


def test_status_module_stays_under_task_size_ceiling():
    import janus.cli.status as status

    assert len(Path(status.__file__).read_text().splitlines()) < 300


def test_table_selection_filters_one_scan_and_uses_the_configured_catalog(
    zone, monkeypatch, capsys
):
    import pyiceberg.catalog
    from pyiceberg.expressions import And, EqualTo, In

    from janus.utils.catalog_properties import derive_pyiceberg_catalog_properties
    from janus.utils.environment import load_environment_config, materialize_runtime_paths

    profile = zone.root / "conf" / "environments" / "local.yaml"
    document = yaml.safe_load(profile.read_text())
    document["spark"]["iceberg"]["catalog_name"] = "configured"
    document["observability"] = {"runs_table": "operator.history"}
    profile.write_text(yaml.safe_dump(document))
    config = load_environment_config("local", zone.root)
    expected_properties = derive_pyiceberg_catalog_properties(
        config, materialize_runtime_paths(config, zone.root)
    )
    selected = ["fixture.fresh", "fixture.undeclared"]
    rows = [
        {"source_id": source.source_id, "ended_at": run.ended_at, "freshness_max_age_seconds": 1}
        for source in zone.sources
        for run in source.runs
        if run.status == "succeeded" and source.source_id in selected
    ]
    scans, catalogs, tables = [], [], []

    def scan(**kwargs):
        scans.append(kwargs)
        assert kwargs["row_filter"] == And(
            EqualTo("status", "succeeded"), In("source_id", selected)
        )
        return SimpleNamespace(to_arrow=lambda: SimpleNamespace(to_pylist=lambda: rows))

    def load_table(identifier):
        tables.append(identifier)
        return SimpleNamespace(scan=scan)

    def load_catalog(name, **properties):
        catalogs.append((name, properties))
        return SimpleNamespace(load_table=load_table)

    monkeypatch.setattr(pyiceberg.catalog, "load_catalog", load_catalog)
    assert (
        _invoke(
            zone,
            "--from-runs-table",
            "--source-id",
            selected[1],
            "--source-id",
            selected[0],
            "--source-id",
            selected[0],
            "--format",
            "json",
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert [row["source_id"] for row in report["sources"]] == selected
    assert report["totals"]["breaching"] == 0
    assert len(scans) == 1
    assert catalogs == [("configured", expected_properties)]
    assert tables == ["operator.history"]

    fresh = next(source for source in zone.sources if source.source_id == "fixture.fresh")
    document = yaml.safe_load(fresh.config_path.read_text())
    document["operations"]["freshness"]["max_age"] = "1h"
    fresh.config_path.write_text(yaml.safe_dump(document))
    assert (
        _invoke(
            zone,
            "--from-runs-table",
            "--source-id",
            selected[0],
            "--source-id",
            selected[1],
            "--format",
            "json",
        )
        == 1
    )
    changed = json.loads(capsys.readouterr().out)
    assert changed["sources"][0]["outcome"] == "stale"
    assert changed["sources"][0]["max_age_seconds"] == 3600


def test_unknown_table_source_is_rejected_before_catalog_access(zone, monkeypatch, capsys):
    import janus.cli.status as status

    def unexpected_read(*args, **kwargs):
        pytest.fail("an unknown selection must not read the catalog")

    monkeypatch.setattr(status, "read_latest_success_from_runs_table", unexpected_read)
    assert _invoke(zone, "--from-runs-table", "--source-id", "fixture.missing") == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fixture.missing" in captured.err


def test_missing_profile_is_a_named_configuration_error(zone, capsys):
    assert _invoke(zone, "--environment", "missing") == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing.yaml" in captured.err
