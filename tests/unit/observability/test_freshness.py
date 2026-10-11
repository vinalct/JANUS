"""FR-7: latest successful metadata evidence and deterministic freshness answers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from janus.lineage.persistence import write_json_atomic
from janus.observability.freshness import (
    BREACHING_OUTCOMES,
    FRESH,
    NEVER_RAN,
    STALE,
    UNDECLARED,
    LatestSuccess,
    SourceFreshness,
    evaluate_freshness,
    evaluate_sources,
    read_latest_success,
)
from janus.registry import SourceNotFoundError, load_registry
from tests.support.metadata_zone_fixture import build_metadata_zone

NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)
MAX_AGE = 86400


@pytest.fixture
def zone(tmp_path):
    return build_metadata_zone(tmp_path / "project", now=NOW, seed=23)


def _write_run(runs_dir: Path, name: str, *, ended_at: datetime, status="succeeded") -> Path:
    return write_json_atomic(
        runs_dir / f"{name}.json", {"status": status, "ended_at": ended_at.isoformat()}
    )


def test_undeclared_window_reports_age_without_a_breach():
    assert evaluate_freshness(
        max_age_seconds=None, last_success_at=NOW - timedelta(days=10), now=NOW
    ) == (UNDECLARED, 864000.0)


def test_declared_window_with_no_success_is_never_ran():
    assert evaluate_freshness(max_age_seconds=MAX_AGE, last_success_at=None, now=NOW) == (
        NEVER_RAN,
        None,
    )


def test_success_within_window_is_fresh():
    assert evaluate_freshness(
        max_age_seconds=MAX_AGE, last_success_at=NOW - timedelta(hours=2), now=NOW
    ) == (FRESH, 7200.0)


def test_success_older_than_window_is_stale():
    assert evaluate_freshness(
        max_age_seconds=MAX_AGE, last_success_at=NOW - timedelta(hours=50), now=NOW
    ) == (STALE, 180000.0)


def test_age_equal_to_window_is_fresh():
    assert evaluate_freshness(
        max_age_seconds=MAX_AGE, last_success_at=NOW - timedelta(seconds=MAX_AGE), now=NOW
    ) == (FRESH, float(MAX_AGE))


def test_age_one_second_over_window_is_stale():
    assert evaluate_freshness(
        max_age_seconds=MAX_AGE, last_success_at=NOW - timedelta(seconds=MAX_AGE + 1), now=NOW
    ) == (STALE, float(MAX_AGE + 1))


def test_future_success_preserves_negative_age():
    assert evaluate_freshness(
        max_age_seconds=MAX_AGE, last_success_at=NOW + timedelta(hours=1), now=NOW
    ) == (FRESH, -3600.0)


def test_undeclared_window_without_history_has_no_age():
    assert evaluate_freshness(max_age_seconds=None, last_success_at=None, now=NOW) == (
        UNDECLARED,
        None,
    )


def test_different_offsets_compare_instants_in_utc():
    offset = timezone(timedelta(hours=-3))
    assert evaluate_freshness(
        max_age_seconds=MAX_AGE,
        last_success_at=(NOW - timedelta(hours=2)).astimezone(offset),
        now=NOW,
    ) == (FRESH, 7200.0)


def test_daylight_saving_fold_uses_elapsed_utc_seconds():
    zone = ZoneInfo("America/New_York")
    first = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=0)
    second = first.replace(fold=1)
    assert evaluate_freshness(max_age_seconds=1800, last_success_at=first, now=second) == (
        STALE,
        3600.0,
    )


@pytest.mark.parametrize("max_age_seconds", [None, MAX_AGE])
@pytest.mark.parametrize("last_success_at", [None, NOW])
def test_naive_now_is_rejected_even_without_a_window_or_history(max_age_seconds, last_success_at):
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        evaluate_freshness(
            max_age_seconds=max_age_seconds,
            last_success_at=last_success_at,
            now=NOW.replace(tzinfo=None),
        )


@pytest.mark.parametrize("max_age_seconds", [None, MAX_AGE])
def test_naive_last_success_is_rejected_even_without_a_window(max_age_seconds):
    with pytest.raises(ValueError, match="last_success_at must be timezone-aware"):
        evaluate_freshness(
            max_age_seconds=max_age_seconds, last_success_at=NOW.replace(tzinfo=None), now=NOW
        )


class UndefinedOffset(tzinfo):
    def utcoffset(self, dt):
        return None


@pytest.mark.parametrize("field_name", ["now", "last_success_at"])
def test_timezone_without_an_offset_is_rejected(field_name):
    arguments = {"max_age_seconds": MAX_AGE, "last_success_at": NOW, "now": NOW}
    arguments[field_name] = NOW.replace(tzinfo=UndefinedOffset())
    with pytest.raises(ValueError, match=f"{field_name} must be timezone-aware"):
        evaluate_freshness(**arguments)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [(FRESH, False), (STALE, True), (NEVER_RAN, True), (UNDECLARED, False)],
)
def test_source_record_has_the_shared_breach_rule(outcome, expected):
    result = SourceFreshness("fixture.source", outcome, MAX_AGE, NOW, 0.0, 1, 0, "/metadata")
    assert frozenset({STALE, NEVER_RAN}) == BREACHING_OUTCOMES
    assert result.is_breach is expected
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.outcome = FRESH


def test_latest_success_record_is_frozen_and_slotted():
    result = LatestSuccess(None, 0, 0)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.runs_scanned = 1


def test_reader_uses_maximum_success_time_not_mtime_or_filename(tmp_path, monkeypatch):
    old = _write_run(tmp_path, "a-old-success", ended_at=NOW - timedelta(hours=50))
    latest = _write_run(tmp_path, "b-latest-success", ended_at=NOW - timedelta(hours=2))
    failed = _write_run(tmp_path, "c-new-failure", ended_at=NOW, status="failed")
    running = _write_run(tmp_path, "d-running", ended_at=NOW + timedelta(hours=1), status="running")
    for path, modified in [(latest, 1), (old, 2), (running, 3), (failed, 4)]:
        os.utime(path, (modified, modified))
    original_glob = Path.glob
    monkeypatch.setattr(
        Path, "glob", lambda path, pattern: iter(sorted(original_glob(path, pattern)))
    )
    assert latest.stat().st_mtime < old.stat().st_mtime < failed.stat().st_mtime
    assert read_latest_success(tmp_path) == LatestSuccess(NOW - timedelta(hours=2), 4, 0)


def test_reader_normalizes_offsets_before_selecting_latest(tmp_path):
    offset = timezone(timedelta(hours=-3))
    _write_run(tmp_path, "utc", ended_at=NOW - timedelta(hours=2))
    _write_run(tmp_path, "offset", ended_at=(NOW - timedelta(hours=1)).astimezone(offset))
    result = read_latest_success(tmp_path)
    assert result == LatestSuccess(NOW - timedelta(hours=1), 2, 0)
    assert result.last_success_at.tzinfo is UTC


@pytest.mark.parametrize(
    "contents",
    [
        b'{"run_id":',
        b"[]",
        b"null",
        b"42",
        b"{}",
        b'{"status": null}',
        b'{"status": 1}',
        b'{"status": "succeeded"}',
        b'{"status": "succeeded", "ended_at": null}',
        b'{"status": "succeeded", "ended_at": 123}',
        b'{"status": "succeeded", "ended_at": "invalid"}',
        b'{"status": "succeeded", "ended_at": "2026-10-10T10:00:00"}',
        b'{"status": "succeeded", "ended_at": "2026-10-10"}',
        b"\xff",
    ],
    ids=[
        "truncated",
        "array",
        "null-document",
        "scalar",
        "missing-status",
        "null-status",
        "numeric-status",
        "missing-end",
        "null-end",
        "numeric-end",
        "invalid-end",
        "naive-end",
        "date-only-end",
        "invalid-utf8",
    ],
)
def test_bad_run_file_is_counted_and_does_not_hide_a_success(tmp_path, contents):
    _write_run(tmp_path, "success", ended_at=NOW)
    (tmp_path / "bad.json").write_bytes(contents)
    assert read_latest_success(tmp_path) == LatestSuccess(NOW, 2, 1)


def test_unreadable_file_is_counted_without_raising(tmp_path, monkeypatch):
    good = _write_run(tmp_path, "success", ended_at=NOW)
    bad = _write_run(tmp_path, "denied", ended_at=NOW + timedelta(hours=1))
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == bad:
            raise PermissionError("permission denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert good.exists()
    assert read_latest_success(tmp_path) == LatestSuccess(NOW, 2, 1)


def test_disappearing_file_is_counted_without_raising(tmp_path, monkeypatch):
    _write_run(tmp_path, "success", ended_at=NOW)
    vanished = _write_run(tmp_path, "vanished", ended_at=NOW + timedelta(hours=1))
    original = Path.exists

    def exists(path, *args, **kwargs):
        return False if path == vanished else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)
    assert read_latest_success(tmp_path) == LatestSuccess(NOW, 2, 1)


def test_failed_and_running_runs_need_no_completion_time(tmp_path):
    write_json_atomic(tmp_path / "failed.json", {"status": "failed", "ended_at": "invalid"})
    write_json_atomic(tmp_path / "running.json", {"status": "running"})
    assert read_latest_success(tmp_path) == LatestSuccess(None, 2, 0)


@pytest.mark.parametrize("directory_exists", [False, True])
def test_missing_or_empty_runs_directory_has_no_history(tmp_path, directory_exists):
    runs_dir = tmp_path / "runs"
    if directory_exists:
        runs_dir.mkdir()
    assert read_latest_success(runs_dir) == LatestSuccess(None, 0, 0)
    assert runs_dir.exists() is directory_exists


def test_atomic_staging_file_is_excluded_by_the_actual_glob(tmp_path, monkeypatch):
    _write_run(tmp_path, "success", ended_at=NOW - timedelta(hours=50))
    original = Path.replace

    def interrupt_replace(path, target):
        raise OSError("interrupted before atomic replace")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt_replace)
        with pytest.raises(OSError, match="interrupted"):
            _write_run(tmp_path, "pending", ended_at=NOW)
    assert Path.replace is original
    staging_files = list(tmp_path.glob(".pending.json.*.tmp"))
    assert len(staging_files) == 1
    candidates = list(tmp_path.glob("*.json"))
    assert staging_files[0] not in candidates
    assert len(candidates) == 1
    assert read_latest_success(tmp_path) == LatestSuccess(NOW - timedelta(hours=50), 1, 0)


def test_real_registry_synthetic_zone_has_all_six_outcomes_and_counts(zone):
    registry = load_registry(zone.root)
    results = evaluate_sources(registry, zone.root, now=zone.now)
    assert isinstance(results, tuple)
    assert {result.source_id: result.outcome for result in results} == zone.expected_outcomes
    assert [result.source_id for result in results] == sorted(zone.expected_outcomes)
    by_id = {result.source_id: result for result in results}
    assert {result.source_id: result.runs_scanned for result in results} == {
        "fixture.empty": 0,
        "fixture.fresh": 3,
        "fixture.never": 2,
        "fixture.stale": 2,
        "fixture.undeclared": 1,
        "fixture.unreadable": 2,
    }
    assert {result.source_id: result.unreadable_run_files for result in results} == {
        source.source_id: source.unreadable_run_files for source in zone.sources
    }
    assert by_id["fixture.stale"].last_success_at == NOW - timedelta(hours=50)
    assert by_id["fixture.stale"].age_seconds == 180000.0
    assert by_id["fixture.stale"].max_age_seconds == MAX_AGE
    assert by_id["fixture.empty"].last_success_at is None
    assert by_id["fixture.empty"].is_breach
    assert by_id["fixture.unreadable"].outcome == FRESH
    assert by_id["fixture.undeclared"].max_age_seconds is None
    assert by_id["fixture.undeclared"].age_seconds == 720000.0
    assert sum(result.is_breach for result in results) == 3
    for source in zone.sources:
        assert by_id[source.source_id].metadata_path == str(source.metadata_dir)


def test_all_unreadable_history_reports_never_ran_and_exposes_the_count(zone):
    source = next(source for source in zone.sources if source.source_id == "fixture.unreadable")
    for run in source.runs:
        (source.metadata_dir / "runs" / f"{run.run_id}.json").unlink()
    (source.metadata_dir / "runs" / "non-mapping.json").write_text("[]", encoding="utf-8")
    (result,) = evaluate_sources(
        load_registry(zone.root), zone.root, now=NOW, source_ids=[source.source_id]
    )
    assert result.outcome == NEVER_RAN
    assert result.is_breach
    assert result.last_success_at is result.age_seconds is None
    assert result.runs_scanned == result.unreadable_run_files == 2


def test_disabled_sources_are_included_and_can_breach(zone):
    source = next(source for source in zone.sources if source.source_id == "fixture.stale")
    document = yaml.safe_load(source.config_path.read_text(encoding="utf-8"))
    document["enabled"] = False
    source.config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    registry = load_registry(zone.root)
    assert not registry.get_source(source.source_id, include_disabled=True).enabled
    results = evaluate_sources(registry, zone.root, now=NOW)
    assert len(results) == len(zone.sources)
    result = next(result for result in results if result.source_id == source.source_id)
    assert result.outcome == STALE
    assert result.is_breach
    assert evaluate_sources(registry, zone.root, now=NOW, source_ids=[source.source_id]) == (
        result,
    )


def test_shared_metadata_path_is_scanned_for_each_source_without_run_id_attribution(zone):
    source = next(source for source in zone.sources if source.source_id == "fixture.stale")
    config = load_registry(zone.root).get_source("fixture.fresh")
    document = yaml.safe_load(source.config_path.read_text(encoding="utf-8"))
    document["outputs"]["metadata"]["path"] = config.outputs.metadata.path
    document["operations"]["freshness"]["max_age"] = "1h"
    source.config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    fresh, stale = evaluate_sources(
        load_registry(zone.root),
        zone.root,
        now=NOW,
        source_ids=["fixture.stale", "fixture.fresh"],
    )
    assert fresh.metadata_path == stale.metadata_path
    assert fresh.last_success_at == stale.last_success_at == NOW - timedelta(hours=2)
    assert fresh.runs_scanned == stale.runs_scanned == 3
    assert fresh.outcome == FRESH
    assert stale.outcome == STALE
    assert stale.max_age_seconds == 3600


def test_absolute_metadata_paths_are_used_as_configured(zone):
    source = next(source for source in zone.sources if source.source_id == "fixture.fresh")
    document = yaml.safe_load(source.config_path.read_text(encoding="utf-8"))
    document["outputs"]["metadata"]["path"] = str(source.metadata_dir)
    source.config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    (result,) = evaluate_sources(
        load_registry(zone.root), zone.root, now=NOW, source_ids=[source.source_id]
    )
    assert result.outcome == FRESH
    assert result.metadata_path == str(source.metadata_dir)


def test_selected_sources_are_unique_sorted_and_do_not_scan_unselected_histories(zone, monkeypatch):
    registry = load_registry(zone.root)
    original = Path.glob
    scanned = []

    def glob(path, *args, **kwargs):
        scanned.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    results = evaluate_sources(
        registry,
        zone.root,
        now=NOW,
        source_ids=["fixture.undeclared", "fixture.fresh", "fixture.fresh"],
    )
    assert [result.source_id for result in results] == ["fixture.fresh", "fixture.undeclared"]
    assert scanned == [Path(result.metadata_path) / "runs" for result in results]


def test_empty_selection_returns_no_sources(zone):
    assert evaluate_sources(load_registry(zone.root), zone.root, now=NOW, source_ids=[]) == ()


def test_unknown_selected_source_is_a_named_registry_error(zone):
    with pytest.raises(SourceNotFoundError, match="fixture.missing"):
        evaluate_sources(
            load_registry(zone.root), zone.root, now=NOW, source_ids=["fixture.missing"]
        )


def test_empty_selection_still_rejects_naive_now(zone):
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        evaluate_sources(
            load_registry(zone.root), zone.root, now=NOW.replace(tzinfo=None), source_ids=[]
        )


def test_import_and_real_zone_evaluation_load_no_compute_or_catalog_engine(zone):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    script = """
import json, sys
from datetime import datetime
from pathlib import Path
import janus.observability.freshness as freshness
engines = ('pyspark', 'pyiceberg', 'pyarrow')
def loaded():
    return sorted(name for name in sys.modules if name.split('.', 1)[0] in engines)
imported = loaded()
from janus.registry import load_registry
root = Path(sys.argv[1])
results = freshness.evaluate_sources(
    load_registry(root), root, now=datetime.fromisoformat(sys.argv[2]))
print(json.dumps({'imported': imported, 'evaluated': loaded(), 'sources': len(results)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(zone.root), NOW.isoformat()],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"imported": [], "evaluated": [], "sources": 6}


def test_freshness_module_stays_under_task_size_ceiling():
    import janus.observability.freshness as freshness

    path = Path(freshness.__file__)
    assert len(path.read_text(encoding="utf-8").splitlines()) < 300
