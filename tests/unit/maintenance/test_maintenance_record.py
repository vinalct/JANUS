"""FR-9 / AC-8: complete, comparable, safe maintenance evidence."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta, timezone
from pathlib import Path

import pytest

from janus.lineage.persistence import MetadataZonePaths
from janus.maintenance import records
from janus.maintenance.planning import PlannedItem, RetentionPlan
from janus.maintenance.records import (
    MAINTENANCE_DIRECTORY,
    MAINTENANCE_ITEM_APPLIED,
    MAINTENANCE_ITEM_FAILED,
    MAINTENANCE_ITEM_PLANNED,
    MAINTENANCE_ITEM_SKIPPED,
    MAINTENANCE_RECORD_SCHEMA_VERSION,
    ItemOutcome,
    MaintenanceRecord,
    MaintenanceRecordStore,
    ZoneSummary,
    default_maintenance_run_id,
    validate_maintenance_run_id,
)
from janus.strategies.http.errors import RESPONSE_BODY_EXCERPT_LIMIT
from janus.utils.logging import REDACTED_VALUE, build_structured_logger
from janus.utils.storage import StorageLayout


@pytest.fixture
def plan(now):
    return RetentionPlan(
        items=(
            PlannedItem(
                "bronze",
                "bronze.example",
                "expire_snapshots",
                {"older_than": now.isoformat(), "retain_last": "2", "snapshot_ids": "[11, 22]"},
            ),
            PlannedItem("metadata", "/metadata/example/runs/old.json", "delete_file", {}, 128),
            PlannedItem("raw", "example", "delete_prefix", {}, skipped_reason="legacy_progress"),
        ),
        protected=(),
        now=now,
        policy_digest="a" * 64,
    )


@pytest.fixture
def store(tmp_path):
    return MaintenanceRecordStore(
        StorageLayout(
            tmp_path,
            tmp_path / "data",
            tmp_path / "raw",
            tmp_path / "bronze",
            tmp_path / "metadata",
        )
    )


def _record(plan, *, dry_run=True, items=None, start_offset=0, end_offset=0.5, **kwargs):
    return MaintenanceRecord.from_plan(
        plan,
        environment="local",
        dry_run=dry_run,
        zones={"bronze", "metadata", "lineage", "raw"},
        started_at=plan.now + timedelta(seconds=start_offset),
        ended_at=plan.now + timedelta(seconds=end_offset),
        items=items,
        **kwargs,
    )


def _applied_items(plan):
    return tuple(
        replace(item, status="applied") if item.status == "planned" else item
        for item in (ItemOutcome.from_planned_item(candidate) for candidate in plan.items)
    )


def _field_diff(left, right, path=""):
    """Compare every field and reject shape changes before comparing values."""
    assert type(left) is type(right), path
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        return set().union(
            *(_field_diff(left[key], right[key], f"{path}.{key}".lstrip(".")) for key in left)
        )
    if isinstance(left, list):
        assert len(left) == len(right), path
        return set().union(
            *(
                _field_diff(a, b, f"{path}[{index}]")
                for index, (a, b) in enumerate(zip(left, right, strict=True))
            )
        )
    return {path} if left != right else set()


def test_dry_run_and_apply_have_the_same_shape_and_plan_with_exact_field_diff(plan, store):
    dry = _record(plan)
    applied = _record(plan, dry_run=False, items=_applied_items(plan), start_offset=1, end_offset=3)
    dry_payload = json.loads(store.persist(dry).read_text())
    applied_payload = json.loads(store.persist(applied).read_text())
    assert dry.plan_digest == applied.plan_digest == plan.digest
    assert dry.policy_digest == applied.policy_digest == plan.policy_digest
    assert _field_diff(dry_payload, applied_payload) == {
        "dry_run",
        "maintenance_run_id",
        "started_at",
        "ended_at",
        "duration_seconds",
        "items[0].status",
        "items[1].status",
        "zone_summaries[0].items_applied",
        "zone_summaries[2].items_applied",
    }
    assert dry_payload["dry_run"] is True
    assert applied_payload["dry_run"] is False


def test_every_planned_item_survives_a_failure_and_other_items_can_apply(plan):
    outcomes = list(_applied_items(plan))
    outcomes[0] = replace(
        outcomes[0],
        status="failed",
        removed_count=None,
        expired_snapshot_ids=(),
        failure_type="CatalogError",
        failure_message="catalog refused expiration",
        duration_seconds=0.25,
    )
    record = _record(plan, dry_run=False, items=tuple(outcomes))
    assert len(record.items) == len(plan.items)
    assert [(item.zone, item.target, item.action) for item in record.items] == [
        (item.zone, item.target, item.action) for item in plan.items
    ]
    assert [item.status for item in record.items] == ["failed", "applied", "skipped"]
    assert record.has_failures
    assert record.zone_summaries[0] == ZoneSummary("bronze", 1, 0, 0, 1, 0, 0)
    assert record.zone_summaries[2] == ZoneSummary("metadata", 1, 1, 0, 0, 1, 128)


@pytest.mark.parametrize("status", ["planned", "applied", "skipped", "failed"])
def test_has_failures_drives_exit_code_for_each_status(plan, status):
    plan = replace(plan, items=plan.items[:1])
    outcome = replace(
        ItemOutcome.from_planned_item(plan.items[0]),
        status=status,
        failure_type="RuntimeError" if status == "failed" else None,
        failure_message="failure" if status == "failed" else None,
    )
    record = _record(plan, dry_run=status == "planned", items=(outcome,))
    assert (1 if record.has_failures else 0) == (1 if status == "failed" else 0)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "reordered", "target", "action", "detail"]
)
def test_applied_records_refuse_dropped_added_or_misattributed_outcomes(plan, change):
    outcomes = _applied_items(plan)
    if change == "missing":
        outcomes = outcomes[:-1]
    elif change == "extra":
        outcomes += outcomes[:1]
    elif change == "duplicate":
        outcomes = (outcomes[0], outcomes[0], outcomes[2])
    elif change == "reordered":
        outcomes = outcomes[::-1]
    else:
        value = {} if change == "detail" else "wrong"
        outcomes = (replace(outcomes[0], **{change: value}), *outcomes[1:])
    with pytest.raises(
        ValueError, match="one outcome per planned item|must match every planned item"
    ):
        _record(plan, dry_run=False, items=outcomes)


def test_apply_requires_explicit_outcomes_and_empty_plans_are_valid(plan):
    with pytest.raises(ValueError, match="requires an outcome for every planned item"):
        _record(plan, dry_run=False)
    empty = replace(plan, items=())
    for dry_run in (True, False):
        record = _record(empty, dry_run=dry_run)
        assert record.items == () and not record.has_failures
        assert all(
            summary.items_planned == summary.removed_bytes == 0 for summary in record.zone_summaries
        )


def test_record_includes_zero_summaries_for_selected_zones_and_sorted_source_filter(plan):
    record = _record(plan, source_ids=("z", "a", "z"))
    assert record.source_ids == ("a", "z")
    assert record.zones == ("bronze", "lineage", "metadata", "raw")
    assert record.zone_summaries[1] == ZoneSummary("lineage", 0, 0, 0, 0, 0, 0)
    assert record.lock == "none"
    assert _record(plan).source_ids == ()


def test_record_refuses_an_item_outside_selected_zones(plan):
    plan = replace(plan, items=(replace(plan.items[0], zone="runs-table"),))
    with pytest.raises(ValueError, match="every item zone must be included"):
        _record(plan)


@pytest.mark.parametrize("dry_run", [True, False])
def test_record_refuses_statuses_inconsistent_with_dry_run_flag(plan, dry_run):
    items = (
        _applied_items(plan)
        if dry_run
        else tuple(ItemOutcome.from_planned_item(item) for item in plan.items)
    )
    with pytest.raises(ValueError, match="invalid for dry_run"):
        _record(plan, dry_run=dry_run, items=items)


def test_dry_run_estimates_and_applied_measurements_share_the_same_fields(plan):
    dry = _record(plan)
    outcomes = list(_applied_items(plan))
    outcomes[1] = replace(outcomes[1], removed_bytes=192)
    applied = _record(plan, dry_run=False, items=tuple(outcomes))
    assert dry.items[1].removed_bytes == dry.zone_summaries[2].removed_bytes == 128
    assert applied.items[1].removed_bytes == applied.zone_summaries[2].removed_bytes == 192
    assert dry.items[1].to_dict().keys() == applied.items[1].to_dict().keys()
    assert dry.items[0].expired_snapshot_ids == (11, 22)
    assert dry.items[0].removed_count == 2
    assert dry.items[2].detail["skipped_reason"] == "legacy_progress"
    assert dry.items[2].removed_count is dry.items[2].removed_bytes is None


def test_summary_retains_measured_partial_removals_on_failure(plan):
    item = replace(
        _applied_items(plan)[1],
        status="failed",
        failure_type="OSError",
        failure_message="partial deletion",
    )
    assert ZoneSummary.from_items("metadata", (item,)) == ZoneSummary(
        "metadata", 1, 0, 0, 1, 1, 128
    )


def test_partition_counts_and_unknown_procedure_counts_are_honest():
    item = PlannedItem("runs-table", "2026-01-01", "delete_partition", {"row_count": "12"})
    assert ItemOutcome.from_planned_item(item).removed_count == 12
    for action in ("remove_orphan_files", "rewrite_data_files", "delete_prefix"):
        assert (
            ItemOutcome.from_planned_item(PlannedItem("bronze", "table", action, {})).removed_count
            is None
        )


def test_id_is_reproducible_for_plan_and_utc_second_and_changes_for_new_plan(plan):
    expected = f"maintenance-20261005T120000Z-{plan.digest[:8]}"
    assert default_maintenance_run_id(plan.digest, plan.now) == expected
    assert (
        default_maintenance_run_id(plan.digest, plan.now + timedelta(microseconds=999999))
        == expected
    )
    local = plan.now.astimezone(timezone(timedelta(hours=-3)))
    assert default_maintenance_run_id(plan.digest, local) == expected
    assert default_maintenance_run_id("b" * 64, plan.now) != expected
    assert default_maintenance_run_id(plan.digest, plan.now + timedelta(seconds=1)) != expected


@pytest.mark.parametrize("digest", ["", "abc", "a" * 63, "g" * 64, "A" * 64, "a" * 64 + "/escape"])
def test_id_derivation_refuses_invalid_digest(plan, digest):
    with pytest.raises(ValueError, match="SHA-256"):
        default_maintenance_run_id(digest, plan.now)


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../escape",
        "/tmp/escape",
        "a/b",
        "a\\b",
        "a..b",
        "a\x00b",
        "a\nb",
        "a\nb\n",
        "a ",
        "a" * 97,
    ],
)
def test_store_and_record_refuse_unsafe_ids_before_writing(plan, store, run_id):
    with pytest.raises(
        ValueError,
        match="maintenance_run_id.*path component",
    ):
        store.record_path(run_id)
    with pytest.raises(ValueError, match="maintenance_run_id.*path component"):
        replace(_record(plan), maintenance_run_id=run_id)
    assert not store.storage_layout.metadata_dir.exists()


def test_store_revalidates_id_before_persisting(plan, store):
    record = _record(plan)
    object.__setattr__(record, "maintenance_run_id", "../escape")
    with pytest.raises(ValueError, match="path component"):
        store.persist(record)
    assert not store.storage_layout.metadata_dir.exists()


def test_valid_id_is_unchanged_and_store_uses_shared_metadata_root(plan, store):
    record = _record(plan)
    assert validate_maintenance_run_id(record.maintenance_run_id) == record.maintenance_run_id
    assert store.record_path(record.maintenance_run_id) == (
        store.storage_layout.metadata_dir
        / MAINTENANCE_DIRECTORY
        / f"{record.maintenance_run_id}.json"
    )


def test_longest_valid_id_can_be_persisted_by_the_atomic_writer(plan, store):
    record = replace(_record(plan), maintenance_run_id="a" * 96)
    assert (
        json.loads(store.persist(record).read_text())["maintenance_run_id"]
        == record.maintenance_run_id
    )


@pytest.mark.parametrize("symlink_kind", ["directory", "file"])
def test_store_refuses_resolved_symlink_escape(plan, store, tmp_path, symlink_kind):
    record = _record(plan)
    root = store.storage_layout.metadata_dir
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    directory = root / MAINTENANCE_DIRECTORY
    if symlink_kind == "directory":
        directory.symlink_to(outside, target_is_directory=True)
    else:
        directory.mkdir()
        (directory / f"{record.maintenance_run_id}.json").symlink_to(outside / "record.json")
    with pytest.raises(ValueError, match="escapes metadata root"):
        store.persist(record)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "message",
    [
        "Catalog failed: jdbc:postgresql://db.example/janus?user=janus&password=planted-password",
        "Catalog failed: jdbc:postgresql://janus:planted-password@db.example/janus",
        "Catalog failed: jdbc:sqlserver://db.example;user=janus;password=planted-password;database=janus",
        "Catalog failed: jdbc:sqlite:/tmp/catalog.db?password=planted-password",
        "Catalog failed: https://user:planted-password@db.example/catalog?token=another-secret",
        "Catalog failed: password = 'planted-password'",
        'Catalog failed: jdbc:sqlserver://db;password="planted-password with spaces";user=janus',
        "Catalog failed: jdbc:sqlserver://db;password={planted-password with spaces};user=janus",
        "x" * (RESPONSE_BODY_EXCERPT_LIMIT - 10) + " password=planted-password",
        "x" * 10000 + " jdbc:postgresql://db/catalog?password=planted-password",
    ],
)
def test_failure_message_is_redacted_and_bounded_in_persisted_json(plan, store, message):
    outcomes = list(_applied_items(plan))
    outcomes[0] = replace(
        outcomes[0], status="failed", failure_type="CatalogError", failure_message=message
    )
    record = _record(plan, dry_run=False, items=tuple(outcomes))
    payload = store.persist(record).read_text()
    assert "planted-password" not in payload
    assert "another-secret" not in payload
    assert "jdbc:" not in payload and "https://" not in payload
    failure = json.loads(payload)["items"][0]["failure_message"]
    assert len(failure) <= RESPONSE_BODY_EXCERPT_LIMIT
    if len(message) < RESPONSE_BODY_EXCERPT_LIMIT:
        assert REDACTED_VALUE in failure
    assert outcomes[0].failure_message == failure


def test_failure_message_normalizes_lines_without_changing_short_diagnostics(plan):
    item = replace(
        _applied_items(plan)[0],
        status="failed",
        failure_type="OSError",
        failure_message="  table failed\n\ttry again  ",
    )
    assert item.failure_message == "table failed try again"


@pytest.mark.parametrize("status", ["planned", "applied", "skipped", "failed"])
def test_structured_events_use_constants_and_only_safe_required_fields(plan, status):
    events = {
        "planned": MAINTENANCE_ITEM_PLANNED,
        "applied": MAINTENANCE_ITEM_APPLIED,
        "skipped": MAINTENANCE_ITEM_SKIPPED,
        "failed": MAINTENANCE_ITEM_FAILED,
    }
    stream = io.StringIO()
    logger = build_structured_logger(f"janus.tests.maintenance.{status}", stream=stream)
    item = ItemOutcome(
        "bronze",
        "jdbc:postgresql://user:planted-password@db/catalog",
        "expire_snapshots",
        status,
        {"payload": "must-not-appear", "password": "detail-secret"},
        failure_type="CatalogError" if status == "failed" else None,
        failure_message="Catalog failed: jdbc:postgresql://db/catalog?password=planted-password"
        if status == "failed"
        else None,
    )
    run_id = default_maintenance_run_id(plan.digest, plan.now)
    item.log(logger, maintenance_run_id=run_id)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == events[status] == f"maintenance_item_{status}"
    assert event["fields"] | {} == {
        "zone": "bronze",
        "target": REDACTED_VALUE,
        "action": "expire_snapshots",
        "maintenance_run_id": run_id,
        **({"failure_message": item.failure_message} if status == "failed" else {}),
    }
    assert event["level"] == ("ERROR" if status == "failed" else "INFO")
    for excluded in ("planted-password", "detail-secret", "must-not-appear", "jdbc:"):
        assert excluded not in lines[0]


def test_json_shape_is_complete_round_trippable_and_byte_stable(plan):
    record = _record(plan)
    payload = record.to_dict()
    assert set(payload) == {
        "maintenance_run_id",
        "schema_version",
        "environment",
        "dry_run",
        "zones",
        "source_ids",
        "policy_digest",
        "plan_digest",
        "lock",
        "started_at",
        "ended_at",
        "duration_seconds",
        "zone_summaries",
        "items",
        "failures",
        "protected",
    }
    assert payload["schema_version"] == MAINTENANCE_RECORD_SCHEMA_VERSION == 1
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == payload
    assert encoded == json.dumps(record.to_dict(), sort_keys=True, allow_nan=False)
    payload["items"][0]["detail"]["retain_last"] = "99"
    assert record.items[0].detail["retain_last"] == "2"


def test_evidence_dataclasses_and_item_arguments_are_immutable(plan, store):
    record = _record(plan)
    for value, field, replacement in (
        (record, "dry_run", False),
        (record.items[0], "status", "applied"),
        (record.zone_summaries[0], "items_planned", 0),
        (store, "storage_layout", None),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)
        assert not hasattr(value, "__dict__")
    detail = {"retain_last": "2"}
    item = ItemOutcome("bronze", "table", "expire_snapshots", "planned", detail)
    detail["retain_last"] = "99"
    assert item.detail["retain_last"] == "2"
    with pytest.raises(TypeError):
        item.detail["retain_last"] = "3"


def test_store_delegates_exactly_once_to_established_atomic_writer(plan, store, monkeypatch):
    calls = []

    def write(path, payload):
        calls.append((path, payload))
        return path

    monkeypatch.setattr(records, "write_json_atomic", write)
    record = _record(plan)
    path = store.persist(record)
    assert calls == [(store.record_path(record.maintenance_run_id), record.to_dict())]
    assert path == calls[0][0]


def test_atomic_replacement_failure_preserves_existing_record(plan, store, monkeypatch):
    record = _record(plan)
    path = store.persist(record)
    before = path.read_bytes()

    def fail_replace(source, target):
        raise OSError("replacement refused")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement refused"):
        store.persist(_record(plan, dry_run=False, items=_applied_items(plan)))
    assert path.read_bytes() == before


def test_validations_directory_property_matches_existing_quality_path_without_io(tmp_path):
    base = tmp_path / "metadata" / "example"
    paths = MetadataZonePaths(
        base,
        base / "runs",
        base / "lineage",
        base / "checkpoints",
        base / "checkpoints/history",
        base / "dead_letters",
        base / "dead_letters/history",
    )
    assert paths.validations_dir == base / "validations"
    assert not base.exists()
    assert not hasattr(paths, "maintenance_dir")


@pytest.mark.parametrize("field", ["started_at", "ended_at"])
def test_record_factory_refuses_naive_input_timestamps(plan, field):
    kwargs = {"started_at": plan.now, "ended_at": plan.now}
    kwargs[field] = plan.now.replace(tzinfo=None)
    with pytest.raises(ValueError, match=f"{field} must be timezone-aware"):
        MaintenanceRecord.from_plan(
            plan, environment="local", dry_run=True, zones={"bronze", "metadata", "raw"}, **kwargs
        )


def test_record_refuses_reversed_times_invalid_schema_and_invalid_duration(plan):
    record = _record(plan)
    with pytest.raises(ValueError, match="must not precede"):
        replace(record, ended_at=plan.now - timedelta(seconds=1))
    with pytest.raises(ValueError, match="schema_version"):
        replace(record, schema_version=2)
    for invalid in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            replace(record, duration_seconds=invalid)


@pytest.mark.parametrize("field", ["removed_count", "removed_bytes", "duration_seconds"])
def test_item_refuses_negative_or_nonfinite_measurements(plan, field):
    for value in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            replace(ItemOutcome.from_planned_item(plan.items[0]), **{field: value})


def test_item_refuses_unknown_status_and_inconsistent_failure_fields(plan):
    item = ItemOutcome.from_planned_item(plan.items[0])
    with pytest.raises(ValueError, match="unsupported maintenance item status"):
        replace(item, status="unknown")
    for updates in ({"status": "failed"}, {"failure_type": "OSError"}, {"failure_message": "oops"}):
        with pytest.raises(ValueError, match="failure_type and failure_message"):
            replace(item, **updates)


def test_record_imports_remain_compute_free_in_a_fresh_interpreter():
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class Tripwire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == root or fullname.startswith(root + '.') for root in
               ('pyspark', 'pyiceberg', 'pyarrow', 'janus.runtime', 'janus.writers')):
            raise AssertionError('record imported ' + fullname)
sys.meta_path.insert(0, Tripwire())
import janus.maintenance.records
"""
    subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[3] / "src")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
