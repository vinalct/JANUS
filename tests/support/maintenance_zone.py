from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType

from janus.models import ExecutionPlan, RunContext
from janus.registry import load_registry
from janus.utils.storage import StorageLayout

SOURCES = ("maintenance_alpha", "maintenance_beta")
FAMILIES = {
    "runs": "started_at",
    "lineage": "emitted_at",
    "checkpoints/history": "recorded_at",
    "validations": "emitted_at",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class PlantedZone:
    root: Path
    protected_paths: frozenset[Path]
    candidate_paths: frozenset[Path]
    run_ids_by_source: Mapping[str, tuple[str, ...]]
    now: datetime


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_maintenance_zone(root: Path, now: datetime) -> PlantedZone:

    if now.utcoffset() is None:
        raise ValueError("fixture now must be timezone-aware")
    protected: set[Path] = set()
    candidates: set[Path] = set()
    run_ids: dict[str, tuple[str, ...]] = {}
    for source_index, source in enumerate(SOURCES):
        ids = tuple(f"{source}-run-{index:02d}" for index in range(25))
        if source_index == 0:
            ids = (f"{source}:run:00", *ids[1:])
        run_ids[source] = ids
        for index, run_id in enumerate(ids):
            started = now - timedelta(days=110 - index)
            for family, field in FAMILIES.items():
                timestamp = started
                if family == "validations" and index == 1:
                    timestamp = now - timedelta(days=1)  # newer than its old run
                if family == "validations" and index == 24:
                    timestamp = now - timedelta(days=120)  # saved solely by run id
                path = _write(
                    root / source / family / f"{run_id}.json",
                    {
                        "source_id": source,
                        "run_id": run_id,
                        field: timestamp.isoformat(),
                    },
                )
                keep = index >= 5 or (source_index == 0 and index == 0)
                keep |= timestamp >= now - timedelta(days=90)
                (protected if keep else candidates).add(path)
        candidates.add(
            _write(
                root / source / "lineage/orphan.json",
                {
                    "source_id": source,
                    "run_id": "orphan",
                    "emitted_at": (now - timedelta(days=120)).isoformat(),
                },
            )
        )
        for family in ("checkpoints", "dead_letters"):
            protected.add(
                _write(
                    root / source / family / "current.json",
                    {
                        "source_id": source,
                        "updated_at": (now - timedelta(days=120)).isoformat(),
                    },
                )
            )
        progress = {"source_id": source, "updated_at": now.isoformat(), "request_index": 1}
        if source_index == 0:
            progress["raw_path_prefix"] = (
                f"runs/ingestion_date={(now - timedelta(days=110)).date()}/run_id={source}-run-00"
            )
        protected.add(_write(root / source / "extraction_progress.json", progress))
        for age in (120, 100, 90, 1, 0):
            day = (now - timedelta(days=age)).date()
            path = _write(
                root / source / "lineage/openlineage" / f"events-{day}.ndjson",
                {
                    "eventTime": now.isoformat(),
                    "run": {"runId": f"seed-{age}"},
                },
            )
            (candidates if age > 90 else protected).add(path)
    for name, pipeline_age in (("old", 120), ("recent", 1), ("null", None)):
        path = _write(
            root / "pipelines" / name / "summary.json",
            {
                "pipeline": {
                    "pipeline_run_id": name,
                    "started_at": None
                    if pipeline_age is None
                    else (now - timedelta(days=pipeline_age)).isoformat(),
                },
            },
        )
        (candidates if pipeline_age == 120 else protected).add(path)
    protected.add(
        _write(
            root / SOURCES[0] / "runs/unaged.json",
            {
                "run_id": "unaged",
                "started_at": "not-a-timestamp",
            },
        )
    )
    invalid = root / SOURCES[0] / "runs/unreadable.json"
    invalid.write_text("{invalid json", encoding="utf-8")
    protected.add(invalid)
    for path in (
        root / SOURCES[0] / "runs/.record.deadbeef.tmp",
        root / "pipelines/.summary.cafebabe.tmp",
    ):
        protected.add(_write(path, {}))
    planted = PlantedZone(
        root, frozenset(protected), frozenset(candidates), MappingProxyType(run_ids), now
    )
    _assert_protected_shapes(planted)
    assert protected.isdisjoint(candidates)
    assert protected | candidates == {path for path in root.rglob("*") if path.is_file()}
    return planted


def _assert_protected_shapes(planted: PlantedZone) -> None:
    """All six AC-3 cases must exist; a declaration alone is insufficient."""
    for source in SOURCES:
        newest = planted.run_ids_by_source[source][-1]
        shapes = (
            planted.root / source / "checkpoints/current.json",
            planted.root / source / "dead_letters/current.json",
            planted.root / source / "extraction_progress.json",
            planted.root / source / "runs" / f"{newest}.json",
            planted.root / source / "validations" / f"{newest}.json",
        )
        assert all(path.is_file() and path in planted.protected_paths for path in shapes)
    live_run = planted.run_ids_by_source[SOURCES[0]][0]
    assert (planted.root / SOURCES[0] / "runs" / f"{live_run}.json") in planted.protected_paths
    for family in FAMILIES:
        assert (planted.root / SOURCES[0] / family / f"{live_run}.json").is_file()


def zone_plans(planted: PlantedZone) -> tuple[ExecutionPlan, ...]:
    """Use existing registry declarations, with source-owned output roots redirected."""
    registry = load_registry(PROJECT_ROOT / "tests/fixtures/full_refresh_history")
    template = registry.get_source("full_refresh_history_unpartitioned")
    plans = []
    for source_id in SOURCES:
        source = replace(
            template,
            source_id=source_id,
            outputs=replace(
                template.outputs,
                metadata=replace(template.outputs.metadata, path=str(planted.root / source_id)),
            ),
        )
        context = RunContext.create(
            run_id="maintenance-fixture",
            environment="local",
            project_root=planted.root.parent,
            started_at=planted.now,
        )
        plans.append(ExecutionPlan.from_source_config(source, context))
    return tuple(plans)


def zone_layout(planted: PlantedZone) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": str(planted.root.parent),
                "metadata_dir": str(planted.root),
                "raw_dir": str(planted.root.parent / "raw"),
                "bronze_dir": str(planted.root.parent / "bronze"),
            }
        },
        planted.root.parent,
    )
