"""Seed a persistent fixture catalog and verify the real Makefile dry-run transcript."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from janus.registry import load_registry
from janus.registry.dependencies import producer_table_identifier
from janus.writers.identifiers import quote_identifier
from tests.support.retention_baseline import filesystem_state
from tests.support.spark_sessions import (
    DEFAULT_CATALOG_NAME,
    DEFAULT_MASTER,
    PROJECT_ROOT,
    sqlite_catalog_target,
    start_session,
)

SOURCE_ID = "full_refresh_history_unpartitioned"
MANIFEST = "smoke-input.json"


def prepare_fixture(root: Path) -> None:
    """The project and catalog outlive the seeding process and container."""
    root = root.resolve()
    fixture = PROJECT_ROOT / "tests/fixtures/full_refresh_history"
    shutil.copytree(fixture / "conf", root / "conf", dirs_exist_ok=True)
    target = sqlite_catalog_target(root / "runtime")
    target.prepare()
    options = target.session_options()
    profile = target.environment_config()

    profile["spark"]["iceberg"].update(runtime_package="", driver_package="")
    profile["spark"].update(
        app_name="janus-maintenance-smoke",
        master=DEFAULT_MASTER,
        warehouse_dir=str(target.resolved_paths["warehouse_dir"]),
        config=options,
    )
    profile["storage"] = {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
    policy_path = PROJECT_ROOT / "tests/fixtures/maintenance/policy.json"
    profile.update(json.loads(policy_path.read_text(encoding="utf-8")))
    profile["maintenance"]["bronze"].update(retain_last=1, older_than_days=0)
    profile["runtime"] = {"log_level": "WARN"}
    profiles = root / "conf/environments"
    profiles.mkdir(exist_ok=True)
    (profiles / "local.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")

    source = load_registry(root).get_source(SOURCE_ID)
    identifier = producer_table_identifier(source)
    assert identifier is not None
    qualified = quote_identifier(f"{DEFAULT_CATALOG_NAME}.{identifier}")
    namespace = quote_identifier(f"{DEFAULT_CATALOG_NAME}.{identifier.rsplit('.', 1)[0]}")
    session = start_session("janus-maintenance-smoke-seed", options)
    try:
        session.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")
        session.sql(
            f"CREATE TABLE IF NOT EXISTS {qualified} (id STRING, value STRING) USING iceberg"
        )
        for index in range(2):
            session.sql(f"INSERT OVERWRITE {qualified} VALUES ('fixture', 'run-{index}')")
        snapshots = session.table(f"{DEFAULT_CATALOG_NAME}.{identifier}.snapshots").collect()
        assert len(snapshots) >= 2
        (current,) = session.table(f"{DEFAULT_CATALOG_NAME}.{identifier}.refs").filter(
            "name = 'main'"
        ).collect()
        manifest = {
            "table": identifier,
            "snapshot_ids": sorted(row.snapshot_id for row in snapshots),
            "current_snapshot_id": current.snapshot_id,
            "warehouse": str(target.warehouse_dir),
            "warehouse_before": filesystem_state(Path(target.warehouse_dir)),
        }
        (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finally:
        session.stop()
    print(f"Seeded {identifier} with {len(snapshots)} snapshots at {root}")


def verify_transcript(root: Path, transcript: Path) -> None:
    """An empty plan, a missing table, or a mutated warehouse fails the smoke."""
    root = root.resolve()
    record = json.loads(transcript.read_text(encoding="utf-8"))
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    assert record["dry_run"] is True and record["environment"] == "local"
    assert record["zones"] == ["bronze"] and record["source_ids"] == [SOURCE_ID]
    assert record["failures"] == []
    (item,) = record["items"]
    assert item["zone"] == "bronze" and item["target"] == manifest["table"]
    assert item["action"] == "expire_snapshots" and item["status"] == "planned"
    expired = set(item["expired_snapshot_ids"])
    assert expired and expired < set(manifest["snapshot_ids"])
    assert manifest["current_snapshot_id"] not in expired
    assert filesystem_state(Path(manifest["warehouse"])) == manifest["warehouse_before"]
    persisted = root / "data/metadata/maintenance" / f"{record['maintenance_run_id']}.json"
    assert json.loads(persisted.read_text(encoding="utf-8")) == record
    print(f"OK: dry-run planned {manifest['table']}; warehouse digest and snapshots unchanged")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("root", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("root", type=Path)
    verify.add_argument("transcript", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_fixture(args.root)
    else:
        verify_transcript(args.root, args.transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
