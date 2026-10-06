"""Compose the existing history and catalog harnesses; no second session builder."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml

from janus.maintenance.settings import MaintenancePolicy, resolve_maintenance_settings
from janus.runtime import SparkSessionProvider
from tests.integration.catalog_commits.conftest import catalog_target, shared_catalog_session
from tests.integration.full_refresh_history.conftest import (
    ENVIRONMENT_CONFIG,
    FIXTURE_PROJECT_ROOT,
    FullRefreshHarness,
    full_refresh_harness,
    full_refresh_registry,
    spark,
)
from tests.support.maintenance_zone import PROJECT_ROOT
from tests.support.operator_cli import run_janus
from tests.support.retention_baseline import filesystem_state

__all__ = [
    "catalog_target",
    "full_refresh_harness",
    "full_refresh_registry",
    "shared_catalog_session",
    "spark",
]


@pytest.fixture
def run_maintenance(tmp_path, monkeypatch):
    """Real dispatcher/command/collectors/executors with an externally owned session.

    Only profile/runtime location resolution and session ownership are supplied by
    the existing isolated harness. All maintenance decisions and actions stay real.
    """
    root = tmp_path / "command-project"
    shutil.copytree(FIXTURE_PROJECT_ROOT / "conf", root / "conf")

    def run(
        session,
        config,
        paths,
        *,
        now,
        zone=None,
        zones=None,
        apply=False,
        source_id=None,
        table_name=None,
        policy=None,
    ):
        command = import_module("janus.cli.maintain")
        paths = {**paths, "metadata_dir": paths.get("metadata_dir", root / "data/metadata")}
        profile = json.loads((PROJECT_ROOT / "tests/fixtures/maintenance/policy.json").read_text())
        config = {**config, **profile}
        catalog_name = session.conf.get("spark.sql.defaultCatalog")
        spark_config = dict(config.get("spark", {}))
        iceberg_config = dict(spark_config.get("iceberg", {}))
        iceberg_config["catalog_name"] = catalog_name
        iceberg_config.setdefault(
            "catalog_type", session.conf.get(f"spark.sql.catalog.{catalog_name}.type")
        )
        spark_config["iceberg"] = iceberg_config
        config["spark"] = spark_config
        metadata = paths["metadata_dir"]
        config["storage"] = {
            "root_dir": str(root / "data"),
            "raw_dir": str(root / "data/raw"),
            "bronze_dir": str(root / "data/bronze"),
            "metadata_dir": str(metadata),
        }
        config["maintenance"]["bronze"].update(retain_last=2, older_than_days=0)
        config["maintenance"]["runs_table"]["older_than_days"] = 30
        if policy is not None:
            config["maintenance"]["bronze"].update(
                retain_last=policy.bronze.retain_last,
                older_than_days=policy.bronze.older_than_days,
            )
            config["maintenance"]["runs_table"]["older_than_days"] = (
                policy.runs_table.older_than_days
            )
            assert resolve_maintenance_settings(config) == policy
        if table_name:
            for path in (root / "conf/sources").glob("*.yaml"):
                source = yaml.safe_load(path.read_text())
                if source["source_id"] == source_id:
                    source["outputs"]["bronze"]["table_name"] = table_name
                    path.write_text(yaml.safe_dump(source))

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=UTC):
                return now.astimezone(tz)

        with monkeypatch.context() as patch:
            patch.setattr(command, "datetime", FixedDateTime)
            patch.setattr(command, "load_environment_config", lambda *args, **kwargs: config)
            patch.setattr(command, "prepare_runtime", lambda *args, **kwargs: paths)
            patch.setattr(SparkSessionProvider, "get", lambda provider: session)
            patch.setattr(SparkSessionProvider, "stop", lambda provider: None)
            argv = [
                "maintain",
                "--environment",
                "local",
                "--project-root",
                str(root),
                "--format",
                "json",
            ]
            for selected_zone in sorted(zones or ({zone} if zone else ())):
                argv.extend(("--zone", selected_zone))
            if apply:
                argv.append("--apply")
            if source_id:
                argv.extend(("--source-id", source_id))
            result = run_janus(argv)
        assert result.exit_code == 0, result.output
        record = json.loads(result.stdout)
        metadata_root = paths["metadata_dir"]
        persisted = metadata_root / "maintenance" / f"{record['maintenance_run_id']}.json"
        assert json.loads(persisted.read_text()) == record
        return record

    return run


@dataclass(frozen=True)
class MaintenanceHarness:
    """History writes plus the real maintenance dispatcher, sharing one session."""

    history: FullRefreshHarness
    command: Callable
    source_id: str

    @property
    def spark(self):
        return self.history.spark

    @property
    def table(self) -> str:
        catalog = self.spark.conf.get("spark.sql.defaultCatalog")
        return f"{catalog}.bronze_full_refresh_history.{self.history.table_name}"

    @staticmethod
    def policy(*, retain_last=2, older_than_days=0) -> MaintenancePolicy:
        profile = json.loads((PROJECT_ROOT / "tests/fixtures/maintenance/policy.json").read_text())
        policy = resolve_maintenance_settings(profile)
        return replace(
            policy,
            bronze=replace(policy.bronze, retain_last=retain_last, older_than_days=older_than_days),
            runs_table=replace(policy.runs_table, older_than_days=30),
        )

    def run_maintain(self, policy, *, apply, now, zones=frozenset({"bronze"})):
        return self.command(
            self.spark,
            ENVIRONMENT_CONFIG,
            {"metadata_dir": self.history.project_root / "metadata"},
            now=now,
            zones=zones,
            apply=apply,
            source_id=self.source_id,
            table_name=self.history.table_name,
            policy=policy,
        )

    def warehouse_digest(self) -> dict[str, object]:
        location = next(
            row.data_type
            for row in self.spark.sql(f"DESCRIBE TABLE EXTENDED {self.table}").collect()
            if row.col_name == "Location"
        )
        return filesystem_state(Path(unquote(urlsplit(location).path)))

    def data_files(self, table: str) -> frozenset[Path]:
        return frozenset(
            Path(unquote(urlsplit(row.file_path).path))
            for row in self.spark.table(f"{table}.files").select("file_path").collect()
        )


@pytest.fixture
def maintenance_harness(full_refresh_harness, run_maintenance, request):
    source_id = getattr(request, "param", "full_refresh_history_unpartitioned")
    return MaintenanceHarness(full_refresh_harness, run_maintenance, source_id)
