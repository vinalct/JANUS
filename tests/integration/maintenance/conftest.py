"""Compose the existing history and catalog harnesses; no second session builder."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from importlib import import_module

import pytest
import yaml

from janus.runtime import SparkSessionProvider
from tests.integration.catalog_commits.conftest import catalog_target, shared_catalog_session
from tests.integration.full_refresh_history.conftest import (
    FIXTURE_PROJECT_ROOT,
    full_refresh_harness,
    full_refresh_registry,
    spark,
)
from tests.support.maintenance_zone import PROJECT_ROOT
from tests.support.operator_cli import run_janus

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

    def run(session, config, paths, *, now, zone, apply=False, source_id=None, table_name=None):
        command = import_module("janus.cli.maintain")
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
        if table_name:
            path = root / "conf/sources/unpartitioned.yaml"
            source = yaml.safe_load(path.read_text())
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
                "--zone",
                zone,
                "--format",
                "json",
            ]
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
