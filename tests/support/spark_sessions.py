"""One place every Spark suite builds its session, over the catalog the profile configures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from janus.utils.environment import (
    ICEBERG_CATALOG_DB_PATH_KEY,
    JDBC_CATALOG_TYPE,
    build_spark_options,
    load_environment_config,
    required_catalog_value,
    resolve_catalog_type,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IVY_JARS_DIR = PROJECT_ROOT / "data" / "metadata" / "ivy" / "jars"

#: The profile whose catalog every Spark suite runs on — the one `make run-local` uses.
PROFILE_NAME = "local"
JANUS_ENV_PREFIX = "JANUS_"

#: The catalog name the profile ships. Suites that need a second catalog pass their own.
DEFAULT_CATALOG_NAME = "janus"
DEFAULT_MASTER = "local[1]"

#: The `spark.iceberg` keys holding a Maven coordinate rather than a path or a name.
CATALOG_PACKAGE_KEYS = ("runtime_package", "driver_package")

#: The only session settings that are the *tests'* rather than the profile's. Deliberately
#: the two every ported builder already pinned by hand: this task changes how a session is
#: built, not how it behaves.
TEST_SESSION_CONFIG = {
    "spark.sql.session.timeZone": "UTC",
    "spark.ui.enabled": "false",
}

JDBC_SQLITE_PREFIX = "jdbc:sqlite:"


def checked_in_iceberg_block() -> dict[str, Any]:
    """The profile's `spark.iceberg` block as a fresh clone gets it.

    `JANUS_*` overrides are scrubbed on purpose: a developer who exports one to poke at a
    warehouse must not thereby re-point the whole Spark suite at another catalog.
    """

    scrubbed = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(JANUS_ENV_PREFIX)
    }
    with mock.patch.dict(os.environ, scrubbed, clear=True):
        config = load_environment_config(PROFILE_NAME, PROJECT_ROOT)
    return dict(config["spark"]["iceberg"])


def vendored_jar(package: str) -> Path:
    """`group:artifact:version` → the seeded jar, in the naming `seed-ivy` and CI use."""

    group, artifact, version = package.split(":")
    return IVY_JARS_DIR / f"{group}_{artifact}-{version}.jar"


def pinned_jars(iceberg: dict[str, Any] | None = None) -> list[Path]:
    """Every jar the profile's catalog needs on the classpath."""

    block = checked_in_iceberg_block() if iceberg is None else iceberg
    jars = []
    for key in CATALOG_PACKAGE_KEYS:
        package = block.get(key)
        if isinstance(package, str) and package.strip():
            jars.append(vendored_jar(package.strip()))
    return jars


def iceberg_session_options(
    *,
    warehouse_dir: Path,
    catalog_db: Path,
    catalog_name: str = DEFAULT_CATALOG_NAME,
    spark_warehouse_dir: Path | None = None,
) -> dict[str, str]:
    """The catalog options the profile emits, pointed at this suite's own locations.

    Pure: the caller owns creating `catalog_db`'s parent directory (`build_iceberg_session`
    does it), because SQLite opens a database file but never creates the path to it.
    """

    iceberg = checked_in_iceberg_block()
    config = {
        "spark": {
            "iceberg": {
                **iceberg,
                "catalog_name": catalog_name,
                "warehouse_dir": str(warehouse_dir),
                "uri": suite_catalog_uri(iceberg, catalog_db),
            },
            "config": dict(TEST_SESSION_CONFIG),
        }
    }
    resolved_paths = {
        "warehouse_dir": spark_warehouse_dir or warehouse_dir.parent / "spark-warehouse",
        "iceberg_warehouse_dir": warehouse_dir,
        ICEBERG_CATALOG_DB_PATH_KEY: catalog_db,
    }

    options = dict(build_spark_options(config, resolved_paths))
    options.pop("spark.jars.packages", None)
    options["spark.jars"] = ",".join(str(jar) for jar in pinned_jars(iceberg))
    return options


def suite_catalog_uri(iceberg: dict[str, Any], catalog_db: Path) -> str:
    """The profile's catalog URI, pointed at this suite's own database file."""

    catalog_type = resolve_catalog_type(iceberg)
    if catalog_type != JDBC_CATALOG_TYPE:
        raise NotImplementedError(_unsupported_catalog(catalog_type, iceberg.get("uri")))

    uri = required_catalog_value(iceberg, "uri", catalog_type)
    if not uri.startswith(JDBC_SQLITE_PREFIX):
        raise NotImplementedError(_unsupported_catalog(catalog_type, uri))

    _path, separator, query = uri[len(JDBC_SQLITE_PREFIX) :].partition("?")
    return f"{JDBC_SQLITE_PREFIX}{catalog_db}{separator}{query}"


def _unsupported_catalog(catalog_type: str, uri: Any) -> str:
    """Why the suites cannot each be given a private copy of what the profile now names."""

    target = str(uri or "").partition("?")[0]
    return (
        f"conf/environments/{PROFILE_NAME}.yaml now configures a {catalog_type!r} catalog"
        + (f" at {target!r}" if target else "")
        + ", which this helper cannot give each suite a private copy of. Extend it — do "
        "not let the suites share one catalog."
    )


def catalog_database_path(root: Path) -> Path:
    """Where a suite's catalog database lives under its temp root.

    Under `metadata/`, mirroring the profile: the catalog is metadata, not data, and must
    not land inside the warehouse it indexes.
    """

    return root / "metadata" / "iceberg-catalog" / "catalog.sqlite"


def require_iceberg_runtime():
    """Skip unless PySpark and every jar the profile's catalog pins are available.

    Returns `pyspark.sql`. A fixture that builds its session lazily calls this at setup so
    an unavailable runtime still skips where it always did — at fixture time, not from
    inside whatever code first asks the factory for a session.
    """

    pyspark_sql = pytest.importorskip("pyspark.sql")
    for jar in pinned_jars():
        if not jar.exists():
            pytest.skip(f"{jar.name} is not available in the local Ivy cache")
    return pyspark_sql


def build_iceberg_session(
    app_name: str,
    root: Path,
    *,
    catalog_name: str = DEFAULT_CATALOG_NAME,
    master: str = DEFAULT_MASTER,
):
    """A live Iceberg session over this suite's own warehouse and catalog database.

    Skips rather than fails when PySpark or a vendored jar is absent, the way every suite
    did before they shared this builder. The caller keeps its own teardown: stop the
    session before its temp root is reclaimed.
    """

    pyspark_sql = require_iceberg_runtime()

    catalog_db = catalog_database_path(root)
    catalog_db.parent.mkdir(parents=True, exist_ok=True)
    options = iceberg_session_options(
        warehouse_dir=root / "iceberg",
        catalog_db=catalog_db,
        catalog_name=catalog_name,
        spark_warehouse_dir=root / "spark-warehouse",
    )

    builder = pyspark_sql.SparkSession.builder.appName(app_name).master(master)
    for key, value in options.items():
        builder = builder.config(key, value)
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session
