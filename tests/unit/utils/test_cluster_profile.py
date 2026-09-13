"""The checked-in `cluster` profile, end to end, without a container in sight."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from janus.utils.catalog_properties import (
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
    derive_pyiceberg_default_namespace,
)
from janus.utils.environment import (
    ICEBERG_WAREHOUSE_PATH_KEY,
    S3_FILE_IO_IMPL,
    build_spark_options,
    load_environment_config,
    materialize_runtime_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE = "cluster"
PROFILE_PATH = PROJECT_ROOT / "conf" / "environments" / f"{PROFILE}.yaml"
ENV_EXAMPLE_PATH = PROJECT_ROOT / "conf" / "environments" / f"{PROFILE}.env.example"
JANUS_ENV_PREFIX = "JANUS_"

CATALOG_PREFIX = "spark.sql.catalog.janus"
WAREHOUSE_URI = "s3://janus-bronze/warehouse"
CATALOG_URI = "jdbc:postgresql://postgres:5432/iceberg"
ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1"
POSTGRES_DRIVER_PACKAGE = "org.postgresql:postgresql:42.7.13"
AWS_BUNDLE_PACKAGE = "org.apache.iceberg:iceberg-aws-bundle:1.10.1"

CATALOG_USER = "janus"
CATALOG_PASSWORD = "cluster-stack-password"


def _checked_in_config(**overrides: str) -> dict[str, Any]:
    """The profile as a fresh clone gets it, plus whatever the stack would export."""

    scrubbed = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(JANUS_ENV_PREFIX)
    }
    with mock.patch.dict(os.environ, {**scrubbed, **overrides}, clear=True):
        return load_environment_config(PROFILE, PROJECT_ROOT)


@pytest.fixture
def cluster_options(tmp_path) -> dict[str, str]:
    config = _checked_in_config(
        JANUS_ICEBERG_CATALOG_USER=CATALOG_USER,
        JANUS_ICEBERG_CATALOG_PASSWORD=CATALOG_PASSWORD,
    )
    return build_spark_options(config, materialize_runtime_paths(config, tmp_path))


# ── what a clean clone ships ─────────────────────────────────────────────────


def test_the_profile_defaults_to_the_safe_catalog_and_an_object_store_warehouse():
    iceberg = _checked_in_config()["spark"]["iceberg"]

    assert iceberg["catalog_type"] == "jdbc"
    assert iceberg["uri"] == CATALOG_URI
    assert iceberg["warehouse_dir"] == WAREHOUSE_URI
    assert iceberg["driver_package"] == POSTGRES_DRIVER_PACKAGE
    assert iceberg["object_store"]["io_impl"] == "S3FileIO"


def test_the_poc_runs_spark_in_the_container_and_says_so():
    """The stub promised `spark://spark-master:7077`; this repository runs no such master."""

    config = _checked_in_config()

    assert config["spark"]["master"] == "local[*]"
    assert _checked_in_config(JANUS_SPARK_MASTER="spark://master:7077")["spark"]["master"] == (
        "spark://master:7077"
    )


def test_only_the_warehouse_moves_off_the_local_volume(tmp_path):
    """The raw and metadata zones stay on the mounted volume in this PoC (task §2)."""

    config = _checked_in_config()
    paths = materialize_runtime_paths(config, tmp_path)

    assert paths[ICEBERG_WAREHOUSE_PATH_KEY] == WAREHOUSE_URI
    for key in ("root_dir", "raw_dir", "bronze_dir", "metadata_dir", "warehouse_dir", "ivy_dir"):
        assert isinstance(paths[key], Path)
        assert paths[key].is_relative_to(tmp_path)


# ── the options a session is built from ──────────────────────────────────────


def test_the_catalog_is_the_postgres_one_at_the_uri_the_profile_declares(cluster_options):
    """A server-backed URI has nothing project-relative in it and must pass through whole."""

    assert cluster_options[f"{CATALOG_PREFIX}.type"] == "jdbc"
    assert cluster_options[f"{CATALOG_PREFIX}.uri"] == CATALOG_URI
    assert cluster_options[f"{CATALOG_PREFIX}.jdbc.schema-version"] == "V1"


def test_the_catalog_login_comes_from_the_environment(cluster_options):
    assert cluster_options[f"{CATALOG_PREFIX}.jdbc.user"] == CATALOG_USER
    assert cluster_options[f"{CATALOG_PREFIX}.jdbc.password"] == CATALOG_PASSWORD


def test_an_unset_login_never_becomes_an_empty_one(tmp_path):
    """The checked-in defaults expand to `""`, which must not reach Spark as a JDBC user."""

    config = _checked_in_config()
    options = build_spark_options(config, materialize_runtime_paths(config, tmp_path))

    assert f"{CATALOG_PREFIX}.jdbc.user" not in options
    assert f"{CATALOG_PREFIX}.jdbc.password" not in options


def test_bronze_is_written_through_s3fileio(cluster_options):
    assert cluster_options[f"{CATALOG_PREFIX}.warehouse"] == WAREHOUSE_URI
    assert cluster_options[f"{CATALOG_PREFIX}.io-impl"] == S3_FILE_IO_IMPL
    assert cluster_options[f"{CATALOG_PREFIX}.s3.endpoint"] == "http://minio:9000"
    assert cluster_options[f"{CATALOG_PREFIX}.s3.path-style-access"] == "true"
    assert cluster_options[f"{CATALOG_PREFIX}.client.region"] == "us-east-1"


def test_there_is_exactly_one_io_path(cluster_options):

    assert not [key for key in cluster_options if "s3a" in key or ".fs." in key]


def test_the_session_resolves_the_engine_the_driver_and_the_io_bundle(cluster_options):
    assert cluster_options["spark.jars.packages"] == (
        f"{ICEBERG_RUNTIME_PACKAGE},{POSTGRES_DRIVER_PACKAGE},{AWS_BUNDLE_PACKAGE}"
    )


def test_pointing_the_profile_at_another_vendor_is_config_only(tmp_path):
    """NFR-1, exercised: real S3 and a managed Postgres, same code, same emitted keys."""

    config = _checked_in_config(
        JANUS_ICEBERG_WAREHOUSE_DIR="s3://acme-lake/janus",
        JANUS_ICEBERG_CATALOG_URI="jdbc:postgresql://db.internal:5432/iceberg?sslmode=require",
        JANUS_S3_ENDPOINT="",
        JANUS_S3_REGION="sa-east-1",
        JANUS_S3_PATH_STYLE_ACCESS="false",
    )
    options = build_spark_options(config, materialize_runtime_paths(config, tmp_path))

    assert options[f"{CATALOG_PREFIX}.warehouse"] == "s3://acme-lake/janus"
    assert options[f"{CATALOG_PREFIX}.uri"] == (
        "jdbc:postgresql://db.internal:5432/iceberg?sslmode=require"
    )
    assert options[f"{CATALOG_PREFIX}.client.region"] == "sa-east-1"
    assert options[f"{CATALOG_PREFIX}.s3.path-style-access"] == "false"
    # An empty endpoint is how AWS itself is addressed: the SDK resolves the region's own.
    assert f"{CATALOG_PREFIX}.s3.endpoint" not in options
    assert options[f"{CATALOG_PREFIX}.io-impl"] == S3_FILE_IO_IMPL


# ── the second engine reads the same profile ─────────────────────────────────


def test_pyiceberg_derives_the_same_catalog_from_the_cluster_profile(tmp_path):
    """Postgres translation, against the profile that actually uses it."""

    config = _checked_in_config(
        JANUS_ICEBERG_CATALOG_USER=CATALOG_USER,
        JANUS_ICEBERG_CATALOG_PASSWORD=CATALOG_PASSWORD,
    )
    properties = derive_pyiceberg_catalog_properties(
        config, materialize_runtime_paths(config, tmp_path)
    )

    assert derive_pyiceberg_catalog_name(config) == "janus"
    assert derive_pyiceberg_default_namespace(config) == "bronze"
    assert properties == {
        "type": "sql",
        "uri": (
            f"postgresql+psycopg2://{CATALOG_USER}:{CATALOG_PASSWORD}@postgres:5432/iceberg"
        ),
        "warehouse": WAREHOUSE_URI,
        # Without these three the second engine resolves the warehouse against AWS itself.
        "s3.endpoint": "http://minio:9000",
        "s3.region": "us-east-1",
        "s3.force-virtual-addressing": "false",
    }


# ── no secret is tracked ─────────────────────────────────────────────────────


def test_the_profile_carries_no_credential_value():
    """Every credential is an expansion; the file itself holds nothing to leak."""

    iceberg = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))["spark"]["iceberg"]

    for value in iceberg["credentials"].values():
        assert value.startswith("${") and value.endswith(":-}")
    assert "access_key" not in iceberg["object_store"]
    assert "secret_key" not in iceberg["object_store"]


def test_the_env_example_ships_placeholders_rather_than_logins():
    """`cluster.env.example` is tracked, so every credential line in it must be empty."""

    lines = [
        line.split("=", 1)
        for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    ]
    credentials = [
        (name, value)
        for name, value in lines
        if any(token in name for token in ("PASSWORD", "SECRET", "ACCESS_KEY", "USER"))
    ]

    assert credentials, "no credential placeholders were found — the sweep read nothing"
    assert all(value == "" for _name, value in credentials), credentials
