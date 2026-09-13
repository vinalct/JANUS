"""The `cluster-rest` overlay: one profile, two catalogs, one bucket."""

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
ENVIRONMENTS_DIR = PROJECT_ROOT / "conf" / "environments"
JDBC_ENV_PATH = ENVIRONMENTS_DIR / "cluster.env.example"
REST_ENV_PATH = ENVIRONMENTS_DIR / "cluster-rest.env.example"
PROFILE_PATH = ENVIRONMENTS_DIR / f"{PROFILE}.yaml"
JANUS_ENV_PREFIX = "JANUS_"

CATALOG_PREFIX = "spark.sql.catalog.janus"
REST_CATALOG_URI = "http://nessie:19120/iceberg/main"
WAREHOUSE_NAME = "janus"
WAREHOUSE_URI = "s3://janus-bronze/warehouse"
ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1"
AWS_BUNDLE_PACKAGE = "org.apache.iceberg:iceberg-aws-bundle:1.10.1"

CATALOG_OVERLAY_KEYS = frozenset(
    {
        "JANUS_ICEBERG_CATALOG_TYPE",
        "JANUS_ICEBERG_CATALOG_URI",
        "JANUS_ICEBERG_WAREHOUSE_DIR",
        "JANUS_ICEBERG_JDBC_DRIVER_PACKAGE",
    }
)

AUTH_OVERLAY_KEYS = frozenset(
    {
        "JANUS_ICEBERG_CATALOG_TOKEN",
        "JANUS_ICEBERG_CATALOG_CREDENTIAL",
        "JANUS_ICEBERG_CATALOG_OAUTH2_SERVER_URI",
        "JANUS_ICEBERG_CATALOG_SCOPE",
    }
)

COSMETIC_OVERLAY_KEYS = frozenset({"JANUS_SPARK_APP_NAME"})


def _env_file(path: Path) -> dict[str, str]:
    """A compose `env_file` as compose reads it: `KEY=VALUE`, comments and blanks skipped."""

    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        entries[name] = value
    return entries


def _overlaid_config(**overrides: str) -> dict[str, Any]:
    """`cluster.yaml` as the `janus-cluster-rest` container loads it."""

    scrubbed = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(JANUS_ENV_PREFIX)
    }
    environment = {**scrubbed, **_env_file(REST_ENV_PATH), **overrides}
    with mock.patch.dict(os.environ, environment, clear=True):
        return load_environment_config(PROFILE, PROJECT_ROOT)


@pytest.fixture
def rest_options(tmp_path) -> dict[str, str]:
    config = _overlaid_config()
    return build_spark_options(config, materialize_runtime_paths(config, tmp_path))


# ── the overlay is confined to the catalog ───────────────────────────────────


def test_the_overlay_changes_the_catalog_and_nothing_else():
    """The justification for one profile file instead of two, made executable."""

    jdbc, rest = _env_file(JDBC_ENV_PATH), _env_file(REST_ENV_PATH)

    assert jdbc and rest, "an env file read as empty would make every assertion vacuous"
    differing = {
        name
        for name in set(jdbc) | set(rest)
        if jdbc.get(name) != rest.get(name)
    }

    assert differing == CATALOG_OVERLAY_KEYS | AUTH_OVERLAY_KEYS | COSMETIC_OVERLAY_KEYS


def test_the_object_store_settings_are_identical_in_both_overlays():
    """AC-3 is not re-litigated by this task: bronze still lands in the same bucket."""

    jdbc, rest = _env_file(JDBC_ENV_PATH), _env_file(REST_ENV_PATH)
    object_store_keys = [name for name in jdbc if name.startswith(("JANUS_S3_", "AWS_"))]

    assert object_store_keys
    for name in object_store_keys:
        assert rest[name] == jdbc[name]


def test_the_profile_itself_is_the_one_both_overlays_read():
    """There is no `cluster-rest.yaml`; a second profile is what the overlay avoids."""

    assert not (ENVIRONMENTS_DIR / "cluster-rest.yaml").exists()
    assert PROFILE_PATH.is_file()


# ── what the overlaid profile emits ──────────────────────────────────────────


def test_the_catalog_is_the_rest_one_at_the_uri_the_overlay_declares(rest_options):
    assert rest_options[f"{CATALOG_PREFIX}.type"] == "rest"
    assert rest_options[f"{CATALOG_PREFIX}.uri"] == REST_CATALOG_URI


def test_the_warehouse_is_the_identifier_the_catalog_resolves(rest_options):
    """The bucket is declared once, in the catalog service, instead of once per client."""

    assert rest_options[f"{CATALOG_PREFIX}.warehouse"] == WAREHOUSE_NAME


def test_no_database_login_reaches_a_catalog_that_opens_no_database(rest_options):
    assert not [key for key in rest_options if key.startswith(f"{CATALOG_PREFIX}.jdbc.")]


def test_the_session_resolves_the_engine_and_the_io_bundle_but_no_driver(rest_options):
    """The Postgres driver belongs to the catalog service now, not to JANUS's classpath."""

    assert rest_options["spark.jars.packages"] == (
        f"{ICEBERG_RUNTIME_PACKAGE},{AWS_BUNDLE_PACKAGE}"
    )


def test_bronze_is_still_written_through_s3fileio(rest_options):
    """The catalog coordinates commits; it does not change how data files are written."""

    assert rest_options[f"{CATALOG_PREFIX}.io-impl"] == S3_FILE_IO_IMPL
    assert rest_options[f"{CATALOG_PREFIX}.s3.endpoint"] == "http://minio:9000"
    assert rest_options[f"{CATALOG_PREFIX}.s3.path-style-access"] == "true"
    assert rest_options[f"{CATALOG_PREFIX}.client.region"] == "us-east-1"


def test_there_is_exactly_one_io_path(rest_options):
    assert not [key for key in rest_options if "s3a" in key or ".fs." in key]


def test_the_stack_the_overlay_ships_is_unauthenticated_and_says_so(rest_options):
    """All four auth variables are empty in the tracked file, so nothing is emitted."""

    assert not [
        key
        for key in rest_options
        if key.endswith((".token", ".credential", ".oauth2-server-uri", ".scope"))
    ]


def test_a_managed_catalog_is_reached_by_exporting_a_token(tmp_path):
    """NFR-1 for the REST variant: no code change, and no tracked file change either."""

    config = _overlaid_config(
        JANUS_ICEBERG_CATALOG_URI="https://catalog.example/v1",
        JANUS_ICEBERG_CATALOG_TOKEN="a-managed-catalog-token",
        JANUS_ICEBERG_CATALOG_SCOPE="catalog:write",
    )
    options = build_spark_options(config, materialize_runtime_paths(config, tmp_path))

    assert options[f"{CATALOG_PREFIX}.uri"] == "https://catalog.example/v1"
    assert options[f"{CATALOG_PREFIX}.token"] == "a-managed-catalog-token"
    assert options[f"{CATALOG_PREFIX}.scope"] == "catalog:write"
    assert f"{CATALOG_PREFIX}.credential" not in options


def test_only_the_warehouse_leaves_the_local_volume(tmp_path):
    """The raw and metadata zones stay on the mounted volume, exactly as under JDBC."""

    config = _overlaid_config()
    paths = materialize_runtime_paths(config, tmp_path)

    assert paths[ICEBERG_WAREHOUSE_PATH_KEY] == WAREHOUSE_NAME
    for key in ("root_dir", "raw_dir", "bronze_dir", "metadata_dir", "warehouse_dir", "ivy_dir"):
        assert isinstance(paths[key], Path)
        assert paths[key].is_relative_to(tmp_path)


# ── the second engine reads the same overlay ─────────────────────────────────


def test_pyiceberg_derives_the_same_rest_catalog(tmp_path):
    config = _overlaid_config()
    properties = derive_pyiceberg_catalog_properties(
        config, materialize_runtime_paths(config, tmp_path)
    )

    assert derive_pyiceberg_catalog_name(config) == "janus"
    assert derive_pyiceberg_default_namespace(config) == "bronze"
    assert properties == {
        "type": "rest",
        "uri": REST_CATALOG_URI,
        "warehouse": WAREHOUSE_NAME,
        "s3.endpoint": "http://minio:9000",
        "s3.region": "us-east-1",
        "s3.force-virtual-addressing": "false",
    }


# ── no secret is tracked ─────────────────────────────────────────────────────


def test_the_profiles_auth_block_carries_no_value():
    """Every auth key is an expansion; the file itself holds nothing to leak."""

    iceberg = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))["spark"]["iceberg"]

    assert set(iceberg["auth"]) == {"token", "credential", "oauth2_server_uri", "scope"}
    for value in iceberg["auth"].values():
        assert value.startswith("${") and value.endswith(":-}")


def test_the_overlay_ships_placeholders_rather_than_secrets():
    """`cluster-rest.env.example` is tracked, so every secret line in it must be empty."""

    secrets = [
        (name, value)
        for name, value in _env_file(REST_ENV_PATH).items()
        if any(
            token in name
            for token in ("PASSWORD", "SECRET", "ACCESS_KEY", "USER", "TOKEN", "CREDENTIAL")
        )
    ]

    assert secrets, "no credential placeholders were found — the sweep read nothing"
    assert all(value == "" for _name, value in secrets), secrets


def test_the_bronze_warehouse_bucket_is_still_the_one_the_stack_creates():
    """The identifier the client sends is a name; the bucket behind it is unchanged."""

    assert _env_file(JDBC_ENV_PATH)["JANUS_ICEBERG_WAREHOUSE_DIR"] == WAREHOUSE_URI
    assert _env_file(REST_ENV_PATH)["JANUS_ICEBERG_WAREHOUSE_DIR"] == WAREHOUSE_NAME
