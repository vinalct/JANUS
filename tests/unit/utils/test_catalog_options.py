"""Per-type Iceberg catalog options emitted by `build_spark_options`.

`build_spark_options` is a pure dict builder, so every rule below is host-testable
without Spark: which keys each catalog type emits, which keys it must refuse to emit,
and which missing values fail closed with a named error instead of reaching Spark.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from janus.utils.environment import (
    ICEBERG_CATALOG_IMPL,
    ICEBERG_SESSION_EXTENSIONS,
    build_spark_options,
    materialize_runtime_paths,
)

ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1"
SQLITE_DRIVER_PACKAGE = "org.xerial:sqlite-jdbc:3.46.0.0"
CATALOG_PREFIX = "spark.sql.catalog.janus"
SECRET_USER = "janus-catalog-user"
SECRET_PASSWORD = "janus-catalog-password"

# The keys every type shares, so a per-type test can assert only its own difference.
COMMON_CATALOG_KEYS = {
    CATALOG_PREFIX,
    f"{CATALOG_PREFIX}.warehouse",
    f"{CATALOG_PREFIX}.default-namespace",
}


def _environment_config(**iceberg: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "catalog_name": "janus",
        "catalog_type": "hadoop",
        "warehouse_dir": "data/bronze/iceberg",
        "runtime_package": ICEBERG_RUNTIME_PACKAGE,
        "default_namespace": "bronze",
    }
    block.update(iceberg)
    return {
        "runtime": {"log_level": "INFO"},
        "spark": {
            "app_name": "janus-test",
            "master": "local[1]",
            "warehouse_dir": "data/metadata/spark-warehouse",
            "ivy_dir": "data/metadata/ivy",
            "iceberg": block,
            "config": {},
        },
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        },
    }


def _build_options(project_root: Path, **iceberg: Any) -> dict[str, str]:
    config = _environment_config(**iceberg)
    return build_spark_options(config, materialize_runtime_paths(config, project_root))


def _catalog_keys(options: dict[str, str]) -> set[str]:
    return {key for key in options if key.startswith(CATALOG_PREFIX)}


# ── one test per catalog type ────────────────────────────────────────────────


def test_hadoop_emits_the_pre_order_13_catalog_block(tmp_path):
    """AC-4 for the seam itself: `catalog_type: hadoop` reproduces today's options."""

    options = _build_options(tmp_path, catalog_type="hadoop")

    assert options["spark.jars.packages"] == ICEBERG_RUNTIME_PACKAGE
    assert options["spark.sql.extensions"] == ICEBERG_SESSION_EXTENSIONS
    assert options["spark.sql.defaultCatalog"] == "janus"
    assert options[CATALOG_PREFIX] == ICEBERG_CATALOG_IMPL
    assert options[f"{CATALOG_PREFIX}.type"] == "hadoop"
    assert options[f"{CATALOG_PREFIX}.warehouse"] == str(tmp_path / "data/bronze/iceberg")
    assert options[f"{CATALOG_PREFIX}.default-namespace"] == "bronze"
    assert _catalog_keys(options) == COMMON_CATALOG_KEYS | {f"{CATALOG_PREFIX}.type"}


def test_jdbc_emits_type_and_uri(tmp_path):
    options = _build_options(
        tmp_path, catalog_type="jdbc", uri="jdbc:sqlite:data/metadata/janus_catalog.db"
    )

    assert options[f"{CATALOG_PREFIX}.type"] == "jdbc"
    assert options[f"{CATALOG_PREFIX}.uri"] == "jdbc:sqlite:data/metadata/janus_catalog.db"
    assert options["spark.jars.packages"] == ICEBERG_RUNTIME_PACKAGE
    assert _catalog_keys(options) == COMMON_CATALOG_KEYS | {
        f"{CATALOG_PREFIX}.type",
        f"{CATALOG_PREFIX}.uri",
    }


def test_jdbc_emits_credentials_and_merges_the_driver_package(tmp_path):
    options = _build_options(
        tmp_path,
        catalog_type="jdbc",
        uri="jdbc:postgresql://catalog-db:5432/janus",
        driver_package=SQLITE_DRIVER_PACKAGE,
        credentials={"user": SECRET_USER, "password": SECRET_PASSWORD},
    )

    assert options[f"{CATALOG_PREFIX}.jdbc.user"] == SECRET_USER
    assert options[f"{CATALOG_PREFIX}.jdbc.password"] == SECRET_PASSWORD
    assert options["spark.jars.packages"] == (
        f"{ICEBERG_RUNTIME_PACKAGE},{SQLITE_DRIVER_PACKAGE}"
    )
    assert _catalog_keys(options) == COMMON_CATALOG_KEYS | {
        f"{CATALOG_PREFIX}.type",
        f"{CATALOG_PREFIX}.uri",
        f"{CATALOG_PREFIX}.jdbc.user",
        f"{CATALOG_PREFIX}.jdbc.password",
    }


def test_rest_emits_type_and_uri_and_nothing_else(tmp_path):
    """
    The credentials and driver package below are deliberately populated: they are
    JDBC-shaped knobs, and a REST catalog must not pick them up.
    """

    options = _build_options(
        tmp_path,
        catalog_type="rest",
        uri="http://catalog:8181",
        driver_package=SQLITE_DRIVER_PACKAGE,
        credentials={"user": SECRET_USER, "password": SECRET_PASSWORD},
    )

    assert options[f"{CATALOG_PREFIX}.type"] == "rest"
    assert options[f"{CATALOG_PREFIX}.uri"] == "http://catalog:8181"
    assert options["spark.jars.packages"] == ICEBERG_RUNTIME_PACKAGE
    assert _catalog_keys(options) == COMMON_CATALOG_KEYS | {
        f"{CATALOG_PREFIX}.type",
        f"{CATALOG_PREFIX}.uri",
    }


# ── the catalog type fails closed ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("label", "block"),
    [
        ("absent", {}),
        ("empty_expansion", {"catalog_type": ""}),
        ("whitespace_expansion", {"catalog_type": "   "}),
    ],
)
def test_missing_catalog_type_names_the_key_and_the_supported_set(tmp_path, label, block):
    """An unset `${JANUS_ICEBERG_CATALOG_TYPE}` expands to `""` — that is missing."""

    del label
    config = _environment_config(**block)
    if not block:
        del config["spark"]["iceberg"]["catalog_type"]
    paths = materialize_runtime_paths(config, tmp_path)

    with pytest.raises(ValueError) as error:
        build_spark_options(config, paths)

    message = str(error.value)
    assert "spark.iceberg.catalog_type" in message
    assert "environment profile" in message
    for supported in ("hadoop", "jdbc", "rest"):
        assert supported in message


def test_unknown_catalog_type_echoes_the_value(tmp_path):
    config = _environment_config(catalog_type="hive")
    paths = materialize_runtime_paths(config, tmp_path)

    with pytest.raises(ValueError, match=re.escape("'hive'")) as error:
        build_spark_options(config, paths)

    message = str(error.value)
    assert "spark.iceberg.catalog_type" in message
    for supported in ("hadoop", "jdbc", "rest"):
        assert supported in message


@pytest.mark.parametrize("catalog_type", ["jdbc", "rest"])
@pytest.mark.parametrize(
    ("label", "block"),
    [
        ("absent", {}),
        ("empty_expansion", {"uri": ""}),
        ("whitespace_expansion", {"uri": "   "}),
    ],
)
def test_uri_is_required_for_the_uri_bearing_types(tmp_path, catalog_type, label, block):
    del label
    config = _environment_config(catalog_type=catalog_type, **block)
    paths = materialize_runtime_paths(config, tmp_path)

    with pytest.raises(ValueError) as error:
        build_spark_options(config, paths)

    message = str(error.value)
    assert "spark.iceberg.uri" in message
    assert repr(catalog_type) in message


def test_hadoop_needs_no_uri(tmp_path):
    """The type that requires nothing beyond today's keys still builds from them."""

    options = _build_options(tmp_path, catalog_type="hadoop", uri="")

    assert f"{CATALOG_PREFIX}.uri" not in options


def test_a_profile_without_an_iceberg_block_needs_no_catalog_type(tmp_path):
    """`catalog_type` is required *when the iceberg block is present*, not always."""

    config = _environment_config()
    del config["spark"]["iceberg"]
    paths = materialize_runtime_paths(config, tmp_path)

    options = build_spark_options(config, paths)

    assert _catalog_keys(options) == set()


# ── overrides and credential containment ─────────────────────────────────────


def test_explicit_spark_config_entries_beat_every_catalog_default(tmp_path):
    """The `setdefault` rule survives the branching: the profile's own values win."""

    config = _environment_config(
        catalog_type="jdbc",
        uri="jdbc:sqlite:data/metadata/janus_catalog.db",
        credentials={"user": SECRET_USER, "password": SECRET_PASSWORD},
    )
    config["spark"]["config"] = {
        "spark.sql.defaultCatalog": "override_catalog",
        CATALOG_PREFIX: "com.example.CustomSparkCatalog",
        f"{CATALOG_PREFIX}.type": "rest",
        f"{CATALOG_PREFIX}.uri": "http://override:8181",
        f"{CATALOG_PREFIX}.jdbc.user": "override_user",
        f"{CATALOG_PREFIX}.jdbc.password": "override_password",
        f"{CATALOG_PREFIX}.warehouse": "/override/warehouse",
        f"{CATALOG_PREFIX}.default-namespace": "override_namespace",
    }
    paths = materialize_runtime_paths(config, tmp_path)

    options = build_spark_options(config, paths)

    assert options["spark.sql.defaultCatalog"] == "override_catalog"
    assert options[CATALOG_PREFIX] == "com.example.CustomSparkCatalog"
    assert options[f"{CATALOG_PREFIX}.type"] == "rest"
    assert options[f"{CATALOG_PREFIX}.uri"] == "http://override:8181"
    assert options[f"{CATALOG_PREFIX}.jdbc.user"] == "override_user"
    assert options[f"{CATALOG_PREFIX}.jdbc.password"] == "override_password"
    assert options[f"{CATALOG_PREFIX}.warehouse"] == "/override/warehouse"
    assert options[f"{CATALOG_PREFIX}.default-namespace"] == "override_namespace"


@pytest.mark.parametrize(
    ("label", "credentials"),
    [
        ("absent", None),
        ("empty_expansion", {"user": "", "password": ""}),
        ("whitespace_expansion", {"user": "  ", "password": "  "}),
        ("password_only", {"user": "", "password": SECRET_PASSWORD}),
    ],
)
def test_credentials_reach_the_options_only_when_non_empty(tmp_path, label, credentials):
    """An unset `${JANUS_ICEBERG_CATALOG_USER}` must not become an empty JDBC user."""

    del label
    block: dict[str, Any] = {
        "catalog_type": "jdbc",
        "uri": "jdbc:sqlite:data/metadata/janus_catalog.db",
    }
    if credentials is not None:
        block["credentials"] = credentials

    options = _build_options(tmp_path, **block)

    assert f"{CATALOG_PREFIX}.jdbc.user" not in options
    expects_password = bool(credentials and credentials.get("password", "").strip())
    assert (f"{CATALOG_PREFIX}.jdbc.password" in options) is expects_password


def test_a_failed_build_never_names_a_credential(tmp_path):
    """The error a broken profile raises is what ends up in logs and tracebacks."""

    config = _environment_config(
        catalog_type="jdbc",
        uri="",
        credentials={"user": SECRET_USER, "password": SECRET_PASSWORD},
    )
    paths = materialize_runtime_paths(config, tmp_path)

    with pytest.raises(ValueError) as error:
        build_spark_options(config, paths)

    message = str(error.value)
    assert SECRET_USER not in message
    assert SECRET_PASSWORD not in message
