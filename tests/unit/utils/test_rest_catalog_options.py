"""The REST catalog's own surface: its auth block, and its warehouse as an identifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from janus.utils.catalog_properties import derive_pyiceberg_catalog_properties
from janus.utils.environment import (
    CATALOG_AUTH_OPTIONS,
    ICEBERG_WAREHOUSE_PATH_KEY,
    build_spark_options,
    materialize_runtime_paths,
    prepare_runtime,
)

ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1"
POSTGRES_DRIVER_PACKAGE = "org.postgresql:postgresql:42.7.13"
CATALOG_PREFIX = "spark.sql.catalog.janus"
REST_URI = "http://catalog:8181"
JDBC_URI = "jdbc:postgresql://catalog-db:5432/janus"

WAREHOUSE_NAME = "janus"

SECRET_TOKEN = "janus-bearer-token"
SECRET_CREDENTIAL = "janus-client:janus-client-secret"
OAUTH2_SERVER_URI = "https://auth.example/oauth/tokens"
SCOPE = "catalog:write"

FULL_AUTH_BLOCK = {
    "token": SECRET_TOKEN,
    "credential": SECRET_CREDENTIAL,
    "oauth2_server_uri": OAUTH2_SERVER_URI,
    "scope": SCOPE,
}

#: What that block must become, in the property names Iceberg's REST spec defines.
EXPECTED_AUTH_OPTIONS = {
    f"{CATALOG_PREFIX}.token": SECRET_TOKEN,
    f"{CATALOG_PREFIX}.credential": SECRET_CREDENTIAL,
    f"{CATALOG_PREFIX}.oauth2-server-uri": OAUTH2_SERVER_URI,
    f"{CATALOG_PREFIX}.scope": SCOPE,
}

SECRET_VALUES = (SECRET_TOKEN, SECRET_CREDENTIAL)


def _environment_config(**iceberg: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "catalog_name": "janus",
        "catalog_type": "rest",
        "uri": REST_URI,
        "warehouse_dir": WAREHOUSE_NAME,
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


def _auth_options(options: dict[str, str]) -> dict[str, str]:
    suffixes = {suffix for _key, suffix in CATALOG_AUTH_OPTIONS}
    return {
        key: value
        for key, value in options.items()
        if key.removeprefix(f"{CATALOG_PREFIX}.") in suffixes
    }


# ── the auth block ───────────────────────────────────────────────────────────


def test_the_auth_block_becomes_the_rest_specs_own_properties(tmp_path):
    """The four names are the spec's, so a managed catalog is reached by config alone."""

    options = _build_options(tmp_path, auth=FULL_AUTH_BLOCK)

    assert _auth_options(options) == EXPECTED_AUTH_OPTIONS


def test_the_profile_key_and_the_property_it_becomes_never_drift(tmp_path):
    """Every key `CATALOG_AUTH_OPTIONS` declares is actually emitted, and under its own name.

    The mapping is walked by both emitters; a row nobody emits would be a property the Spark
    session silently never receives while `pyiceberg` receives it.
    """

    options = _build_options(tmp_path, auth=FULL_AUTH_BLOCK)

    assert CATALOG_AUTH_OPTIONS
    for key, suffix in CATALOG_AUTH_OPTIONS:
        assert options[f"{CATALOG_PREFIX}.{suffix}"] == FULL_AUTH_BLOCK[key]


@pytest.mark.parametrize(
    ("label", "auth"),
    [
        ("absent", None),
        ("empty_block", {}),
        ("empty_expansions", dict.fromkeys(FULL_AUTH_BLOCK, "")),
        ("whitespace_expansions", dict.fromkeys(FULL_AUTH_BLOCK, "   ")),
    ],
)
def test_an_unset_expansion_never_becomes_a_blank_credential(tmp_path, label, auth):
    """The checked-in profile ships `${JANUS_ICEBERG_CATALOG_TOKEN:-}`; unset means absent.

    A blank bearer token is worse than none: it turns "this deployment is unauthenticated"
    into an authentication attempt the catalog rejects.
    """

    del label
    block: dict[str, Any] = {} if auth is None else {"auth": auth}

    assert _auth_options(_build_options(tmp_path, **block)) == {}


def test_one_half_of_the_block_reaches_the_options_without_the_other(tmp_path):
    """A bearer-token deployment sets `token` alone; the other three stay out."""

    options = _build_options(tmp_path, auth={"token": SECRET_TOKEN, "scope": ""})

    assert _auth_options(options) == {f"{CATALOG_PREFIX}.token": SECRET_TOKEN}


def test_an_explicit_spark_config_entry_still_beats_the_derived_property(tmp_path):
    """The `setdefault` rule holds for the auth block too."""

    config = _environment_config(auth=FULL_AUTH_BLOCK)
    config["spark"]["config"] = {f"{CATALOG_PREFIX}.token": "override-token"}

    options = build_spark_options(config, materialize_runtime_paths(config, tmp_path))

    assert options[f"{CATALOG_PREFIX}.token"] == "override-token"


# ── the block is inert where it has nowhere to go ────────────────────────────


def test_a_jdbc_catalog_ignores_the_auth_block_entirely(tmp_path):
    """The mirror of "a REST catalog ignores `credentials`"."""

    options = _build_options(
        tmp_path,
        catalog_type="jdbc",
        uri=JDBC_URI,
        driver_package=POSTGRES_DRIVER_PACKAGE,
        auth=FULL_AUTH_BLOCK,
    )

    assert _auth_options(options) == {}
    assert not [value for value in SECRET_VALUES if value in str(options)]


def test_a_rest_catalog_resolves_no_jdbc_driver(tmp_path):
    """A REST client opens no database, so the profile's driver package is not its business."""

    options = _build_options(tmp_path, driver_package=POSTGRES_DRIVER_PACKAGE)

    assert options["spark.jars.packages"] == ICEBERG_RUNTIME_PACKAGE


# ── the block fails closed ───────────────────────────────────────────────────


@pytest.mark.parametrize("catalog_type", ["jdbc", "rest"])
def test_an_unknown_auth_key_is_rejected_rather_than_ignored(tmp_path, catalog_type):
    """Refused under *both* types: the typo is in one file, whichever overlay is running."""

    block: dict[str, Any] = {"catalog_type": catalog_type}
    if catalog_type == "jdbc":
        block["uri"] = JDBC_URI
    config = _environment_config(**block, auth={"bearer_token": SECRET_TOKEN})
    paths = materialize_runtime_paths(config, tmp_path)

    with pytest.raises(ValueError) as error:
        build_spark_options(config, paths)

    message = str(error.value)
    assert "spark.iceberg.auth" in message
    assert "bearer_token" in message
    for supported, _suffix in CATALOG_AUTH_OPTIONS:
        assert supported in message


def test_the_rejection_names_the_key_but_never_the_secret_it_held(tmp_path):
    """The error a broken profile raises is what ends up in logs and tracebacks."""

    config = _environment_config(auth={"bearer_token": SECRET_TOKEN, **FULL_AUTH_BLOCK})
    paths = materialize_runtime_paths(config, tmp_path)

    with pytest.raises(ValueError) as error:
        build_spark_options(config, paths)

    message = str(error.value)
    assert not [value for value in SECRET_VALUES if value in message]


# ── the warehouse is an identifier, not a directory ──────────────────────────


def test_the_warehouse_reaches_spark_as_the_name_the_profile_wrote(tmp_path):
    """Project-resolving it would send the catalog a path off this machine's filesystem."""

    options = _build_options(tmp_path)

    assert options[f"{CATALOG_PREFIX}.warehouse"] == WAREHOUSE_NAME


def test_the_warehouse_identifier_is_not_a_resolved_path(tmp_path):
    """`materialize_runtime_paths` hands it on verbatim, as it does a location URI."""

    config = _environment_config()

    warehouse = materialize_runtime_paths(config, tmp_path)[ICEBERG_WAREHOUSE_PATH_KEY]

    assert warehouse == WAREHOUSE_NAME
    assert not isinstance(warehouse, Path)


def test_the_warehouse_identifier_is_never_created_on_disk(tmp_path):
    """The storage behind it is the catalog's; nothing here may mkdir a name."""

    config = _environment_config()

    prepare_runtime(config, tmp_path)

    assert not (tmp_path / WAREHOUSE_NAME).exists()
    # The zones that really are directories are still created beside it.
    assert (tmp_path / "data" / "raw").is_dir()


def test_a_catalog_that_owns_no_storage_still_resolves_its_warehouse(tmp_path):
    """The regression pin: only a catalog-managed warehouse changed meaning."""

    config = _environment_config(
        catalog_type="jdbc", uri=JDBC_URI, warehouse_dir="data/bronze/iceberg"
    )

    paths = materialize_runtime_paths(config, tmp_path)

    assert paths[ICEBERG_WAREHOUSE_PATH_KEY] == tmp_path / "data/bronze/iceberg"


@pytest.mark.parametrize("warehouse", ["", "   "])
def test_an_empty_warehouse_identifier_fails_closed(tmp_path, warehouse):
    """An unset `${JANUS_ICEBERG_WAREHOUSE_DIR}` must not reach the catalog as a lookup for ''."""

    config = _environment_config(warehouse_dir=warehouse)

    with pytest.raises(ValueError) as error:
        materialize_runtime_paths(config, tmp_path)

    message = str(error.value)
    assert "spark.iceberg.warehouse_dir" in message
    assert "'rest'" in message


# ── the second engine reads the same block ───────────────────────────────────


def test_pyiceberg_receives_the_same_identifier_and_the_same_auth(tmp_path):
    """Both engines must reach the same warehouse with the same credentials, or neither."""

    config = _environment_config(auth=FULL_AUTH_BLOCK)
    properties = derive_pyiceberg_catalog_properties(
        config, materialize_runtime_paths(config, tmp_path)
    )

    assert properties == {
        "type": "rest",
        "uri": REST_URI,
        "warehouse": WAREHOUSE_NAME,
        "token": SECRET_TOKEN,
        "credential": SECRET_CREDENTIAL,
        "oauth2-server-uri": OAUTH2_SERVER_URI,
        "scope": SCOPE,
    }


def test_pyiceberg_receives_no_blank_credential_either(tmp_path):
    """The empty-guard is the profile's, not each emitter's."""

    config = _environment_config(auth=dict.fromkeys(FULL_AUTH_BLOCK, ""))
    properties = derive_pyiceberg_catalog_properties(
        config, materialize_runtime_paths(config, tmp_path)
    )

    assert set(properties) == {"type", "uri", "warehouse"}
