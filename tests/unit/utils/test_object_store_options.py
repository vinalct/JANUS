"""The `object_store` block: the FileIO configuration a warehouse on object storage needs."""

from __future__ import annotations

from typing import Any

import pytest

from janus.utils.environment import (
    FILE_IO_IMPL_OPTION,
    OBJECT_STORE_KEY,
    S3_FILE_IO_IMPL,
    SUPPORTED_FILE_IO_IMPLS,
    SUPPORTED_OBJECT_STORE_KEYS,
    build_spark_options,
    materialize_runtime_paths,
)

ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1"
AWS_BUNDLE_PACKAGE = "org.apache.iceberg:iceberg-aws-bundle:1.10.1"
POSTGRES_DRIVER_PACKAGE = "org.postgresql:postgresql:42.7.13"
CATALOG_PREFIX = "spark.sql.catalog.janus"
OBJECT_STORE_WAREHOUSE = "s3://janus-bronze/warehouse"
SECRET_KEY = "janus-object-store-secret"

#: The block the `cluster` profile ships, as a mapping.
OBJECT_STORE_BLOCK = {
    "io_impl": "S3FileIO",
    "io_package": AWS_BUNDLE_PACKAGE,
    "endpoint": "http://minio:9000",
    "path_style_access": "true",
    "region": "us-east-1",
}

#: What the block adds to the catalog options, keyed by Iceberg's own property names.
EXPECTED_OBJECT_STORE_OPTIONS = {
    f"{CATALOG_PREFIX}.io-impl": S3_FILE_IO_IMPL,
    f"{CATALOG_PREFIX}.s3.endpoint": "http://minio:9000",
    f"{CATALOG_PREFIX}.s3.path-style-access": "true",
    f"{CATALOG_PREFIX}.client.region": "us-east-1",
}


def _config(*, object_store: dict[str, Any] | None, **iceberg: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "catalog_name": "janus",
        "catalog_type": "jdbc",
        "warehouse_dir": OBJECT_STORE_WAREHOUSE,
        "runtime_package": ICEBERG_RUNTIME_PACKAGE,
        "default_namespace": "bronze",
        "uri": "jdbc:postgresql://postgres:5432/iceberg",
        "driver_package": POSTGRES_DRIVER_PACKAGE,
    }
    block.update(iceberg)
    if object_store is not None:
        block[OBJECT_STORE_KEY] = object_store
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


def _build(project_root, *, object_store=OBJECT_STORE_BLOCK, **iceberg) -> dict[str, str]:
    config = _config(object_store=object_store, **iceberg)
    return build_spark_options(config, materialize_runtime_paths(config, project_root))


# ── what the block emits ─────────────────────────────────────────────────────


def test_the_block_emits_the_file_io_and_its_settings(tmp_path):
    options = _build(tmp_path)

    for key, value in EXPECTED_OBJECT_STORE_OPTIONS.items():
        assert options[key] == value


def test_the_emitted_property_names_are_icebergs_own(tmp_path):
    """Read from the vendored jar, not from memory — see this module's docstring."""

    options = _build(tmp_path)

    assert options[f"{CATALOG_PREFIX}.{FILE_IO_IMPL_OPTION}"] == S3_FILE_IO_IMPL
    assert FILE_IO_IMPL_OPTION == "io-impl"
    assert set(EXPECTED_OBJECT_STORE_OPTIONS) <= set(options)


def test_the_io_bundle_joins_the_packages_the_session_resolves(tmp_path):
    """S3FileIO is in the Iceberg runtime jar; the AWS SDK it calls is not."""

    options = _build(tmp_path)

    assert options["spark.jars.packages"] == (
        f"{ICEBERG_RUNTIME_PACKAGE},{POSTGRES_DRIVER_PACKAGE},{AWS_BUNDLE_PACKAGE}"
    )


def test_an_object_store_without_a_package_emits_only_the_settings(tmp_path):
    """A deployment whose image already ships the bundle declares no coordinate."""

    block = {key: value for key, value in OBJECT_STORE_BLOCK.items() if key != "io_package"}

    options = _build(tmp_path, object_store=block)

    assert options["spark.jars.packages"] == (
        f"{ICEBERG_RUNTIME_PACKAGE},{POSTGRES_DRIVER_PACKAGE}"
    )
    assert options[f"{CATALOG_PREFIX}.{FILE_IO_IMPL_OPTION}"] == S3_FILE_IO_IMPL


def test_an_empty_expansion_drops_the_setting_rather_than_emitting_a_blank(tmp_path):
    """An unset `${JANUS_S3_REGION}` expands to `""`, and `""` is not a region."""

    block = {**OBJECT_STORE_BLOCK, "region": "", "endpoint": "   "}

    options = _build(tmp_path, object_store=block)

    assert f"{CATALOG_PREFIX}.client.region" not in options
    assert f"{CATALOG_PREFIX}.s3.endpoint" not in options
    assert options[f"{CATALOG_PREFIX}.{FILE_IO_IMPL_OPTION}"] == S3_FILE_IO_IMPL


def test_the_block_is_independent_of_the_catalog_type(tmp_path):
    """Where a table's *files* live is not what coordinates its commits."""

    options = _build(tmp_path, catalog_type="rest", uri="http://catalog:8181")

    assert options[f"{CATALOG_PREFIX}.type"] == "rest"
    for key, value in EXPECTED_OBJECT_STORE_OPTIONS.items():
        assert options[key] == value


def test_an_explicit_spark_config_entry_still_wins(tmp_path):
    """The `setdefault` rule holds here too: the profile's own value beats the derived one."""

    config = _config(object_store=OBJECT_STORE_BLOCK)
    config["spark"]["config"] = {
        f"{CATALOG_PREFIX}.{FILE_IO_IMPL_OPTION}": "com.example.CustomFileIO",
        f"{CATALOG_PREFIX}.s3.endpoint": "http://override:9000",
    }

    options = build_spark_options(config, materialize_runtime_paths(config, tmp_path))

    assert options[f"{CATALOG_PREFIX}.{FILE_IO_IMPL_OPTION}"] == "com.example.CustomFileIO"
    assert options[f"{CATALOG_PREFIX}.s3.endpoint"] == "http://override:9000"


# ── a profile without the block is untouched ─────────────────────────────────


@pytest.mark.parametrize(
    ("label", "object_store"), [("absent", None), ("empty", {})]
)
def test_a_profile_with_no_object_store_gets_no_file_io_configuration(
    tmp_path, label, object_store
):
    """AC-4 for this block: the local profile's options must not move because of it."""

    del label
    options = _build(
        tmp_path, object_store=object_store, warehouse_dir="data/bronze/iceberg"
    )

    assert not [key for key in options if ".s3." in key or key.endswith(".io-impl")]
    assert not [key for key in options if key.endswith(".client.region")]
    assert options["spark.jars.packages"] == (
        f"{ICEBERG_RUNTIME_PACKAGE},{POSTGRES_DRIVER_PACKAGE}"
    )


# ── fail closed ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("label", "block"),
    [
        ("absent", {}),
        ("empty_expansion", {"io_impl": ""}),
        ("whitespace_expansion", {"io_impl": "   "}),
    ],
)
def test_a_block_without_a_file_io_names_the_key_and_the_supported_set(
    tmp_path, label, block
):
    del label
    object_store = {**OBJECT_STORE_BLOCK, **block}
    if not block:
        del object_store["io_impl"]

    with pytest.raises(ValueError) as error:
        _build(tmp_path, object_store=object_store)

    message = str(error.value)
    assert f"spark.iceberg.{OBJECT_STORE_KEY}.io_impl" in message
    for supported in SUPPORTED_FILE_IO_IMPLS:
        assert supported in message


def test_an_unverified_file_io_is_refused_by_name(tmp_path):
    """A fully-qualified class in YAML would put any class on the classpath into a session."""

    object_store = {**OBJECT_STORE_BLOCK, "io_impl": "org.apache.iceberg.aws.s3.S3FileIO"}

    with pytest.raises(ValueError) as error:
        _build(tmp_path, object_store=object_store)

    message = str(error.value)
    assert "org.apache.iceberg.aws.s3.S3FileIO" in message
    assert "S3FileIO" in message


def test_an_unknown_block_key_is_rejected_rather_than_ignored(tmp_path):
    """Silently ignoring a key is how a setting somebody believed in never took effect."""

    object_store = {**OBJECT_STORE_BLOCK, "pathStyleAccess": "true"}

    with pytest.raises(ValueError) as error:
        _build(tmp_path, object_store=object_store)

    message = str(error.value)
    assert "pathStyleAccess" in message
    for supported in SUPPORTED_OBJECT_STORE_KEYS:
        assert supported in message


@pytest.mark.parametrize("credential_key", ["access_key", "secret_key", "session_token"])
def test_an_object_store_credential_cannot_be_configured_at_all(tmp_path, credential_key):
    """The hygiene rule, enforced rather than documented.

    S3FileIO reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY itself, so JANUS never needs a
    credential key — and a profile that grew one would be a tracked file inviting a secret.
    """

    object_store = {**OBJECT_STORE_BLOCK, credential_key: SECRET_KEY}

    with pytest.raises(ValueError) as error:
        _build(tmp_path, object_store=object_store)

    message = str(error.value)
    assert credential_key in message
    assert SECRET_KEY not in message, "the rejection echoed the credential it refused"
    assert "AWS_ACCESS_KEY_ID" in message


@pytest.mark.parametrize(
    ("declared", "emitted"),
    [("true", "true"), ("True", "true"), ("yes", "true"), ("1", "true"),
     ("false", "false"), ("False", "false"), ("no", "false"), ("0", "false")],
)
def test_a_boolean_setting_is_canonicalised_before_either_engine_sees_it(
    tmp_path, declared, emitted
):
    """Both engines must read the same answer, and one of them reads it inverted."""

    options = _build(
        tmp_path, object_store={**OBJECT_STORE_BLOCK, "path_style_access": declared}
    )

    assert options[f"{CATALOG_PREFIX}.s3.path-style-access"] == emitted


def test_an_unquoted_yaml_boolean_is_accepted(tmp_path):
    """`path_style_access: true` with no quotes is a bool by the time it reaches here."""

    options = _build(
        tmp_path, object_store={**OBJECT_STORE_BLOCK, "path_style_access": True}
    )

    assert options[f"{CATALOG_PREFIX}.s3.path-style-access"] == "true"


def test_a_typo_in_a_boolean_setting_fails_closed(tmp_path):
    """Java reads anything that is not `true` as false, so a typo would silently flip it.

    Against an S3-compatible store that is not AWS, the flipped value addresses the bucket
    by hostname: the run fails at the first read, against the store, with a DNS error.
    """

    with pytest.raises(ValueError) as error:
        _build(tmp_path, object_store={**OBJECT_STORE_BLOCK, "path_style_access": "ture"})

    message = str(error.value)
    assert "path_style_access" in message
    assert "'ture'" in message
    assert "true" in message and "false" in message


def test_the_supported_sets_this_module_parametrizes_over_are_not_empty():
    """A table that emptied would leave several assertions above vacuously true."""

    assert SUPPORTED_FILE_IO_IMPLS
    assert SUPPORTED_OBJECT_STORE_KEYS
    assert SUPPORTED_FILE_IO_IMPLS["S3FileIO"] == S3_FILE_IO_IMPL
