"""The shared test session builder, covered where PySpark is not.

Twelve suites now build their session through `tests/support/spark_sessions.py`, and on a
host without PySpark every one of them skips — so nothing would exercise the builder's own
logic until CI. These tests do, over the checked-in `local` profile and no Spark at all:
the catalog comes from the profile rather than from a hand-written block, each suite gets
its own database, and no session can reach Maven.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from janus.utils.environment import JDBC_CATALOG_TYPE, SUPPORTED_CATALOG_TYPES
from tests.support import spark_sessions
from tests.support.spark_sessions import (
    JDBC_SQLITE_PREFIX,
    TEST_SESSION_CONFIG,
    catalog_database_path,
    checked_in_iceberg_block,
    iceberg_session_options,
    suite_catalog_uri,
    vendored_jar,
)

CATALOG_PREFIX = "spark.sql.catalog.janus"


def _attached_jars(options) -> list[Path]:
    return [Path(jar) for jar in options["spark.jars"].split(",")]


def _options(root, **kwargs):
    return iceberg_session_options(
        warehouse_dir=root / "iceberg",
        catalog_db=catalog_database_path(root),
        spark_warehouse_dir=root / "spark-warehouse",
        **kwargs,
    )


# ── the catalog comes from the profile, not from the test ────────────────────


def test_the_emitted_catalog_type_is_the_one_the_profile_declares(tmp_path):
    """The whole point of the task: a suite cannot pin a catalog of its own choosing."""

    declared = checked_in_iceberg_block()["catalog_type"]

    assert _options(tmp_path)[f"{CATALOG_PREFIX}.type"] == declared


def test_a_janus_env_override_cannot_repoint_the_suite(tmp_path, monkeypatch):
    """A developer poking at a warehouse must not thereby move the whole Spark suite."""

    monkeypatch.setenv("JANUS_ICEBERG_CATALOG_TYPE", "rest")
    monkeypatch.setenv("JANUS_ICEBERG_CATALOG_NAME", "not-janus")

    options = _options(tmp_path)

    assert options[f"{CATALOG_PREFIX}.type"] == "jdbc"
    assert options["spark.sql.defaultCatalog"] == "janus"


def test_the_session_settings_are_the_two_the_ported_builders_pinned(tmp_path):
    """Construction moved to the helper; behaviour did not move with it."""

    options = _options(tmp_path)

    for key, value in TEST_SESSION_CONFIG.items():
        assert options[key] == value


def test_the_profiles_own_runtime_settings_are_not_inherited(tmp_path):
    """`local.yaml` sizes a real run — 8g driver, 256 shuffle partitions. A suite is not one."""

    options = _options(tmp_path)

    assert "spark.driver.memory" not in options
    assert "spark.sql.shuffle.partitions" not in options


# ── every suite gets its own catalog, and its own warehouse ──────────────────


def test_each_suite_gets_its_own_catalog_database(tmp_path):
    """A shared catalog file would couple suites through SQLite's database lock."""

    first, second = tmp_path / "suite-a", tmp_path / "suite-b"

    first_uri = _options(first)[f"{CATALOG_PREFIX}.uri"]
    second_uri = _options(second)[f"{CATALOG_PREFIX}.uri"]

    assert first_uri != second_uri
    assert str(first) in first_uri
    assert str(second) in second_uri


def test_the_catalog_database_never_lands_inside_the_warehouse(tmp_path):
    """The catalog is metadata, not data, applied to the suites."""

    database = catalog_database_path(tmp_path)

    assert not database.is_relative_to(tmp_path / "iceberg")
    assert "metadata" in database.parts


def test_the_warehouse_and_spark_warehouse_stay_where_the_ported_builders_put_them(tmp_path):
    """These paths are what bronze assertions resolve against; they must not move."""

    options = _options(tmp_path)

    assert options[f"{CATALOG_PREFIX}.warehouse"] == str(tmp_path / "iceberg")
    assert options["spark.sql.warehouse.dir"] == str(tmp_path / "spark-warehouse")


def test_a_named_catalog_renames_every_catalog_key(tmp_path):
    """A suite needing a second catalog gets a whole block, not a half-renamed one."""

    options = _options(tmp_path, catalog_name="janus_second")

    assert options["spark.sql.defaultCatalog"] == "janus_second"
    assert options["spark.sql.catalog.janus_second.type"] == "jdbc"
    assert not [key for key in options if key.startswith(f"{CATALOG_PREFIX}.")]


# ── the driver query, and the offline-jar rule ───────────────────────────────


def test_the_driver_query_the_profile_ships_survives_the_redirection(tmp_path):
    """WAL and the busy timeout are why concurrent committers serialize instead of failing.

    Dropping them while replacing the path would leave the suites committing under settings
    a run never uses — the drift this task closes, one layer down.
    """

    profile_query = checked_in_iceberg_block()["uri"].partition("?")[2]
    emitted = _options(tmp_path)[f"{CATALOG_PREFIX}.uri"]

    assert profile_query, "the profile's catalog URI carries no driver query to preserve"
    assert emitted.endswith(f"?{profile_query}")
    assert emitted.startswith(f"{JDBC_SQLITE_PREFIX}{catalog_database_path(tmp_path)}?")


def test_no_session_can_send_the_test_gate_to_maven(tmp_path):
    """offline-jar rule. `spark.jars.packages` is a fetch; `spark.jars` is a path."""

    iceberg = checked_in_iceberg_block()
    options = _options(tmp_path)

    assert "spark.jars.packages" not in options
    assert _attached_jars(options) == [
        vendored_jar(iceberg["runtime_package"]),
        vendored_jar(iceberg["driver_package"]),
    ]


def test_the_jars_are_taken_from_the_directory_seed_ivy_and_ci_fill(tmp_path):
    """A jar resolved from anywhere else would pass here and fail in the container."""

    for jar in _attached_jars(_options(tmp_path)):
        assert jar.parent == spark_sessions.IVY_JARS_DIR


# ── fail-closed when the profile outgrows the helper ─────────────────────────

OTHER_CATALOG_TYPES = sorted(SUPPORTED_CATALOG_TYPES - {JDBC_CATALOG_TYPE})


@pytest.mark.parametrize("catalog_type", OTHER_CATALOG_TYPES)
def test_a_catalog_type_the_helper_cannot_privately_copy_is_refused(catalog_type, tmp_path):
    """Silently sharing one catalog across twelve suites is the failure worth preventing."""

    block = {"catalog_type": catalog_type, "uri": "http://catalog:8181"}

    with pytest.raises(NotImplementedError) as error:
        suite_catalog_uri(block, catalog_database_path(tmp_path))

    assert "share one catalog" in str(error.value)


def test_the_supported_types_the_parametrization_reads_are_not_empty():
    """A set that shrank to `{jdbc}` would leave the refusal untested and the test green."""

    assert OTHER_CATALOG_TYPES


def test_a_server_backed_jdbc_catalog_is_refused_too(tmp_path):
    """A Postgres catalog is one database; twelve suites sharing it is not isolation."""

    block = {"catalog_type": JDBC_CATALOG_TYPE, "uri": "jdbc:postgresql://db/janus"}

    with pytest.raises(NotImplementedError) as error:
        suite_catalog_uri(block, catalog_database_path(tmp_path))

    assert "share one catalog" in str(error.value)


def test_the_refusal_names_the_profile_a_reader_has_to_edit(tmp_path):
    """An error that does not say where the decision lives sends the reader hunting."""

    with pytest.raises(NotImplementedError) as error:
        suite_catalog_uri({"catalog_type": "rest", "uri": "http://catalog:8181"}, tmp_path)

    assert "conf/environments/local.yaml" in str(error.value)
