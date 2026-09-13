"""URI-aware runtime locations: a warehouse on object storage is not a directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import janus.utils.environment as environment_utils
from janus.utils.catalog_properties import derive_pyiceberg_catalog_properties
from janus.utils.environment import (
    ICEBERG_WAREHOUSE_PATH_KEY,
    build_spark_options,
    is_location_uri,
    materialize_runtime_paths,
    prepare_runtime,
)

ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1"
OBJECT_STORE_WAREHOUSE = "s3://janus-bronze/warehouse"
LOCAL_WAREHOUSE = "data/bronze/iceberg"
CATALOG_PREFIX = "spark.sql.catalog.janus"

#: Schemes are matched structurally, never from a list. These are here to show that the
#: rule already covers stores JANUS has never been pointed at.
LOCATION_URIS = (
    "s3://janus-bronze/warehouse",
    "s3a://janus-bronze/warehouse",
    "gs://janus-bronze/warehouse",
    "abfss://bronze@janus.dfs.core.windows.net/warehouse",
    "file:///srv/janus/warehouse",
)

#: Values that are paths, however URI-ish they look.
LOCAL_VALUES = (
    "data/bronze/iceberg",
    "/srv/janus/warehouse",
    "./relative/warehouse",
    "data/bronze/iceberg:snapshot",
)


def _config(warehouse: str, **spark: Any) -> dict[str, Any]:
    block = {
        "app_name": "janus-test",
        "master": "local[1]",
        "warehouse_dir": "data/metadata/spark-warehouse",
        "ivy_dir": "data/metadata/ivy",
        "iceberg": {
            "catalog_name": "janus",
            "catalog_type": "jdbc",
            "warehouse_dir": warehouse,
            "runtime_package": ICEBERG_RUNTIME_PACKAGE,
            "default_namespace": "bronze",
            "uri": "jdbc:sqlite:data/metadata/iceberg-catalog/catalog.sqlite",
        },
        "config": {},
    }
    block.update(spark)
    return {
        "runtime": {"log_level": "INFO"},
        "spark": block,
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        },
    }


# ── the predicate ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", LOCATION_URIS)
def test_a_scheme_bearing_value_is_a_location(value):
    assert is_location_uri(value)


@pytest.mark.parametrize("value", LOCAL_VALUES)
def test_a_path_is_not_a_location(value):
    assert not is_location_uri(value)


def test_a_resolved_path_object_is_not_a_location():
    """The predicate reads configured text; a `Path` has already lost the authority."""

    assert not is_location_uri(Path("s3://janus-bronze/warehouse"))


def test_the_mangling_this_rule_exists_to_prevent_is_real():
    """The premise, asserted rather than remembered: `Path` eats the empty authority."""

    assert str(Path(OBJECT_STORE_WAREHOUSE)) == "s3:/janus-bronze/warehouse"
    assert str(Path(OBJECT_STORE_WAREHOUSE)) != OBJECT_STORE_WAREHOUSE


# ── resolution ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("warehouse", LOCATION_URIS)
def test_a_warehouse_uri_is_carried_verbatim(tmp_path, warehouse):
    """Not project-resolved, and never round-tripped through `Path`."""

    paths = materialize_runtime_paths(_config(warehouse), tmp_path)

    assert paths[ICEBERG_WAREHOUSE_PATH_KEY] == warehouse
    assert isinstance(paths[ICEBERG_WAREHOUSE_PATH_KEY], str)
    assert str(tmp_path) not in str(paths[ICEBERG_WAREHOUSE_PATH_KEY])


def test_a_local_warehouse_still_resolves_exactly_as_before(tmp_path):
    paths = materialize_runtime_paths(_config(LOCAL_WAREHOUSE), tmp_path)

    assert paths[ICEBERG_WAREHOUSE_PATH_KEY] == tmp_path / LOCAL_WAREHOUSE
    assert isinstance(paths[ICEBERG_WAREHOUSE_PATH_KEY], Path)


def test_only_the_iceberg_warehouse_is_uri_aware(tmp_path):
    """The scope boundary, stated as a test."""

    config = _config(OBJECT_STORE_WAREHOUSE)
    config["storage"]["raw_dir"] = "s3://janus-raw/zone"

    paths = materialize_runtime_paths(config, tmp_path)

    assert isinstance(paths["raw_dir"], Path)
    assert paths["raw_dir"].is_relative_to(tmp_path)


# ── prepare_runtime leaves a store alone ─────────────────────────────────────


def test_a_warehouse_uri_is_never_created_on_disk(tmp_path):
    paths = prepare_runtime(_config(OBJECT_STORE_WAREHOUSE), tmp_path)

    assert paths[ICEBERG_WAREHOUSE_PATH_KEY] == OBJECT_STORE_WAREHOUSE
    assert not (tmp_path / "s3:").exists()
    assert [entry.name for entry in tmp_path.iterdir()] == ["data"]


def test_the_local_zones_beside_a_warehouse_uri_are_still_created(tmp_path):
    """The realistic cluster shape: bronze on object storage, everything else on the volume."""

    paths = prepare_runtime(_config(OBJECT_STORE_WAREHOUSE), tmp_path)

    for key in ("root_dir", "raw_dir", "bronze_dir", "metadata_dir", "warehouse_dir", "ivy_dir"):
        assert isinstance(paths[key], Path)
        assert paths[key].is_dir()


def test_a_warehouse_uri_is_never_permission_probed(tmp_path, monkeypatch):
    """Nothing may `mkdir` or write a probe file into a location this process does not own."""

    probed: list[Path] = []
    original = environment_utils._ensure_writable_directory

    def record(path: Path) -> None:
        probed.append(path)
        original(path)

    monkeypatch.setattr(environment_utils, "_ensure_writable_directory", record)

    prepare_runtime(_config(OBJECT_STORE_WAREHOUSE), tmp_path)

    assert probed, "the probe never ran at all — this test would pass vacuously"
    assert not [path for path in probed if "s3:" in str(path)]


def test_a_warehouse_uri_never_falls_back_to_the_scratch_directory(tmp_path, monkeypatch):
    """An unreachable bucket is not fixed by writing somewhere else, so there is no fallback.

    The Iceberg warehouse *is* a fallback-eligible key when it is a local directory, which
    is what makes this worth pinning: the eligibility must follow the value, not the key.
    """

    project_root = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("JANUS_RUNTIME_SCRATCH_DIR", str(scratch))
    monkeypatch.setattr(
        environment_utils,
        "_ensure_writable_directory",
        _refuse_the_spark_scratch_paths(),
    )

    paths = prepare_runtime(_config(OBJECT_STORE_WAREHOUSE), project_root)

    assert (scratch / "workspace" / "ivy_dir").is_dir()
    assert (scratch / "workspace" / "warehouse_dir").is_dir()
    assert not (scratch / "workspace" / ICEBERG_WAREHOUSE_PATH_KEY).exists()
    assert paths[ICEBERG_WAREHOUSE_PATH_KEY] == OBJECT_STORE_WAREHOUSE


SPARK_SCRATCH_DIRECTORY_NAMES = frozenset({"spark-warehouse", "ivy"})


def _refuse_the_spark_scratch_paths():
    """An unwritable mount under the Spark scratch directories, and nothing else."""

    original = environment_utils._ensure_writable_directory

    def ensure(path: Path) -> None:
        if path.name in SPARK_SCRATCH_DIRECTORY_NAMES:
            raise PermissionError(13, "Permission denied", str(path))
        original(path)

    return ensure


# ── both emitters agree on what the location is ──────────────────────────────


def test_spark_receives_the_warehouse_uri_unchanged(tmp_path):
    config = _config(OBJECT_STORE_WAREHOUSE)

    options = build_spark_options(config, materialize_runtime_paths(config, tmp_path))

    assert options[f"{CATALOG_PREFIX}.warehouse"] == OBJECT_STORE_WAREHOUSE


def test_both_engines_are_pointed_at_the_same_location(tmp_path):
    """A second engine picks its own FileIO from the scheme, so the scheme must survive."""

    config = _config(OBJECT_STORE_WAREHOUSE)
    paths = materialize_runtime_paths(config, tmp_path)

    options = build_spark_options(config, paths)
    properties = derive_pyiceberg_catalog_properties(config, paths)

    assert properties["warehouse"] == options[f"{CATALOG_PREFIX}.warehouse"]
    assert properties["warehouse"] == OBJECT_STORE_WAREHOUSE
