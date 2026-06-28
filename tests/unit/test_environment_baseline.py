from pathlib import Path

import janus.utils.environment as environment_utils
from janus.main import default_project_root, main
from janus.utils.environment import (
    ICEBERG_CATALOG_IMPL,
    ICEBERG_SESSION_EXTENSIONS,
    build_spark_options,
    load_environment_config,
    materialize_runtime_paths,
    prepare_runtime,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1"


def test_default_project_root_prefers_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.delenv("JANUS_PROJECT_ROOT", raising=False)

    assert default_project_root() == PROJECT_ROOT


def test_default_project_root_prefers_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("JANUS_PROJECT_ROOT", str(tmp_path))

    assert default_project_root() == tmp_path


def test_load_local_environment_uses_checked_in_defaults(monkeypatch):
    monkeypatch.delenv("JANUS_SPARK_MASTER", raising=False)
    monkeypatch.delenv("JANUS_SPARK_DRIVER_BIND_ADDRESS", raising=False)
    monkeypatch.delenv("JANUS_SPARK_DRIVER_HOST", raising=False)
    monkeypatch.delenv("JANUS_RAW_DIR", raising=False)
    monkeypatch.delenv("JANUS_ICEBERG_RUNTIME_PACKAGE", raising=False)

    config = load_environment_config("local", PROJECT_ROOT)
    assert config["name"] == "local"
    assert config["spark"]["master"] == "local[*]"
    assert config["spark"]["config"]["spark.driver.bindAddress"] == "127.0.0.1"
    assert config["spark"]["config"]["spark.driver.host"] == "127.0.0.1"
    assert config["spark"]["ivy_dir"] == "data/metadata/ivy"
    assert config["spark"]["iceberg"]["catalog_name"] == "janus"
    assert config["spark"]["iceberg"]["runtime_package"] == ICEBERG_RUNTIME_PACKAGE
    assert config["storage"]["raw_dir"] == "data/raw"


def test_environment_variables_override_checked_in_defaults(monkeypatch):
    monkeypatch.setenv("JANUS_RAW_DIR", "/tmp/janus-raw")
    monkeypatch.setenv("JANUS_ICEBERG_WAREHOUSE_DIR", "/tmp/janus-iceberg")

    config = load_environment_config("local", PROJECT_ROOT)
    assert config["storage"]["raw_dir"] == "/tmp/janus-raw"
    assert config["spark"]["iceberg"]["warehouse_dir"] == "/tmp/janus-iceberg"


def test_prepare_runtime_creates_expected_directories(tmp_path):
    config = {
        "runtime": {"log_level": "INFO"},
        "spark": {
            "app_name": "janus-test",
            "master": "local[1]",
            "warehouse_dir": "data/metadata/spark-warehouse",
            "ivy_dir": "data/metadata/ivy",
            "iceberg": {
                "catalog_name": "janus",
                "warehouse_dir": "data/bronze/iceberg",
                "runtime_package": ICEBERG_RUNTIME_PACKAGE,
                "default_namespace": "bronze",
            },
            "config": {},
        },
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        },
    }

    paths = prepare_runtime(config, tmp_path)
    assert paths == materialize_runtime_paths(config, tmp_path)
    assert paths["raw_dir"].is_dir()
    assert paths["bronze_dir"].is_dir()
    assert paths["metadata_dir"].is_dir()
    assert paths["warehouse_dir"].is_dir()
    assert paths["ivy_dir"].is_dir()
    assert paths["iceberg_warehouse_dir"].is_dir()


def test_build_spark_options_merges_iceberg_runtime_settings(tmp_path):
    config = {
        "runtime": {"log_level": "INFO"},
        "spark": {
            "app_name": "janus-test",
            "master": "local[1]",
            "warehouse_dir": "data/metadata/spark-warehouse",
            "ivy_dir": "data/metadata/ivy",
            "iceberg": {
                "catalog_name": "janus",
                "warehouse_dir": "data/bronze/iceberg",
                "runtime_package": ICEBERG_RUNTIME_PACKAGE,
                "default_namespace": "bronze",
            },
            "config": {
                "spark.jars.packages": "com.example:demo:1.0.0",
                "spark.sql.extensions": "com.example.CustomSparkExtensions",
                "spark.driver.host": "127.0.0.1",
                "spark.sql.shuffle.partitions": "2",
            },
        },
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        },
    }

    paths = materialize_runtime_paths(config, tmp_path)
    options = build_spark_options(config, paths)

    assert options["spark.sql.warehouse.dir"] == str(paths["warehouse_dir"])
    assert options["spark.jars.ivy"] == str(paths["ivy_dir"])
    assert options["spark.driver.host"] == "127.0.0.1"
    assert options["spark.jars.packages"] == (
        f"com.example:demo:1.0.0,{ICEBERG_RUNTIME_PACKAGE}"
    )
    assert options["spark.sql.extensions"] == (
        f"com.example.CustomSparkExtensions,{ICEBERG_SESSION_EXTENSIONS}"
    )
    assert options["spark.sql.defaultCatalog"] == "janus"
    assert options["spark.sql.catalog.janus"] == ICEBERG_CATALOG_IMPL
    assert options["spark.sql.catalog.janus.type"] == "hadoop"
    assert options["spark.sql.catalog.janus.warehouse"] == str(paths["iceberg_warehouse_dir"])
    assert options["spark.sql.catalog.janus.default-namespace"] == "bronze"
    assert options["spark.sql.shuffle.partitions"] == "2"



def test_main_reports_runtime_permission_error_cleanly(monkeypatch, capsys):
    config = {
        "name": "local",
        "spark": {"app_name": "janus-test", "master": "local[1]", "config": {}},
    }

    monkeypatch.setattr(
        "janus.main.load_environment_config",
        lambda environment, project_root: config,
    )

    def raise_permission_error(config, project_root):
        raise PermissionError(13, "Permission denied", "/workspace/data/metadata/ivy")

    monkeypatch.setattr("janus.main.prepare_runtime", raise_permission_error)

    exit_code = main(["--environment", "local"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "/workspace/data/metadata/ivy" in captured.err
    assert "make down && make up" in captured.err
    assert captured.out == ""



def test_prepare_runtime_falls_back_only_for_spark_scratch_paths(monkeypatch, tmp_path):
    project_root = tmp_path / "workspace"
    scratch_root = tmp_path / "scratch"
    config = _runtime_config()
    original_ensure = environment_utils._ensure_writable_directory
    spark_scratch_suffixes = {
        "spark-warehouse",
        "ivy",
        "iceberg",
    }

    def fake_ensure_writable_directory(path):
        if path.name in spark_scratch_suffixes:
            raise PermissionError(13, "Permission denied", str(path))
        original_ensure(path)

    monkeypatch.setenv("JANUS_RUNTIME_SCRATCH_DIR", str(scratch_root))
    monkeypatch.setattr(
        environment_utils,
        "_ensure_writable_directory",
        fake_ensure_writable_directory,
    )

    paths = prepare_runtime(config, project_root)

    assert paths["root_dir"] == project_root / "data"
    assert paths["raw_dir"] == project_root / "data" / "raw"
    assert paths["bronze_dir"] == project_root / "data" / "bronze"
    assert paths["metadata_dir"] == project_root / "data" / "metadata"
    assert paths["warehouse_dir"] == scratch_root / "workspace" / "warehouse_dir"
    assert paths["ivy_dir"] == scratch_root / "workspace" / "ivy_dir"
    assert paths["iceberg_warehouse_dir"] == (
        scratch_root / "workspace" / "iceberg_warehouse_dir"
    )
    assert config["storage"]["raw_dir"] == "data/raw"
    assert config["storage"]["metadata_dir"] == "data/metadata"
    assert config["spark"]["ivy_dir"] == str(paths["ivy_dir"])
    assert config["spark"]["iceberg"]["warehouse_dir"] == str(
        paths["iceberg_warehouse_dir"]
    )
    assert all(path.is_dir() for path in paths.values())


def test_prepare_runtime_raises_when_storage_zone_is_unwritable(monkeypatch, tmp_path):
    project_root = tmp_path / "workspace"
    config = _runtime_config()
    original_ensure = environment_utils._ensure_writable_directory

    def fake_ensure_writable_directory(path):
        if path == project_root / "data" / "raw":
            raise PermissionError(13, "Permission denied", str(path))
        original_ensure(path)

    monkeypatch.setattr(
        environment_utils,
        "_ensure_writable_directory",
        fake_ensure_writable_directory,
    )

    try:
        prepare_runtime(config, project_root)
    except PermissionError as exc:
        assert exc.filename == str(project_root / "data" / "raw")
    else:
        raise AssertionError("prepare_runtime should fail for unwritable JANUS data zones")


def _runtime_config() -> dict:
    return {
        "runtime": {"log_level": "INFO"},
        "spark": {
            "app_name": "janus-test",
            "master": "local[1]",
            "warehouse_dir": "data/metadata/spark-warehouse",
            "ivy_dir": "data/metadata/ivy",
            "iceberg": {
                "catalog_name": "janus",
                "warehouse_dir": "data/bronze/iceberg",
                "runtime_package": ICEBERG_RUNTIME_PACKAGE,
                "default_namespace": "bronze",
            },
            "config": {},
        },
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        },
    }
