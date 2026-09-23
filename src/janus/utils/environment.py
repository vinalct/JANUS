"""Environment profiles: loading them, materializing their paths, building a Spark session."""

from __future__ import annotations

import errno
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from janus.utils.catalog_options import (
    CATALOG_AUTH_KEY,
    CATALOG_AUTH_OPTIONS,
    CATALOG_TYPE_KEY,
    CATALOG_TYPES_REQUIRING_URI,
    CATALOG_TYPES_WITH_AUTH,
    CATALOG_TYPES_WITH_CATALOG_MANAGED_WAREHOUSE,
    FILE_IO_IMPL_KEY,
    FILE_IO_IMPL_OPTION,
    HADOOP_CATALOG_TYPE,
    ICEBERG_CATALOG_DB_PATH_KEY,
    ICEBERG_CATALOG_IMPL,
    ICEBERG_SESSION_EXTENSIONS,
    ICEBERG_WAREHOUSE_PATH_KEY,
    JDBC_AUTHORITY_PREFIX,
    JDBC_CATALOG_TYPE,
    JDBC_CREDENTIAL_OPTIONS,
    JDBC_SCHEMA_VERSION_OPTION,
    JDBC_URI_PREFIX,
    LOCATION_URI_PATTERN,
    OBJECT_STORE_ENDPOINT_KEY,
    OBJECT_STORE_FLAG_KEYS,
    OBJECT_STORE_KEY,
    OBJECT_STORE_OPTIONS,
    OBJECT_STORE_PACKAGE_KEY,
    OBJECT_STORE_PATH_STYLE_KEY,
    OBJECT_STORE_REGION_KEY,
    REST_CATALOG_TYPE,
    S3_FILE_IO_IMPL,
    SUPPORTED_CATALOG_AUTH_KEYS,
    SUPPORTED_CATALOG_TYPES,
    SUPPORTED_FILE_IO_IMPLS,
    SUPPORTED_OBJECT_STORE_KEYS,
    WAREHOUSE_DIR_KEY,
    RuntimeLocation,
    apply_iceberg_catalog_options,
    catalog_auth_block,
    catalog_database_path,
    catalog_managed_warehouse,
    declared_catalog_type,
    is_location_uri,
    merge_csv_values,
    non_empty_text,
    object_store_block,
    object_store_flag,
    object_store_value,
    required_catalog_value,
    resolve_catalog_type,
    resolve_catalog_uri,
    resolve_file_io_impl,
    warehouse_is_catalog_managed,
)

ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)(?::-(?P<default>[^}]*))?\}")
RUNTIME_SCRATCH_DIR_ENV = "JANUS_RUNTIME_SCRATCH_DIR"
FALLBACK_RUNTIME_PATH_KEYS = frozenset({"warehouse_dir", ICEBERG_WAREHOUSE_PATH_KEY})
RUNTIME_FILE_PATH_KEYS = frozenset({ICEBERG_CATALOG_DB_PATH_KEY})
_PROCESS_FALLBACK_ROOT: Path | None = None
_RUNTIME_PATH_CONFIG_LOCATIONS = {
    "root_dir": ("storage", "root_dir"),
    "raw_dir": ("storage", "raw_dir"),
    "bronze_dir": ("storage", "bronze_dir"),
    "metadata_dir": ("storage", "metadata_dir"),
    "warehouse_dir": ("spark", "warehouse_dir"),
    "ivy_dir": ("spark", "ivy_dir"),
    ICEBERG_WAREHOUSE_PATH_KEY: ("spark", "iceberg", "warehouse_dir"),
}


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(
            lambda match: os.getenv(match.group("name"), match.group("default") or ""),
            value,
        )
    return value


def load_environment_config(environment: str, project_root: Path) -> dict[str, Any]:
    config_path = project_root / "conf" / "environments" / f"{environment}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Environment config must be a mapping: {config_path}")

    return expand_env_vars(data)


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def resolve_runtime_location(project_root: Path, value: str) -> RuntimeLocation:
    """A location URI verbatim; anything else resolved project-relative, exactly as before."""

    return value if is_location_uri(value) else resolve_project_path(project_root, value)


def materialize_runtime_paths(
    config: dict[str, Any], project_root: Path
) -> dict[str, RuntimeLocation]:
    """Every runtime location the profile declares, resolved once.

    Only the Iceberg warehouse may be something other than a directory: it is the one
    location an engine reaches through its own storage layer, and — under a catalog that
    manages its own storage — the one the *catalog server* resolves rather than this
    process. The raw, bronze and metadata zones, the Spark warehouse and the Ivy cache are
    directories this process writes to directly, so they keep their existing
    project-relative semantics — putting those on object storage is a different change
    (it touches raw artifact writing, sidecars and resume-state rediscovery).
    """

    storage = config.get("storage", {})
    spark = config.get("spark", {})

    paths: dict[str, RuntimeLocation] = {
        "root_dir": resolve_project_path(project_root, storage["root_dir"]),
        "raw_dir": resolve_project_path(project_root, storage["raw_dir"]),
        "bronze_dir": resolve_project_path(project_root, storage["bronze_dir"]),
        "metadata_dir": resolve_project_path(project_root, storage["metadata_dir"]),
        "warehouse_dir": resolve_project_path(project_root, spark["warehouse_dir"]),
    }

    if "ivy_dir" in spark:
        paths["ivy_dir"] = resolve_project_path(project_root, spark["ivy_dir"])

    iceberg = spark.get("iceberg", {})
    if isinstance(iceberg, dict) and WAREHOUSE_DIR_KEY in iceberg:
        paths[ICEBERG_WAREHOUSE_PATH_KEY] = (
            catalog_managed_warehouse(iceberg)
            if warehouse_is_catalog_managed(iceberg)
            else resolve_runtime_location(project_root, iceberg[WAREHOUSE_DIR_KEY])
        )

    database_path = catalog_database_path(iceberg) if isinstance(iceberg, dict) else None
    if database_path is not None:
        paths[ICEBERG_CATALOG_DB_PATH_KEY] = resolve_project_path(
            project_root, database_path
        )

    return paths


def prepare_runtime(
    config: dict[str, Any], project_root: Path
) -> dict[str, RuntimeLocation]:
    paths = materialize_runtime_paths(config, project_root)
    for key, path in tuple(paths.items()):
        if not isinstance(path, Path):
            continue
        try:
            _ensure_writable_directory(_directory_to_materialize(key, path))
        except PermissionError:
            if key not in FALLBACK_RUNTIME_PATH_KEYS:
                raise
            fallback = _fallback_runtime_path(project_root, key)
            _ensure_writable_directory(fallback)
            paths[key] = fallback
            _set_config_path(config, key, fallback)
    return paths


def build_spark_options(
    config: dict[str, Any], resolved_paths: Mapping[str, RuntimeLocation]
) -> dict[str, str]:
    spark_config = config.get("spark", {})
    options: dict[str, str] = {
        "spark.sql.warehouse.dir": str(resolved_paths["warehouse_dir"]),
    }
    options.update({key: str(value) for key, value in spark_config.get("config", {}).items()})

    ivy_dir = resolved_paths.get("ivy_dir")
    if ivy_dir is not None:
        options.setdefault("spark.jars.ivy", str(ivy_dir))

    iceberg = spark_config.get("iceberg")
    if isinstance(iceberg, dict) and iceberg:
        apply_iceberg_catalog_options(options, iceberg, resolved_paths)

    return options


def build_spark_session(
    config: dict[str, Any], resolved_paths: Mapping[str, RuntimeLocation]
):
    from pyspark.sql import SparkSession

    spark_config = config.get("spark", {})
    builder = SparkSession.builder.appName(spark_config["app_name"]).master(
        spark_config["master"]
    )

    for key, value in build_spark_options(config, resolved_paths).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(config.get("runtime", {}).get("log_level", "WARN"))
    return spark


def _directory_to_materialize(key: str, path: Path) -> Path:
    """The directory `prepare_runtime` must create for a resolved path."""

    return path.parent if key in RUNTIME_FILE_PATH_KEYS else path


def _ensure_writable_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
    except PermissionError:
        raise
    except OSError as exc:
        raise PermissionError(exc.errno, exc.strerror, str(path)) from exc


def _fallback_runtime_path(project_root: Path, key: str) -> Path:
    scratch_root = _fallback_runtime_root()
    scratch_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_private_directory(scratch_root)
    project_segment = _safe_path_segment(project_root.resolve().name or "project")
    return scratch_root / project_segment / key


def _fallback_runtime_root() -> Path:
    """Choose an explicit, user-runtime, or process-private fallback root."""

    configured_root = os.getenv(RUNTIME_SCRATCH_DIR_ENV)
    if configured_root:
        return Path(configured_root)

    xdg_runtime_dir = os.getenv("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        return Path(xdg_runtime_dir) / "janus"

    # All warehouse fallbacks in this process must share the same generated root.
    global _PROCESS_FALLBACK_ROOT  # noqa: PLW0603 - intentional process singleton
    if _PROCESS_FALLBACK_ROOT is None:
        _PROCESS_FALLBACK_ROOT = Path(tempfile.mkdtemp(prefix="janus-runtime-"))
    return _PROCESS_FALLBACK_ROOT


def _assert_private_directory(path: Path) -> None:
    """Refuse a fallback root another user owns or can modify."""

    directory_stat = path.stat()
    if directory_stat.st_uid != os.getuid():
        raise PermissionError(
            errno.EPERM,
            "runtime fallback directory is not owned by the current user",
            str(path),
        )
    if directory_stat.st_mode & 0o022:
        raise PermissionError(
            errno.EPERM,
            "runtime fallback directory is writable by group or other",
            str(path),
        )


def _safe_path_segment(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._")
    return normalized or "project"


def _set_config_path(config: dict[str, Any], key: str, path: Path) -> None:
    location = _RUNTIME_PATH_CONFIG_LOCATIONS.get(key)
    if location is None:
        return

    current: Any = config
    for segment in location[:-1]:
        if not isinstance(current, dict):
            return
        current = current.setdefault(segment, {})
    if isinstance(current, dict):
        current[location[-1]] = str(path)
