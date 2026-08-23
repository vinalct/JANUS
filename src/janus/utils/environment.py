from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)(?::-(?P<default>[^}]*))?\}")
ICEBERG_SESSION_EXTENSIONS = (
    "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
)
ICEBERG_CATALOG_IMPL = "org.apache.iceberg.spark.SparkCatalog"
JDBC_CATALOG_TYPE = "jdbc"
REST_CATALOG_TYPE = "rest"
HADOOP_CATALOG_TYPE = "hadoop"
SUPPORTED_CATALOG_TYPES = frozenset(
    {JDBC_CATALOG_TYPE, REST_CATALOG_TYPE, HADOOP_CATALOG_TYPE}
)
CATALOG_TYPES_REQUIRING_URI = frozenset({JDBC_CATALOG_TYPE, REST_CATALOG_TYPE})
CATALOG_TYPE_KEY = "catalog_type"
JDBC_CREDENTIAL_OPTIONS = (("user", "jdbc.user"), ("password", "jdbc.password"))
RUNTIME_SCRATCH_DIR_ENV = "JANUS_RUNTIME_SCRATCH_DIR"
DEFAULT_RUNTIME_SCRATCH_DIR = "/tmp/janus/runtime"
FALLBACK_RUNTIME_PATH_KEYS = frozenset(
    {"warehouse_dir", "ivy_dir", "iceberg_warehouse_dir"}
)
_RUNTIME_PATH_CONFIG_LOCATIONS = {
    "root_dir": ("storage", "root_dir"),
    "raw_dir": ("storage", "raw_dir"),
    "bronze_dir": ("storage", "bronze_dir"),
    "metadata_dir": ("storage", "metadata_dir"),
    "warehouse_dir": ("spark", "warehouse_dir"),
    "ivy_dir": ("spark", "ivy_dir"),
    "iceberg_warehouse_dir": ("spark", "iceberg", "warehouse_dir"),
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


def materialize_runtime_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    storage = config.get("storage", {})
    spark = config.get("spark", {})

    paths = {
        "root_dir": resolve_project_path(project_root, storage["root_dir"]),
        "raw_dir": resolve_project_path(project_root, storage["raw_dir"]),
        "bronze_dir": resolve_project_path(project_root, storage["bronze_dir"]),
        "metadata_dir": resolve_project_path(project_root, storage["metadata_dir"]),
        "warehouse_dir": resolve_project_path(project_root, spark["warehouse_dir"]),
    }

    if "ivy_dir" in spark:
        paths["ivy_dir"] = resolve_project_path(project_root, spark["ivy_dir"])

    iceberg = spark.get("iceberg", {})
    if isinstance(iceberg, dict) and "warehouse_dir" in iceberg:
        paths["iceberg_warehouse_dir"] = resolve_project_path(
            project_root, iceberg["warehouse_dir"]
        )

    return paths


def prepare_runtime(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    paths = materialize_runtime_paths(config, project_root)
    for key, path in tuple(paths.items()):
        try:
            _ensure_writable_directory(path)
        except PermissionError:
            if key not in FALLBACK_RUNTIME_PATH_KEYS:
                raise
            fallback = _fallback_runtime_path(project_root, key)
            _ensure_writable_directory(fallback)
            paths[key] = fallback
            _set_config_path(config, key, fallback)
    return paths


def merge_csv_values(existing: str | None, value: str) -> str:
    items = [item.strip() for item in (existing or "").split(",") if item.strip()]
    if value not in items:
        items.append(value)
    return ",".join(items)


def build_spark_options(
    config: dict[str, Any], resolved_paths: dict[str, Path]
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
        _apply_iceberg_catalog_options(options, iceberg, resolved_paths)

    return options


def build_spark_session(config: dict[str, Any], resolved_paths: dict[str, Path]):
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


def _apply_iceberg_catalog_options(
    options: dict[str, str], iceberg: dict[str, Any], resolved_paths: dict[str, Path]
) -> None:
    """Emit the Iceberg catalog block for the catalog type the profile declares.

    Explicit `spark.config` entries are already in `options` by the time this runs, so
    every catalog key is written with `setdefault` — a profile override still wins.
    """

    catalog_name = iceberg["catalog_name"]
    catalog_type = _resolve_catalog_type(iceberg)
    catalog_prefix = f"spark.sql.catalog.{catalog_name}"

    packages = merge_csv_values(
        options.get("spark.jars.packages"), iceberg["runtime_package"]
    )
    for package in _catalog_jar_packages(catalog_type, iceberg):
        packages = merge_csv_values(packages, package)
    options["spark.jars.packages"] = packages

    options["spark.sql.extensions"] = merge_csv_values(
        options.get("spark.sql.extensions"), ICEBERG_SESSION_EXTENSIONS
    )
    options.setdefault("spark.sql.defaultCatalog", catalog_name)
    options.setdefault(catalog_prefix, ICEBERG_CATALOG_IMPL)

    for suffix, value in _catalog_type_options(catalog_type, iceberg):
        options.setdefault(f"{catalog_prefix}.{suffix}", value)

    iceberg_warehouse_dir = resolved_paths.get("iceberg_warehouse_dir")
    if iceberg_warehouse_dir is None:
        raise KeyError("Resolved Iceberg warehouse path is missing")

    options.setdefault(f"{catalog_prefix}.warehouse", str(iceberg_warehouse_dir))

    default_namespace = iceberg.get("default_namespace")
    if default_namespace:
        options.setdefault(
            f"{catalog_prefix}.default-namespace", str(default_namespace)
        )


def _resolve_catalog_type(iceberg: dict[str, Any]) -> str:
    """The declared catalog type, or a named error. There is no default on purpose.

    A code-side default is how an unsafe catalog sneaks back into a profile that
    forgot the key; the default belongs to the profile.
    """

    catalog_type = _non_empty_text(iceberg.get(CATALOG_TYPE_KEY))
    supported = ", ".join(sorted(SUPPORTED_CATALOG_TYPES))
    if catalog_type is None:
        raise ValueError(
            f"Environment config must set spark.iceberg.{CATALOG_TYPE_KEY} in the "
            f"environment profile; supported values: {supported}"
        )
    if catalog_type not in SUPPORTED_CATALOG_TYPES:
        raise ValueError(
            f"Environment config has an unsupported spark.iceberg.{CATALOG_TYPE_KEY}: "
            f"{catalog_type!r}; supported values: {supported}"
        )
    return catalog_type


def _catalog_type_options(
    catalog_type: str, iceberg: dict[str, Any]
) -> list[tuple[str, str]]:
    emitted = [("type", catalog_type)]
    if catalog_type in CATALOG_TYPES_REQUIRING_URI:
        emitted.append(("uri", _required_catalog_value(iceberg, "uri", catalog_type)))
    if catalog_type == JDBC_CATALOG_TYPE:
        emitted.extend(_jdbc_credential_options(iceberg))
    return emitted


def _catalog_jar_packages(catalog_type: str, iceberg: dict[str, Any]) -> list[str]:
    if catalog_type != JDBC_CATALOG_TYPE:
        return []
    driver_package = _non_empty_text(iceberg.get("driver_package"))
    return [driver_package] if driver_package is not None else []


def _jdbc_credential_options(iceberg: dict[str, Any]) -> list[tuple[str, str]]:
    credentials = iceberg.get("credentials")
    if not isinstance(credentials, dict):
        return []
    emitted = []
    for key, suffix in JDBC_CREDENTIAL_OPTIONS:
        value = _non_empty_text(credentials.get(key))
        if value is not None:
            emitted.append((suffix, value))
    return emitted


def _required_catalog_value(
    iceberg: dict[str, Any], key: str, catalog_type: str
) -> str:
    """Read a per-type required key, treating an unset `${VAR:-}` expansion as missing.

    Env expansion yields `""` for an unset variable, and `""` must never reach Spark
    as a catalog URI.
    """

    value = _non_empty_text(iceberg.get(key))
    if value is None:
        raise ValueError(
            f"Environment config must set a non-empty spark.iceberg.{key} for "
            f"spark.iceberg.{CATALOG_TYPE_KEY} {catalog_type!r} in the environment profile"
        )
    return value


def _non_empty_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
    scratch_root = Path(os.getenv(RUNTIME_SCRATCH_DIR_ENV, DEFAULT_RUNTIME_SCRATCH_DIR))
    project_segment = _safe_path_segment(project_root.resolve().name or "project")
    return scratch_root / project_segment / key


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
