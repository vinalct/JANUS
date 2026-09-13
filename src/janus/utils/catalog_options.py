"""The Iceberg catalog block of an environment profile, and the Spark options it emits."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOCATION_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
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
WAREHOUSE_DIR_KEY = "warehouse_dir"

CATALOG_TYPES_WITH_CATALOG_MANAGED_WAREHOUSE = frozenset({REST_CATALOG_TYPE})

JDBC_CREDENTIAL_OPTIONS = (("user", "jdbc.user"), ("password", "jdbc.password"))
JDBC_SCHEMA_VERSION_OPTION = ("jdbc.schema-version", "V1")
JDBC_URI_PREFIX = "jdbc:"
JDBC_AUTHORITY_PREFIX = "//"
ICEBERG_CATALOG_DB_PATH_KEY = "iceberg_catalog_db"
ICEBERG_WAREHOUSE_PATH_KEY = "iceberg_warehouse_dir"

CATALOG_AUTH_KEY = "auth"

CATALOG_AUTH_OPTIONS = (
    ("token", "token"),
    ("credential", "credential"),
    ("oauth2_server_uri", "oauth2-server-uri"),
    ("scope", "scope"),
)
SUPPORTED_CATALOG_AUTH_KEYS = frozenset(key for key, _ in CATALOG_AUTH_OPTIONS)

#: Catalog types with somewhere to put an auth block. The JDBC catalog authenticates
#: through `credentials` (a database login); Hadoop authenticates through the filesystem.
CATALOG_TYPES_WITH_AUTH = frozenset({REST_CATALOG_TYPE})

OBJECT_STORE_KEY = "object_store"
FILE_IO_IMPL_KEY = "io_impl"
OBJECT_STORE_PACKAGE_KEY = "io_package"
FILE_IO_IMPL_OPTION = "io-impl"
S3_FILE_IO_IMPL = "org.apache.iceberg.aws.s3.S3FileIO"
SUPPORTED_FILE_IO_IMPLS = {"S3FileIO": S3_FILE_IO_IMPL}

OBJECT_STORE_ENDPOINT_KEY = "endpoint"
OBJECT_STORE_PATH_STYLE_KEY = "path_style_access"
OBJECT_STORE_REGION_KEY = "region"

OBJECT_STORE_OPTIONS = (
    (OBJECT_STORE_ENDPOINT_KEY, "s3.endpoint"),
    (OBJECT_STORE_PATH_STYLE_KEY, "s3.path-style-access"),
    (OBJECT_STORE_REGION_KEY, "client.region"),
)

OBJECT_STORE_FLAG_KEYS = frozenset({OBJECT_STORE_PATH_STYLE_KEY})
TRUE_TEXT = frozenset({"true", "t", "yes", "y", "on", "1"})
FALSE_TEXT = frozenset({"false", "f", "no", "n", "off", "0"})

SUPPORTED_OBJECT_STORE_KEYS = frozenset(
    {FILE_IO_IMPL_KEY, OBJECT_STORE_PACKAGE_KEY, *(key for key, _ in OBJECT_STORE_OPTIONS)}
)

RuntimeLocation = Path | str


def is_location_uri(value: Any) -> bool:
    """Whether a configured value names a store rather than a directory on this filesystem.

    Any scheme, by design: `s3://`, `s3a://` and whatever comes next are the same case —
    a location this process configures an engine to reach, and never one it creates,
    probes or resolves against the project root.
    """

    return isinstance(value, str) and LOCATION_URI_PATTERN.match(value) is not None


def non_empty_text(value: Any) -> str | None:
    """The single empty-guard both catalog emitters apply to an expanded profile value."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def merge_csv_values(existing: str | None, value: str) -> str:
    items = [item.strip() for item in (existing or "").split(",") if item.strip()]
    if value not in items:
        items.append(value)
    return ",".join(items)


def apply_iceberg_catalog_options(
    options: dict[str, str],
    iceberg: dict[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
) -> None:
    """Emit the Iceberg catalog block for the catalog type the profile declares.

    Explicit `spark.config` entries are already in `options` by the time this runs, so
    every catalog key is written with `setdefault` — a profile override still wins.
    """

    catalog_name = iceberg["catalog_name"]
    catalog_type = resolve_catalog_type(iceberg)
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

    for suffix, value in _catalog_type_options(catalog_type, iceberg, resolved_paths):
        options.setdefault(f"{catalog_prefix}.{suffix}", value)

    for suffix, value in _object_store_options(iceberg):
        options.setdefault(f"{catalog_prefix}.{suffix}", value)

    iceberg_warehouse_dir = resolved_paths.get(ICEBERG_WAREHOUSE_PATH_KEY)
    if iceberg_warehouse_dir is None:
        raise KeyError("Resolved Iceberg warehouse path is missing")

    options.setdefault(f"{catalog_prefix}.warehouse", str(iceberg_warehouse_dir))

    default_namespace = iceberg.get("default_namespace")
    if default_namespace:
        options.setdefault(
            f"{catalog_prefix}.default-namespace", str(default_namespace)
        )


def resolve_catalog_type(iceberg: dict[str, Any]) -> str:
    """The declared catalog type, or a named error. There is no default on purpose."""

    catalog_type = declared_catalog_type(iceberg)
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


def declared_catalog_type(iceberg: dict[str, Any]) -> str | None:
    """What the profile *says*, with no judgement passed on whether it is supported."""

    return non_empty_text(iceberg.get(CATALOG_TYPE_KEY))


def warehouse_is_catalog_managed(iceberg: dict[str, Any]) -> bool:
    """Whether `warehouse_dir` names a warehouse the catalog server resolves for itself."""

    return declared_catalog_type(iceberg) in CATALOG_TYPES_WITH_CATALOG_MANAGED_WAREHOUSE


def catalog_managed_warehouse(iceberg: dict[str, Any]) -> str:
    """The warehouse identifier a catalog-managed warehouse names, verbatim."""

    warehouse = non_empty_text(iceberg.get(WAREHOUSE_DIR_KEY))
    if warehouse is None:
        raise ValueError(
            f"Environment config must set a non-empty spark.iceberg.{WAREHOUSE_DIR_KEY} "
            f"for spark.iceberg.{CATALOG_TYPE_KEY} {REST_CATALOG_TYPE!r}: the catalog "
            "resolves the warehouse by the identifier the client sends, so there is no "
            "location to fall back on"
        )
    return warehouse


def _catalog_type_options(
    catalog_type: str,
    iceberg: dict[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
) -> list[tuple[str, str]]:
    emitted = [("type", catalog_type)]
    if catalog_type in CATALOG_TYPES_REQUIRING_URI:
        emitted.append(("uri", resolve_catalog_uri(iceberg, resolved_paths)))
    if catalog_type == JDBC_CATALOG_TYPE:
        emitted.append(JDBC_SCHEMA_VERSION_OPTION)
        emitted.extend(_jdbc_credential_options(iceberg))
    emitted.extend(_catalog_auth_options(catalog_type, iceberg))
    return emitted


def catalog_database_path(iceberg: dict[str, Any]) -> str | None:
    """The filesystem path a file-backed JDBC URI names, or ``None`` when it names a server."""

    uri = non_empty_text(iceberg.get("uri"))
    split = _split_file_backed_jdbc_uri(uri) if uri is not None else None
    return None if split is None else split[1]


def resolve_catalog_uri(
    iceberg: dict[str, Any], resolved_paths: Mapping[str, RuntimeLocation]
) -> str:
    """The catalog URI as an engine must receive it, with a file-backed path made absolute."""

    catalog_type = resolve_catalog_type(iceberg)
    uri = required_catalog_value(iceberg, "uri", catalog_type)

    split = _split_file_backed_jdbc_uri(uri)
    if split is None:
        return uri

    prefix, _, suffix = split
    database = resolved_paths.get(ICEBERG_CATALOG_DB_PATH_KEY)
    if database is None:
        raise KeyError("Resolved Iceberg catalog database path is missing")
    return f"{prefix}{database}{suffix}"


def _split_file_backed_jdbc_uri(uri: str) -> tuple[str, str, str] | None:

    if not uri.startswith(JDBC_URI_PREFIX):
        return None

    backend, separator, target = uri[len(JDBC_URI_PREFIX) :].partition(":")
    if not separator or target.startswith(JDBC_AUTHORITY_PREFIX):
        return None

    path, question, options = target.partition("?")
    if not path:
        return None
    return f"{JDBC_URI_PREFIX}{backend}:", path, f"{question}{options}"


def _catalog_jar_packages(catalog_type: str, iceberg: dict[str, Any]) -> list[str]:
    """Maven coordinates this catalog needs beyond the Iceberg runtime itself."""

    candidates = []
    if catalog_type == JDBC_CATALOG_TYPE:
        candidates.append(non_empty_text(iceberg.get("driver_package")))

    object_store = object_store_block(iceberg)
    if object_store is not None:
        candidates.append(non_empty_text(object_store.get(OBJECT_STORE_PACKAGE_KEY)))

    return [package for package in candidates if package is not None]


def object_store_block(iceberg: dict[str, Any]) -> dict[str, Any] | None:
    """The profile's validated `object_store` block, or `None` when it declares none."""

    object_store = iceberg.get(OBJECT_STORE_KEY)
    if not isinstance(object_store, dict) or not object_store:
        return None

    _reject_unsupported_object_store_keys(object_store)
    resolve_file_io_impl(object_store)
    for key in OBJECT_STORE_FLAG_KEYS:
        object_store_flag(object_store, key)
    return object_store


def object_store_flag(object_store: dict[str, Any], key: str) -> bool | None:
    """A boolean object-store setting, parsed strictly, or `None` when it is unset."""

    value = non_empty_text(object_store.get(key))
    if value is None:
        return None
    if value.lower() in TRUE_TEXT:
        return True
    if value.lower() in FALSE_TEXT:
        return False

    accepted = ", ".join(sorted(TRUE_TEXT | FALSE_TEXT))
    raise ValueError(
        f"Environment config has a non-boolean spark.iceberg.{OBJECT_STORE_KEY}.{key}: "
        f"{value!r}; accepted values: {accepted}"
    )


def object_store_value(object_store: dict[str, Any], key: str) -> str | None:
    """One setting as both engines receive it: booleans canonicalised, the rest verbatim."""

    if key in OBJECT_STORE_FLAG_KEYS:
        flag = object_store_flag(object_store, key)
        return None if flag is None else str(flag).lower()
    return non_empty_text(object_store.get(key))


def _object_store_options(iceberg: dict[str, Any]) -> list[tuple[str, str]]:
    """The FileIO block for a warehouse on object storage, keyed by Iceberg's own names."""

    object_store = object_store_block(iceberg)
    if object_store is None:
        return []

    emitted = [(FILE_IO_IMPL_OPTION, resolve_file_io_impl(object_store))]
    for key, suffix in OBJECT_STORE_OPTIONS:
        value = object_store_value(object_store, key)
        if value is not None:
            emitted.append((suffix, value))
    return emitted


def resolve_file_io_impl(object_store: dict[str, Any]) -> str:
    """The FileIO class the declared `io_impl` names, or a named error."""

    supported = ", ".join(sorted(SUPPORTED_FILE_IO_IMPLS))
    name = non_empty_text(object_store.get(FILE_IO_IMPL_KEY))
    if name is None:
        raise ValueError(
            f"Environment config must set spark.iceberg.{OBJECT_STORE_KEY}."
            f"{FILE_IO_IMPL_KEY} when the {OBJECT_STORE_KEY} block is present; supported "
            f"values: {supported}"
        )

    impl = SUPPORTED_FILE_IO_IMPLS.get(name)
    if impl is None:
        raise ValueError(
            f"Environment config has an unsupported spark.iceberg.{OBJECT_STORE_KEY}."
            f"{FILE_IO_IMPL_KEY}: {name!r}; supported values: {supported}"
        )
    return impl


def _reject_unsupported_object_store_keys(object_store: dict[str, Any]) -> None:
    """Fail closed on a key the block does not define, naming the key but never its value."""

    unsupported = sorted(set(object_store) - SUPPORTED_OBJECT_STORE_KEYS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_OBJECT_STORE_KEYS))
        raise ValueError(
            f"Environment config has unsupported spark.iceberg.{OBJECT_STORE_KEY} key(s): "
            f"{', '.join(unsupported)}; supported keys: {supported}. Object-store "
            "credentials are read from the standard AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY environment variables, never from a profile"
        )


def catalog_auth_block(iceberg: dict[str, Any]) -> dict[str, Any] | None:
    """The profile's validated `auth` block, or `None` when it declares none."""

    auth = iceberg.get(CATALOG_AUTH_KEY)
    if not isinstance(auth, dict) or not auth:
        return None

    _reject_unsupported_auth_keys(auth)
    return auth


def _catalog_auth_options(
    catalog_type: str, iceberg: dict[str, Any]
) -> list[tuple[str, str]]:
    """The REST spec's auth properties, from whichever of them the profile actually sets."""

    auth = catalog_auth_block(iceberg)
    if auth is None or catalog_type not in CATALOG_TYPES_WITH_AUTH:
        return []

    emitted = []
    for key, suffix in CATALOG_AUTH_OPTIONS:
        value = non_empty_text(auth.get(key))
        if value is not None:
            emitted.append((suffix, value))
    return emitted


def _reject_unsupported_auth_keys(auth: dict[str, Any]) -> None:
    """Fail closed on a key the block does not define, naming the key but never its value."""

    unsupported = sorted(set(auth) - SUPPORTED_CATALOG_AUTH_KEYS)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_CATALOG_AUTH_KEYS))
        raise ValueError(
            f"Environment config has unsupported spark.iceberg.{CATALOG_AUTH_KEY} key(s): "
            f"{', '.join(unsupported)}; supported keys: {supported}"
        )


def _jdbc_credential_options(iceberg: dict[str, Any]) -> list[tuple[str, str]]:
    credentials = iceberg.get("credentials")
    if not isinstance(credentials, dict):
        return []
    emitted = []
    for key, suffix in JDBC_CREDENTIAL_OPTIONS:
        value = non_empty_text(credentials.get(key))
        if value is not None:
            emitted.append((suffix, value))
    return emitted


def required_catalog_value(
    iceberg: dict[str, Any], key: str, catalog_type: str
) -> str:
    """Read a per-type required key, treating an unset `${VAR:-}` expansion as missing."""

    value = non_empty_text(iceberg.get(key))
    if value is None:
        raise ValueError(
            f"Environment config must set a non-empty spark.iceberg.{key} for "
            f"spark.iceberg.{CATALOG_TYPE_KEY} {catalog_type!r} in the environment profile"
        )
    return value
