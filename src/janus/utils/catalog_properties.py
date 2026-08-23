"""One derivation of the Iceberg catalog properties a *second* engine needs.

:func:`janus.utils.environment.build_spark_options` configures Spark's ``SparkCatalog`` from the
``spark.iceberg`` block of an environment profile. ``pyiceberg`` reaches the same catalog through
a different vocabulary, and the two dialects diverge in several places that are easy to get
subtly — and silently — wrong by hand. Two hand-written configs that happen to agree today is the
drift order-12's shared-catalog gate exists to prevent, so both emitters read the **same** mapping
and every known divergence is encoded here, unit-tested, rather than discovered against a live
catalog at 3 a.m.

Translation table
=================

=================  ======================================  =======================================
Concept            Spark (``build_spark_options``)         ``pyiceberg`` (this module)
=================  ======================================  =======================================
catalog type       ``type = jdbc``                         ``type = sql`` (the ``SqlCatalog``)
SQLite URI         ``jdbc:sqlite:<path>`` (JDBC form)      ``sqlite:///<path>`` (SQLAlchemy form)
file-backed path   resolved absolute, once, in              the same resolved path — never the
                   ``materialize_runtime_paths``            profile's project-relative string
driver options     ``?journal_mode=…`` — PRAGMAs the       carried through; SQLAlchemy ignores
                   xerial driver applies per connection     unknown query keys (see below)
Postgres URI       ``jdbc:postgresql://host/db``           ``postgresql+psycopg2://host/db``
credentials        ``jdbc.user`` / ``jdbc.password``       userinfo inside the URI authority
warehouse          local path or ``s3://…``                same location, local paths ``file://``
default namespace  ``default-namespace`` conf              no property — identifiers qualify it
REST               ``type = rest``, ``uri``                ``type = rest``, ``uri`` (the easy one)
Hadoop             ``type = hadoop``                       unrepresentable — raises
=================  ======================================  =======================================

**Credentials: the canonical mapping is the URI authority, and it is not a preference.**
``SqlCatalog.__init__`` reads exactly one connection property — ``uri`` — and hands it straight to
SQLAlchemy's ``create_engine``. There is no user property and no password property to map onto, so
userinfo in the URI is the only representation that exists. A URI with no authority to carry it
(SQLite names a file, not a server) fails closed rather than dropping the credential silently.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from janus.utils.environment import (
    CATALOG_TYPE_KEY,
    HADOOP_CATALOG_TYPE,
    JDBC_AUTHORITY_PREFIX,
    JDBC_CATALOG_TYPE,
    JDBC_URI_PREFIX,
    REST_CATALOG_TYPE,
    non_empty_text,
    resolve_catalog_type,
    resolve_catalog_uri,
)

ICEBERG_WAREHOUSE_PATH_KEY = "iceberg_warehouse_dir"

# `pyiceberg`'s own vocabulary (pyiceberg.catalog.TYPE / URI / WAREHOUSE_LOCATION).
PYICEBERG_TYPE_KEY = "type"
PYICEBERG_URI_KEY = "uri"
PYICEBERG_WAREHOUSE_KEY = "warehouse"

# `pyiceberg.catalog.CatalogType` members this project can reach.
PYICEBERG_SQL_CATALOG_TYPE = "sql"
PYICEBERG_REST_CATALOG_TYPE = "rest"

# JDBC subprotocol → the driver-qualified SQLAlchemy scheme `SqlCatalog` can open.
SQLALCHEMY_SCHEMES = {
    "sqlite": "sqlite",
    "postgresql": "postgresql+psycopg2",
}

_URI_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


class CatalogPropertyError(ValueError):
    """A catalog this profile declares cannot be expressed in `pyiceberg`'s vocabulary."""


class HadoopCatalogUnrepresentableError(CatalogPropertyError):
    """`catalog_type: hadoop` has no `pyiceberg` counterpart"""


def derive_pyiceberg_catalog_properties(
    config: dict[str, Any], resolved_paths: dict[str, Path]
) -> dict[str, str]:
    """Translate one environment profile into `pyiceberg.catalog.load_catalog` properties.

    Takes the *same* mapping and resolved paths ``build_spark_options`` takes, and reuses its
    readers, so a profile the Spark emitter rejects is rejected here with the same message. A
    value that cannot be derived from what Spark already gets means the profile schema is wrong
    and must be amended there — never patched with a second config surface here.
    """

    iceberg = _iceberg_block(config)
    catalog_type = resolve_catalog_type(iceberg)

    if catalog_type == HADOOP_CATALOG_TYPE:
        raise HadoopCatalogUnrepresentableError(
            "pyiceberg implements no Hadoop catalog: its commits are filesystem renames, which "
            "cannot be coordinated between two engines (nor safely on an object store). Set "
            f"spark.iceberg.{CATALOG_TYPE_KEY} to {JDBC_CATALOG_TYPE!r} or {REST_CATALOG_TYPE!r} "
            "in the environment profile — that swap is what order-13 exists to make possible"
        )

    builder = _PROPERTY_BUILDERS.get(catalog_type)
    if builder is None:
        raise CatalogPropertyError(
            f"No pyiceberg translation is defined for spark.iceberg.{CATALOG_TYPE_KEY} "
            f"{catalog_type!r}; a catalog type the Spark emitter supports needs its counterpart "
            "in utils/catalog_properties.py, or the two engines have drifted"
        )

    properties = builder(iceberg, resolved_paths)
    properties[PYICEBERG_WAREHOUSE_KEY] = _warehouse_location(resolved_paths)
    return properties


def derive_pyiceberg_catalog_name(config: dict[str, Any]) -> str:
    """The name to pass as `load_catalog(name, **properties)`.

    It is an argument rather than a property, so it would otherwise be the one value a caller
    hand-wrote next to a derived dict — which is exactly the drift this module exists to close.
    """

    name = non_empty_text(_iceberg_block(config).get("catalog_name"))
    if name is None:
        raise CatalogPropertyError(
            "Environment config must set a non-empty spark.iceberg.catalog_name in the "
            "environment profile"
        )
    return name


def derive_pyiceberg_default_namespace(config: dict[str, Any]) -> str | None:
    """The namespace a `pyiceberg` identifier must carry, or `None` when the profile sets none.

    This is deliberately **not** a catalog property: `pyiceberg` has no `default-namespace`
    setting, so what Spark resolves implicitly through
    ``spark.sql.catalog.<name>.default-namespace`` a `pyiceberg` caller must spell out in every
    identifier. Handing it back from the same profile is what keeps both engines pointed at the
    same tables.
    """

    return non_empty_text(_iceberg_block(config).get("default_namespace"))


def _jdbc_properties(
    iceberg: dict[str, Any], resolved_paths: dict[str, Path]
) -> dict[str, str]:
    """Spark's JDBC catalog is `pyiceberg`'s `SqlCatalog`, over a SQLAlchemy URI."""

    return {
        PYICEBERG_TYPE_KEY: PYICEBERG_SQL_CATALOG_TYPE,
        PYICEBERG_URI_KEY: _sqlalchemy_uri(
            resolve_catalog_uri(iceberg, resolved_paths), *_uri_credentials(iceberg)
        ),
    }


def _rest_properties(
    iceberg: dict[str, Any], resolved_paths: dict[str, Path]
) -> dict[str, str]:
    """The easy case: both engines take `type` and `uri` verbatim."""

    return {
        PYICEBERG_TYPE_KEY: PYICEBERG_REST_CATALOG_TYPE,
        PYICEBERG_URI_KEY: resolve_catalog_uri(iceberg, resolved_paths),
    }


_PROPERTY_BUILDERS: dict[
    str, Callable[[dict[str, Any], dict[str, Path]], dict[str, str]]
] = {
    JDBC_CATALOG_TYPE: _jdbc_properties,
    REST_CATALOG_TYPE: _rest_properties,
}


def _iceberg_block(config: dict[str, Any]) -> dict[str, Any]:
    spark = config.get("spark")
    iceberg = spark.get("iceberg") if isinstance(spark, dict) else None
    if not isinstance(iceberg, dict) or not iceberg:
        raise CatalogPropertyError(
            "Environment config declares no spark.iceberg block, so there is no catalog for a "
            "second engine to share"
        )
    return iceberg


def _uri_credentials(iceberg: dict[str, Any]) -> tuple[str | None, str | None]:
    """The same read `_jdbc_credential_options` performs Spark-side, including its empty-guard.

    An unset `${JANUS_ICEBERG_CATALOG_USER}` expands to `""`, and `""` is not a user.
    """

    credentials = iceberg.get("credentials")
    if not isinstance(credentials, dict):
        return None, None
    return (
        non_empty_text(credentials.get("user")),
        non_empty_text(credentials.get("password")),
    )


def _sqlalchemy_uri(jdbc_uri: str, user: str | None, password: str | None) -> str:
    """Rewrite a JDBC URL as the SQLAlchemy URL `SqlCatalog` opens.

    Only the two backends the translation table documents are accepted. Emitting an untranslated
    scheme would hand SQLAlchemy a URL whose driver resolution nobody has verified — a failure
    that would surface as a broken catalog rather than as a broken profile.
    """

    if not jdbc_uri.startswith(JDBC_URI_PREFIX):
        raise CatalogPropertyError(
            f"spark.iceberg.uri must be a JDBC URL of the form 'jdbc:<backend>:…' for "
            f"spark.iceberg.{CATALOG_TYPE_KEY} {JDBC_CATALOG_TYPE!r}; got a "
            f"{_scheme_hint(jdbc_uri)!r} URI"
        )

    backend, separator, target = jdbc_uri[len(JDBC_URI_PREFIX) :].partition(":")
    scheme = SQLALCHEMY_SCHEMES.get(backend.lower()) if separator else None
    if scheme is None:
        supported = ", ".join(sorted(SQLALCHEMY_SCHEMES))
        raise CatalogPropertyError(
            f"No pyiceberg translation is defined for the JDBC URI {_scheme_hint(jdbc_uri)!r}; "
            f"translatable backends: {supported}"
        )

    if target.startswith(JDBC_AUTHORITY_PREFIX):
        authority = target[len(JDBC_AUTHORITY_PREFIX) :]
        return f"{scheme}://{_userinfo(user, password)}{authority}"

    if user is not None or password is not None:
        raise CatalogPropertyError(
            f"spark.iceberg.credentials cannot be handed to pyiceberg for the "
            f"{_scheme_hint(jdbc_uri)!r} URI: it names a file rather than a server, so the "
            "SQLAlchemy URI has no authority to carry them. Drop the credentials or point the "
            "catalog at a server"
        )
    # An empty authority, then the path: `sqlite:///relative/db` — and an absolute target keeps
    # its own leading slash, which is how `sqlite:////absolute/db` gets its fourth.
    return f"{scheme}:{JDBC_AUTHORITY_PREFIX}/{target}"


def _userinfo(user: str | None, password: str | None) -> str:
    """`user:password@`, percent-encoded, for whichever halves the profile actually set."""

    if user is None and password is None:
        return ""
    encoded = quote(user or "", safe="")
    if password is not None:
        encoded = f"{encoded}:{quote(password, safe='')}"
    return f"{encoded}@"


def _warehouse_location(resolved_paths: dict[str, Path]) -> str:
    """The warehouse as a location URI.

    `pyiceberg` picks its FileIO from the location's scheme, so a bare local path — which is what
    the profile resolves to today — must be spelled `file://`. A value that already carries a
    scheme is a location in its own right and passes through untouched, which is how an
    object-store warehouse stays a config-only change.
    """

    warehouse = resolved_paths.get(ICEBERG_WAREHOUSE_PATH_KEY)
    if warehouse is None:
        raise CatalogPropertyError("Resolved Iceberg warehouse path is missing")

    location = str(warehouse)
    if _URI_SCHEME_PATTERN.match(location):
        return location

    path = Path(location)
    if not path.is_absolute():
        raise CatalogPropertyError(
            "The resolved Iceberg warehouse path must be absolute before it can become a "
            f"file:// URI; got {location!r}"
        )
    return path.as_uri()


def _scheme_hint(uri: str) -> str:
    """The scheme of a URI and nothing after it — an authority can carry a password."""

    scheme, separator, _ = uri.partition("://")
    if separator:
        return scheme
    return ":".join(uri.split(":")[:2])
