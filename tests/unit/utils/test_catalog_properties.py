"""The one derivation of `pyiceberg` catalog properties, and its parity with the Spark one."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pytest

import janus
import janus.utils.catalog_properties as catalog_properties
from janus.utils.catalog_properties import (
    _PROPERTY_BUILDERS,
    PYICEBERG_REST_CATALOG_TYPE,
    PYICEBERG_SQL_CATALOG_TYPE,
    SQLALCHEMY_SCHEMES,
    CatalogPropertyError,
    HadoopCatalogUnrepresentableError,
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
    derive_pyiceberg_default_namespace,
)
from janus.utils.environment import (
    HADOOP_CATALOG_TYPE,
    SUPPORTED_CATALOG_TYPES,
    build_spark_options,
    materialize_runtime_paths,
)

ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1"
CATALOG_PREFIX = "spark.sql.catalog.janus"
SQLITE_DATABASE = "data/metadata/janus_catalog.db"
POSTGRES_JDBC_URI = "jdbc:postgresql://catalog-db:5432/janus"
REST_URI = "http://catalog:8181"
SECRET_USER = "janus catalog user"
SECRET_PASSWORD = "p@ss:w/rd?"

# The profile fixtures both emitters are driven from. One dict per catalog type, so the parity
# test and the per-type tests can never diverge on what "a jdbc profile" means.
CATALOG_PROFILES: dict[str, dict[str, Any]] = {
    "jdbc_sqlite": {"catalog_type": "jdbc", "uri": f"jdbc:sqlite:{SQLITE_DATABASE}"},
    "jdbc_postgres": {
        "catalog_type": "jdbc",
        "uri": POSTGRES_JDBC_URI,
        "credentials": {"user": SECRET_USER, "password": SECRET_PASSWORD},
    },
    "rest": {"catalog_type": "rest", "uri": REST_URI},
}


def _decoded_credentials(parsed: Any) -> tuple[str | None, str | None]:
    """Userinfo as SQLAlchemy reads it: `_parse_url` percent-decodes both halves."""

    return (
        unquote(parsed.username) if parsed.username is not None else None,
        unquote(parsed.password) if parsed.password is not None else None,
    )


def _environment_config(**iceberg: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "catalog_name": "janus",
        "catalog_type": "jdbc",
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


def _derive(project_root: Path, **iceberg: Any) -> dict[str, str]:
    config = _environment_config(**iceberg)
    return derive_pyiceberg_catalog_properties(
        config, materialize_runtime_paths(config, project_root)
    )


# ── jdbc → the SqlCatalog ────────────────────────────────────────────────────


def test_a_sqlite_profile_becomes_a_sql_catalog_over_a_sqlalchemy_uri(tmp_path):
    """`type` is renamed, the JDBC URL is rewritten, and the warehouse gains a scheme.

    The database path arrives already resolved: `materialize_runtime_paths` makes it absolute
    once, for both engines, so neither depends on the working directory it was started from.
    """

    properties = _derive(tmp_path, **CATALOG_PROFILES["jdbc_sqlite"])

    assert properties == {
        "type": PYICEBERG_SQL_CATALOG_TYPE,
        "uri": f"sqlite:///{tmp_path / SQLITE_DATABASE}",
        "warehouse": (tmp_path / "data/bronze/iceberg").as_uri(),
    }
    assert properties["uri"].startswith("sqlite:////")


def test_both_engines_receive_the_same_resolved_database_path(tmp_path):
    """A project-relative catalog path must not resolve differently per engine or per cwd."""

    config = _environment_config(**CATALOG_PROFILES["jdbc_sqlite"])
    paths = materialize_runtime_paths(config, tmp_path)

    spark_uri = build_spark_options(config, paths)[f"{CATALOG_PREFIX}.uri"]
    derived_uri = derive_pyiceberg_catalog_properties(config, paths)["uri"]

    database = str(tmp_path / SQLITE_DATABASE)
    assert spark_uri == f"jdbc:sqlite:{database}"
    assert derived_uri == f"sqlite:///{database}"


def test_an_absolute_sqlite_path_keeps_its_own_leading_slash(tmp_path):
    """SQLAlchemy spells an absolute SQLite file with four slashes; three is a relative path."""

    database = tmp_path / "metadata" / "janus_catalog.db"

    properties = _derive(tmp_path, catalog_type="jdbc", uri=f"jdbc:sqlite:{database}")

    assert properties["uri"] == f"sqlite:///{database}"
    assert properties["uri"].startswith("sqlite:////")


def test_a_postgres_profile_is_driver_qualified_and_carries_its_credentials(tmp_path):
    """The canonical credential mapping: userinfo in the URI, because `SqlCatalog` has no other."""

    properties = _derive(tmp_path, **CATALOG_PROFILES["jdbc_postgres"])

    assert properties["type"] == PYICEBERG_SQL_CATALOG_TYPE
    parsed = urlparse(properties["uri"])
    assert parsed.scheme == "postgresql+psycopg2"
    assert (parsed.hostname, parsed.port, parsed.path) == ("catalog-db", 5432, "/janus")
    assert _decoded_credentials(parsed) == (SECRET_USER, SECRET_PASSWORD)


def test_jdbc_query_parameters_survive_the_rewrite(tmp_path):
    """A JDBC URL carries connection options (`?sslmode=require`); only the scheme is rewritten."""

    properties = _derive(
        tmp_path,
        catalog_type="jdbc",
        uri=f"{POSTGRES_JDBC_URI}?sslmode=require",
        credentials={"user": "janus", "password": "secret"},
    )

    assert properties["uri"] == (
        "postgresql+psycopg2://janus:secret@catalog-db:5432/janus?sslmode=require"
    )


def test_credentials_are_percent_encoded_into_the_authority(tmp_path):
    """A password holding `@`, `:` or `/` must not be able to rewrite the host it points at."""

    properties = _derive(tmp_path, **CATALOG_PROFILES["jdbc_postgres"])

    authority, _, _ = properties["uri"].partition("://")[2].partition("/")
    assert authority.endswith("@catalog-db:5432")
    assert SECRET_PASSWORD not in authority
    assert "%40" in authority and "%3A" in authority and "%2F" in authority


@pytest.mark.parametrize(
    ("label", "credentials"),
    [
        ("absent", None),
        ("empty_expansion", {"user": "", "password": ""}),
        ("whitespace_expansion", {"user": "  ", "password": "  "}),
    ],
)
def test_unset_credentials_leave_the_authority_alone(tmp_path, label, credentials):
    """An unset `${JANUS_ICEBERG_CATALOG_USER}` expands to `""`, and `""` is not a user."""

    del label
    block: dict[str, Any] = {"catalog_type": "jdbc", "uri": POSTGRES_JDBC_URI}
    if credentials is not None:
        block["credentials"] = credentials

    properties = _derive(tmp_path, **block)

    assert properties["uri"] == "postgresql+psycopg2://catalog-db:5432/janus"


def test_a_password_without_a_user_still_reaches_the_uri(tmp_path):
    """Spark emits a lone `jdbc.password`; the two engines must not disagree about that."""

    properties = _derive(
        tmp_path,
        catalog_type="jdbc",
        uri=POSTGRES_JDBC_URI,
        credentials={"user": "", "password": SECRET_PASSWORD},
    )

    parsed = urlparse(properties["uri"])
    assert _decoded_credentials(parsed) == ("", SECRET_PASSWORD)


def test_credentials_on_a_file_backed_catalog_fail_closed(tmp_path):
    """A SQLite URI names a file, not a server: there is no authority to carry a credential."""

    with pytest.raises(CatalogPropertyError) as error:
        _derive(
            tmp_path,
            catalog_type="jdbc",
            uri=f"jdbc:sqlite:{SQLITE_DATABASE}",
            credentials={"user": SECRET_USER, "password": SECRET_PASSWORD},
        )

    assert "jdbc:sqlite" in str(error.value)


# ── rest → the easy case ─────────────────────────────────────────────────────


def test_a_rest_profile_passes_its_uri_through(tmp_path):
    """Near-identical dialects: only the warehouse needs a scheme."""

    properties = _derive(tmp_path, **CATALOG_PROFILES["rest"])

    assert properties == {
        "type": PYICEBERG_REST_CATALOG_TYPE,
        "uri": REST_URI,
        "warehouse": (tmp_path / "data/bronze/iceberg").as_uri(),
    }


def test_a_rest_profile_emits_no_credentials(tmp_path):
    """The Spark side emits none either"""

    properties = _derive(
        tmp_path,
        catalog_type="rest",
        uri=REST_URI,
        credentials={"user": SECRET_USER, "password": SECRET_PASSWORD},
    )

    assert properties["uri"] == REST_URI
    assert SECRET_USER not in str(properties) and SECRET_PASSWORD not in str(properties)


# ── hadoop → the named error that is the point of the order ──────────────────


def test_hadoop_raises_a_named_error_explaining_why(tmp_path):
    """`pyiceberg` implements no Hadoop catalog. Saying so *here* is the whole justification."""

    with pytest.raises(HadoopCatalogUnrepresentableError) as error:
        _derive(tmp_path, catalog_type=HADOOP_CATALOG_TYPE)

    message = str(error.value)
    assert "pyiceberg" in message
    assert "Hadoop catalog" in message
    assert "rename" in message
    assert "order-13" in message


def test_the_hadoop_error_is_a_catalog_property_error():
    """One base class, so a caller can handle "this profile has no second-engine config"."""

    assert issubclass(HadoopCatalogUnrepresentableError, CatalogPropertyError)
    assert issubclass(CatalogPropertyError, ValueError)


# ── the two emitters cover the same catalog types ────────────────────────────


def test_every_supported_catalog_type_is_translated_or_named_unrepresentable():
    """The drift test, both directions: a type Spark gains needs a decision here, and vice versa."""

    assert set(_PROPERTY_BUILDERS) | {HADOOP_CATALOG_TYPE} == set(SUPPORTED_CATALOG_TYPES)
    assert set(_PROPERTY_BUILDERS) == set(SUPPORTED_CATALOG_TYPES) - {HADOOP_CATALOG_TYPE}


def test_the_supported_set_is_referenced_not_redeclared():

    module_source = _catalog_properties_source()

    assert "SUPPORTED_CATALOG_TYPES = " not in module_source
    assert "from janus.utils.environment import" in module_source


# ── profile-level failures fail closed, through the shared readers ───────────


def test_a_profile_without_an_iceberg_block_fails_closed(tmp_path):
    config = _environment_config()
    del config["spark"]["iceberg"]
    paths = materialize_runtime_paths(config, tmp_path)

    with pytest.raises(CatalogPropertyError, match=re.escape("spark.iceberg")):
        derive_pyiceberg_catalog_properties(config, paths)


@pytest.mark.parametrize(
    ("label", "block", "expected"),
    [
        ("missing_catalog_type", {"catalog_type": ""}, "spark.iceberg.catalog_type"),
        ("unknown_catalog_type", {"catalog_type": "hive"}, "'hive'"),
        ("missing_uri", {"catalog_type": "jdbc", "uri": ""}, "spark.iceberg.uri"),
    ],
)
def test_profile_errors_come_from_the_same_readers_the_spark_emitter_uses(
    tmp_path, label, block, expected
):
    """A profile the Spark emitter rejects is rejected here, with the same message."""

    del label
    config = _environment_config(**block)
    paths = materialize_runtime_paths(config, tmp_path)

    with pytest.raises(ValueError) as derived_error:
        derive_pyiceberg_catalog_properties(config, paths)
    with pytest.raises(ValueError) as spark_error:
        build_spark_options(config, paths)

    assert expected in str(derived_error.value)
    assert str(derived_error.value) == str(spark_error.value)


@pytest.mark.parametrize(
    ("label", "uri"),
    [
        ("not_a_jdbc_url", "postgresql://catalog-db:5432/janus"),
        ("untranslated_backend", "jdbc:mysql://catalog-db:3306/janus"),
        ("no_backend_separator", "jdbc:sqlite"),
    ],
)
def test_an_untranslatable_jdbc_uri_fails_closed(tmp_path, label, uri):
    """Only the documented backends are emitted; an unverified driver is a broken catalog."""

    del label
    with pytest.raises(CatalogPropertyError) as error:
        _derive(tmp_path, catalog_type="jdbc", uri=uri)

    message = str(error.value)
    assert "jdbc" in message.lower()
    if "mysql" in uri:
        assert all(backend in message for backend in SQLALCHEMY_SCHEMES)


@pytest.mark.parametrize(
    ("label", "uri"),
    [
        ("untranslated_backend", "jdbc:mysql://user:hunter2@catalog-db:3306/janus"),
        ("not_a_jdbc_url", "postgresql://user:hunter2@catalog-db:5432/janus"),
    ],
)
def test_a_uri_error_never_echoes_the_authority(tmp_path, label, uri):
    """The URI itself can carry a password, so an error may name its scheme and nothing more."""

    del label
    with pytest.raises(CatalogPropertyError) as error:
        _derive(tmp_path, catalog_type="jdbc", uri=uri)

    assert "hunter2" not in str(error.value)
    assert "catalog-db" not in str(error.value)


@pytest.mark.parametrize(
    ("label", "block"),
    [
        ("hadoop", {"catalog_type": HADOOP_CATALOG_TYPE}),
        ("missing_uri", {"catalog_type": "jdbc", "uri": ""}),
        ("untranslated_backend", {"catalog_type": "jdbc", "uri": "jdbc:mysql://db/janus"}),
        ("sqlite_with_credentials", {"catalog_type": "jdbc", "uri": "jdbc:sqlite:janus.db"}),
    ],
)
def test_no_credential_appears_in_any_exception_text(tmp_path, label, block):
    """Every failure path runs with credentials configured: the message is what logs keep."""

    del label
    with pytest.raises(ValueError) as error:
        _derive(
            tmp_path,
            credentials={"user": SECRET_USER, "password": SECRET_PASSWORD},
            **block,
        )

    message = str(error.value)
    assert SECRET_USER not in message
    assert SECRET_PASSWORD not in message


# ── the object store the warehouse lives in ──────────────────────────────────

OBJECT_STORE_BLOCK = {
    "io_impl": "S3FileIO",
    "io_package": "org.apache.iceberg:iceberg-aws-bundle:1.10.1",
    "endpoint": "http://minio:9000",
    "path_style_access": "true",
    "region": "us-east-1",
}


def _object_store_properties(tmp_path, **overrides):
    """Derived properties for a jdbc profile whose warehouse is on object storage."""

    block = {**OBJECT_STORE_BLOCK, **overrides}
    config = _environment_config(
        **{
            **CATALOG_PROFILES["jdbc_postgres"],
            "warehouse_dir": "s3://janus-bronze/warehouse",
            "object_store": {
                key: value for key, value in block.items() if value is not None
            },
        }
    )
    return derive_pyiceberg_catalog_properties(
        config, materialize_runtime_paths(config, tmp_path)
    )


def test_the_second_engine_is_told_where_the_object_store_is(tmp_path):
    """The gap a live cluster run found: without these, pyarrow talks to AWS itself."""

    properties = _object_store_properties(tmp_path)

    assert properties["s3.endpoint"] == "http://minio:9000"
    assert properties["s3.region"] == "us-east-1"


def test_path_style_addressing_is_spelled_the_other_way_round(tmp_path):
    """Spark says path-style; `pyiceberg` says virtual-addressing. They are inverses."""

    assert (
        _object_store_properties(tmp_path, path_style_access="true")[
            "s3.force-virtual-addressing"
        ]
        == "false"
    )
    assert (
        _object_store_properties(tmp_path, path_style_access="false")[
            "s3.force-virtual-addressing"
        ]
        == "true"
    )


def test_both_engines_read_one_block(tmp_path):
    """Every object-store value either reaches both engines or reaches neither."""

    block = {**OBJECT_STORE_BLOCK}
    config = _environment_config(
        **{
            **CATALOG_PROFILES["jdbc_postgres"],
            "warehouse_dir": "s3://janus-bronze/warehouse",
            "object_store": block,
        }
    )
    paths = materialize_runtime_paths(config, tmp_path)

    options = build_spark_options(config, paths)
    properties = derive_pyiceberg_catalog_properties(config, paths)
    prefix = "spark.sql.catalog.janus"

    assert properties["s3.endpoint"] == options[f"{prefix}.s3.endpoint"]
    assert properties["s3.region"] == options[f"{prefix}.client.region"]
    assert properties["warehouse"] == options[f"{prefix}.warehouse"]
    # The inverted one, stated as the round trip depends on it.
    assert properties["s3.force-virtual-addressing"] == "false"
    assert options[f"{prefix}.s3.path-style-access"] == "true"


def test_no_object_store_credential_is_ever_derived(tmp_path):
    """pyarrow falls back to the AWS environment chain; JANUS hands it nothing."""

    properties = _object_store_properties(tmp_path)

    assert not [key for key in properties if "access-key" in key or "secret" in key]


def test_a_profile_without_an_object_store_derives_no_file_io_settings(tmp_path):
    """The local profile must be untouched by any of this."""

    config = _environment_config(**CATALOG_PROFILES["jdbc_sqlite"])
    properties = derive_pyiceberg_catalog_properties(
        config, materialize_runtime_paths(config, tmp_path)
    )

    assert set(properties) == {"type", "uri", "warehouse"}


def test_a_block_the_spark_emitter_rejects_is_rejected_here_too(tmp_path):
    """One reader, so a profile cannot configure one engine and quietly skip the other."""

    for override in ({"io_impl": "GlacierFileIO"}, {"path_style_access": "ture"}):
        with pytest.raises(ValueError):
            _object_store_properties(tmp_path, **override)


# ── the warehouse location ───────────────────────────────────────────────────


def test_a_warehouse_that_already_carries_a_scheme_passes_through(tmp_path):
    """Object storage is a config-only swap: a location URI is a location, not a path to resolve."""

    config = _environment_config(**CATALOG_PROFILES["jdbc_sqlite"])
    paths = materialize_runtime_paths(config, tmp_path)
    paths["iceberg_warehouse_dir"] = "s3://janus-bronze/iceberg"  # type: ignore[assignment]

    properties = derive_pyiceberg_catalog_properties(config, paths)

    assert properties["warehouse"] == "s3://janus-bronze/iceberg"


def test_a_warehouse_uri_mangled_into_a_path_is_rejected(tmp_path):
    """The other half of the finding above: `s3:/bucket/x` is not a location, and never was."""

    config = _environment_config(**CATALOG_PROFILES["jdbc_sqlite"])
    paths = materialize_runtime_paths(config, tmp_path)
    paths["iceberg_warehouse_dir"] = Path("s3://janus-bronze/iceberg")

    with pytest.raises(CatalogPropertyError, match="absolute"):
        derive_pyiceberg_catalog_properties(config, paths)


def test_a_missing_resolved_warehouse_fails_closed(tmp_path):
    config = _environment_config(**CATALOG_PROFILES["jdbc_sqlite"])
    paths = materialize_runtime_paths(config, tmp_path)
    del paths["iceberg_warehouse_dir"]

    with pytest.raises(CatalogPropertyError, match="warehouse"):
        derive_pyiceberg_catalog_properties(config, paths)


def test_a_relative_warehouse_cannot_become_a_file_uri(tmp_path):
    """`materialize_runtime_paths` resolves project-relative paths; an unresolved one is a bug."""

    config = _environment_config(**CATALOG_PROFILES["jdbc_sqlite"])
    paths = materialize_runtime_paths(config, tmp_path)
    paths["iceberg_warehouse_dir"] = Path("data/bronze/iceberg")

    with pytest.raises(CatalogPropertyError, match="absolute"):
        derive_pyiceberg_catalog_properties(config, paths)


# ── the catalog identity a caller needs next to the properties ───────────────


def test_the_catalog_name_and_namespace_come_from_the_same_profile(tmp_path):
    config = _environment_config(**CATALOG_PROFILES["jdbc_sqlite"])

    assert derive_pyiceberg_catalog_name(config) == "janus"
    assert derive_pyiceberg_default_namespace(config) == "bronze"


def test_a_profile_without_a_default_namespace_reports_none():
    """Spark omits the conf when the profile sets none; a caller here gets the same answer."""

    config = _environment_config(default_namespace="")

    assert derive_pyiceberg_default_namespace(config) is None


def test_an_empty_catalog_name_fails_closed():
    config = _environment_config(catalog_name="")

    with pytest.raises(CatalogPropertyError, match="catalog_name"):
        derive_pyiceberg_catalog_name(config)


# ── parity: one profile, two emitters, the same catalog ──────────────────────


def _spark_catalog_option(options: dict[str, str], suffix: str) -> str:
    return options[f"{CATALOG_PREFIX}.{suffix}"]


def _sqlite_database(uri: str) -> str:
    """The file both dialects name, extracted without reusing the translation under test."""

    if uri.startswith("jdbc:sqlite:"):
        return uri.removeprefix("jdbc:sqlite:")
    return uri.split(":///", 1)[1]


@pytest.mark.parametrize("profile", sorted(CATALOG_PROFILES))
def test_both_emitters_describe_the_same_catalog(tmp_path, profile):
    """The executable form of "one source of truth"."""

    config = _environment_config(**CATALOG_PROFILES[profile])
    paths = materialize_runtime_paths(config, tmp_path)

    options = build_spark_options(config, paths)
    properties = derive_pyiceberg_catalog_properties(config, paths)

    # same catalog name: it is the Spark key prefix, and pyiceberg's `load_catalog` argument
    name = derive_pyiceberg_catalog_name(config)
    assert f"spark.sql.catalog.{name}.type" in options

    # same warehouse location
    assert urlparse(properties["warehouse"]).path == _spark_catalog_option(options, "warehouse")

    # same namespace
    assert derive_pyiceberg_default_namespace(config) == _spark_catalog_option(
        options, "default-namespace"
    )

    # URIs pointing at the same database
    spark_uri = _spark_catalog_option(options, "uri")
    derived_uri = properties["uri"]
    if profile == "jdbc_sqlite":
        assert _sqlite_database(derived_uri) == _sqlite_database(spark_uri)
    elif profile == "jdbc_postgres":
        spark_target = urlparse(spark_uri.removeprefix("jdbc:"))
        derived_target = urlparse(derived_uri)
        assert (derived_target.hostname, derived_target.port, derived_target.path) == (
            spark_target.hostname,
            spark_target.port,
            spark_target.path,
        )
    else:
        assert derived_uri == spark_uri


def test_the_postgres_credentials_are_the_same_on_both_sides(tmp_path):
    """Spark's `jdbc.user`/`jdbc.password` confs and pyiceberg's userinfo carry the same pair."""

    config = _environment_config(**CATALOG_PROFILES["jdbc_postgres"])
    paths = materialize_runtime_paths(config, tmp_path)

    options = build_spark_options(config, paths)
    parsed = urlparse(derive_pyiceberg_catalog_properties(config, paths)["uri"])

    assert _decoded_credentials(parsed) == (
        _spark_catalog_option(options, "jdbc.user"),
        _spark_catalog_option(options, "jdbc.password"),
    )


# ── containment: no second-engine dependency enters `src/janus` ──────────────

FORBIDDEN_RUNTIME_IMPORTS = ("pyiceberg", "sqlalchemy", "duckdb")


def _catalog_properties_source() -> str:
    return Path(catalog_properties.__file__).read_text(encoding="utf-8")


def _imported_root_modules(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    return roots


def _source_modules() -> list[Path]:
    return sorted(Path(janus.__file__).parent.rglob("*.py"))


def test_no_module_under_src_imports_a_second_engine():
    """The derivation maps strings to strings; the live `pyiceberg` lives in tests only.

    Package-scoped on purpose: a sweep pinned to `catalog_properties.py` would go quiet the moment
    the import it guards against appeared in a neighbour.
    """

    offenders = {
        module.name: sorted(imported)
        for module in _source_modules()
        if (
            imported := _imported_root_modules(module.read_text(encoding="utf-8"))
            & set(FORBIDDEN_RUNTIME_IMPORTS)
        )
    }

    assert not offenders, (
        "src/janus must keep its three runtime dependencies — a second engine is a dev/test "
        f"dependency, not a runtime one: {offenders}"
    )


def test_the_import_sweep_actually_read_the_package():
    """A glob matching nothing would make the assertion above vacuously green."""

    modules = _source_modules()

    assert len(modules) > 1
    assert any(module.name == "catalog_properties.py" for module in modules)
