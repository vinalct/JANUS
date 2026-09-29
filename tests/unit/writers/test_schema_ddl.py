"""The pure SQL builders every bronze statement is rendered by."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from janus.writers import build_insert_overwrite_sql

RED_TASK_2 = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE = "bronze.events"
VIEW = "janus_bronze_events_0f"


# ── the by-name insert and the one projection renderer ─────────────


def test_insert_into_renders_an_explicit_projection():
    from janus.writers.schema_ddl import build_insert_into_sql

    sql = build_insert_into_sql(
        table_identifier=TABLE, source_view=VIEW, projection=("event_id", "amount")
    )

    assert sql == (
        "INSERT INTO `bronze`.`events`\n"
        "SELECT `event_id`, `amount` FROM `janus_bronze_events_0f`"
    )


def test_insert_into_quotes_every_column_with_the_identifier_defence():
    from janus.writers.schema_ddl import build_insert_into_sql

    sql = build_insert_into_sql(table_identifier=TABLE, source_view=VIEW, projection=("we`ird",))

    assert "SELECT `we``ird` FROM" in sql


def test_an_empty_projection_is_refused():
    from janus.writers.schema_ddl import build_insert_into_sql, render_projection

    with pytest.raises(ValueError):
        build_insert_into_sql(table_identifier=TABLE, source_view=VIEW, projection=())
    with pytest.raises(ValueError):
        render_projection(())


def test_insert_overwrite_and_insert_into_share_one_projection_renderer():
    from janus.writers.schema_ddl import build_insert_into_sql, render_projection

    projection = ("event_id", "amount", "ingestion_date")
    rendered = render_projection(projection)
    overwrite = build_insert_overwrite_sql(
        table_identifier=TABLE, source_view=VIEW, projection=projection
    )
    insert = build_insert_into_sql(table_identifier=TABLE, source_view=VIEW, projection=projection)

    assert rendered == "`event_id`, `amount`, `ingestion_date`"
    assert overwrite == f"INSERT OVERWRITE `bronze`.`events`\nSELECT {rendered} FROM `{VIEW}`"
    assert insert == f"INSERT INTO `bronze`.`events`\nSELECT {rendered} FROM `{VIEW}`"


def test_build_add_columns_sql_moved_but_keeps_its_public_import_path():
    from janus.writers import build_add_columns_sql as exported
    from janus.writers.schema_ddl import build_add_columns_sql

    assert exported is build_add_columns_sql
    assert build_add_columns_sql(table_identifier=TABLE, columns=[("note", "string")]) == (
        "ALTER TABLE `bronze`.`events` ADD COLUMNS (`note` string)"
    )
    assert build_add_columns_sql(table_identifier=TABLE, columns=[]) is None


def test_the_ddl_module_imports_no_spark():
    import_paths = (str(PROJECT_ROOT / "src"), *(entry for entry in sys.path if entry))
    command = (
        "import sys; "
        f"sys.path[:0] = {import_paths!r}; "
        "import janus.writers.schema_ddl; "
        "assert not {'pyspark', 'pyiceberg', 'pyarrow'} & sys.modules.keys()"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


# ── in-place evolution and the contract stamp ───────────────────────


@RED_TASK_2
def test_alter_column_type_renders_the_iceberg_promotion():
    from janus.writers.schema_ddl import build_alter_column_type_sql

    sql = build_alter_column_type_sql(table_identifier="t", column="amount", spark_type="bigint")

    assert sql == "ALTER TABLE `t` ALTER COLUMN `amount` TYPE bigint"


@RED_TASK_2
def test_the_contract_stamp_single_quotes_and_escapes_every_value():
    from janus.writers.schema_ddl import (
        CONTRACT_PROPERTY_KEYS,
        build_set_contract_properties_sql,
    )

    sql = build_set_contract_properties_sql(
        table_identifier=TABLE,
        contract_id="test.o'brien",
        contract_version="1.0.0",
        schema_version="ab" * 32,
    )

    assert CONTRACT_PROPERTY_KEYS == (
        "janus.contract_id",
        "janus.contract_version",
        "janus.schema_version",
    )
    assert sql == (
        "ALTER TABLE `bronze`.`events` SET TBLPROPERTIES ("
        "'janus.contract_id' = 'test.o''brien', "
        "'janus.contract_version' = '1.0.0', "
        f"'janus.schema_version' = '{'ab' * 32}')"
    )


@RED_TASK_2
def test_the_stamp_is_read_back_with_show_tblproperties():
    from janus.writers.schema_ddl import build_show_contract_properties_sql

    assert build_show_contract_properties_sql(table_identifier=TABLE) == (
        "SHOW TBLPROPERTIES `bronze`.`events`"
    )


@RED_TASK_2
@pytest.mark.parametrize(
    ("physical_type", "ddl_type"),
    [
        ("boolean", "boolean"),
        ("integer", "int"),
        ("long", "bigint"),
        ("float", "float"),
        ("double", "double"),
        ("decimal(18,2)", "decimal(18,2)"),
        ("string", "string"),
        ("binary", "binary"),
        ("date", "date"),
        ("timestamp", "timestamp_ntz"),
        ("timestamptz", "timestamp"),
    ],
)
def test_every_scalar_has_one_ddl_spelling(physical_type, ddl_type):
    from janus.models.data_contracts.vocabulary import spark_sql_type

    assert spark_sql_type(physical_type) == ddl_type
