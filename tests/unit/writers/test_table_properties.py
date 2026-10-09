from importlib import import_module
from types import SimpleNamespace

import pytest

from janus.writers.identifiers import quote_identifier
from janus.writers.overwrite import (
    build_create_table_as_select_sql,
    build_replace_table_as_select_sql,
)

SIZE_KEY = "write.target-file-size-bytes"
MODE_KEY = "write.distribution-mode"
DESIRED = {SIZE_KEY: "134217728", MODE_KEY: "hash"}
ESCAPED_TABLE = "bronze.data" + chr(96) + "set"


def _properties():
    return import_module("janus.writers.table_properties")


def test_declaration_renders_exact_sorted_quoted_properties_with_size_in_bytes():
    from janus.models.config.types import SparkWriteConfig

    module = _properties()
    properties = module.desired_table_properties(
        SparkWriteConfig(target_file_size_mb=128, distribution_mode="hash"),
    )
    assert properties == DESIRED
    assert module.render_tblproperties_clause(properties) == (
        "TBLPROPERTIES ('write.distribution-mode'='hash', "
        "'write.target-file-size-bytes'='134217728')"
    )


def test_property_literal_escaping_sort_order_and_empty_clause():
    render = _properties().render_tblproperties_clause
    assert render({"z": "O'Brien", "a": "x"}) == "TBLPROPERTIES ('a'='x', 'z'='O''Brien')"
    assert render({"operator's.key": "value"}) == "TBLPROPERTIES ('operator''s.key'='value')"
    assert render({}) == ""
    assert render(dict(reversed(list(DESIRED.items())))) == render(DESIRED)


@pytest.mark.parametrize(
    "builder",
    [build_create_table_as_select_sql, build_replace_table_as_select_sql],
)
def test_ctas_and_replace_place_properties_before_select_and_quote_identifiers(builder):
    sql = builder(
        table_identifier=ESCAPED_TABLE,
        source_view="source.view",
        partition_columns=("event_date",),
        table_properties=DESIRED,
    )
    assert quote_identifier(ESCAPED_TABLE) in sql
    assert quote_identifier("source.view") in sql
    assert sql.index("PARTITIONED BY") < sql.index("TBLPROPERTIES") < sql.index("AS SELECT")
    assert "'write.distribution-mode'='hash'" in sql
    assert "'write.target-file-size-bytes'='134217728'" in sql
    assert "`write.distribution-mode`" not in sql
    assert "`write.target-file-size-bytes`" not in sql
    assert "UNSET" not in sql.upper()


def test_omitting_properties_keeps_existing_create_and_replace_sql():
    for builder in (build_create_table_as_select_sql, build_replace_table_as_select_sql):
        sql = builder(table_identifier="bronze.data", source_view="source", partition_columns=())
        assert "TBLPROPERTIES" not in sql
        assert "UNSET" not in sql.upper()
        assert quote_identifier("bronze.data") in sql
        assert builder(
            table_identifier="bronze.data", source_view="source", partition_columns=(),
            table_properties={},
        ) == sql


@pytest.mark.parametrize("mode", ["none", "hash", "range"])
def test_desired_properties_manage_only_file_size_and_distribution(mode):
    from janus.models.config.types import SparkWriteConfig

    module = _properties()
    properties = module.desired_table_properties(
        SparkWriteConfig(target_file_size_mb=256, distribution_mode=mode, partitions=4)
    )
    assert tuple(sorted(properties)) == module.MANAGED_TABLE_PROPERTIES == (MODE_KEY, SIZE_KEY)
    assert properties == {MODE_KEY: mode, SIZE_KEY: "268435456"}
    assert "UNSET" not in module.render_tblproperties_clause(properties).upper()


@pytest.mark.xfail(strict=True, reason="green")
def test_drift_sets_only_changed_managed_key_in_one_alter_statement():
    module = _properties()
    plan = module.plan_table_property_reconciliation(
        current={SIZE_KEY: "67108864", MODE_KEY: "hash"},
        desired=DESIRED,
    )
    assert plan.needs_alter
    assert dict(plan.to_set) == {SIZE_KEY: "134217728"}
    assert plan.unchanged == (MODE_KEY,)
    assert module.build_set_tblproperties_sql(
        table_identifier="bronze.data",
        properties=plan.to_set,
    ) == (
        "ALTER TABLE "
        + quote_identifier("bronze.data")
        + " SET TBLPROPERTIES ('write.target-file-size-bytes'='134217728')"
    )


@pytest.mark.xfail(strict=True, reason="green")
def test_equal_properties_issue_no_statement_at_all(monkeypatch):
    module = _properties()
    plan = module.plan_table_property_reconciliation(current=DESIRED, desired=DESIRED)
    assert not plan.needs_alter and dict(plan.to_set) == {}
    statements = []

    def sql(statement):
        statements.append(statement)
        return SimpleNamespace(
            collect=lambda: [
                SimpleNamespace(key=key, value=value) for key, value in DESIRED.items()
            ]
        )

    from janus.models.config.types import SparkWriteConfig
    from janus.writers import spark as spark_writer

    monkeypatch.setattr(module, "read_table_properties", lambda spark, table: dict(DESIRED))
    metadata = spark_writer._reconcile_table_properties(
        SimpleNamespace(sql=sql),
        "bronze.data",
        SparkWriteConfig(target_file_size_mb=128, distribution_mode="hash"),
    )
    assert not any(statement.upper().startswith("ALTER TABLE") for statement in statements)
    assert metadata["table_properties_reconciled"] == "none"


@pytest.mark.xfail(strict=True, reason="green in TASK-15")
def test_unmanaged_keys_are_preserved_and_never_mentioned_in_sql():
    module = _properties()
    plan = module.plan_table_property_reconciliation(
        current={SIZE_KEY: "1", MODE_KEY: "hash", "operator.note": "keep"},
        desired={**DESIRED, "operator.note": "do not manage"},
    )
    assert dict(plan.to_set) == {SIZE_KEY: "134217728"}
    assert plan.unmanaged == ("operator.note",)
    sql = module.build_set_tblproperties_sql(table_identifier="bronze.data", properties=plan.to_set)
    assert "operator.note" not in sql
    assert "UNSET" not in sql.upper()


@pytest.mark.xfail(strict=True, reason="green")
def test_missing_managed_keys_are_drift():
    plan = _properties().plan_table_property_reconciliation(current={}, desired=DESIRED)
    assert plan.needs_alter
    assert dict(plan.to_set) == DESIRED


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize("identifier", ["bronze.name.with.dots", ESCAPED_TABLE])
def test_alter_quotes_each_table_identifier_segment_and_never_unsets(identifier):
    sql = _properties().build_set_tblproperties_sql(
        table_identifier=identifier,
        properties=DESIRED,
    )
    assert sql.startswith("ALTER TABLE " + quote_identifier(identifier) + " SET TBLPROPERTIES ")
    assert "UNSET" not in sql.upper()


@pytest.mark.xfail(strict=True, reason="green")
def test_empty_alter_properties_are_refused():
    with pytest.raises(ValueError):
        _properties().build_set_tblproperties_sql(table_identifier="bronze.data", properties={})
