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


def test_missing_managed_keys_are_drift():
    plan = _properties().plan_table_property_reconciliation(current={}, desired=DESIRED)
    assert plan.needs_alter
    assert dict(plan.to_set) == DESIRED


@pytest.mark.parametrize("identifier", ["bronze.name.with.dots", ESCAPED_TABLE])
def test_alter_quotes_each_table_identifier_segment_and_never_unsets(identifier):
    sql = _properties().build_set_tblproperties_sql(
        table_identifier=identifier,
        properties=DESIRED,
    )
    assert sql.startswith("ALTER TABLE " + quote_identifier(identifier) + " SET TBLPROPERTIES ")
    assert "UNSET" not in sql.upper()


def test_empty_alter_properties_are_refused():
    with pytest.raises(ValueError):
        _properties().build_set_tblproperties_sql(table_identifier="bronze.data", properties={})


@pytest.mark.parametrize("numeric_side", ["current", "desired", "both"])
def test_rendered_numeric_values_do_not_cause_permanent_drift(numeric_side):
    current, desired = dict(DESIRED), dict(DESIRED)
    if numeric_side in {"current", "both"}:
        current[SIZE_KEY] = 134217728
    if numeric_side in {"desired", "both"}:
        desired[SIZE_KEY] = 134217728
    reconciliation = _properties().plan_table_property_reconciliation(
        current=current, desired=desired,
    )
    assert not reconciliation.needs_alter
    assert reconciliation.unchanged == (MODE_KEY, SIZE_KEY)


def test_omitted_managed_key_is_never_removed_and_plan_cannot_be_mutated():
    current = {**DESIRED, "z.foreign": "z", "a.foreign": "a"}
    desired = {MODE_KEY: "range"}
    reconciliation = _properties().plan_table_property_reconciliation(
        current=current, desired=desired,
    )
    assert dict(reconciliation.to_set) == {MODE_KEY: "range"}
    assert reconciliation.unmanaged == ("a.foreign", "z.foreign")
    assert reconciliation.unchanged == ()
    assert current[MODE_KEY] == "hash" and desired == {MODE_KEY: "range"}
    with pytest.raises(TypeError):
        reconciliation.to_set[MODE_KEY] = "none"


def test_alter_escapes_property_literals_and_sorts_keys():
    sql = _properties().build_set_tblproperties_sql(
        table_identifier=ESCAPED_TABLE, properties={"z": "O'Brien", "operator's.key": "x"},
    )
    assert sql == (
        f"ALTER TABLE {quote_identifier(ESCAPED_TABLE)} SET "
        "TBLPROPERTIES ('operator''s.key'='x', 'z'='O''Brien')"
    )


def test_catalog_read_quotes_identifier_and_normalizes_values():
    statements = []

    def sql(statement):
        statements.append(statement)
        return SimpleNamespace(collect=lambda: [{"key": SIZE_KEY, "value": 134217728}])

    current = _properties().read_table_properties(SimpleNamespace(sql=sql), ESCAPED_TABLE)
    assert current == {SIZE_KEY: "134217728"}
    assert statements == [f"SHOW TBLPROPERTIES {quote_identifier(ESCAPED_TABLE)}"]
    assert not getattr(current, "read_failed", False)


@pytest.mark.parametrize("stage", ["sql", "collect"])
def test_failed_read_is_logged_and_distinguishable_from_missing_properties(stage, caplog):
    def fail():
        raise RuntimeError("property lookup unavailable")

    def sql(_statement):
        return fail() if stage == "sql" else SimpleNamespace(collect=fail)

    module = _properties()
    current = module.read_table_properties(SimpleNamespace(sql=sql), "bronze.data")
    assert current == {} and current.read_failed
    assert "Could not read table properties for bronze.data" in caplog.text
    empty = module.read_table_properties(
        SimpleNamespace(sql=lambda _: SimpleNamespace(collect=lambda: [])), "bronze.data",
    )
    assert empty == {} and not getattr(empty, "read_failed", False)


@pytest.mark.parametrize("failure", ["read", "alter"])
def test_reconciler_degrades_lookup_and_alter_failures(failure, caplog):
    from janus.models.config.types import SparkWriteConfig
    from janus.writers.spark import _reconcile_table_properties

    statements = []

    def sql(statement):
        statements.append(statement)
        if failure == "read" or statement.startswith("ALTER"):
            raise RuntimeError("catalog unavailable")
        return SimpleNamespace(collect=lambda: [])

    metadata = _reconcile_table_properties(
        SimpleNamespace(sql=sql), "bronze.data", SparkWriteConfig(),
    )
    assert metadata["table_properties_reconciled"] == f"{failure}_failed"
    assert len(statements) == (1 if failure == "read" else 2)
    assert "Could not" in caplog.text


def test_one_alter_records_changed_keys_and_unmanaged_keys():
    from janus.models.config.types import SparkWriteConfig
    from janus.writers.spark import _reconcile_table_properties

    statements = []

    def sql(statement):
        statements.append(statement)
        return SimpleNamespace(collect=lambda: [{"key": "operator.note", "value": "keep"}])

    metadata = _reconcile_table_properties(
        SimpleNamespace(sql=sql), "bronze.data", SparkWriteConfig(distribution_mode="hash"),
    )
    assert metadata == {
        "table_properties_reconciled": f"{MODE_KEY},{SIZE_KEY}",
        "table_properties_unmanaged": "operator.note",
    }
    assert len(statements) == 2 and statements[1].startswith("ALTER TABLE")
    assert "operator.note" not in statements[1] and "UNSET" not in statements[1].upper()
