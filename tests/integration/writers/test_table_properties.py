"""Managed properties are created and reconciled across every bronze write strategy."""

from __future__ import annotations

import json
from dataclasses import replace
from time import perf_counter
from unittest.mock import patch

import pytest

from janus.models import BronzeWriteIntent
from janus.models.config.types import SparkWriteConfig
from janus.normalizers import BaseNormalizer
from janus.utils.storage import StorageLayout
from janus.writers import MANAGED_TABLE_PROPERTIES, SparkDatasetWriter, quote_identifier
from janus.writers.spark import _reconcile_table_properties
from janus.writers.table_properties import (
    desired_table_properties,
    plan_table_property_reconciliation,
    read_table_properties,
)
from tests.fixtures.handoff.generate_pages import synthetic_plan
from tests.support.spark_sessions import catalog_target_params, start_session
from tests.support.writer_contracts import contract_for_frame

pytestmark = pytest.mark.spark

STRATEGIES = ["skip_if_exists", "insert", "create", "replace_table", "merge_on_keys"]
DECLARATIONS = [
    pytest.param(64, "none", (), id="unpartitioned-none-64MiB"),
    pytest.param(128, "hash", ("group_key",), id="partitioned-hash-128MiB"),
    pytest.param(256, "range", ("group_key",), id="partitioned-range-256MiB"),
]
ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "runtime", "raw_dir": "runtime/raw",
        "bronze_dir": "runtime/bronze", "metadata_dir": "runtime/metadata",
    }
}


@pytest.fixture(scope="module", params=catalog_target_params())
def catalog_target(request, tmp_path_factory):
    target = request.param(tmp_path_factory.mktemp("table-properties-catalog"))
    target.prepare()
    return target


@pytest.fixture(scope="module")
def spark(catalog_target):
    session = start_session("janus-table-properties", catalog_target.session_options())
    yield session
    session.stop()


def _plan_and_writer(tmp_path, table_name, declaration, partitions):
    plan = synthetic_plan(tmp_path / "raw")
    plan = replace(
        plan,
        source_config=replace(
            plan.source_config,
            spark=replace(plan.source_config.spark, write=declaration, partition_by=partitions),
        ),
        bronze_output=replace(plan.bronze_output, table_name=table_name),
    )
    writer = SparkDatasetWriter(StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path))
    return plan, writer


def _frame_and_contract(spark, plan):
    frame = spark.createDataFrame(
        [(1, "first", "A"), (2, "second", "B")], "id long, value string, group_key string"
    )
    plan = plan.with_data_contract(
        contract_for_frame(frame, source_id=plan.source.source_id, primary_key=("id",))
    )
    return BaseNormalizer().normalize(frame, plan), plan


def _properties(spark, identifier):
    return {
        row.key: row.value
        for row in spark.sql(f"SHOW TBLPROPERTIES {quote_identifier(identifier)}").collect()
    }


def _assert_declared_properties(spark, identifier, declaration):
    properties = _properties(spark, identifier)
    assert {key: properties[key] for key in MANAGED_TABLE_PROPERTIES} == {
        "write.distribution-mode": declaration.distribution_mode,
        "write.target-file-size-bytes": str(declaration.target_file_size_bytes),
    }
    print("FR6_PROPERTIES=" + json.dumps({"table": identifier, **properties}), flush=True)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(("size_mb", "mode", "partitions"), DECLARATIONS)
def test_first_write_applies_declared_properties(
    spark, tmp_path, strategy, size_mb, mode, partitions
):
    declaration = SparkWriteConfig(target_file_size_mb=size_mb, distribution_mode=mode)
    plan, writer = _plan_and_writer(
        tmp_path, f"properties_{strategy}_{mode}", declaration, partitions
    )
    frame, plan = _frame_and_contract(spark, plan)
    result = writer.write(
        frame, plan, "bronze", intent=BronzeWriteIntent(strategy, "overwrite", merge_keys=("id",)),
        apply_repartition=False, metadata={"caller": "preserved"},
    )
    _assert_declared_properties(spark, result.path, declaration)
    assert result.metadata_as_dict()["table_properties_applied"] == "created"
    assert "table_properties_reconciled" not in result.metadata_as_dict()
    assert result.metadata_as_dict()["caller"] == "preserved"
    assert spark.table(result.path).schema.simpleString() == frame.schema.simpleString()
    assert sorted(spark.table(result.path).collect()) == sorted(frame.collect())


@pytest.mark.parametrize(("size_mb", "mode", "partitions"), DECLARATIONS)
def test_replacement_applies_current_declaration(spark, tmp_path, size_mb, mode, partitions):
    declaration = SparkWriteConfig(target_file_size_mb=size_mb, distribution_mode=mode)
    plan, writer = _plan_and_writer(
        tmp_path, f"properties_replacement_{mode}",
        SparkWriteConfig(target_file_size_mb=32, distribution_mode="none"), partitions,
    )
    frame, plan = _frame_and_contract(spark, plan)
    intent = BronzeWriteIntent("replace_table", "overwrite")
    first = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    _assert_declared_properties(spark, first.path, plan.source_config.spark.write)
    spark.sql(
        f"ALTER TABLE {quote_identifier(first.path)} SET TBLPROPERTIES ('operator.note'='keep')"
    )
    # A partition-spec change executes the writer's real REPLACE TABLE path.
    plan = replace(
        plan, source_config=replace(
            plan.source_config,
            spark=replace(plan.source_config.spark, write=declaration, partition_by=("value",)),
        ),
    )
    result = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    assert result.metadata_as_dict()["overwrite_mechanism"] == "replace_table"
    assert "partition spec changed" in result.metadata_as_dict()["history_reset_reason"]
    _assert_declared_properties(spark, result.path, declaration)
    assert _properties(spark, result.path)["operator.note"] == "keep"
    assert sorted(spark.table(result.path).collect()) == sorted(frame.collect())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_existing_table_drift_is_reconciled_and_unmanaged_properties_survive(
    spark, tmp_path, strategy,
):
    original = SparkWriteConfig(target_file_size_mb=64, distribution_mode="none")
    plan, writer = _plan_and_writer(tmp_path, f"properties_existing_{strategy}", original, ())
    frame, plan = _frame_and_contract(spark, plan)
    intent = BronzeWriteIntent(strategy, "overwrite", merge_keys=("id",))
    first = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    spark.sql(
        f"ALTER TABLE {quote_identifier(first.path)} SET TBLPROPERTIES ('operator.note'='keep')"
    )
    before_rows = spark.table(first.path).collect()
    before_snapshots = _snapshots(spark, first.path)
    plan = replace(
        plan, source_config=replace(
            plan.source_config,
            spark=replace(
                plan.source_config.spark,
                write=SparkWriteConfig(target_file_size_mb=256, distribution_mode="hash"),
            ),
        ),
    )
    with patch.object(spark, "sql", wraps=spark.sql) as sql_spy:
        result = writer.write(
            frame.where("id = 1"), plan, "bronze", intent=intent, apply_repartition=False,
        )
    statements = [call.args[0] for call in sql_spy.call_args_list]
    managed_alters = [statement for statement in statements if (
        statement.startswith("ALTER TABLE") and "'write.distribution-mode'" in statement
    )]
    assert len(managed_alters) == 1
    assert all("UNSET" not in statement.upper() for statement in statements)
    assert "operator.note" not in managed_alters[0]
    after = _properties(spark, result.path)
    _assert_declared_properties(spark, result.path, plan.source_config.spark.write)
    assert after["operator.note"] == "keep"
    assert "table_properties_applied" not in result.metadata_as_dict()
    assert result.metadata_as_dict()["table_properties_reconciled"] == ",".join(
        MANAGED_TABLE_PROPERTIES
    )
    assert "operator.note" in result.metadata_as_dict()["table_properties_unmanaged"].split(",")
    reconciliation = plan_table_property_reconciliation(
        current=after, desired=desired_table_properties(plan.source_config.spark.write),
    )
    assert "operator.note" in reconciliation.unmanaged
    if strategy == "skip_if_exists":
        assert spark.table(first.path).collect() == before_rows
        assert _snapshots(spark, first.path) == before_snapshots
        assert not any(statement.startswith(("INSERT", "MERGE", "CREATE TABLE", "REPLACE TABLE"))
                       for statement in statements)
    print("FR6_RECONCILED=" + json.dumps({
        "strategy": strategy, "properties": after, "metadata": result.metadata_as_dict(),
        "managed_alters": managed_alters,
    }), flush=True)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_matching_properties_issue_no_alter_at_all(spark, tmp_path, strategy):
    declaration = SparkWriteConfig(target_file_size_mb=128, distribution_mode="hash")
    plan, writer = _plan_and_writer(tmp_path, f"properties_equal_{strategy}", declaration, ())
    frame, plan = _frame_and_contract(spark, plan)
    intent = BronzeWriteIntent(strategy, "overwrite", merge_keys=("id",))
    first = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    spark.sql(
        f"ALTER TABLE {quote_identifier(first.path)} SET TBLPROPERTIES ('operator.note'='keep')"
    )
    before = _properties(spark, first.path)
    with patch.object(spark, "sql", wraps=spark.sql) as sql_spy:
        result = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    statements = [call.args[0] for call in sql_spy.call_args_list]
    assert not any(statement.startswith("ALTER TABLE") for statement in statements)
    assert result.metadata_as_dict()["table_properties_reconciled"] == "none"
    assert "operator.note" in result.metadata_as_dict()["table_properties_unmanaged"].split(",")
    after = _properties(spark, first.path)
    for key in (*MANAGED_TABLE_PROPERTIES, "operator.note"):
        assert after[key] == before[key]
    print("FR6_EQUAL=" + json.dumps({
        "strategy": strategy, "properties": after, "alter_count": 0,
    }), flush=True)


RECONCILIATION_STRATEGIES = ["replace_table", "insert", "merge_on_keys"]


@pytest.mark.parametrize("strategy", RECONCILIATION_STRATEGIES)
def test_legacy_table_with_no_managed_properties_is_reconciled(spark, tmp_path, strategy):
    declaration = SparkWriteConfig(target_file_size_mb=128, distribution_mode="hash")
    plan, writer = _plan_and_writer(tmp_path, f"properties_legacy_{strategy}", declaration, ())
    frame, plan = _frame_and_contract(spark, plan)
    identifier = f"bronze.properties_legacy_{strategy}"
    spark.sql("CREATE NAMESPACE IF NOT EXISTS bronze")
    frame.createOrReplaceTempView("legacy_property_source")
    try:
        spark.sql(
            f"CREATE TABLE {quote_identifier(identifier)} USING iceberg "
            "AS SELECT * FROM legacy_property_source"
        )
    finally:
        spark.catalog.dropTempView("legacy_property_source")
    assert not set(MANAGED_TABLE_PROPERTIES) & _properties(spark, identifier).keys()
    result = writer.write(
        frame, plan, "bronze", intent=BronzeWriteIntent(strategy, "overwrite", merge_keys=("id",)),
        apply_repartition=False,
    )
    assert result.metadata_as_dict()["table_properties_reconciled"] == ",".join(
        MANAGED_TABLE_PROPERTIES
    )
    _assert_declared_properties(spark, identifier, declaration)


@pytest.mark.parametrize("strategy", RECONCILIATION_STRATEGIES)
@pytest.mark.parametrize("failure", ["read", "alter"])
def test_property_catalog_failures_do_not_prevent_data_writes(
    spark, tmp_path, strategy, failure,
):
    original = SparkWriteConfig(target_file_size_mb=64, distribution_mode="none")
    plan, writer = _plan_and_writer(
        tmp_path, f"properties_{failure}_failed_{strategy}", original, (),
    )
    frame, plan = _frame_and_contract(spark, plan)
    intent = BronzeWriteIntent(strategy, "overwrite", merge_keys=("id",))
    first = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    plan = replace(plan, source_config=replace(
        plan.source_config,
        spark=replace(plan.source_config.spark, write=SparkWriteConfig(distribution_mode="hash")),
    ))
    # Inject failure into the optional property read, leaving contract reads authoritative.
    original_sql = spark.sql
    injected = []

    def sql(statement, *args, **kwargs):
        if not injected and (
            (failure == "read" and statement.startswith("SHOW TBLPROPERTIES"))
            or (failure == "alter" and statement.startswith("ALTER TABLE")
                and "'write.distribution-mode'" in statement)
        ):
            injected.append(statement)
            raise RuntimeError("injected optional property catalog failure")
        return original_sql(statement, *args, **kwargs)

    before_snapshots = _snapshots(spark, first.path)
    with patch.object(spark, "sql", side_effect=sql):
        result = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    assert len(injected) == 1
    assert result.metadata_as_dict()["table_properties_reconciled"] == f"{failure}_failed"
    assert len(_snapshots(spark, first.path)) == len(before_snapshots) + 1
    assert spark.table(first.path).count() == (4 if strategy == "insert" else 2)
    _assert_declared_properties(spark, first.path, original)
    print("FR6_DEGRADED=" + json.dumps({
        "strategy": strategy, "failure": failure, "metadata": result.metadata_as_dict(),
    }), flush=True)


def _snapshots(spark, identifier):
    return sorted(
        row.snapshot_id for row in spark.table(f"{identifier}.snapshots").collect()
    )


def test_property_alter_is_metadata_only_and_measure_lookup_cost(spark, tmp_path):
    original = SparkWriteConfig(target_file_size_mb=64, distribution_mode="none")
    plan, writer = _plan_and_writer(tmp_path, "properties_metadata_commit", original, ())
    frame, plan = _frame_and_contract(spark, plan)
    first = writer.write(frame, plan, "bronze", apply_repartition=False)
    before_snapshots = _snapshots(spark, first.path)
    before_history = spark.table(f"{first.path}.history").collect()
    started = perf_counter()
    properties = read_table_properties(spark, first.path)
    lookup_ms = (perf_counter() - started) * 1000
    assert properties["write.target-file-size-bytes"] == "67108864"
    metadata = _reconcile_table_properties(
        spark, first.path, SparkWriteConfig(target_file_size_mb=128, distribution_mode="hash"),
    )
    assert metadata["table_properties_reconciled"] == ",".join(MANAGED_TABLE_PROPERTIES)
    assert _snapshots(spark, first.path) == before_snapshots
    assert spark.table(f"{first.path}.history").collect() == before_history
    assert sorted(spark.table(first.path).collect()) == sorted(frame.collect())
    print("FR6_METADATA_COMMIT=" + json.dumps({
        "lookup_ms": lookup_ms, "snapshots_before": before_snapshots,
        "snapshots_after": _snapshots(spark, first.path), "history_unchanged": True,
        "properties": _properties(spark, first.path),
    }), flush=True)
