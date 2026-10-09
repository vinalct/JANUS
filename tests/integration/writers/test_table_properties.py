"""Creation and replacement apply declared properties to a real Iceberg catalog."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from janus.models import BronzeWriteIntent
from janus.models.config.types import SparkWriteConfig
from janus.normalizers import BaseNormalizer
from janus.utils.storage import StorageLayout
from janus.writers import MANAGED_TABLE_PROPERTIES, SparkDatasetWriter, quote_identifier
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
    assert sorted(spark.table(result.path).collect()) == sorted(frame.collect())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_existing_table_properties_are_untouched(spark, tmp_path, strategy):
    original = SparkWriteConfig(target_file_size_mb=64, distribution_mode="none")
    plan, writer = _plan_and_writer(tmp_path, f"properties_existing_{strategy}", original, ())
    frame, plan = _frame_and_contract(spark, plan)
    intent = BronzeWriteIntent(strategy, "overwrite", merge_keys=("id",))
    first = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    spark.sql(
        f"ALTER TABLE {quote_identifier(first.path)} SET TBLPROPERTIES ('operator.note'='keep')"
    )
    preserved_keys = (*MANAGED_TABLE_PROPERTIES, "operator.note")
    properties = _properties(spark, first.path)
    before = {key: properties[key] for key in preserved_keys}
    plan = replace(
        plan, source_config=replace(
            plan.source_config,
            spark=replace(
                plan.source_config.spark,
                write=SparkWriteConfig(target_file_size_mb=256, distribution_mode="hash"),
            ),
        ),
    )
    result = writer.write(frame, plan, "bronze", intent=intent, apply_repartition=False)
    after = _properties(spark, result.path)
    assert {key: after[key] for key in preserved_keys} == before
    assert "table_properties_applied" not in result.metadata_as_dict()
    assert "table_properties_reconciled" not in result.metadata_as_dict()
