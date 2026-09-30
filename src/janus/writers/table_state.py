"""Thin Spark reads and writes for an Iceberg bronze table's schema and contract stamp."""

from __future__ import annotations

from typing import Any

from janus.models.data_contracts import DataContract, physical_type_from_spark_json
from janus.writers.evolution import LiveColumn
from janus.writers.schema_ddl import (
    CONTRACT_PROPERTY_KEYS,
    build_set_contract_properties_sql,
    build_show_contract_properties_sql,
)


def read_target_columns(spark: Any, table_identifier: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field.name, field.dataType.simpleString())
        for field in spark.table(table_identifier).schema.fields
    )


def read_live_columns(spark: Any, table_identifier: str) -> tuple[LiveColumn, ...]:
    return tuple(
        LiveColumn(
            field.name,
            physical_type_from_spark_json(field.dataType.jsonValue()),
            not getattr(field, "nullable", True),
        )
        for field in spark.table(table_identifier).schema.fields
    )


def read_target_partitions(spark: Any, table_identifier: str) -> tuple[str, ...] | None:
    """Read identity partition columns for a full-refresh overwrite."""
    try:
        partitions_schema = spark.table(f"{table_identifier}.partitions").schema
        partition_field = next(
            (field for field in partitions_schema.fields if field.name == "partition"),
            None,
        )
        if partition_field is None:
            return ()
        return tuple(field.name for field in partition_field.dataType.fields)
    except Exception:
        return None


def read_contract_stamp(spark: Any, table_identifier: str) -> dict[str, str]:
    statement = build_show_contract_properties_sql(table_identifier=table_identifier)
    rows = spark.sql(statement).collect()
    return {row["key"]: row["value"] for row in rows if row["key"] in CONTRACT_PROPERTY_KEYS}


def stamp_contract(
    spark: Any,
    table_identifier: str,
    contract: DataContract,
    recorded_stamp: dict[str, str] | None,
) -> None:
    desired = {
        "janus.contract_id": contract.id,
        "janus.contract_version": contract.version,
        "janus.schema_version": contract.schema_version,
    }
    if recorded_stamp == desired:
        return
    spark.sql(
        build_set_contract_properties_sql(
            table_identifier=table_identifier,
            contract_id=contract.id,
            contract_version=contract.version,
            schema_version=contract.schema_version,
        )
    )
