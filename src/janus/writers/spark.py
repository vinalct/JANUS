from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from janus.models import ExecutionPlan, WriteResult
from janus.utils.storage import StorageLayout, bronze_table_identifier

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

SUPPORTED_SPARK_WRITE_FORMATS = frozenset({"csv", "json", "jsonl", "parquet", "text"})


class SparkDatasetWriter:
    """Spark writer for Iceberg-backed bronze outputs and path-based Spark outputs."""

    def __init__(self, storage_layout: StorageLayout) -> None:
        self.storage_layout = storage_layout

    def write(
        self,
        dataframe: DataFrame,
        plan: ExecutionPlan,
        zone: str,
        *,
        path_suffix: str | None = None,
        format_name: str | None = None,
        mode: str | None = None,
        partition_by: tuple[str, ...] | None = None,
        options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, str] | None = None,
        records_written: int | None = None,
        count_records: bool = False,
        apply_repartition: bool = True,
    ) -> WriteResult:
        resolved_target = self.storage_layout.resolve_output(plan, zone)
        resolved_format = format_name or resolved_target.format
        if zone == "bronze" and resolved_format.strip().lower() == "iceberg":
            return self._write_bronze_iceberg(
                dataframe,
                plan,
                configured_format=resolved_format,
                mode=mode,
                partition_by=partition_by,
                metadata=metadata,
                records_written=records_written,
                count_records=count_records,
                apply_repartition=apply_repartition,
            )

        path = (
            resolved_target.resolved_path
            if path_suffix is None
            else resolved_target.child(path_suffix)
        )
        spark_format = _spark_write_format(resolved_format)
        write_mode = mode or plan.source_config.spark.write_mode
        partition_columns = partition_by or plan.source_config.spark.partition_by

        prepared_frame = _rebalance_for_write(
            dataframe,
            target_partitions=plan.source_config.spark.repartition,
            apply_repartition=apply_repartition and zone == "bronze",
        )

        resolved_records_written = records_written
        if count_records and resolved_records_written is None:
            resolved_records_written = prepared_frame.count()

        writer = prepared_frame.write.mode(write_mode).format(spark_format)
        for key, value in _normalize_options(options).items():
            writer = writer.option(key, value)
        if partition_columns:
            writer = writer.partitionBy(*partition_columns)
        writer.save(str(path))

        return WriteResult.from_plan(
            plan,
            zone,
            path=str(path),
            format_name=resolved_format,
            mode=write_mode,
            records_written=resolved_records_written,
            partition_by=partition_columns,
            metadata=metadata,
        )

    def _write_bronze_iceberg(
        self,
        dataframe: DataFrame,
        plan: ExecutionPlan,
        *,
        configured_format: str,
        mode: str | None,
        partition_by: tuple[str, ...] | None,
        metadata: Mapping[str, str] | None,
        records_written: int | None,
        count_records: bool,
        apply_repartition: bool,
    ) -> WriteResult:
        if configured_format.strip().lower() != "iceberg":
            raise ValueError("bronze outputs must use the 'iceberg' format")

        write_mode = mode or plan.source_config.spark.write_mode
        partition_columns = partition_by or plan.source_config.spark.partition_by

        prepared_frame = _rebalance_for_write(
            dataframe,
            target_partitions=plan.source_config.spark.repartition,
            apply_repartition=apply_repartition,
        )

        resolved_records_written = records_written
        if count_records and resolved_records_written is None:
            resolved_records_written = prepared_frame.count()

        table_identifier = bronze_table_identifier(
            plan.bronze_output.path,
            fallback_name=plan.source.source_id,
            namespace=plan.bronze_output.namespace,
            table_name=plan.bronze_output.table_name,
        )
        namespace_identifier = table_identifier.rsplit(".", 1)[0]
        spark = prepared_frame.sparkSession
        temp_view_name = f"janus_bronze_{plan.source.source_id}_{uuid4().hex}"

        prepared_frame.createOrReplaceTempView(temp_view_name)
        try:
            spark.sql(
                f"CREATE NAMESPACE IF NOT EXISTS {_quote_identifier(namespace_identifier)}"
            )

            quoted_table = _quote_identifier(table_identifier)
            quoted_temp_view = _quote_identifier(temp_view_name)
            partition_clause = _partition_clause(partition_columns)
            table_exists = spark.catalog.tableExists(table_identifier)

            if write_mode == "ignore" and table_exists:
                pass
            elif write_mode == "append":
                if table_exists:
                    spark.sql(f"INSERT INTO {quoted_table} SELECT * FROM {quoted_temp_view}")
                else:
                    spark.sql(
                        f"CREATE TABLE {quoted_table} USING iceberg "
                        f"{partition_clause} AS SELECT * FROM {quoted_temp_view}"
                    )
            elif write_mode == "overwrite":
                if table_exists:
                    spark.sql(
                        f"REPLACE TABLE {quoted_table} USING iceberg "
                        f"{partition_clause} AS SELECT * FROM {quoted_temp_view}"
                    )
                else:
                    spark.sql(
                        f"CREATE TABLE {quoted_table} USING iceberg "
                        f"{partition_clause} AS SELECT * FROM {quoted_temp_view}"
                    )
            else:
                allowed = ", ".join(sorted({"append", "ignore", "overwrite"}))
                raise ValueError(f"mode must be one of: {allowed}")
        finally:
            spark.catalog.dropTempView(temp_view_name)

        return WriteResult.from_plan(
            plan,
            "bronze",
            path=table_identifier,
            format_name="iceberg",
            mode=write_mode,
            records_written=resolved_records_written,
            partition_by=partition_columns,
            metadata=metadata,
        )


def _normalize_options(options: Mapping[str, Any] | None) -> dict[str, str]:
    if not options:
        return {}
    return {str(key): str(value) for key, value in options.items()}


def _spark_write_format(format_name: str) -> str:
    normalized = format_name.strip().lower()
    if normalized not in SUPPORTED_SPARK_WRITE_FORMATS:
        allowed = ", ".join(sorted(SUPPORTED_SPARK_WRITE_FORMATS))
        raise ValueError(f"format_name must be one of: {allowed}")
    if normalized == "jsonl":
        return "json"
    return normalized


def _rebalance_for_write(
    dataframe: DataFrame,
    *,
    target_partitions: int | None,
    apply_repartition: bool,
) -> DataFrame:
    if not apply_repartition or not target_partitions:
        return dataframe

    current_partitions = dataframe.rdd.getNumPartitions()
    if target_partitions > current_partitions:
        return dataframe.repartition(target_partitions)
    return dataframe


def _partition_clause(partition_columns: tuple[str, ...]) -> str:
    if not partition_columns:
        return ""
    # AC-5 PROOF: deliberate Iceberg-writer regression - partition the bronze CTAS
    # by a non-existent column so the write fails. REVERT before merge; main never carries this.
    rendered_columns = ", ".join(
        _quote_identifier(f"{column}_nonexistent") for column in partition_columns
    )
    return f"PARTITIONED BY ({rendered_columns})"


def _quote_identifier(identifier: str) -> str:
    return ".".join(f"`{part.replace('`', '``')}`" for part in identifier.split("."))
