from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from janus.models import (
    BronzeWriteIntent,
    ExecutionPlan,
    WriteResult,
    resolve_bronze_write_intent,
)
from janus.utils.storage import StorageLayout, bronze_table_identifier

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

SUPPORTED_SPARK_WRITE_FORMATS = frozenset({"csv", "json", "jsonl", "parquet", "text"})

# Transient columns the source-side dedup stamps on the batch and
# drops again before the MERGE — never persisted to bronze.
_MERGE_SEQUENCE_COLUMN = "_janus_merge_seq"
_MERGE_RANK_COLUMN = "_janus_merge_rank"


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
        intent: BronzeWriteIntent | None = None,
        partition_by: tuple[str, ...] | None = None,
        options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, str] | None = None,
        records_written: int | None = None,
        count_records: bool = False,
        apply_repartition: bool = True,
    ) -> WriteResult:
        """Write ``dataframe`` to ``zone``.

        For bronze Iceberg writes the resolved :class:`BronzeWriteIntent` decides how the
        table is written. ``intent`` may be passed explicitly (the materializer does so);
        when it is not, the writer derives it from the plan via
        :func:`resolve_bronze_write_intent`, so a hand-rolled ``write(df, plan, "bronze")``
        is idempotent too. ``mode`` is only consulted for the non-upsert bronze branches and
        for non-bronze zones. Non-bronze / non-iceberg zones ignore ``intent`` entirely.
        """
        resolved_target = self.storage_layout.resolve_output(plan, zone)
        resolved_format = format_name or resolved_target.format
        if zone == "bronze" and resolved_format.strip().lower() == "iceberg":
            return self._write_bronze_iceberg(
                dataframe,
                plan,
                configured_format=resolved_format,
                mode=mode,
                intent=intent,
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
        intent: BronzeWriteIntent | None,
        partition_by: tuple[str, ...] | None,
        metadata: Mapping[str, str] | None,
        records_written: int | None,
        count_records: bool,
        apply_repartition: bool,
    ) -> WriteResult:
        if configured_format.strip().lower() != "iceberg":
            raise ValueError("bronze outputs must use the 'iceberg' format")

        resolved_intent = intent or resolve_bronze_write_intent(plan)
        write_mode = mode or plan.source_config.spark.write_mode
        partition_columns = partition_by or plan.source_config.spark.partition_by

        prepared_frame = _rebalance_for_write(
            dataframe,
            target_partitions=plan.source_config.spark.repartition,
            apply_repartition=apply_repartition,
        )

        table_identifier = bronze_table_identifier(
            plan.bronze_output.path,
            fallback_name=plan.source.source_id,
            namespace=plan.bronze_output.namespace,
            table_name=plan.bronze_output.table_name,
        )
        namespace_identifier = table_identifier.rsplit(".", 1)[0]

        if resolved_intent.strategy == "merge_on_keys":
            return self._merge_bronze_iceberg(
                prepared_frame,
                plan,
                resolved_intent,
                table_identifier=table_identifier,
                namespace_identifier=namespace_identifier,
                partition_columns=partition_columns,
                metadata=metadata,
                records_written=records_written,
                count_records=count_records,
            )

        resolved_records_written = records_written
        if count_records and resolved_records_written is None:
            resolved_records_written = prepared_frame.count()

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

    def _merge_bronze_iceberg(
        self,
        prepared_frame: DataFrame,
        plan: ExecutionPlan,
        intent: BronzeWriteIntent,
        *,
        table_identifier: str,
        namespace_identifier: str,
        partition_columns: tuple[str, ...],
        metadata: Mapping[str, str] | None,
        records_written: int | None,
        count_records: bool,
    ) -> WriteResult:
        """Make a bronze write idempotent on ``intent.merge_keys`` via Iceberg ``MERGE INTO``."""
        merge_keys = intent.merge_keys
        _reject_complex_merge_keys(prepared_frame, merge_keys)

        deduped, duplicates_dropped, deduped_count = _dedupe_for_merge(
            prepared_frame, merge_keys, count=count_records
        )

        write_metadata: dict[str, str] = dict(metadata or {})
        write_metadata["write_strategy"] = "merge_on_keys"
        write_metadata["merge_keys"] = ",".join(merge_keys)
        if duplicates_dropped is not None:
            write_metadata["in_batch_duplicates_dropped"] = str(duplicates_dropped)

        resolved_records_written = records_written
        if count_records and resolved_records_written is None:
            resolved_records_written = deduped_count

        if count_records and resolved_records_written == 0:
            write_metadata["write_skipped"] = "empty_batch"
            return WriteResult.from_plan(
                plan,
                "bronze",
                path=table_identifier,
                format_name="iceberg",
                mode=intent.reported_mode,
                records_written=0,
                partition_by=partition_columns,
                metadata=write_metadata,
            )

        spark = deduped.sparkSession
        table_exists = spark.catalog.tableExists(table_identifier)

        merge_source = deduped.localCheckpoint(eager=True) if table_exists else deduped

        temp_view_name = f"janus_bronze_{plan.source.source_id}_{uuid4().hex}"
        merge_source.createOrReplaceTempView(temp_view_name)
        try:
            spark.sql(
                f"CREATE NAMESPACE IF NOT EXISTS {_quote_identifier(namespace_identifier)}"
            )
            quoted_table = _quote_identifier(table_identifier)
            quoted_temp_view = _quote_identifier(temp_view_name)

            if not table_exists:
                partition_clause = _partition_clause(partition_columns)
                spark.sql(
                    f"CREATE TABLE {quoted_table} USING iceberg "
                    f"{partition_clause} AS SELECT * FROM {quoted_temp_view}"
                )
                write_metadata["write_strategy"] = "create"
                write_metadata["requested_strategy"] = "merge_on_keys"
            else:
                evolved_columns = _reconcile_merge_schema(
                    spark, merge_source, table_identifier, plan
                )
                if evolved_columns:
                    write_metadata["schema_evolved_columns"] = ",".join(evolved_columns)
                spark.sql(
                    build_merge_sql(
                        table_identifier=table_identifier,
                        source_view=temp_view_name,
                        merge_keys=merge_keys,
                    )
                )
        finally:
            spark.catalog.dropTempView(temp_view_name)

        return WriteResult.from_plan(
            plan,
            "bronze",
            path=table_identifier,
            format_name="iceberg",
            mode=intent.reported_mode,
            records_written=resolved_records_written,
            partition_by=partition_columns,
            metadata=write_metadata,
        )


def build_merge_sql(
    *,
    table_identifier: str,
    source_view: str,
    merge_keys: Sequence[str],
) -> str:
    """Render the idempotent ``MERGE INTO`` for a bronze upsert.

    Pure — strings in, string out, no Spark. ``<=>`` (null-safe equality) keeps the write
    idempotent even for a null key (the quality gate fails the run afterwards, which is the
    correct division of labour). ``UPDATE SET *`` is last-seen-wins. Aliases are
    ``janus_target`` / ``janus_source`` so a payload column named ``t``/``s`` cannot shadow
    them, and every key column is quoted with the same defence as the table identifier.
    """
    if not merge_keys:
        raise ValueError("merge_on_keys requires at least one merge key")

    quoted_table = _quote_identifier(table_identifier)
    quoted_view = _quote_identifier(source_view)
    conditions = "\n   AND ".join(
        f"janus_target.{_quote_identifier(key)} <=> janus_source.{_quote_identifier(key)}"
        for key in merge_keys
    )
    return (
        f"MERGE INTO {quoted_table} AS janus_target\n"
        f"USING {quoted_view} AS janus_source\n"
        f"ON {conditions}\n"
        "WHEN MATCHED THEN UPDATE SET *\n"
        "WHEN NOT MATCHED THEN INSERT *"
    )


def build_add_columns_sql(
    *,
    table_identifier: str,
    columns: Sequence[tuple[str, str]],
) -> str | None:
    """Render ``ALTER TABLE ... ADD COLUMNS`` for schema evolution, or ``None`` if empty."""
    if not columns:
        return None

    quoted_table = _quote_identifier(table_identifier)
    rendered = ", ".join(
        f"{_quote_identifier(name)} {column_type}" for name, column_type in columns
    )
    return f"ALTER TABLE {quoted_table} ADD COLUMNS ({rendered})"


def _dedupe_for_merge(
    frame: DataFrame,
    merge_keys: tuple[str, ...],
    *,
    count: bool,
) -> tuple[DataFrame, int | None, int | None]:
    """Keep exactly one row per key so MERGE never sees a many-to-one match.

    The survivor is the last-observed row within the batch: ``ingestion_timestamp`` is
    constant within a run, so the real tiebreaker is the monotonic sequence — deterministic
    within the run, which is all "exactly one row per key" needs; ordering across runs is
    MERGE's job. Returns ``(deduped, duplicates_dropped, deduped_count)``; the two counts are
    ``None`` when ``count`` is false so the file family does not pay for an extra action.
    """
    from pyspark.sql.functions import col, monotonically_increasing_id, row_number
    from pyspark.sql.window import Window

    ordering = [
        col("ingestion_timestamp").desc_nulls_last(),
        col(_MERGE_SEQUENCE_COLUMN).asc(),
    ]
    window = Window.partitionBy(*[col(key) for key in merge_keys]).orderBy(*ordering)
    deduped = (
        frame.withColumn(_MERGE_SEQUENCE_COLUMN, monotonically_increasing_id())
        .withColumn(_MERGE_RANK_COLUMN, row_number().over(window))
        .where(col(_MERGE_RANK_COLUMN) == 1)
        .drop(_MERGE_RANK_COLUMN, _MERGE_SEQUENCE_COLUMN)
    )

    if not count:
        return deduped, None, None

    before = frame.count()
    after = deduped.count()
    return deduped, before - after, after


def _reject_complex_merge_keys(frame: DataFrame, merge_keys: tuple[str, ...]) -> None:
    """Reject nested/complex-typed key columns up front with a clear message."""
    from pyspark.sql.types import ArrayType, MapType, StructType

    field_types = {field.name: field.dataType for field in frame.schema.fields}
    offending = [
        key
        for key in merge_keys
        if isinstance(field_types.get(key), (ArrayType, MapType, StructType))
    ]
    if offending:
        raise ValueError(
            "merge keys must be scalar columns; these have nested/complex types and cannot "
            f"be used as idempotency keys: {', '.join(offending)}"
        )


def _reconcile_merge_schema(
    spark: Any,
    deduped: DataFrame,
    table_identifier: str,
    plan: ExecutionPlan,
) -> list[str]:
    """Reconcile source columns absent from the target before ``UPDATE SET *`` runs."""
    source_fields = deduped.schema.fields
    target_columns = set(spark.table(table_identifier).columns)
    missing_in_target = [
        field.name for field in source_fields if field.name not in target_columns
    ]
    if not missing_in_target:
        return []

    if not plan.source_config.quality.allow_schema_evolution:
        raise ValueError(
            f"bronze target {table_identifier!r} is missing columns present in this batch: "
            f"{', '.join(missing_in_target)}; set quality.allow_schema_evolution to add "
            "them automatically"
        )

    source_types = {field.name: field.dataType.simpleString() for field in source_fields}
    add_columns_sql = build_add_columns_sql(
        table_identifier=table_identifier,
        columns=[(name, source_types[name]) for name in missing_in_target],
    )
    if add_columns_sql is not None:
        spark.sql(add_columns_sql)
    return missing_in_target


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
    rendered_columns = ", ".join(_quote_identifier(column) for column in partition_columns)
    return f"PARTITIONED BY ({rendered_columns})"


def _quote_identifier(identifier: str) -> str:
    return ".".join(f"`{part.replace('`', '``')}`" for part in identifier.split("."))
