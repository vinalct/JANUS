"""Spark writers for the bronze zone and for path-based Spark outputs.

A full refresh normally uses ``INSERT OVERWRITE`` to preserve snapshot ancestry.
Contract compatibility decides schema changes before each write, and a declared
breaking change on a full refresh may choose ``REPLACE TABLE``. The writer records
the operation and its effect on history in write metadata.

The cost of retaining history is real and deliberate: **snapshots are never
expired here.** Every full refresh keeps the previous run's data files until someone expires
them, so the bronze zone now grows per run where it previously did not. For the current source
set — small daily API refreshes plus periodic archives — that is negligible; at CNPJ scale it
is the first thing to revisit. Expiration needs a retention window, a schedule and somewhere to
run, which makes it an operational policy rather than a writer concern.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from janus.models import (
    BronzeWriteIntent,
    ExecutionPlan,
    WriteResult,
    resolve_bronze_write_intent,
)
from janus.models.data_contracts import spark_sql_type
from janus.quality.contract_checks import MissingContractError
from janus.utils.storage import StorageLayout, bronze_table_identifier
from janus.writers.errors import SchemaEvolutionRefusedError
from janus.writers.evolution import PLAN_METADATA_KEY, EvolutionPlan, plan_schema_evolution
from janus.writers.identifiers import build_bronze_temp_view_name, quote_identifier
from janus.writers.overwrite import build_create_table_as_select_sql
from janus.writers.overwrite_exec import execute_full_refresh_overwrite
from janus.writers.schema_ddl import (
    build_add_columns_sql,
    build_alter_column_type_sql,
    build_insert_into_sql,
    project_append_columns,
)
from janus.writers.table_state import (
    read_contract_stamp,
    read_live_columns,
    read_target_columns,
    stamp_contract,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

SUPPORTED_SPARK_WRITE_FORMATS = frozenset({"csv", "json", "jsonl", "parquet", "text"})

# Every non-merge bronze strategy maps to exactly one Iceberg write mode
_BRONZE_STRATEGY_ICEBERG_MODE = {
    "skip_if_exists": "ignore",
    "insert": "append",
    "create": "append",
    "replace_table": "overwrite",
}

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
        batch_index: int = 1,
    ) -> WriteResult:
        """Write ``dataframe`` to ``zone``.

        For bronze Iceberg writes the resolved :class:`BronzeWriteIntent` is the *only*
        thing that decides how the table is written. ``intent`` may be passed explicitly
        (the materializer does so); when it is not, the writer derives it from the plan via
        :func:`resolve_bronze_write_intent`, so a hand-rolled ``write(df, plan, "bronze")``
        is idempotent too. ``mode`` is ignored for bronze Iceberg writes — the intent always
        wins — and is only consulted for non-bronze / non-iceberg zones, which ignore
        ``intent`` entirely.
        """
        if zone == "bronze" and plan.data_contract is None:
            raise MissingContractError(plan.source.source_id)
        resolved_target = self.storage_layout.resolve_output(plan, zone)
        resolved_format = format_name or resolved_target.format
        if zone == "bronze" and resolved_format.strip().lower() == "iceberg":
            return self._write_bronze_iceberg(
                dataframe,
                plan,
                configured_format=resolved_format,
                intent=intent,
                partition_by=partition_by,
                metadata=metadata,
                records_written=records_written,
                count_records=count_records,
                apply_repartition=apply_repartition,
                batch_index=batch_index,
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
            metadata=(
                {**(metadata or {}), PLAN_METADATA_KEY: "none"} if zone == "bronze" else metadata
            ),
        )

    def _write_bronze_iceberg(
        self,
        dataframe: DataFrame,
        plan: ExecutionPlan,
        *,
        configured_format: str,
        intent: BronzeWriteIntent | None,
        partition_by: tuple[str, ...] | None,
        metadata: Mapping[str, str] | None,
        records_written: int | None,
        count_records: bool,
        apply_repartition: bool,
        batch_index: int,
    ) -> WriteResult:
        contract = plan.data_contract
        if contract is None:
            raise MissingContractError(plan.source.source_id)
        if configured_format.strip().lower() != "iceberg":
            raise ValueError("bronze outputs must use the 'iceberg' format")

        resolved_intent = intent or resolve_bronze_write_intent(plan)
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
                batch_index=batch_index,
            )

        effective_mode = _BRONZE_STRATEGY_ICEBERG_MODE.get(resolved_intent.strategy)
        if effective_mode is None:
            raise ValueError(f"unsupported bronze write strategy: {resolved_intent.strategy!r}")

        resolved_records_written = records_written
        if count_records and resolved_records_written is None:
            resolved_records_written = prepared_frame.count()

        write_metadata: dict[str, str] = dict(metadata or {})
        write_metadata[PLAN_METADATA_KEY] = "none"
        spark = prepared_frame.sparkSession
        temp_view_name = build_bronze_temp_view_name(plan.source.source_id)

        prepared_frame.createOrReplaceTempView(temp_view_name)
        try:
            spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {quote_identifier(namespace_identifier)}")

            table_exists = spark.catalog.tableExists(table_identifier)

            recorded_stamp = None
            evolution = None
            if table_exists and effective_mode != "ignore":
                evolution, recorded_stamp = _evolve_target_for_batch(
                    spark, table_identifier, plan, resolved_intent, batch_index=batch_index
                )
                write_metadata[PLAN_METADATA_KEY] = evolution.render()

            if effective_mode == "ignore" and table_exists:
                pass
            elif effective_mode == "overwrite" and table_exists:
                overwrite_plan = execute_full_refresh_overwrite(
                    spark,
                    table_identifier,
                    temp_view_name,
                    prepared_frame,
                    partition_columns,
                    evolution,
                )
                write_metadata["overwrite_mechanism"] = overwrite_plan.mechanism
                if not overwrite_plan.preserves_history:
                    write_metadata["history_reset_reason"] = overwrite_plan.reason
                stamp_contract(
                    spark,
                    table_identifier,
                    contract,
                    None if overwrite_plan.mechanism == "replace_table" else recorded_stamp,
                )
            elif effective_mode == "append" and table_exists:
                target_columns = read_target_columns(spark, table_identifier)
                projection = project_append_columns(
                    source_names={field.name for field in prepared_frame.schema.fields},
                    target_names=tuple(name for name, _ in target_columns),
                    required_names=set(contract.required_columns),
                    table_identifier=table_identifier,
                )
                spark.sql(
                    build_insert_into_sql(
                        table_identifier=table_identifier,
                        source_view=temp_view_name,
                        projection=projection,
                    )
                )
                stamp_contract(spark, table_identifier, contract, recorded_stamp)
            else:
                # A first write for any of ignore/append/overwrite creates the table.
                spark.sql(
                    build_create_table_as_select_sql(
                        table_identifier=table_identifier,
                        source_view=temp_view_name,
                        partition_columns=partition_columns,
                    )
                )
                stamp_contract(spark, table_identifier, contract, None)
        finally:
            spark.catalog.dropTempView(temp_view_name)

        return WriteResult.from_plan(
            plan,
            "bronze",
            path=table_identifier,
            format_name="iceberg",
            mode=effective_mode,
            records_written=resolved_records_written,
            partition_by=partition_columns,
            metadata=write_metadata,
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
        batch_index: int,
    ) -> WriteResult:
        """Make a bronze write idempotent on ``intent.merge_keys`` via Iceberg ``MERGE INTO``."""
        contract = plan.data_contract
        if contract is None:
            raise MissingContractError(plan.source.source_id)
        merge_keys = intent.merge_keys
        _reject_complex_merge_keys(prepared_frame, merge_keys)

        deduped, duplicates_dropped, deduped_count = _dedupe_for_merge(
            prepared_frame, merge_keys, count=count_records
        )

        write_metadata: dict[str, str] = dict(metadata or {})
        write_metadata[PLAN_METADATA_KEY] = "none"
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

        temp_view_name = build_bronze_temp_view_name(plan.source.source_id)
        merge_source.createOrReplaceTempView(temp_view_name)
        try:
            spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {quote_identifier(namespace_identifier)}")

            if not table_exists:
                spark.sql(
                    build_create_table_as_select_sql(
                        table_identifier=table_identifier,
                        source_view=temp_view_name,
                        partition_columns=partition_columns,
                    )
                )
                write_metadata["write_strategy"] = "create"
                write_metadata["requested_strategy"] = "merge_on_keys"
                stamp_contract(spark, table_identifier, contract, None)
            else:
                evolution, recorded_stamp = _evolve_target_for_batch(
                    spark, table_identifier, plan, intent, batch_index=batch_index
                )
                write_metadata[PLAN_METADATA_KEY] = evolution.render()
                spark.sql(
                    build_merge_sql(
                        table_identifier=table_identifier,
                        source_view=temp_view_name,
                        merge_keys=merge_keys,
                    )
                )
                stamp_contract(spark, table_identifier, contract, recorded_stamp)
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

    quoted_table = quote_identifier(table_identifier)
    quoted_view = quote_identifier(source_view)
    conditions = "\n   AND ".join(
        f"janus_target.{quote_identifier(key)} <=> janus_source.{quote_identifier(key)}"
        for key in merge_keys
    )
    return (
        f"MERGE INTO {quoted_table} AS janus_target\n"
        f"USING {quoted_view} AS janus_source\n"
        f"ON {conditions}\n"
        "WHEN MATCHED THEN UPDATE SET *\n"
        "WHEN NOT MATCHED THEN INSERT *"
    )


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
        if isinstance(field_types.get(key), ArrayType | MapType | StructType)
    ]
    if offending:
        raise ValueError(
            "merge keys must be scalar columns; these have nested/complex types and cannot "
            f"be used as idempotency keys: {', '.join(offending)}"
        )


def _evolve_target_for_batch(
    spark: Any,
    table_identifier: str,
    plan: ExecutionPlan,
    intent: BronzeWriteIntent,
    *,
    batch_index: int,
) -> tuple[EvolutionPlan, dict[str, str]]:
    """Read the live table once, decide from its contract, and apply allowed DDL."""
    contract = plan.data_contract
    if contract is None:
        raise MissingContractError(plan.source.source_id)
    live_columns = read_live_columns(spark, table_identifier)
    recorded_stamp = read_contract_stamp(spark, table_identifier)
    evolution = plan_schema_evolution(
        contract=contract,
        live_columns=live_columns,
        recorded_contract_version=recorded_stamp.get("janus.contract_version"),
        write_strategy=intent.strategy,
        batch_index=batch_index,
    )
    if evolution.outcome == "refused":
        raise SchemaEvolutionRefusedError(evolution)
    if evolution.outcome == "evolve":
        add_sql = build_add_columns_sql(
            table_identifier=table_identifier,
            columns=[(name, spark_sql_type(type_)) for name, type_ in evolution.add_columns],
        )
        if add_sql is not None:
            spark.sql(add_sql)
        for name, _before, after in evolution.promote_columns:
            spark.sql(
                build_alter_column_type_sql(
                    table_identifier=table_identifier,
                    column=name,
                    spark_type=spark_sql_type(after),
                )
            )
    return evolution, recorded_stamp


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
