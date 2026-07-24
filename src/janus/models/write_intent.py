from __future__ import annotations

from dataclasses import dataclass, replace

from janus.models.contracts import ExecutionPlan

BRONZE_WRITE_STRATEGIES = frozenset(
    {
        "insert",  # INSERT INTO            (append, table exists)
        "create",  # CREATE TABLE ... AS    (table absent)
        "replace_table",  # REPLACE TABLE ... AS   (full refresh)
        "skip_if_exists",  # ignore mode, table exists
        "merge_on_keys",  # MERGE INTO ... ON keys 
        "overwrite_partitions",  # partition overwrite    
    }
)

_UPSERT_STRATEGIES = frozenset({"merge_on_keys", "overwrite_partitions"})

NORMALIZATION_METADATA_COLUMNS = (
    "janus_run_id",
    "janus_source_id",
    "janus_source_name",
    "janus_environment",
    "janus_strategy_family",
    "janus_strategy_variant",
    "ingestion_timestamp",
    "ingestion_date",
)


@dataclass(frozen=True, slots=True)
class BronzeWriteIntent:
    """How one run must write bronze, derived once from the source contract."""

    strategy: str  # one of BRONZE_WRITE_STRATEGIES
    configured_mode: str  # spark.write_mode, verbatim (audit trail)
    merge_keys: tuple[str, ...] = ()  # quality.unique_fields when upserting
    partition_columns: tuple[str, ...] = ()  # spark.partition_by
    reason: str = ""  # human-readable derivation, for logs/metadata

    def __post_init__(self) -> None:
        if self.strategy not in BRONZE_WRITE_STRATEGIES:
            allowed = ", ".join(sorted(BRONZE_WRITE_STRATEGIES))
            raise ValueError(f"strategy must be one of: {allowed}")
        if self.strategy in _UPSERT_STRATEGIES and not self.merge_keys:
            raise ValueError("upsert strategies require merge_keys")

    @property
    def is_upsert(self) -> bool:
        """Return whether this run de-duplicates bronze on its declared keys."""
        return self.strategy in _UPSERT_STRATEGIES

    @property
    def reported_mode(self) -> str:
        """Value carried into WriteResult.mode"""
        return "upsert" if self.is_upsert else self.configured_mode

    def for_batch(self, batch_index: int) -> BronzeWriteIntent:
        """Second and later batches of a multi-batch file handoff never re-replace."""
        if batch_index <= 1:
            return self
        if self.strategy in {"replace_table", "create"}:
            downgrade_reason = (
                f"{self.reason}; batch {batch_index}: downgraded to insert so a multi-batch "
                "handoff does not re-replace the table"
            ).lstrip("; ")
            return replace(self, strategy="insert", reason=downgrade_reason)
        return self


def resolve_bronze_write_intent(plan: ExecutionPlan) -> BronzeWriteIntent:
    """Derive the bronze write decision from the source contract alone.

    Pure: no Spark, no I/O, no ``source_id`` conditionals. The four contract facts that
    already exist — ``extraction.mode``, ``spark.write_mode``, ``quality.unique_fields`` and
    ``spark.partition_by`` — fully determine the write. Every non-``incremental`` source
    resolves to today's exact behaviour.
    """
    source_config = plan.source_config
    mode = source_config.extraction.mode
    write_mode = source_config.spark.write_mode
    unique_fields = source_config.quality.unique_fields
    partition_columns = source_config.spark.partition_by

    if write_mode == "ignore":
        return BronzeWriteIntent(
            strategy="skip_if_exists",
            configured_mode=write_mode,
            partition_columns=partition_columns,
            reason="write_mode 'ignore': an existing bronze table is left untouched",
        )

    if mode in {"full_refresh", "snapshot"}:
        if write_mode == "overwrite":
            return BronzeWriteIntent(
                strategy="replace_table",
                configured_mode=write_mode,
                partition_columns=partition_columns,
                reason=f"{mode}+overwrite: each run replaces the bronze table",
            )
        return BronzeWriteIntent(
            strategy="insert",
            configured_mode=write_mode,
            partition_columns=partition_columns,
            reason=f"{mode}+append: rows are appended to the bronze table",
        )

    if write_mode == "overwrite":
        return BronzeWriteIntent(
            strategy="replace_table",
            configured_mode=write_mode,
            partition_columns=partition_columns,
            reason=(
                "incremental+overwrite: each run replaces the table; prior windows "
                "are discarded"
            ),
        )

    if not unique_fields:
        raise ValueError(
            "incremental sources require quality.unique_fields to derive an idempotent "
            f"bronze write; none declared for {source_config.source_id!r}"
        )

    merge_keys = tuple(unique_fields)
    if _partitions_align_with_window(plan, merge_keys):
        return BronzeWriteIntent(
            strategy="overwrite_partitions",
            configured_mode=write_mode,
            merge_keys=merge_keys,
            partition_columns=partition_columns,
            reason=(
                "incremental+append with partitions bounded by the re-fetched window: "
                "whole partitions are overwritten"
            ),
        )

    return BronzeWriteIntent(
        strategy="merge_on_keys",
        configured_mode=write_mode,
        merge_keys=merge_keys,
        partition_columns=partition_columns,
        reason=(
            "incremental+append: MERGE on "
            f"{', '.join(merge_keys)} keeps the re-fetched window idempotent"
        ),
    )


def _partitions_align_with_window(
    plan: ExecutionPlan, merge_keys: tuple[str, ...]
) -> bool:
    """Return whether re-ingesting the window rewrites *whole* partitions."""
    
    partitions = plan.source_config.spark.partition_by
    if not partitions:
        return False
    if any(column in NORMALIZATION_METADATA_COLUMNS for column in partitions):
        return False  # ingestion_date & friends are run stamps, never windows
    checkpoint_field = plan.source_config.extraction.checkpoint_field
    allowed = set(merge_keys) | ({checkpoint_field} if checkpoint_field else set())
    return all(column in allowed for column in partitions)
