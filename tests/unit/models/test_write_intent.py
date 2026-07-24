from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.models import (
    BRONZE_WRITE_STRATEGIES,
    BronzeWriteIntent,
    ExecutionPlan,
    RunContext,
    resolve_bronze_write_intent,
)
from janus.models.write_intent import (
    NORMALIZATION_METADATA_COLUMNS as WRITE_INTENT_METADATA_COLUMNS,
)
from janus.normalizers.base import NORMALIZATION_METADATA_COLUMNS
from janus.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _plan_with(
    *,
    mode: str,
    write_mode: str,
    unique_fields: tuple[str, ...] = (),
    partition_by: tuple[str, ...] = (),
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
) -> ExecutionPlan:
    """Build an execution plan whose contract facts drive the resolver, no Spark or YAML."""
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        source_config,
        extraction=replace(
            source_config.extraction,
            mode=mode,
            checkpoint_field=checkpoint_field,
            checkpoint_strategy=checkpoint_strategy,
        ),
        spark=replace(
            source_config.spark,
            write_mode=write_mode,
            partition_by=partition_by,
        ),
        quality=replace(source_config.quality, unique_fields=unique_fields),
    )
    run_context = RunContext.create(
        run_id="run-write-intent-001",
        environment="local",
        project_root=PROJECT_ROOT,
        started_at=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


# --- 1. Resolution matrix -----------------------------------


@pytest.mark.parametrize(
    ("mode", "write_mode", "unique_fields", "partition_by", "checkpoint_field", "expected"),
    [
        ("full_refresh", "ignore", (), (), None, ("skip_if_exists", (), "ignore")),
        ("snapshot", "ignore", (), (), None, ("skip_if_exists", (), "ignore")),
        ("incremental", "ignore", ("id",), (), "id", ("skip_if_exists", (), "ignore")),
        ("full_refresh", "overwrite", (), (), None, ("replace_table", (), "overwrite")),
        ("full_refresh", "append", (), (), None, ("insert", (), "append")),
        ("snapshot", "overwrite", (), (), None, ("replace_table", (), "overwrite")),
        ("snapshot", "append", (), (), None, ("insert", (), "append")),
        (
            "incremental",
            "append",
            ("id",),
            ("ingestion_date",),
            "updated_at",
            ("merge_on_keys", ("id",), "upsert"),
        ),
        (
            "incremental",
            "append",
            ("event_id",),
            ("event_date",),
            "event_date",
            ("overwrite_partitions", ("event_id",), "upsert"),
        ),
        (
            "incremental",
            "overwrite",
            ("id",),
            (),
            "updated_at",
            ("replace_table", (), "overwrite"),
        ),
    ],
)
def test_resolution_matrix(
    mode: str,
    write_mode: str,
    unique_fields: tuple[str, ...],
    partition_by: tuple[str, ...],
    checkpoint_field: str | None,
    expected: tuple[str, tuple[str, ...], str],
) -> None:
    expected_strategy, expected_keys, expected_reported_mode = expected
    plan = _plan_with(
        mode=mode,
        write_mode=write_mode,
        unique_fields=unique_fields,
        partition_by=partition_by,
        checkpoint_field=checkpoint_field,
        checkpoint_strategy="max_value" if mode == "incremental" else "none",
    )

    intent = resolve_bronze_write_intent(plan)

    assert intent.strategy == expected_strategy
    assert intent.merge_keys == expected_keys
    assert intent.reported_mode == expected_reported_mode
    assert intent.configured_mode == write_mode


def test_incremental_append_without_unique_fields_is_rejected() -> None:
    plan = _plan_with(
        mode="incremental",
        write_mode="append",
        unique_fields=(),
        checkpoint_field="updated_at",
        checkpoint_strategy="max_value",
    )

    with pytest.raises(ValueError, match=r"require quality\.unique_fields"):
        resolve_bronze_write_intent(plan)


def test_incremental_overwrite_reason_names_the_contradiction() -> None:
    plan = _plan_with(
        mode="incremental",
        write_mode="overwrite",
        unique_fields=("id",),
        checkpoint_field="updated_at",
        checkpoint_strategy="max_value",
    )

    intent = resolve_bronze_write_intent(plan)

    assert intent.strategy == "replace_table"
    assert "prior windows are discarded" in intent.reason


# --- 2. Legacy parity over the real conf/sources tree ----------------------------------


def test_every_non_incremental_source_keeps_its_configured_write_mode() -> None:
    registry = load_registry(PROJECT_ROOT)
    checked = 0
    for source_config in registry.list_sources(enabled_only=False):
        if source_config.extraction.mode == "incremental":
            continue
        run_context = RunContext.create(
            run_id="run-parity-001",
            environment="local",
            project_root=PROJECT_ROOT,
            started_at=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
        )
        plan = ExecutionPlan.from_source_config(source_config, run_context)

        intent = resolve_bronze_write_intent(plan)

        assert intent.reported_mode == source_config.spark.write_mode, source_config.source_id
        assert not intent.is_upsert, source_config.source_id
        checked += 1

    assert checked > 0, "expected at least one non-incremental source in conf/sources"


# --- 3. Alignment predicate ------------------------------------------------------------


@pytest.mark.parametrize(
    ("partition_by", "unique_fields", "checkpoint_field", "expected_strategy"),
    [
        (("ingestion_date",), ("id",), "updated_at", "merge_on_keys"),
        ((), ("id",), "updated_at", "merge_on_keys"),
        (("event_date",), ("id",), "event_date", "overwrite_partitions"),
        (("region",), ("event_id",), None, "merge_on_keys"),
    ],
)
def test_alignment_predicate_selects_the_fast_path_only_when_bounded(
    partition_by: tuple[str, ...],
    unique_fields: tuple[str, ...],
    checkpoint_field: str | None,
    expected_strategy: str,
) -> None:
    plan = _plan_with(
        mode="incremental",
        write_mode="append",
        unique_fields=unique_fields,
        partition_by=partition_by,
        checkpoint_field=checkpoint_field,
        checkpoint_strategy="max_value",
    )

    intent = resolve_bronze_write_intent(plan)

    assert intent.strategy == expected_strategy


# --- 4. Metadata-column guard ----------------------------------------------------------


def test_metadata_columns_stay_pinned_to_the_normalizer() -> None:
    assert WRITE_INTENT_METADATA_COLUMNS == NORMALIZATION_METADATA_COLUMNS


# --- 5. for_batch ----------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["replace_table", "create"])
def test_replace_and_create_downgrade_to_insert_after_first_batch(strategy: str) -> None:
    intent = BronzeWriteIntent(
        strategy=strategy,
        configured_mode="overwrite",
        reason="full_refresh+overwrite: each run replaces the bronze table",
    )

    downgraded = intent.for_batch(2)

    assert downgraded.strategy == "insert"
    assert "downgraded to insert" in downgraded.reason
    assert intent.for_batch(1) is intent


@pytest.mark.parametrize(
    "intent",
    [
        BronzeWriteIntent(strategy="merge_on_keys", configured_mode="append", merge_keys=("id",)),
        BronzeWriteIntent(
            strategy="overwrite_partitions", configured_mode="append", merge_keys=("id",)
        ),
        BronzeWriteIntent(strategy="insert", configured_mode="append"),
        BronzeWriteIntent(strategy="skip_if_exists", configured_mode="ignore"),
    ],
)
def test_idempotent_and_append_strategies_pass_through_every_batch(
    intent: BronzeWriteIntent,
) -> None:
    assert intent.for_batch(1) is intent
    assert intent.for_batch(2) == intent


# --- 6. Invariants ---------------------------------------------------------------------


def test_upsert_strategy_requires_merge_keys() -> None:
    with pytest.raises(ValueError, match="require merge_keys"):
        BronzeWriteIntent(strategy="merge_on_keys", configured_mode="append", merge_keys=())


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="strategy must be one of"):
        BronzeWriteIntent(strategy="truncate_load", configured_mode="append")


def test_all_declared_strategies_are_recognised() -> None:
    assert "merge_on_keys" in BRONZE_WRITE_STRATEGIES
    assert "overwrite_partitions" in BRONZE_WRITE_STRATEGIES
