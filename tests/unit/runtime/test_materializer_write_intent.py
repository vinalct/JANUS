"""The resolved write intent is the only thing that decides how bronze is written.

These tests drive :class:`BronzeMaterializer` with fakes — no Spark — and assert on the
``intent`` the materializer hands the writer per batch. They pin the contract: 
one resolution per run, the preserved multi-batch downgrade for full refresh, and
merge-on-keys with no downgrade for incremental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from janus.models import (
    BronzeWriteIntent,
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    RunContext,
    SourceConfig,
    WriteResult,
)
from janus.planner import PlannedRun
from janus.runtime import materialize as materialize_module
from janus.runtime.materialize import BronzeMaterializer
from janus.utils.storage import StorageLayout

ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}


@dataclass(slots=True)
class FakeReader:
    def read_extraction_result(
        self, spark, extraction_result, format_name=None, schema=None, options=None
    ):
        del spark, extraction_result, format_name, schema, options
        return object()


@dataclass(slots=True)
class FakeNormalizer:
    def normalize(self, dataframe, plan):
        del dataframe, plan
        return object()


@dataclass(slots=True)
class IntentRecordingWriter:
    """Captures the ``intent`` handed to every bronze write."""

    bronze_path: str
    intents: list[BronzeWriteIntent] = field(default_factory=list)

    def write(self, dataframe, plan, zone, *, intent=None, count_records=False, **kwargs):
        del dataframe, count_records, kwargs
        assert intent is not None, "materializer must always pass an intent for bronze"
        self.intents.append(intent)
        return WriteResult.from_plan(
            plan,
            zone,
            path=self.bronze_path,
            format_name="iceberg",
            mode=intent.reported_mode,
            records_written=None,
            partition_by=intent.partition_columns,
        )


def _materialize(planned_run, plan, handoff, tmp_path):
    writer = IntentRecordingWriter(bronze_path=plan.bronze_output.path)
    materializer = BronzeMaterializer(
        reader=FakeReader(),
        normalizer=FakeNormalizer(),
        writer_factory=lambda storage_layout: writer,
    )
    materializer.materialize(
        planned_run,
        plan,
        SimpleNamespace(),  # spark — untouched by the fakes
        handoff,
        StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path),
        None,
    )
    return writer


def test_full_refresh_single_batch_resolves_to_replace_table(tmp_path):
    plan, planned_run = _plan(tmp_path, mode="full_refresh", write_mode="overwrite")
    handoff = _handoff(plan, artifact_count=1)

    writer = _materialize(planned_run, plan, handoff, tmp_path)

    assert [intent.strategy for intent in writer.intents] == ["replace_table"]
    assert writer.intents[0].reported_mode == "overwrite"


def test_full_refresh_multi_batch_preserves_the_insert_downgrade(tmp_path):
    plan, planned_run = _plan(
        tmp_path, mode="full_refresh", write_mode="overwrite", strategy_family="file"
    )
    # 11 artifacts over a batch limit of 5 => batches of 5, 5, 1.
    handoff = _handoff(plan, artifact_count=11)

    writer = _materialize(planned_run, plan, handoff, tmp_path)

    assert [intent.strategy for intent in writer.intents] == [
        "replace_table",
        "insert",
        "insert",
    ]
    # Batch 1 overwrites the table — now via INSERT OVERWRITE, which keeps the snapshot log —
    # and batches 2+ append into what it just wrote instead of overwriting it again.
    assert [intent.reported_mode for intent in writer.intents] == ["overwrite"] * 3
    assert all(
        "downgraded to insert" in intent.reason for intent in writer.intents[1:]
    )


def test_incremental_with_keys_merges_every_batch_without_downgrade(tmp_path):
    plan, planned_run = _plan(
        tmp_path,
        mode="incremental",
        write_mode="append",
        unique_fields=["event_id"],
        strategy_family="file",
    )
    handoff = _handoff(plan, artifact_count=11)

    writer = _materialize(planned_run, plan, handoff, tmp_path)

    assert [intent.strategy for intent in writer.intents] == ["merge_on_keys"] * 3
    assert {intent.merge_keys for intent in writer.intents} == {("event_id",)}


def test_ignore_mode_skips_if_exists_on_every_batch(tmp_path):
    plan, planned_run = _plan(
        tmp_path, mode="full_refresh", write_mode="ignore", strategy_family="file"
    )
    handoff = _handoff(plan, artifact_count=11)

    writer = _materialize(planned_run, plan, handoff, tmp_path)

    assert [intent.strategy for intent in writer.intents] == ["skip_if_exists"] * 3


def test_intent_is_resolved_exactly_once_per_run(tmp_path, monkeypatch):
    plan, planned_run = _plan(
        tmp_path, mode="full_refresh", write_mode="overwrite", strategy_family="file"
    )
    handoff = _handoff(plan, artifact_count=11)

    real_resolver = materialize_module.resolve_bronze_write_intent
    calls: list[ExecutionPlan] = []

    def spy(resolved_plan):
        calls.append(resolved_plan)
        return real_resolver(resolved_plan)

    monkeypatch.setattr(materialize_module, "resolve_bronze_write_intent", spy)

    writer = _materialize(planned_run, plan, handoff, tmp_path)

    # Three batches, but the resolver runs once — per-batch intent comes from `for_batch`.
    assert len(calls) == 1
    assert len(writer.intents) == 3


def _handoff(plan: ExecutionPlan, *, artifact_count: int) -> ExtractionResult:
    raw_root = Path(plan.run_context.project_root) / "data" / "raw"
    artifacts = tuple(
        ExtractedArtifact(
            path=str(raw_root / f"page-{index:04d}.json"),
            format="json",
            checksum=f"sum{index}",
        )
        for index in range(artifact_count)
    )
    return ExtractionResult.from_plan(plan, artifacts=artifacts, records_extracted=artifact_count)


def _plan(
    tmp_path: Path,
    *,
    mode: str,
    write_mode: str,
    unique_fields: list[str] | None = None,
    strategy_family: str = "api",
) -> tuple[ExecutionPlan, PlannedRun]:
    source_config = _source_config(
        tmp_path, mode=mode, write_mode=write_mode, unique_fields=unique_fields
    )
    run_context = RunContext.create(
        run_id="run-materializer-intent-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 7, 8, 12, 0, tzinfo=UTC),
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)
    planned_run = PlannedRun(
        plan=plan,
        strategy=SimpleNamespace(strategy_family=strategy_family),
        hook=None,
    )
    return plan, planned_run


def _source_config(
    tmp_path: Path,
    *,
    mode: str,
    write_mode: str,
    unique_fields: list[str] | None,
) -> SourceConfig:
    source_id = "materializer_intent_fixture"
    unique = unique_fields or []
    extraction: dict[str, Any] = {
        "mode": mode,
        "dead_letter_max_items": 0,
        "retry": {"max_attempts": 3, "backoff_strategy": "fixed", "backoff_seconds": 1},
    }
    if mode == "incremental":
        extraction["checkpoint_field"] = "event_date"
        extraction["checkpoint_strategy"] = "max_value"
    payload: dict[str, Any] = {
        "source_id": source_id,
        "name": source_id,
        "owner": "janus",
        "enabled": True,
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "federation_level": "federal",
        "domain": "example",
        "public_access": True,
        "access": {
            "base_url": "https://example.invalid",
            "path": "/events",
            "method": "GET",
            "format": "json",
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 10,
            },
            "rate_limit": {"requests_per_minute": None, "concurrency": 1, "backoff_seconds": 5},
        },
        "extraction": extraction,
        "schema": {"mode": "infer"},
        "spark": {
            "input_format": "json",
            "write_mode": write_mode,
            "repartition": 1,
            "partition_by": [],
        },
        "outputs": {
            "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
            "bronze": {
                "path": f"data/bronze/example/{source_id}",
                "format": "iceberg",
                "namespace": "bronze_test",
                "table_name": source_id,
            },
            "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
        },
        "quality": {
            "required_fields": ["event_id", "event_date"],
            "unique_fields": unique,
            "allow_schema_evolution": True,
        },
    }
    config_path = tmp_path / "conf" / "sources" / f"{source_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return SourceConfig.from_mapping(payload, config_path)
