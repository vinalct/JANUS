from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    RunContext,
    SourceConfig,
)
from janus.strategies.api import ApiHook, ApiStrategy
from janus.strategies.api.records import _default_records_from_payload
from janus.strategies.common import _raw_run_path_prefix
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter
from tests.support.contracts import CONTRACT_SCHEMA_BLOCK

FIXTURE_ROOT = Path(__file__).parent
ENVIRONMENT_CONFIG = {
    "storage": {
        "root_dir": "data",
        "raw_dir": "data/raw",
        "bronze_dir": "data/bronze",
        "metadata_dir": "data/metadata",
    }
}


def source_mapping(*, spark: dict[str, Any] | None = None, raw_format: str = "json"):
    """A valid mapping without retired schema keys or the legacy repartition knob."""
    return {
        "source_id": "order22_handoff",
        "name": "Order 22 handoff fixture",
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
            "path": "/records",
            "method": "GET",
            "format": raw_format,
            "timeout_seconds": 30,
            "auth": {"type": "none"},
            "pagination": {
                "type": "page_number",
                "page_param": "page",
                "size_param": "page_size",
                "page_size": 10,
            },
            "rate_limit": {"requests_per_minute": None, "concurrency": 1},
        },
        "extraction": {
            "mode": "full_refresh",
            "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
        },
        "schema": dict(CONTRACT_SCHEMA_BLOCK),
        "spark": {"input_format": "json", "write_mode": "overwrite", **(spark or {})},
        "outputs": {
            "raw": {"path": "data/raw/order22", "format": raw_format},
            "bronze": {"path": "data/bronze/order22", "format": "iceberg"},
            "metadata": {"path": "data/metadata/order22", "format": "json"},
        },
        "quality": {},
    }


@dataclass(frozen=True)
class RawZone:
    plan: ExecutionPlan
    layout: StorageLayout
    writer: RawArtifactWriter
    extraction: ExtractionResult
    prefix: Path

    @property
    def run_root(self) -> Path:
        return self.layout.resolve_output(self.plan, "raw").resolved_path / self.prefix

    def strategy(self) -> ApiStrategy:
        return ApiStrategy(storage_layout_factory=lambda plan: self.layout)

    def replay_plan(self) -> ExecutionPlan:
        return replace(self.plan, raw_output=replace(self.plan.raw_output, path=str(self.run_root)))


def build_raw_zone(
    project_root: Path,
    case: str = "single_input",
    *,
    spark: dict[str, Any] | None = None,
    raw_format: str | None = None,
) -> RawZone:
    format_name = raw_format or ("text" if case == "text_input" else "json")
    config = SourceConfig.from_mapping(
        source_mapping(spark=spark, raw_format=format_name), project_root / "source.yaml"
    )
    plan = ExecutionPlan.from_source_config(
        config,
        RunContext.create(
            run_id="fixture",
            environment="local",
            project_root=project_root,
            started_at=datetime(2026, 9, 16, 12, tzinfo=UTC),
        ),
    )
    layout = StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, project_root)
    prefix = _raw_run_path_prefix(plan)
    assert prefix is not None
    writer = RawArtifactWriter(layout, raw_path_prefix=prefix)
    artifacts: list[ExtractedArtifact] = []
    record_count = 0
    for path in sorted((FIXTURE_ROOT / case).rglob("page-*")):
        relative = path.relative_to(FIXTURE_ROOT / case)
        if format_name == "text":
            persisted = writer.write_text(plan, relative, path.read_text(encoding="utf-8"))
        elif format_name == "binary":
            persisted = writer.write_bytes(plan, relative.with_suffix(".bin"), path.read_bytes())
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = _default_records_from_payload(payload)
            record_count += len(records)
            if format_name == "jsonl":
                persisted = writer.write_json_lines(plan, relative.with_suffix(".jsonl"), records)
            else:
                persisted = writer.write_json(plan, relative, payload)
        artifacts.append(persisted.artifact)
    assert artifacts, f"fixture {case!r} contains no pages"
    extraction = ExtractionResult.from_plan(
        plan,
        artifacts=tuple(artifacts),
        records_extracted=record_count,
        checkpoint_value="fixture-checkpoint",
        metadata={"fixture": case},
    )
    return RawZone(plan, layout, writer, extraction, prefix)


class RecordsHook(ApiHook):
    """Response-dependent extraction cannot be reconstructed from saved pages."""

    def extract_records(self, plan, request, response, payload):
        raise AssertionError("extract_records must not be called at handoff time")


class HandoffHook(ApiHook):
    """Like IBGE, refuses JSONL input and supplies its own normalized artifact."""

    def __init__(self, zone: RawZone):
        self.zone = zone
        self.received: ExtractionResult | None = None

    def on_normalization_handoff(self, plan, extraction_result):
        self.received = extraction_result
        assert all(artifact.format == "json" for artifact in extraction_result.artifacts)
        artifact = self.zone.writer.write_json_lines(
            plan, "normalized/hook-records.jsonl", [{"id": "hook-output"}]
        ).artifact
        return replace(extraction_result, artifacts=(artifact,)).with_metadata(
            "fixture_hook", "true"
        )


def read_records(artifacts) -> list[Any]:
    return [
        json.loads(line)
        for artifact in artifacts
        for line in Path(artifact.path).read_text(encoding="utf-8").splitlines()
    ]


def page_snapshot(zone: RawZone) -> dict[str, tuple[bytes, bytes, int]]:
    return {
        artifact.path: (
            Path(artifact.path).read_bytes(),
            Path(artifact.path + ".sha256").read_bytes(),
            Path(artifact.path).stat().st_mtime_ns,
        )
        for artifact in zone.extraction.artifacts
    }
