from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from janus.lineage.persistence import MetadataZonePaths, write_json_atomic
from janus.models import ExecutionPlan
from janus.quality.models import ValidationReport


@dataclass(frozen=True, slots=True)
class PersistedValidationReport:
    """Validation report plus its persisted metadata-zone path."""

    report: ValidationReport
    path: Path


@dataclass(slots=True)
class ValidationReportStore:
    """Persist validation outcomes as run-scoped JSON artifacts in the metadata zone."""

    directory_name: str = "validations"

    def write(self, plan: ExecutionPlan, report: ValidationReport) -> Path:
        if report.run_id != plan.run_context.run_id:
            raise ValueError("report.run_id must match plan.run_context.run_id")
        metadata_paths = MetadataZonePaths.from_plan(plan)
        target_path = metadata_paths.validations_dir / f"{report.run_id}.json"
        return write_json_atomic(target_path, report.to_dict())
