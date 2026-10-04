from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from janus.models import ExecutionPlan
from janus.utils.environment import resolve_project_path


@dataclass(frozen=True, slots=True)
class MetadataZonePaths:
    """Resolved metadata-zone paths shared by run metadata, lineage, and checkpoints."""

    base_dir: Path
    runs_dir: Path
    lineage_dir: Path
    checkpoints_dir: Path
    checkpoint_history_dir: Path
    dead_letters_dir: Path
    dead_letter_history_dir: Path

    @classmethod
    def from_plan(cls, plan: ExecutionPlan) -> MetadataZonePaths:
        base_dir = resolve_project_path(plan.run_context.project_root, plan.metadata_output.path)
        return cls(
            base_dir=base_dir,
            runs_dir=base_dir / "runs",
            lineage_dir=base_dir / "lineage",
            checkpoints_dir=base_dir / "checkpoints",
            checkpoint_history_dir=base_dir / "checkpoints" / "history",
            dead_letters_dir=base_dir / "dead_letters",
            dead_letter_history_dir=base_dir / "dead_letters" / "history",
        )

    def run_metadata_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def lineage_path(self, run_id: str) -> Path:
        return self.lineage_dir / f"{run_id}.json"

    @property
    def validations_dir(self) -> Path:
        return self.base_dir / "validations"

    @property
    def checkpoint_state_path(self) -> Path:
        return self.checkpoints_dir / "current.json"

    def checkpoint_history_path(self, run_id: str) -> Path:
        return self.checkpoint_history_dir / f"{run_id}.json"

    @property
    def dead_letter_state_path(self) -> Path:
        return self.dead_letters_dir / "current.json"

    def dead_letter_history_path(self, name: str) -> Path:
        return self.dead_letter_history_dir / f"{name}.json"


def read_json_mapping(path: Path) -> dict[str, Any] | None:
    """Load one JSON document when present and validate the top-level shape."""
    if not path.exists():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected a JSON object at {path}")
    return dict(raw)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write one JSON document atomically to keep metadata files rerun-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    temporary_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path
