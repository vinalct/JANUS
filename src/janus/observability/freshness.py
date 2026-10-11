"""Per-source run-history reads and pure evaluation of declared freshness windows."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from janus.lineage.persistence import read_json_mapping
from janus.utils.environment import resolve_project_path

if TYPE_CHECKING:
    from janus.registry import SourceRegistry

FRESH = "fresh"
STALE = "stale"
NEVER_RAN = "never_ran"
UNDECLARED = "undeclared"
BREACHING_OUTCOMES = frozenset({STALE, NEVER_RAN})


@dataclass(frozen=True, slots=True)
class SourceFreshness:
    """One source's answer, with the evidence needed to render or alert on it."""

    source_id: str
    outcome: str
    max_age_seconds: int | None
    last_success_at: datetime | None
    age_seconds: float | None
    runs_scanned: int
    unreadable_run_files: int
    metadata_path: str

    @property
    def is_breach(self) -> bool:
        return self.outcome in BREACHING_OUTCOMES


@dataclass(frozen=True, slots=True)
class LatestSuccess:
    """The latest successful completion and counts for the entire scanned history."""

    last_success_at: datetime | None
    runs_scanned: int
    unreadable_run_files: int


def evaluate_freshness(
    *, max_age_seconds: int | None, last_success_at: datetime | None, now: datetime
) -> tuple[str, float | None]:
    """Return the outcome and elapsed UTC seconds without reading a clock or any evidence.

    Both instants must be timezone-aware. Equality with the window is fresh, as
    is a future success; its negative age is preserved to expose clock disagreement.
    An undeclared window still reports the age when a successful run is known.
    """
    _validate_timezone_aware("now", now)
    age_seconds = None
    if last_success_at is not None:
        _validate_timezone_aware("last_success_at", last_success_at)
        age_seconds = (now.astimezone(UTC) - last_success_at.astimezone(UTC)).total_seconds()
    if max_age_seconds is None:
        return UNDECLARED, age_seconds
    if age_seconds is None:
        return NEVER_RAN, None
    return (STALE if age_seconds > max_age_seconds else FRESH), age_seconds


def read_latest_success(runs_dir: Path) -> LatestSuccess:
    """Read every ``*.json`` candidate and take the maximum succeeded ``ended_at``.

    A missing directory is empty history. Atomic ``.json.<uuid>.tmp`` staging
    files are excluded by the glob. Malformed or unreadable files are counted and
    skipped; failed and running runs do not need a completion timestamp.
    The scan is deliberately uncapped: retention bounds the history, and a cap
    could hide the source's actual latest success.
    """
    last_success_at = None
    runs_scanned = 0
    unreadable_run_files = 0
    for path in runs_dir.glob("*.json"):
        runs_scanned += 1
        try:
            document = read_json_mapping(path)
            if document is None or not isinstance(document.get("status"), str):
                raise ValueError("Run metadata must have a status")
            if document["status"] != "succeeded":
                continue
            ended_at = datetime.fromisoformat(document["ended_at"])
            _validate_timezone_aware("ended_at", ended_at)
            ended_at = ended_at.astimezone(UTC)
        except (OSError, ValueError, TypeError, KeyError):
            unreadable_run_files += 1
            continue
        if last_success_at is None or ended_at > last_success_at:
            last_success_at = ended_at
    return LatestSuccess(last_success_at, runs_scanned, unreadable_run_files)


def evaluate_sources(
    registry: SourceRegistry,
    project_root: Path,
    *,
    now: datetime,
    source_ids: Collection[str] | None = None,
) -> tuple[SourceFreshness, ...]:
    """Evaluate selected registry sources in lexicographic ``source_id`` order.

    Disabled sources are included. Each source uses its own configured metadata
    path, without planning. Shared paths are scanned independently, using all
    runs in that directory rather than filtering their embedded source ids.
    Unknown selections raise the registry's named lookup error.
    """
    _validate_timezone_aware("now", now)
    sources = (
        registry.list_sources(enabled_only=False)
        if source_ids is None
        else tuple(
            registry.get_source(source_id, include_disabled=True)
            for source_id in sorted(set(source_ids))
        )
    )
    results = []
    for config in sorted(sources, key=lambda source: source.source_id):
        metadata_dir = resolve_project_path(project_root, config.outputs.metadata.path)
        latest = read_latest_success(metadata_dir / "runs")
        freshness = config.operations.freshness
        max_age_seconds = None if freshness is None else freshness.max_age_seconds
        outcome, age_seconds = evaluate_freshness(
            max_age_seconds=max_age_seconds, last_success_at=latest.last_success_at, now=now
        )
        results.append(
            SourceFreshness(
                source_id=config.source_id,
                outcome=outcome,
                max_age_seconds=max_age_seconds,
                last_success_at=latest.last_success_at,
                age_seconds=age_seconds,
                runs_scanned=latest.runs_scanned,
                unreadable_run_files=latest.unreadable_run_files,
                metadata_path=str(metadata_dir),
            )
        )
    return tuple(results)


def _validate_timezone_aware(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
