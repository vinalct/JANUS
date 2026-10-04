"""Frozen input records for retention decisions. Collectors are added separately."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from janus.models import BronzeRetentionConfig


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    snapshot_id: int
    committed_at: datetime
    is_current: bool
    parent_id: int | None = None


@dataclass(frozen=True, slots=True)
class BronzeTableInventory:
    table_identifier: str
    source_ids: tuple[str, ...]
    snapshots: tuple[SnapshotEntry, ...]
    override: BronzeRetentionConfig | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunArtifactEntry:
    kind: str
    path: Path
    source_id: str | None
    run_id: str | None
    timestamp: datetime | None
    read_error: str | None = None


@dataclass(frozen=True, slots=True)
class MetadataZoneInventory:
    artifacts: tuple[RunArtifactEntry, ...]
    protected_paths: frozenset[Path]
    live_raw_run_segments: typing.Mapping[str, str | None]


@dataclass(frozen=True, slots=True)
class EventFileEntry:
    path: Path
    day: date


@dataclass(frozen=True, slots=True)
class RunsTablePartitionEntry:
    emitted_at_day: date
    row_count: int | None = None


@dataclass(frozen=True, slots=True)
class RawRunPrefixEntry:
    path: Path
    source_id: str
    run_segment: str
    ingestion_date: date
    run_succeeded: bool | None


@dataclass(frozen=True, slots=True)
class MaintenanceInventory:
    bronze: tuple[BronzeTableInventory, ...] = ()
    metadata: MetadataZoneInventory | None = None
    lineage_events: tuple[EventFileEntry, ...] = ()
    runs_table: tuple[RunsTablePartitionEntry, ...] = ()
    raw: tuple[RawRunPrefixEntry, ...] = ()
