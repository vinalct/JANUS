"""Read raw prefixes and run statuses without guessing identities or filesystem ages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

from janus.lineage.persistence import MetadataZonePaths, read_json_mapping
from janus.maintenance.inventory import RawRunPrefixEntry
from janus.models import ExecutionPlan
from janus.strategies.common import raw_run_path_segment
from janus.utils.storage import StorageLayout


def collect_raw_run_statuses(
    plans: Iterable[ExecutionPlan],
) -> dict[str, dict[str, str | None]]:
    """Match in sanitized space; unreadable or conflicting records protect their prefix.

    A running record is also unknown for deletion: only an explicit terminal status
    establishes whether a run belongs in the successful-run floor or can be removed.
    """
    result: dict[str, dict[str, str | None]] = {}
    for plan in plans:
        source_id = plan.source.source_id
        statuses = result.setdefault(source_id, {})
        for path in sorted(MetadataZonePaths.from_plan(plan).runs_dir.glob("*.json")):
            if not path.is_file():
                continue
            segment = raw_run_path_segment(path.stem)
            status = None
            try:
                record = read_json_mapping(path)
                if (
                    record is not None
                    and record.get("run_id") == path.stem
                    and record.get("source_id", source_id) == source_id
                    and record.get("status") in {"succeeded", "failed"}
                ):
                    status = record["status"]
            except (OSError, ValueError, TypeError):
                pass

            statuses[segment] = (
                status if segment not in statuses or statuses[segment] == status else None
            )
    return result


def collect_raw_prefixes(
    plans: Iterable[ExecutionPlan],
    storage_layout: StorageLayout,
    *,
    source_ids: frozenset[str] | None,
    run_status_by_segment: Mapping[str, Mapping[str, str | None]],
) -> tuple[RawRunPrefixEntry, ...]:
    entries = []
    for plan in plans:
        source_id = plan.source.source_id
        if source_ids is not None and source_id not in source_ids:
            continue

        root = storage_layout.resolve_output(plan, "raw").resolved_path
        if root.exists() and any(path.name != "runs" for path in root.iterdir()):
            entries.append(
                RawRunPrefixEntry(root, source_id, "", None, None, root, "flat_layout_present")
            )
        statuses = run_status_by_segment.get(source_id, {})
        for path in sorted(root.glob("runs/ingestion_date=*/run_id=*")):
            if path.is_dir():
                entries.append(_prefix_entry(path, source_id, root, statuses))
    return tuple(entries)


def _prefix_entry(
    path: Path, source_id: str, root: Path, statuses: Mapping[str, str | None]
) -> RawRunPrefixEntry:
    segment = path.name.removeprefix("run_id=")
    day_text = path.parent.name.removeprefix("ingestion_date=")
    try:
        day = date.fromisoformat(day_text)
        if day.isoformat() != day_text or not segment or raw_run_path_segment(segment) != segment:
            raise ValueError("raw prefixes require a canonical date and a normalized run segment")
    except ValueError:
        return RawRunPrefixEntry(path, source_id, segment, None, None, root, "invalid_raw_prefix")
    status = statuses.get(segment)
    succeeded = status == "succeeded" if status in {"succeeded", "failed"} else None
    return RawRunPrefixEntry(path, source_id, segment, day, succeeded, root)
