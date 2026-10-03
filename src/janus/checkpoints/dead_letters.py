from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.lineage.persistence import MetadataZonePaths, read_json_mapping, write_json_atomic
from janus.models import ExecutionPlan


@dataclass(frozen=True, slots=True)
class DeadLetterEntry:
    """One execution item skipped after exhausting request-level retries."""

    item_key: str
    item_type: str
    error_type: str
    error_message: str
    recorded_at: datetime
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.item_key.strip():
            raise ValueError("item_key must not be empty")
        if not self.item_type.strip():
            raise ValueError("item_type must not be empty")
        if not self.error_type.strip():
            raise ValueError("error_type must not be empty")
        if not self.error_message.strip():
            raise ValueError("error_message must not be empty")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeadLetterEntry:
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("dead letter metadata must be a mapping")
        return cls(
            item_key=_require_string(payload, "item_key"),
            item_type=_require_string(payload, "item_type"),
            error_type=_require_string(payload, "error_type"),
            error_message=_require_string(payload, "error_message"),
            recorded_at=_parse_datetime(_require_string(payload, "recorded_at"), "recorded_at"),
            metadata=_freeze_string_mapping(
                {str(key): str(value) for key, value in metadata.items()}
            ),
        )

    def metadata_as_dict(self) -> dict[str, str]:
        return dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_key": self.item_key,
            "item_type": self.item_type,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "recorded_at": self.recorded_at.isoformat(),
            "metadata": self.metadata_as_dict(),
        }


@dataclass(frozen=True, slots=True)
class DeadLetterState:
    """Source-scoped dead-letter state persisted in the metadata zone."""

    run_id: str
    source_id: str
    strategy_family: str
    strategy_variant: str
    updated_at: datetime
    entries: tuple[DeadLetterEntry, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.strategy_family.strip():
            raise ValueError("strategy_family must not be empty")
        if not self.strategy_variant.strip():
            raise ValueError("strategy_variant must not be empty")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeadLetterState:
        raw_entries = payload.get("entries") or []
        if not isinstance(raw_entries, list):
            raise ValueError("dead letter entries must be a list")
        return cls(
            run_id=_require_string(payload, "run_id"),
            source_id=_require_string(payload, "source_id"),
            strategy_family=_require_string(payload, "strategy_family"),
            strategy_variant=_require_string(payload, "strategy_variant"),
            updated_at=_parse_datetime(_require_string(payload, "updated_at"), "updated_at"),
            entries=tuple(DeadLetterEntry.from_dict(entry) for entry in raw_entries),
        )

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def item_keys(self) -> frozenset[str]:
        return frozenset(entry.item_key for entry in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_id": self.source_id,
            "strategy_family": self.strategy_family,
            "strategy_variant": self.strategy_variant,
            "updated_at": self.updated_at.isoformat(),
            "entries": [entry.to_dict() for entry in self.entries],
        }


class DeadLetterReleaseError(ValueError):
    """A release that cannot be applied as asked. Raised before anything is written."""


@dataclass(frozen=True, slots=True)
class DeadLetterReleaseRecord:
    """One operator release: what was let go, by whom, why, and what remained."""

    source_id: str
    released_at: datetime
    operator: str
    reason: str
    released_entries: tuple[DeadLetterEntry, ...]
    remaining_item_keys: tuple[str, ...]
    state_run_id: str

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.operator.strip():
            raise ValueError("operator must not be empty")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if not self.state_run_id.strip():
            raise ValueError("state_run_id must not be empty")
        if not self.released_entries:
            raise ValueError("released_entries must not be empty")
        if self.released_at.tzinfo is None or self.released_at.utcoffset() is None:
            raise ValueError("released_at must be timezone-aware")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "released_at": self.released_at.isoformat(),
            "operator": self.operator,
            "reason": self.reason,
            "state_run_id": self.state_run_id,
            "released_entries": [entry.to_dict() for entry in self.released_entries],
            "remaining_item_keys": list(self.remaining_item_keys),
        }


@dataclass(slots=True)
class DeadLetterStore:
    """Persist dead-lettered execution items so resume can skip them safely."""

    def load(self, plan: ExecutionPlan) -> DeadLetterState | None:
        payload = read_json_mapping(self.path(plan))
        if payload is None:
            return None

        state = DeadLetterState.from_dict(payload)
        if state.source_id != plan.source.source_id:
            return None
        return state

    def record(
        self,
        plan: ExecutionPlan,
        *,
        item_key: str,
        item_type: str,
        error: Exception,
        metadata: Mapping[str, str] | None = None,
        recorded_at: datetime | None = None,
    ) -> DeadLetterState:
        normalized_item_key = item_key.strip()
        if not normalized_item_key:
            raise ValueError("item_key must not be empty")

        existing_state = self.load(plan)
        if existing_state is not None and normalized_item_key in existing_state.item_keys:
            return existing_state

        resolved_recorded_at = recorded_at or datetime.now(tz=UTC)
        if resolved_recorded_at.tzinfo is None or resolved_recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")

        entry = DeadLetterEntry(
            item_key=normalized_item_key,
            item_type=item_type,
            error_type=type(error).__name__,
            error_message=(str(error).strip() or type(error).__name__),
            recorded_at=resolved_recorded_at,
            metadata=_freeze_string_mapping(metadata),
        )
        entries = (existing_state.entries if existing_state is not None else ()) + (entry,)
        state = DeadLetterState(
            run_id=plan.run_context.run_id,
            source_id=plan.source.source_id,
            strategy_family=plan.source.strategy,
            strategy_variant=plan.source.strategy_variant,
            updated_at=resolved_recorded_at,
            entries=entries,
        )
        write_json_atomic(self.path(plan), state.to_dict())
        return state

    def release(
        self,
        plan: ExecutionPlan,
        *,
        item_keys: Sequence[str] | None = None,
        operator: str,
        reason: str,
        released_at: datetime | None = None,
    ) -> DeadLetterReleaseRecord:
        """Remove exactly these entries, atomically, and record the removal.

        Deletes `current.json` when nothing remains, so `load` returns None and a
        resuming run sees no skip set — the same end state `clear` produces, reached
        without forgetting the entries that were not named. `item_keys=None` releases
        every entry.

        The history record is written first. If that write fails, the state is untouched
        and the release did not happen; the reverse order could drop entries with no record
        of who let them go. Every refusal is raised before anything is written.
        """
        resolved_released_at = released_at or datetime.now(tz=UTC)
        if resolved_released_at.tzinfo is None or resolved_released_at.utcoffset() is None:
            raise ValueError("released_at must be timezone-aware")

        state = self.load(plan)
        if state is None or not state.entries:
            raise DeadLetterReleaseError(
                f"No dead letters are recorded for source {plan.source.source_id!r} "
                f"at {self.path(plan)}; there is nothing to release"
            )

        released, remaining = _split_released_entries(state, item_keys)
        record = DeadLetterReleaseRecord(
            source_id=state.source_id,
            released_at=resolved_released_at,
            operator=operator.strip(),
            reason=reason.strip(),
            released_entries=released,
            remaining_item_keys=tuple(entry.item_key for entry in remaining),
            state_run_id=state.run_id,
        )
        history_path = self.history_path(plan, record)
        if history_path.exists():
            raise DeadLetterReleaseError(
                f"A release of dead letters recorded by run {state.run_id!r} is already on "
                f"record at {history_path}; release again in a later second rather than "
                "overwrite it"
            )

        write_json_atomic(history_path, record.to_dict())
        if remaining:
            write_json_atomic(
                self.path(plan),
                replace(state, updated_at=resolved_released_at, entries=remaining).to_dict(),
            )
        else:
            self.clear(plan)
        return record

    def clear(self, plan: ExecutionPlan) -> None:
        path = self.path(plan)
        if path.exists():
            path.unlink()

    def path(self, plan: ExecutionPlan) -> Path:
        return MetadataZonePaths.from_plan(plan).dead_letter_state_path

    def history_path(self, plan: ExecutionPlan, record: DeadLetterReleaseRecord) -> Path:
        """`dead_letters/history/<released_at>-<state run id>.json`.

        The run id passes through the planner's `normalize_run_id_segment`, so a state file
        whose `run_id` holds a slash or `..` cannot steer the record out of the directory.
        The import is deferred: the planner sits above this store (it binds the strategies
        that record dead letters), and `janus.checkpoints` is imported by modules that must
        never load it.
        """
        from janus.planner import normalize_run_id_segment

        timestamp = record.released_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{timestamp}-{normalize_run_id_segment(record.state_run_id)}"
        return MetadataZonePaths.from_plan(plan).dead_letter_history_path(name)


def _split_released_entries(
    state: DeadLetterState, item_keys: Sequence[str] | None
) -> tuple[tuple[DeadLetterEntry, ...], tuple[DeadLetterEntry, ...]]:
    """Split the entries into (released, remaining), both in their recorded order.

    A key that is not recorded refuses the whole release: releasing the rest of a request
    that held a typo is how an operator lets go of the wrong entry.
    """
    if item_keys is None:
        return state.entries, ()

    requested = tuple(dict.fromkeys(key.strip() for key in item_keys))
    if not requested or "" in requested:
        raise ValueError(
            "item_keys must name at least one non-empty key; pass None to release every entry"
        )

    unknown = [key for key in requested if key not in state.item_keys]
    if unknown:
        raise DeadLetterReleaseError(
            f"Cannot release dead letters for source {state.source_id!r}: unknown item "
            f"key(s) {_render_keys(unknown)}; recorded item keys: "
            f"{_render_keys(entry.item_key for entry in state.entries)}"
        )

    released_keys = frozenset(requested)
    return (
        tuple(entry for entry in state.entries if entry.item_key in released_keys),
        tuple(entry for entry in state.entries if entry.item_key not in released_keys),
    )


def _render_keys(keys: Iterable[str]) -> str:
    return ", ".join(repr(key) for key in keys)


def _freeze_string_mapping(values: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()

    frozen_items: list[tuple[str, str]] = []
    for key, value in values.items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if not normalized_key:
            raise ValueError("mapping keys must be non-empty strings")
        if not normalized_value:
            raise ValueError("mapping values must be non-empty strings")
        frozen_items.append((normalized_key, normalized_value))
    return tuple(sorted(frozen_items))


def _parse_datetime(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid datetime for {field_name}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed


def _require_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None:
        raise ValueError(f"Missing field: {field_name}")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Field {field_name} must not be empty")
    return normalized


def can_continue_after_dead_letter(
    *,
    total_item_count: int,
    dead_letter_count: int,
    dead_letter_max_items: int,
) -> bool:
    return total_item_count > 1 and dead_letter_count <= dead_letter_max_items
