"""`janus checkpoint`: see where a source's next run starts, and move it with a record.

A checkpoint is the one value, `<metadata>/checkpoints/current.json`, that an incremental
source's next run starts from; every write of it leaves a file under
`checkpoints/history/`. The metadata zone is found by planning the source (D-10), disabled
sources included (D-9), as `janus dead-letters` finds it. No action reads an environment
profile, starts Spark or runs a source.

- `show` prints the stored state and the last N history entries, newest first. A source
  that declares no checkpoint, and one with nothing stored, are healthy answers. A stored
  state that no longer matches the plan is refused with the store's message: the contract
  changed under it, and printing the value would mislead. An unreadable history file is
  skipped with a warning on stderr; it never hides the current state.
- `set` moves the checkpoint through `CheckpointStore.reset`. The new value must keep the
  kind (`datetime`, `decimal` or `text`) of the value it replaces, as the store's own
  `normalize_checkpoint_value` reads it (D-14): across kinds the next run's comparison falls
  back to comparing text. The transition and its direction are printed before anything is
  written (PRD §8 risk 4), to stderr under `--format json` so stdout stays one document.
- `clear` forgets the checkpoint through `CheckpointStore.clear_state`, which records the
  forgotten value before it deletes `current.json` (D-13). With nothing stored it writes
  nothing.

`set` and `clear` require `--reason`. The operator is `resolve_operator()` (D-11) and the
history file is named by the synthesized `manual-...` run id (D-12); there is no `--run-id`.
In `--format json` every action prints the same keys: `history` holds the entries `show`
read, or the one entry `set` or `clear` wrote. Markers are ASCII, as in `janus validate`.

Exit codes: 0 success; 2 for arguments, an unknown source, `set` on a source that declares
no checkpoint, a value of another kind, or a stored state that does not match the plan.
Never 1: this verb executes nothing. Every refusal is printed as worded, and nothing is
written before one.
"""

from __future__ import annotations

import argparse
import functools
import heapq
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.checkpoints import (
    CheckpointHistoryEntry,
    CheckpointState,
    CheckpointStore,
    CheckpointWriteResult,
    compare_checkpoint_values,
    normalize_checkpoint_value,
)
from janus.cli.common import build_parent_parser
from janus.cli.operator import require_reason, resolve_operator
from janus.lineage import MetadataZonePaths, read_json_mapping
from janus.models import ExecutionPlan
from janus.planner import Planner, PlannerError, PlanningRequest
from janus.registry import SourceNotFoundError

ARGUMENT_ERROR = 2
PLAN_ATTRIBUTES = {"trigger": "checkpoint"}
NO_CHECKPOINT = "none"
DEFAULT_HISTORY = 5
EMPTY_CELL = "-"
CLEARED_CELL = "(cleared)"
HISTORY_COLUMNS = ("recorded_at", "decision", "advanced", "previous", "stored", "run")
COLUMN_GAP = "  "
LABEL_WIDTH = 8
DETAIL_INDENT = 6

_EPILOG = (
    "A checkpoint is the value an incremental source's next run starts from. Every action "
    "plans the source to find its metadata zone, disabled sources included; none reads an "
    "environment profile or runs anything. set and clear are recorded in the history "
    "directory with the operator and the reason. `janus checkpoint ACTION --help` describes "
    "each action."
)

_SET_EPILOG = (
    "The value is taken as written, never coerced, and must keep the kind of the stored "
    "value (a timestamp, a number or text): the next run compares the two to decide whether "
    "it advanced, and across kinds it would compare text. With nothing stored, any non-empty "
    "value is accepted. The previous value and the direction of the move are printed before "
    "anything is written. The next run advances from the value set: a higher value moves the "
    "checkpoint on, a lower one is retained."
)


class _Refusal(Exception):
    """Printed to stderr as worded, with exit 2. Raised before anything is written."""


def configure(parser: argparse.ArgumentParser) -> None:
    parser.epilog = _EPILOG
    actions = parser.add_subparsers(title="actions", dest="action", required=True)

    show = _add_action(
        actions,
        "show",
        _show,
        summary="Print a source's checkpoint and its last history entries.",
        description="Print the stored checkpoint and the last N history entries, newest "
        "first. A source that declares no checkpoint, or has none stored, is a healthy answer "
        "and exits 0.",
    )
    show.add_argument(
        "--history",
        type=_history_count,
        default=DEFAULT_HISTORY,
        metavar="N",
        help=f"How many history entries to print, newest first. Defaults to {DEFAULT_HISTORY}; "
        "0 prints none and leaves the history directory unread. Otherwise every file in it is "
        "read, because entries are ordered by when they were recorded, not by name.",
    )
    _add_format(show)

    set_action = _add_action(
        actions,
        "set",
        _set,
        summary="Move the checkpoint to a value, backwards or forwards, with a recorded reason.",
        description="Move the checkpoint to a value and write a history entry naming the "
        "previous value, the operator and the reason.",
        epilog=_SET_EPILOG,
    )
    set_action.add_argument(
        "--to",
        required=True,
        metavar="VALUE",
        help="The new checkpoint value, as the checkpoint field holds it. Join a value that "
        "begins with '-' to the option with '='.",
    )
    _add_reason(set_action, "Why the checkpoint moves")
    _add_format(set_action)

    clear = _add_action(
        actions,
        "clear",
        _clear,
        summary="Forget the checkpoint, so the next run starts without one, with a recorded "
        "reason.",
        description="Forget the checkpoint. The forgotten value is recorded in the history "
        "directory before the state file is deleted; with nothing stored, nothing is written.",
    )
    _add_reason(clear, "Why the checkpoint is forgotten")
    _add_format(clear)


def checkpoint_command(args: argparse.Namespace) -> int:
    """Run the action the parser selected; each one is bound to its own parser."""
    perform: Callable[[argparse.Namespace], int] = args.perform
    try:
        return perform(args)
    except _Refusal as exc:
        print(str(exc), file=sys.stderr)
        return ARGUMENT_ERROR


def _add_action(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    perform: Callable[[argparse.ArgumentParser, argparse.Namespace], int],
    *,
    summary: str,
    description: str,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    action = actions.add_parser(
        name,
        help=summary,
        description=description,
        epilog=epilog,
        parents=[build_parent_parser(suppress_defaults=True)],
        allow_abbrev=False,
    )
    action.add_argument(
        "--source-id",
        required=True,
        help="Configured source_id whose checkpoint this acts on; disabled sources included.",
    )
    # Bound to its own parser, so an action refuses a bad value the way argparse does.
    action.set_defaults(perform=functools.partial(perform, action))
    return action


def _add_reason(parser: argparse.ArgumentParser, purpose: str) -> None:
    parser.add_argument(
        "--reason",
        required=True,
        help=f"{purpose}. Recorded in the history file with the operator ($JANUS_OPERATOR, "
        "else $USER). Required: every state change is recorded.",
    )


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format. Defaults to table.",
    )


def _history_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        count = -1
    if count < 0:
        raise argparse.ArgumentTypeError(f"must be a whole number, 0 or more: {value!r}")
    return count


def _show(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    plan = _plan(args, project_root)
    if not _declares_checkpoint(plan):
        document = _document(plan, project_root)
        print(_render_json(document) if args.format == "json" else _no_checkpoint_line(plan))
        return 0

    document = _document(plan, project_root, _load(plan))
    if args.history:
        history, document["history_total"] = _read_history(plan, project_root, args.history)
        document["history"] = [entry.to_dict() for entry in history]
    print(_render_json(document) if args.format == "json" else _render_show(document))
    return 0


def _set(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    reason = require_reason(parser, args.reason)
    value = args.to.strip()
    if not value:
        parser.error("--to must not be empty: a checkpoint value is never blank")
    project_root = args.project_root.resolve()
    plan = _plan(args, project_root)
    if not _declares_checkpoint(plan):
        raise _Refusal(
            f"Source {plan.source.source_id!r} declares no checkpoint (strategy: "
            f"{plan.checkpoint_strategy}); no run would read a value set here"
        )

    current = _load(plan)
    _require_comparable(plan, value, current)
    direction = _direction(value, current)
    _announce(args, _transition_line(plan, current, value, direction))
    operator = resolve_operator()
    result = _record(CheckpointStore().reset, plan, value, operator=operator, reason=reason)

    document = _change_document(
        plan, project_root, result, current, direction=direction, operator=operator, reason=reason
    )
    print(_render_json(document) if args.format == "json" else _render_change(document))
    return 0


def _clear(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    reason = require_reason(parser, args.reason)
    project_root = args.project_root.resolve()
    plan = _plan(args, project_root)
    current = _load(plan) if _declares_checkpoint(plan) else None
    if current is None:
        document = _document(plan, project_root, decision="skipped")
        if args.format == "json":
            print(_render_json(document))
        elif not _declares_checkpoint(plan):
            print(f"{_no_checkpoint_line(plan)}; nothing was cleared")
        else:
            print(f"no checkpoint recorded for {plan.source.source_id}; nothing was cleared")
        return 0

    _announce(
        args,
        f"{plan.source.source_id}: clearing {plan.checkpoint_field} {current.checkpoint_value} "
        f"({plan.checkpoint_strategy}; the next run starts without a checkpoint)",
    )
    operator = resolve_operator()
    result = _record(CheckpointStore().clear_state, plan, operator=operator, reason=reason)

    document = _change_document(
        plan, project_root, result, current, direction=None, operator=operator, reason=reason
    )
    print(_render_json(document) if args.format == "json" else _render_change(document))
    return 0


def _plan(args: argparse.Namespace, project_root: Path) -> ExecutionPlan:
    """Plan the source for its metadata zone, disabled sources included."""
    try:
        planned = Planner().plan(
            PlanningRequest.create(
                source_id=args.source_id,
                environment=args.environment,
                project_root=project_root,
                attributes=PLAN_ATTRIBUTES,
                include_disabled=True,
            )
        )
    except (FileNotFoundError, PlannerError, SourceNotFoundError, ValueError) as exc:
        raise _Refusal(str(exc)) from exc
    return planned.plan


def _declares_checkpoint(plan: ExecutionPlan) -> bool:
    """The store's own test: with no strategy or no field, it reads and writes nothing."""
    return plan.checkpoint_strategy != NO_CHECKPOINT and plan.checkpoint_field is not None


def _load(plan: ExecutionPlan) -> CheckpointState | None:
    """The stored state, refused when it no longer matches the plan (the store's check)."""
    try:
        return CheckpointStore().load(plan)
    except (OSError, ValueError) as exc:
        path = MetadataZonePaths.from_plan(plan).checkpoint_state_path
        raise _Refusal(f"Cannot read the checkpoint state at {path}: {exc}") from exc


def _require_comparable(plan: ExecutionPlan, value: str, current: CheckpointState | None) -> None:
    label = f"{plan.checkpoint_field} of {plan.source.source_id!r}"
    if current is not None:
        kind = normalize_checkpoint_value(value)[0]
        stored_kind = normalize_checkpoint_value(current.checkpoint_value)[0]
        if kind != stored_kind:
            raise _Refusal(
                f"Refusing to set {label} to {value!r}: it reads as a {kind} value, and the "
                f"stored checkpoint {current.checkpoint_value!r} is a {stored_kind} value. "
                "The next run would compare the two as text, so the new value must be a "
                f"{stored_kind} value too"
            )
    reference = value if current is None else current.checkpoint_value
    try:
        compare_checkpoint_values(value, reference)
    except ArithmeticError as exc:
        raise _Refusal(
            f"Refusing to set {label} to {value!r}: the next run could not compare it "
            f"({type(exc).__name__})"
        ) from exc


def _direction(value: str, current: CheckpointState | None) -> str | None:
    if current is None:
        return None
    comparison = compare_checkpoint_values(value, current.checkpoint_value)
    return "backwards" if comparison < 0 else "forwards" if comparison > 0 else "unchanged"


def _transition_line(
    plan: ExecutionPlan, current: CheckpointState | None, value: str, direction: str | None
) -> str:
    previous = "(none)" if current is None else current.checkpoint_value
    note = {
        None: "no checkpoint was stored",
        "backwards": "this moves the checkpoint backwards",
        "forwards": "this moves the checkpoint forwards",
        "unchanged": "the value is unchanged",
    }[direction]
    return (
        f"{plan.source.source_id}: {plan.checkpoint_field} {previous} -> {value} "
        f"({plan.checkpoint_strategy}; {note})"
    )


def _announce(args: argparse.Namespace, line: str) -> None:
    """The change, printed before it is written. Under JSON it goes to stderr: a format
    must not be a way to skip it."""
    print(line, file=sys.stderr if args.format == "json" else sys.stdout, flush=True)


def _record(
    operation: Callable[..., CheckpointWriteResult],
    plan: ExecutionPlan,
    *values: str,
    operator: str,
    reason: str,
) -> CheckpointWriteResult:
    """Call `reset` or `clear_state`; their refusals are this command's refusals."""
    recorded_at = datetime.now(tz=UTC).replace(microsecond=0)
    try:
        return operation(plan, *values, operator=operator, reason=reason, recorded_at=recorded_at)
    except (OSError, ValueError) as exc:
        raise _Refusal(
            f"Cannot record the checkpoint change of {plan.source.source_id!r}: {exc}"
        ) from exc


def _read_history(
    plan: ExecutionPlan, project_root: Path, limit: int
) -> tuple[list[CheckpointHistoryEntry], int]:
    """The newest `limit` entries, and how many entries were readable.

    File names are run ids of several shapes (planner, batch, `manual-...`, an operator's own
    `--run-id` on `janus run`), so only the recorded instant orders them, and every file is
    read. `--history 0` is how a caller skips the directory.
    """
    history_dir = MetadataZonePaths.from_plan(plan).checkpoint_history_dir
    entries = [
        entry
        for path in sorted(history_dir.glob("*.json"))
        if (entry := _read_history_entry(path, project_root)) is not None
    ]
    newest = heapq.nlargest(limit, entries, key=lambda entry: (entry.recorded_at, entry.run_id))
    return newest, len(entries)


def _read_history_entry(path: Path, project_root: Path) -> CheckpointHistoryEntry | None:
    try:
        payload = read_json_mapping(path)
        return None if payload is None else CheckpointHistoryEntry.from_dict(payload)
    except (OSError, ValueError) as exc:
        print(
            "warning: skipped the unreadable checkpoint history file "
            f"{_display_path(path, project_root)}: {exc}",
            file=sys.stderr,
        )
        return None


def _document(
    plan: ExecutionPlan,
    project_root: Path,
    state: CheckpointState | None = None,
    *,
    decision: str | None = None,
) -> dict[str, Any]:
    """Every key every action prints; `value`, `run_id` and `updated_at` are the state after
    the action. An action fills in the rest of its own keys and adds none."""
    declared = _declares_checkpoint(plan)
    state_path = MetadataZonePaths.from_plan(plan).checkpoint_state_path
    return {
        "source_id": plan.source.source_id,
        "checkpoint_field": plan.checkpoint_field,
        "checkpoint_strategy": plan.checkpoint_strategy,
        "value": state.checkpoint_value if state is not None else None,
        "run_id": state.run_id if state is not None else None,
        "updated_at": state.updated_at.isoformat() if state is not None else None,
        "previous_value": None,
        "decision": decision,
        "direction": None,
        "operator": None,
        "reason": None,
        "current_path": _display_path(state_path, project_root) if declared else None,
        "history_path": None,
        "history": [],
        "history_total": None,
    }


def _change_document(
    plan: ExecutionPlan,
    project_root: Path,
    result: CheckpointWriteResult,
    current: CheckpointState | None,
    *,
    direction: str | None,
    operator: str,
    reason: str,
) -> dict[str, Any]:
    """What `set` or `clear` wrote, with the history entry read back from its file."""
    document = _document(plan, project_root, result.state, decision=result.decision)
    document.update(
        previous_value=current.checkpoint_value if current is not None else None,
        direction=direction,
        operator=operator,
        reason=reason,
    )
    if result.history_path is not None:
        document["history_path"] = _display_path(result.history_path, project_root)
        written = _read_history_entry(result.history_path, project_root)
        document["history"] = [] if written is None else [written.to_dict()]
    return document


def _no_checkpoint_line(plan: ExecutionPlan) -> str:
    return f"{plan.source.source_id} declares no checkpoint (strategy: {plan.checkpoint_strategy})"


def _display_path(path: Path, project_root: Path) -> str:
    """Relative to the project root when inside it; an absolute metadata zone as it is."""
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _render_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True)


def _render_show(document: dict[str, Any]) -> str:
    field_row = ("field", document["checkpoint_field"], "strategy", document["checkpoint_strategy"])
    if document["value"] is None:
        lines = [
            f"no checkpoint recorded for {document['source_id']}",
            *_pairs([field_row]),
            _field("state", f"{document['current_path']} (absent)"),
        ]
    else:
        lines = [
            f"{document['source_id']} - checkpoint",
            *_pairs([field_row, ("value", document["value"], "updated", document["updated_at"])]),
            _field("run", document["run_id"]),
            _field("state", document["current_path"]),
        ]
    return "\n".join((*lines, *_render_history(document)))


def _render_history(document: dict[str, Any]) -> list[str]:
    entries, total = document["history"], document["history_total"]
    if total is None:
        return []
    if not entries:
        return ["", "no checkpoint history recorded"]

    rows = [HISTORY_COLUMNS, *(_history_row(entry) for entry in entries)]
    widths = [max(len(row[index]) for row in rows) for index in range(len(HISTORY_COLUMNS) - 1)]
    lines = [
        "",
        f"history (last {len(entries)} of {total}, newest first)",
        _table_row(rows[0], widths),
    ]
    for row, entry in zip(rows[1:], entries, strict=True):
        lines.append(_table_row(row, widths))
        if entry["decision"] == "reset":
            # The reason an operator gave is the point of the record: print it with the row.
            metadata = entry["metadata"]
            reason = " ".join(metadata.get("reason", EMPTY_CELL).split())
            operator = metadata.get("operator", EMPTY_CELL)
            lines.append(f'{" " * DETAIL_INDENT}operator {operator}; reason "{reason}"')
    return lines


def _history_row(entry: dict[str, Any]) -> tuple[str, ...]:
    cleared = entry["metadata"].get("cleared") == "true"
    return (
        entry["recorded_at"],
        entry["decision"],
        "yes" if entry["advanced"] else "no",
        entry.get("previous_value") or EMPTY_CELL,
        CLEARED_CELL if cleared else entry["stored_value"],
        entry["run_id"],
    )


def _render_change(document: dict[str, Any]) -> str:
    state = document["current_path"]
    if document["value"] is None:
        state = f"{state} (deleted)"
    return "\n".join(
        (
            _field("history", document["history_path"] or EMPTY_CELL),
            _field("state", state),
            _field("operator", document["operator"]),
            _field("reason", document["reason"]),
        )
    )


def _pairs(rows: Sequence[tuple[str, str, str, str]]) -> list[str]:
    """Two `label  value` columns per line, each column padded to a common width."""
    value_width = max(len(row[1]) for row in rows)
    label_width = max(len(row[2]) for row in rows)
    return [
        _field(label, COLUMN_GAP.join((value.ljust(value_width), other.ljust(label_width), more)))
        for label, value, other, more in rows
    ]


def _table_row(row: Sequence[str], widths: Sequence[int]) -> str:
    # The last column is left unpadded, so no line carries trailing whitespace.
    cells = [cell.ljust(width) for cell, width in zip(row[:-1], widths, strict=True)]
    return "  " + COLUMN_GAP.join((*cells, row[-1]))


def _field(label: str, value: str) -> str:
    return f"  {label:<{LABEL_WIDTH}}{COLUMN_GAP}{value}".rstrip()
