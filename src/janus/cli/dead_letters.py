"""`janus dead-letters`: see what a source's runs gave up on, release it, replay it.

A dead letter is one item (a request input, a download candidate, a catalog entity) that a
run skipped after exhausting its retries. The state lives at
`<metadata>/dead_letters/current.json`, and the metadata zone is found by planning the
source (D-10): `outputs.metadata.path` is configured per source and may be absolute, so a
path derived here from `--project-root` would be a second containment rule. Planning is
pure, reads no profile and starts nothing; it includes disabled sources (D-9), because
state outlives the enabled flag.

- `list` prints every entry whole. The error message carries the response excerpt the
  transport redacted when the entry was recorded; it is the only record of *why*, so it is
  neither truncated nor redacted again here. No state is a healthy answer and exits 0.
- `release` drops exactly the named entries through `DeadLetterStore.release`, which writes
  the history record before it rewrites or deletes `current.json`. Releasing everything
  takes `--all`; a `release` with no selector is refused.
- `replay` is the release and its run as one command (Q4). Without `--execute` it is a dry
  run built on `preview_release`, so it refuses what the release would refuse and writes
  nothing. With `--execute` it releases (`replay: true` in the history record), then hands
  `--execute --resume` to `run_command`, the body of `janus run`: one executor call site
  for both commands. Its stdout is that run's summary; the release report goes to stderr.

Only `replay --execute` reads an environment profile or may start Spark, and it reads the
profile and refuses a disabled source (without `--include-disabled`) before it releases.
Markers are ASCII, as in `janus validate`: a glyph cannot be encoded on every stream.

Exit codes: 0 success; 1 only when `replay --execute` ran and the source failed; 2 for
arguments, configuration, an unknown source or key, or no state where a release needs one.
Every refusal is printed as worded, and nothing is written before one.
"""

from __future__ import annotations

import argparse
import functools
import json
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from janus.checkpoints import (
    DeadLetterReleaseError,
    DeadLetterReleaseRecord,
    DeadLetterStore,
    ExtractionProgressStore,
)
from janus.cli import run
from janus.cli.common import build_parent_parser
from janus.cli.operator import require_reason, resolve_operator
from janus.models import ExecutionPlan, SourceConfig
from janus.observability.openlineage import resolve_openlineage_settings
from janus.planner import Planner, PlannerError, PlanningRequest
from janus.registry import SourceNotFoundError
from janus.utils.environment import load_environment_config

ARGUMENT_ERROR = 2
PLAN_ATTRIBUTES = {"trigger": "dead-letters"}
RELEASE_METADATA = {"source": "cli"}
REPLAY_METADATA = {"source": "cli", "replay": "true"}
EMPTY_CELL = "-"
ENTRY_COLUMNS = ("item_key", "type", "recorded_at")
COLUMN_GAP = "  "
DETAIL_INDENT = 6

_KEY_NOTE = "Quote a key that holds spaces; join one that begins with '-' to the option with '='."

_EPILOG = (
    "A dead letter is an item a run skipped after exhausting its retries. Every action plans "
    "the source to find its metadata zone, disabled sources included. Only replay with "
    "--execute reads an environment profile or runs anything. `janus dead-letters ACTION "
    "--help` describes each action."
)

_REPLAY_EPILOG = (
    "Without --execute this is a dry run: nothing is released, nothing runs and nothing is "
    "written. It prints what a resuming run would retry, what stays skipped and where the "
    "run picks up. With --execute the entries are released, the history record noting the "
    "replay, and the source runs as `janus run` runs it with --execute and --resume. A "
    "release followed by that run is the same operation in two steps."
)


class _Refusal(Exception):
    """Printed to stderr as worded, with exit 2. Raised before anything is written."""


def configure(parser: argparse.ArgumentParser) -> None:
    parser.epilog = _EPILOG
    actions = parser.add_subparsers(title="actions", dest="action", required=True)

    listing = _add_action(
        actions,
        "list",
        _list,
        summary="Print a source's dead letters, each with its error and metadata.",
        description="Print every dead letter recorded for one source, with its error and "
        "metadata. No state is a healthy answer and exits 0.",
    )
    listing.add_argument(
        "--item-key",
        action="append",
        default=[],
        metavar="KEY",
        help=f"Print only this recorded item key; repeat for several. {_KEY_NOTE}",
    )
    _add_format(listing)

    release = _add_action(
        actions,
        "release",
        _release,
        summary="Release dead letters with a recorded reason, so the next resuming run "
        "retries them.",
        description="Remove the named dead letters from the state and write a history record "
        "naming the operator and the reason. The next --execute --resume run retries them and "
        "keeps skipping the rest.",
    )
    _add_selection(release)
    _add_reason(release, "Why they are released. Recorded in the history file with the operator")
    _add_format(release)

    replay = _add_action(
        actions,
        "replay",
        _replay,
        summary="Show what a resume would retry after a release (a dry run, the default), "
        "or release and run it with --execute.",
        description="Show what a release followed by a resuming run would do, and, with "
        "--execute, do it.",
        epilog=_REPLAY_EPILOG,
    )
    _add_selection(replay)
    _add_reason(
        replay,
        "Why they are released. With --execute it is recorded in the history file with the "
        "operator",
    )
    replay.add_argument(
        "--execute",
        action="store_true",
        help="Release, then run the source with --execute --resume. Without it, nothing is "
        "written.",
    )
    replay.add_argument(
        "--include-disabled",
        action="store_true",
        help="Allow --execute to run a source that is configured but disabled.",
    )


def dead_letters_command(args: argparse.Namespace) -> int:
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
        help="Configured source_id whose dead letters to read; disabled sources included.",
    )
    # Bound to its own parser, so an action refuses a bad value the way argparse does.
    action.set_defaults(perform=functools.partial(perform, action))
    return action


def _add_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--item-key",
        action="append",
        metavar="KEY",
        help=f"A recorded item key, as `list` prints it; repeat for several. {_KEY_NOTE}",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Every recorded entry. Releasing them all is asked for by name, never by default.",
    )


def _add_reason(parser: argparse.ArgumentParser, purpose: str) -> None:
    parser.add_argument(
        "--reason",
        required=True,
        help=f"{purpose} ($JANUS_OPERATOR, else $USER). Required: every state change is recorded.",
    )


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format. Defaults to table.",
    )


def _list(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """The recorded state as written, its entries narrowed to the requested keys."""
    requested = _requested_keys(parser, args.item_key)
    project_root = args.project_root.resolve()
    plan = _plan(args, project_root)
    store = DeadLetterStore()
    try:
        state = store.load(plan)
    except (OSError, ValueError) as exc:
        raise _Refusal(f"Cannot read the dead-letter state at {store.path(plan)}: {exc}") from exc

    document: dict[str, Any] = (
        state.to_dict()
        if state is not None
        else {
            "run_id": None,
            "source_id": plan.source.source_id,
            "strategy_family": plan.source.strategy,
            "strategy_variant": plan.source.strategy_variant,
            "updated_at": None,
            "entries": [],
        }
    )
    recorded = [entry["item_key"] for entry in document["entries"]]
    document.update(
        entries=[
            entry
            for entry in document["entries"]
            if not requested or entry["item_key"] in requested
        ],
        recorded_count=len(recorded),
        recorded_item_keys=recorded,
        missing_item_keys=[key for key in requested if key not in recorded],
        state_path=_display_path(store.path(plan), project_root),
    )
    print(_render_json(document) if args.format == "json" else _render_listing(document))
    return 0


def _release(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    reason = require_reason(parser, args.reason)
    item_keys = _selection(parser, args)
    project_root = args.project_root.resolve()
    plan = _plan(args, project_root)
    record = _apply_release(DeadLetterStore().release, plan, item_keys, reason, RELEASE_METADATA)

    document = _release_document(plan, record, project_root)
    if args.format == "json":
        print(_render_json(document))
    else:
        next_step = _resume_command(args.environment, plan.source_config)
        print(_render_release(document, next_step=next_step))
    return 0


def _replay(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    reason = require_reason(parser, args.reason)
    item_keys = _selection(parser, args)
    project_root = args.project_root.resolve()
    plan = _plan(args, project_root)
    if args.execute:
        return _replay_execute(args, plan, item_keys, reason, project_root)

    # The dry run: every check the release makes, through `preview_release`, and no write.
    record = _apply_release(
        DeadLetterStore().preview_release, plan, item_keys, reason, REPLAY_METADATA
    )
    try:
        progress = ExtractionProgressStore().load(plan)
    except (OSError, ValueError) as exc:
        raise _Refusal(
            f"Cannot read the extraction progress of {plan.source.source_id!r}: {exc}"
        ) from exc

    released = [_entry_summary(entry.to_dict()) for entry in record.released_entries]
    lines = [
        f"{record.source_id} - replay dry run: nothing was released and nothing ran",
        *_field("would release", released),
        *_field("stays skipped", record.remaining_item_keys),
        *_field("resume from", [_describe_progress(progress)]),
        *_field("operator", [record.operator]),
        *_field("reason", [record.reason]),
    ]
    if not plan.source_config.enabled and not args.include_disabled:
        note = f"{record.source_id} is disabled: --execute also needs --include-disabled"
        lines.extend(_field("note", [note]))
    lines.extend(
        (
            "",
            "With --execute, the release is recorded and the source runs as",
            f"  {_resume_command(args.environment, plan.source_config)}",
            "which retries the released item(s) and keeps skipping the rest.",
        )
    )
    print("\n".join(lines))
    return 0


def _replay_execute(
    args: argparse.Namespace,
    plan: ExecutionPlan,
    item_keys: Sequence[str] | None,
    reason: str,
    project_root: Path,
) -> int:
    """Release, then run the source exactly as `janus run --execute --resume` does.

    The run must carry `resume=true`: without it `ResumeState.load` clears both stores, so
    the release would be pointless and every other dead letter forgotten. `run_command`
    sets it from `--resume`, the flag the two-step form passes. The profile is read here
    first, creating nothing, so a misspelt `--environment` is refused before the release.
    """
    if not plan.source_config.enabled and not args.include_disabled:
        raise _Refusal(
            f"Source {plan.source.source_id!r} is configured but disabled; "
            "replay --execute requires --include-disabled to run it"
        )
    try:
        resolve_openlineage_settings(load_environment_config(args.environment, project_root))
    except (FileNotFoundError, ValueError) as exc:
        raise _Refusal(str(exc)) from exc
    record = _apply_release(DeadLetterStore().release, plan, item_keys, reason, REPLAY_METADATA)
    report = _render_release(_release_document(plan, record, project_root), next_step=None)
    print(report, file=sys.stderr)

    # Each value is bound to its flag with `=`, so one beginning with '-' stays a value.
    argv = [
        f"--environment={args.environment}",
        f"--project-root={project_root}",
        f"--source-id={args.source_id}",
        "--execute",
        "--resume",
    ]
    if args.include_disabled:
        argv.append("--include-disabled")
    return run.run_command(argv)


def _plan(args: argparse.Namespace, project_root: Path) -> ExecutionPlan:
    """Plan the source for its metadata zone (D-10), disabled sources included (D-9)."""
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


def _apply_release(
    operation: Callable[..., DeadLetterReleaseRecord],
    plan: ExecutionPlan,
    item_keys: Sequence[str] | None,
    reason: str,
    metadata: dict[str, str],
) -> DeadLetterReleaseRecord:
    """Call `release` or `preview_release`; their refusals are this command's refusals."""
    try:
        return operation(
            plan,
            item_keys=item_keys,
            operator=resolve_operator(),
            reason=reason,
            metadata=metadata,
        )
    except DeadLetterReleaseError as exc:
        raise _Refusal(str(exc)) from exc
    except (OSError, ValueError) as exc:
        path = DeadLetterStore().path(plan)
        raise _Refusal(f"Cannot release the dead letters at {path}: {exc}") from exc


def _requested_keys(
    parser: argparse.ArgumentParser, values: Sequence[str] | None
) -> tuple[str, ...]:
    """Stripped and de-duplicated in order. A blank key is an argument error: an empty
    selector must never act as a wildcard."""
    keys = tuple(dict.fromkeys(value.strip() for value in values or ()))
    if "" in keys:
        parser.error("--item-key must not be empty")
    return keys


def _selection(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[str, ...] | None:
    """`None` is every entry, the store's convention, and only `--all` asks for it."""
    return None if args.all else _requested_keys(parser, args.item_key)


def _release_document(
    plan: ExecutionPlan, record: DeadLetterReleaseRecord, project_root: Path
) -> dict[str, Any]:
    store = DeadLetterStore()
    return {
        **record.to_dict(),
        "history_path": _display_path(store.history_path(plan, record), project_root),
        "state_path": _display_path(store.path(plan), project_root),
        "state_deleted": not record.remaining_item_keys,
    }


def _resume_command(environment: str, source: SourceConfig) -> str:
    """The documented second step of a release, for this source, as an operator types it."""
    words = ["janus", "--environment", environment, "--source-id", source.source_id]
    if not source.enabled:
        words.append("--include-disabled")
    return shlex.join((*words, "--execute", "--resume"))


def _describe_progress(progress: dict[str, Any] | None) -> str:
    """Where a resuming run picks up, from the progress record as the last run left it."""
    if progress is None:
        return "no extraction progress recorded: the run starts from the beginning"
    parts = []
    if progress.get("current_input_key"):
        parts.append(
            f"request input {progress.get('current_input_index', '?')} of "
            f"{progress.get('request_input_count', '?')} ({progress['current_input_key']})"
        )
    if progress.get("last_page_number") is not None:
        parts.append(f"last page {progress['last_page_number']}")
    elif progress.get("last_offset") is not None:
        parts.append(f"last offset {progress['last_offset']}")
    elif progress.get("last_cursor") is not None:
        parts.append("a recorded cursor")
    parts.append(f"{len(progress.get('completed_inputs') or ())} input(s) completed")
    if progress.get("updated_at"):
        parts.append(f"recorded {progress['updated_at']}")
    return "; ".join(parts)


def _display_path(path: Path, project_root: Path) -> str:
    """Relative to the project root when inside it; an absolute metadata zone as it is."""
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _render_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True)


def _render_listing(document: dict[str, Any]) -> str:
    source_id, count = document["source_id"], document["recorded_count"]
    if count == 0:
        absent = "absent" if document["run_id"] is None else "holds no entries"
        state = f"  state  {document['state_path']} ({absent})"
        return f"no dead letters recorded for {source_id}\n{state}"

    entries, missing = document["entries"], document["missing_item_keys"]
    shown = f"{len(entries)} of {count}" if missing or len(entries) < count else str(count)
    lines = [
        f"{source_id} - {shown} dead letter(s), recorded by run {document['run_id']}",
        *_field("state", [document["state_path"]], width=7),
        *_field("updated", [document["updated_at"]], width=7),
    ]
    if entries:
        lines.extend(("", *_entry_table(entries)))
    if missing:
        lines.append("")
        lines.extend(_field("not recorded", [repr(key) for key in missing]))
        lines.extend(_field("recorded", [repr(key) for key in document["recorded_item_keys"]]))
    return "\n".join(lines)


def _entry_table(entries: Sequence[dict[str, Any]]) -> list[str]:
    """One row per entry, then its error and metadata in full on the lines below it."""
    rows = [
        ENTRY_COLUMNS,
        *((entry["item_key"], entry["item_type"], entry["recorded_at"]) for entry in entries),
    ]
    widths = [max(len(row[index]) for row in rows) for index in range(len(ENTRY_COLUMNS) - 1)]
    details = [
        [("error", f"{entry['error_type']}: {entry['error_message']}"), *entry["metadata"].items()]
        for entry in entries
    ]
    label_width = max(len(label) for detail in details for label, _ in detail)
    lines = [_table_row(rows[0], widths)]
    for row, detail in zip(rows[1:], details, strict=True):
        lines.append(_table_row(row, widths))
        for label, value in detail:
            lines.extend(_field(label, [value], width=label_width, indent=DETAIL_INDENT))
    return lines


def _table_row(row: Sequence[str], widths: Sequence[int]) -> str:
    # The last column is left unpadded, so no line carries trailing whitespace.
    cells = [cell.ljust(width) for cell, width in zip(row[:-1], widths, strict=True)]
    return "  " + COLUMN_GAP.join((*cells, row[-1]))


def _render_release(document: dict[str, Any], *, next_step: str | None) -> str:
    released, remaining = document["released_entries"], document["remaining_item_keys"]
    state = "deleted: nothing remains" if document["state_deleted"] else "rewritten"
    lines = [
        f"{document['source_id']} - released {len(released)} dead letter(s); "
        f"{len(remaining)} remain",
        *_field("released", [_entry_summary(entry) for entry in released]),
        *_field("remaining", remaining),
        *_field("operator", [document["operator"]]),
        *_field("reason", [document["reason"]]),
        *_field("history", [document["history_path"]]),
        *_field("state", [f"{document['state_path']} ({state})"]),
    ]
    if next_step is not None:
        lines.extend(
            (
                "",
                "The next resuming run retries the released item(s) and keeps skipping the rest:",
                f"  {next_step}",
            )
        )
    return "\n".join(lines)


def _entry_summary(entry: dict[str, Any]) -> str:
    return f"{entry['item_key']}  ({entry['item_type']}, {entry['error_type']})"


def _field(label: str, values: Sequence[str], *, width: int = 13, indent: int = 2) -> list[str]:
    """`label  value`, then each further value, or line of a value, aligned under the first.
    An empty field reads `-`; trailing whitespace is never printed."""
    lines = [line for value in values for line in str(value).splitlines()] or [EMPTY_CELL]
    padding = " " * (indent + width + len(COLUMN_GAP))
    first, *rest = lines
    return [
        f"{' ' * indent}{label:<{width}}{COLUMN_GAP}{first}".rstrip(),
        *(f"{padding}{line}".rstrip() for line in rest),
    ]
