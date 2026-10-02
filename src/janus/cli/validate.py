"""`janus validate`: load the registry, plan every source against it, and report.

Loading runs every structural check, resolves the graph and applies the semantic pass
(`janus.registry.semantics`). This module reports what they found and never re-implements
a rule: a load failure is printed to stderr exactly as the loader worded it. Planning then
runs once per source through one `Planner` against the one snapshot, always with
`include_disabled=True`, because a validation that honoured the enabled flag would check
one source of thirty-one. A source that cannot be planned is that source's issue; its
peers are still planned and reported.

The report is deterministic: no run id, timestamp, duration, or absolute path outside
`--project-root`; sources are sorted by id and issues by (source, path, message). Its
markers are ASCII (`ok`, `FAIL`, `note`). That is decided here, once: a glyph cannot be
encoded on a non-UTF-8 stream, and a validation command that crashes printing its own
verdict is worse than a plain one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.models.source_config import SourceConfig
from janus.planner import PlannedRun, Planner, PlannerError, PlanningRequest
from janus.registry import (
    SourceNotFoundError,
    SourceRegistry,
    load_registry,
    unverified_required_fields,
)

ISSUES_FOUND = 2
VALIDATE_INSTANT = datetime(1970, 1, 1, tzinfo=UTC)
PLAN_ISSUE_PATH = "plan"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-id",
        help="Plan only this source; the semantic pass still covers the whole registry.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report format. Defaults to text.",
    )


def validate_command(args: argparse.Namespace) -> int:
    """Exit 0 when the registry loads and every selected source plans, 2 otherwise."""
    try:
        registry = load_registry(args.project_root.resolve())
        selected = _selected_sources(registry, args.source_id)
    except (FileNotFoundError, SourceNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return ISSUES_FOUND

    planner = _build_planner()
    sources = [_plan_source(planner, registry, source, args.environment) for source in selected]
    report = _build_report(registry, sources)
    print(_render_json(report) if args.format == "json" else _render_text(report))
    return ISSUES_FOUND if report["issues"] else 0


def _build_planner() -> Planner:
    """The one planner every source is planned through, behind an injectable seam."""
    return Planner()


def _selected_sources(registry: SourceRegistry, source_id: str | None) -> tuple[SourceConfig, ...]:
    """Narrow the plan step only: the registry was loaded, and so checked, whole."""
    if source_id is not None:
        return (registry.get_source(source_id, include_disabled=True),)
    return tuple(
        sorted(registry.list_sources(enabled_only=False), key=lambda source: source.source_id)
    )


def _plan_source(
    planner: Planner, registry: SourceRegistry, source: SourceConfig, environment: str
) -> dict[str, Any]:
    planned: PlannedRun | None = None
    issues: list[dict[str, str]] = []
    try:
        planned = planner.plan(
            PlanningRequest.create(
                source_id=source.source_id,
                environment=environment,
                project_root=registry.project_root,
                started_at=VALIDATE_INSTANT,
                attributes={"trigger": "validate"},
                include_disabled=True,
            ),
            registry=registry,
        )
    except (PlannerError, ValueError) as exc:
        # The planning refusals `janus run` reports with exit 2: a hook or binding that
        # does not resolve, and a strategy rejecting its config. Anything else is a defect
        # and keeps its traceback.
        issues.append({"source_id": source.source_id, "path": PLAN_ISSUE_PATH, "message": str(exc)})
    hook = planned.hook if planned is not None else None
    return {
        "source_id": source.source_id,
        "family": source.strategy,
        "variant": source.strategy_variant,
        "dispatch_path": planned.dispatch_path if planned is not None else None,
        "enabled": source.enabled,
        "hook": source.source_hook,
        "hook_implementation": type(hook).__name__ if hook is not None else None,
        "planned": planned is not None,
        "issues": issues,
    }


def _build_report(registry: SourceRegistry, sources: Sequence[dict[str, Any]]) -> dict[str, Any]:
    issues = sorted(
        (issue for source in sources for issue in source["issues"]),
        key=lambda issue: (issue["source_id"], issue["path"], issue["message"]),
    )
    notes = [
        {"kind": "unverified_required_fields", "source_id": source_id, "fields": list(fields)}
        for source_id, fields in unverified_required_fields(
            registry.sources, contracts=registry.contracts
        )
    ]
    enabled = sum(1 for source in registry.sources if source.enabled)
    sources_dir = registry.app_config.registry.resolve_sources_dir(registry.project_root)
    return {
        "project_root": str(registry.project_root),
        "sources_dir": _project_relative(sources_dir, registry.project_root),
        "counts": {
            "sources": len(registry.sources),
            "enabled": enabled,
            "disabled": len(registry.sources) - enabled,
            "nodes": len(registry.graph.nodes),
            "edges": len(registry.graph.edges),
            "issues": len(issues),
            "unverified_required_fields": len(notes),
        },
        "sources": list(sources),
        "issues": issues,
        "notes": notes,
    }


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def _render_text(report: dict[str, Any]) -> str:
    counts = report["counts"]
    sources = report["sources"]
    width = max((len(source["source_id"]) for source in sources), default=0)
    lines = [
        f"JANUS registry validation - {report['sources_dir']}",
        f"{counts['sources']} sources ({counts['enabled']} enabled, "
        f"{counts['disabled']} disabled), {counts['nodes']} nodes, {counts['edges']} edges",
        "",
    ]
    for source in sources:
        marker = "FAIL" if source["issues"] else "ok"
        dispatch = source["dispatch_path"] or "(not planned)"
        lines.append(f"  {marker:<4}  {source['source_id']:<{width}}  {dispatch}")
        lines.extend(f"          {issue['path']}: {issue['message']}" for issue in source["issues"])
    if report["notes"]:
        lines.extend(("", "notes"))
        for note in report["notes"]:
            lines.append(
                f"  note  {note['source_id']}: required_fields declared with no contract "
                "to verify them against"
            )
            lines.append(f"          ({', '.join(note['fields'])})")
    failing = sum(1 for source in sources if source["issues"])
    summary = (
        f"{counts['issues']} issue(s) in {failing} source(s); "
        f"{counts['unverified_required_fields']} unverified required-field declaration(s)"
    )
    return "\n".join([*lines, "", summary])
