"""`janus validate`: load the registry, plan every source against it, and report.

Loading runs every structural check, resolves the graph and applies the semantic pass
(`janus.registry.semantics`). This module reports what they found and never re-implements
a rule: a load failure is printed to stderr exactly as the loader worded it. Planning then
runs once per source through one `Planner` against the one snapshot, always with
`include_disabled=True`, because a validation that honoured the enabled flag would check
one source of thirty-one. A source that cannot be planned is that source's issue; its
peers are still planned and reported.

With `--environment` the named profile is checked too, in the order a run reads it: load,
runtime paths, OpenLineage transport, Spark options. Nothing is created unless `--prepare`
asks for it, and no session is built: `build_spark_options` is a mapping, not a JVM. The
two halves are independent. A refusal from either is printed to stderr verbatim, profile
first, and a refusal leaves stdout empty, so a report only ever describes a registry and a
profile that both resolved. Spark options are reported by key only, never by value: an
option value is where a catalog credential lives, and a report that prints none cannot
leak one.

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

from janus.cli.common import format_runtime_permission_error
from janus.models.source_config import SourceConfig
from janus.observability.openlineage import (
    FileOpenLineageTransport,
    HttpOpenLineageTransport,
    OpenLineageTransport,
    resolve_openlineage_settings,
    resolve_openlineage_transport,
)
from janus.planner import PlannedRun, Planner, PlannerError, PlanningRequest
from janus.registry import (
    SourceNotFoundError,
    SourceRegistry,
    load_registry,
    unverified_required_fields,
)
from janus.utils.catalog_options import (
    ICEBERG_CATALOG_DB_PATH_KEY,
    RuntimeLocation,
    resolve_catalog_type,
)
from janus.utils.environment import (
    build_spark_options,
    environment_config_path,
    load_environment_config,
    materialize_runtime_paths,
    prepare_runtime,
)

ISSUES_FOUND = 2
VALIDATE_INSTANT = datetime(1970, 1, 1, tzinfo=UTC)
PLAN_ISSUE_PATH = "plan"

_EPILOG = (
    "With --environment, the named profile is checked as well: its catalog, its "
    "OpenLineage transport, the Spark options it renders and the runtime paths it "
    "resolves. No Spark session is started, and no directory is created unless "
    "--prepare is given."
)


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
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="With --environment, also create the runtime directories the profile "
        "resolves. Off by default: validate creates nothing.",
    )
    parser.epilog = _EPILOG
    # The parent defaults --environment for every verb. Here the flag also asks for the
    # profile half, so an absent flag must stay visible: its default moves to
    # `planning_environment`, which only feeds a run context that is never printed.
    parser.set_defaults(planning_environment=parser.get_default("environment"), environment=None)


def validate_command(args: argparse.Namespace) -> int:
    """Exit 0 when everything resolves and every selected source plans, 2 otherwise."""
    project_root = args.project_root.resolve()
    if args.prepare and args.environment is None:
        return _refuse(["--prepare requires --environment"])

    refusals: list[str] = []
    environment: dict[str, Any] | None = None
    if args.environment is not None:
        try:
            environment = _validate_environment(
                args.environment, project_root, prepare=args.prepare
            )
        except PermissionError as exc:
            refusals.append(format_runtime_permission_error(exc))
        except (FileNotFoundError, ValueError) as exc:
            refusals.append(str(exc))
        except KeyError as exc:
            refusals.append(f"Environment config is incomplete: {exc}")

    try:
        registry = load_registry(project_root)
        selected = _selected_sources(registry, args.source_id)
    except (FileNotFoundError, SourceNotFoundError, ValueError) as exc:
        return _refuse([*refusals, str(exc)])
    if refusals:
        return _refuse(refusals)

    planner = _build_planner()
    planning_environment = args.environment or args.planning_environment
    sources = [_plan_source(planner, registry, source, planning_environment) for source in selected]
    report = _build_report(registry, sources)
    if environment is not None:
        report["environment"] = environment
    print(_render_json(report) if args.format == "json" else _render_text(report))
    return ISSUES_FOUND if report["issues"] else 0


def _refuse(messages: Sequence[str]) -> int:
    for message in messages:
        print(message, file=sys.stderr)
    return ISSUES_FOUND


def _validate_environment(name: str, project_root: Path, *, prepare: bool) -> dict[str, Any]:
    """The profile half, step by step as a run reads it; the first refusal raises.

    D-19: the dry run resolves every location with `materialize_runtime_paths`, which
    creates nothing; only `--prepare` reaches `prepare_runtime` and its fallback.
    """
    config = load_environment_config(name, project_root)
    resolve_paths = prepare_runtime if prepare else materialize_runtime_paths
    paths = resolve_paths(config, project_root)
    transport = resolve_openlineage_transport(resolve_openlineage_settings(config), paths)
    options = build_spark_options(config, paths)
    iceberg = config.get("spark", {}).get("iceberg")
    catalog_type = resolve_catalog_type(iceberg) if isinstance(iceberg, dict) and iceberg else None
    config_path = environment_config_path(name, project_root)
    return {
        "name": config.get("name", name),
        "config_path": _project_relative(config_path, project_root),
        "catalog_type": catalog_type,
        "openlineage_transport": transport.kind,
        "openlineage_target": _transport_target(transport, project_root),
        "spark_option_keys": sorted(options),
        "paths": {key: _location(value, project_root) for key, value in paths.items()},
        "prepared": prepare,
    }


def _transport_target(transport: OpenLineageTransport, project_root: Path) -> str | None:
    """Where events would go: the events directory, or the URL redacted as the transport
    itself redacts it. Never a token: the HTTP target carries none, and auth is a header."""
    if isinstance(transport, FileOpenLineageTransport):
        return _project_relative(transport.directory, project_root)
    if isinstance(transport, HttpOpenLineageTransport):
        return transport.target
    return None


def _location(value: RuntimeLocation, project_root: Path) -> str:
    """A path relative to the project root; a location URI or warehouse name verbatim."""
    return _project_relative(value, project_root) if isinstance(value, Path) else value


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
    if "environment" in report:
        lines.extend(("", *_render_environment(report["environment"])))
    failing = sum(1 for source in sources if source["issues"])
    summary = (
        f"{counts['issues']} issue(s) in {failing} source(s); "
        f"{counts['unverified_required_fields']} unverified required-field declaration(s)"
    )
    return "\n".join([*lines, "", summary])


def _render_environment(environment: dict[str, Any]) -> list[str]:
    paths = environment["paths"]
    catalog = environment["catalog_type"] or "none"
    if ICEBERG_CATALOG_DB_PATH_KEY in paths:
        catalog += f"  (database file: {paths[ICEBERG_CATALOG_DB_PATH_KEY]})"
    elif environment["catalog_type"] is None:
        catalog += "  (no spark.iceberg block)"
    openlineage = environment["openlineage_transport"]
    if environment["openlineage_target"] is not None:
        openlineage += f"  ({environment['openlineage_target']})"
    created = "[materialized: --prepare]" if environment["prepared"] else "[not created: dry run]"
    width = max((len(key) for key in paths), default=0)
    return [
        f"environment: {environment['name']}  ({environment['config_path']})",
        f"  catalog        {catalog}",
        f"  openlineage    {openlineage}",
        f"  spark options  {len(environment['spark_option_keys'])} keys",
        f"  paths          {created}",
        *(f"    {key:<{width}}  {value}" for key, value in paths.items()),
    ]
