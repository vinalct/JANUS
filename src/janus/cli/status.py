"""Report declared source freshness without planning a run or opening a Spark session."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import yaml

from janus.cli.common import parse_started_at
from janus.observability.freshness import (
    BREACHING_OUTCOMES,
    FRESH,
    NEVER_RAN,
    STALE,
    UNDECLARED,
    SourceFreshness,
    evaluate_sources,
    read_latest_success_from_runs_table,
)
from janus.observability.metrics.settings import resolve_metrics_settings
from janus.observability.openlineage.settings import resolve_openlineage_settings
from janus.registry import SourceNotFoundError, SourceRegistry, load_registry
from janus.utils.environment import (
    load_environment_config,
    materialize_runtime_paths,
    read_spark_write_defaults,
)

ARGUMENT_ERROR = 2
OUTCOMES = (FRESH, STALE, NEVER_RAN, UNDECLARED)
TABLE_COLUMNS = ("SOURCE", "OUTCOME", "LAST SUCCESS", "AGE", "MAX AGE")


def configure_status_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-id",
        action="append",
        help="Evaluate only this source; repeat to select several. Disabled sources are included.",
    )
    parser.add_argument(
        "--from-runs-table",
        action="store_true",
        help="Read successful runs through PyIceberg instead of the per-source metadata zone.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--now",
        type=parse_started_at,
        help="Evaluate at this timezone-aware ISO-8601 instant. Defaults to the current UTC time.",
    )


def run_status(args: argparse.Namespace) -> int:
    """Exit 1 on a freshness breach, 2 when the question cannot be evaluated, else 0."""
    project_root = args.project_root.resolve()
    now = (args.now or datetime.now(UTC)).astimezone(UTC)
    try:
        config = load_environment_config(args.environment, project_root)
        materialize_runtime_paths(config, project_root)
        resolve_openlineage_settings(config)
        resolve_metrics_settings(config)
        registry = load_registry(project_root, write_defaults=read_spark_write_defaults(config))
        selected_ids = (
            [source.source_id for source in registry.list_sources(enabled_only=False)]
            if args.source_id is None
            else sorted(set(args.source_id))
        )
        for source_id in selected_ids:
            registry.get_source(source_id, include_disabled=True)
    except KeyError as exc:
        print(f"Environment config is incomplete: {exc}", file=sys.stderr)
        return ARGUMENT_ERROR
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        SourceNotFoundError,
        yaml.YAMLError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return ARGUMENT_ERROR

    latest_successes = None
    if args.from_runs_table:
        try:
            from pyiceberg.catalog import load_catalog

            latest_successes = read_latest_success_from_runs_table(
                config, selected_ids, project_root=project_root, catalog_loader=load_catalog
            )
        except Exception as exc:
            print(f"Could not read runs table: {exc}", file=sys.stderr)
            return ARGUMENT_ERROR
    try:
        results = evaluate_sources(
            registry,
            project_root,
            now=now,
            source_ids=selected_ids,
            latest_successes=latest_successes,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return ARGUMENT_ERROR

    report = _build_report(
        results, registry, environment=args.environment, now=now, from_table=args.from_runs_table
    )
    print(json.dumps(report, indent=2) if args.format == "json" else _render_text(report))
    return int(any(result.is_breach for result in results))


def _build_report(
    results: Sequence[SourceFreshness],
    registry: SourceRegistry,
    *,
    environment: str,
    now: datetime,
    from_table: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "environment": environment,
        "evaluated_at": now.isoformat(),
        "source": "runs_table" if from_table else "metadata_zone",
        "totals": {
            **{
                outcome: sum(result.outcome == outcome for result in results)
                for outcome in OUTCOMES
            },
            "breaching": sum(result.is_breach for result in results),
        },
        "sources": [
            {
                "source_id": result.source_id,
                "enabled": registry.get_source(result.source_id, include_disabled=True).enabled,
                "outcome": result.outcome,
                "last_success_at": (
                    result.last_success_at.isoformat()
                    if result.last_success_at is not None
                    else None
                ),
                "age_seconds": result.age_seconds,
                "max_age_seconds": result.max_age_seconds,
                "runs_scanned": result.runs_scanned,
                "unreadable_run_files": result.unreadable_run_files,
                "metadata_path": result.metadata_path,
            }
            for result in results
        ],
    }


def _render_text(report: dict[str, Any]) -> str:
    rows = [_text_row(source) for source in report["sources"]]
    widths = [
        max(len(row[index]) for row in [TABLE_COLUMNS, *rows])
        for index in range(len(TABLE_COLUMNS))
    ]
    lines = [
        "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip()
        for row in [TABLE_COLUMNS, *rows]
    ]
    shared: dict[str, list[str]] = {}
    for source in report["sources"]:
        shared.setdefault(source["metadata_path"], []).append(source["source_id"])
    for source_ids in shared.values():
        if len(source_ids) > 1:
            lines.append(f"Note: {', '.join(source_ids)} share a metadata path.")
    unreadable = sum(source["unreadable_run_files"] for source in report["sources"])
    if unreadable:
        lines.append(f"Unreadable run files: {unreadable}")
    breaches = [
        source["source_id"]
        for source in report["sources"]
        if source["outcome"] in BREACHING_OUTCOMES
    ]
    lines.extend(
        [
            "",
            f"{len(breaches)} of {len(rows)} sources breach their freshness window"
            + (f": {', '.join(breaches)}" if breaches else "."),
        ]
    )
    return "\n".join(lines)


def _text_row(source: dict[str, Any]) -> tuple[str, ...]:
    outcome = source["outcome"]
    if source["unreadable_run_files"]:
        outcome += f" ({source['unreadable_run_files']} unreadable)"
    return (
        source["source_id"] + (" (disabled)" if not source["enabled"] else ""),
        outcome,
        source["last_success_at"].replace("+00:00", "Z") if source["last_success_at"] else "-",
        _format_age(source["age_seconds"]),
        _format_age(source["max_age_seconds"]),
    )


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    hours, minutes = divmod(int(abs(seconds) // 60), 60)
    sign = "-" if seconds < 0 else ""
    return f"{sign}{hours}h" + (f" {minutes}m" if minutes else "")
