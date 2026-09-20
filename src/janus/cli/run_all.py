"""The one-shot ``janus run-all`` batch command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from janus.cli.common import (
    default_project_root,
    format_runtime_permission_error,
    parse_started_at,
)
from janus.orchestration import (
    BatchPlan,
    BatchPlanner,
    BatchPlanRequest,
    BatchSelection,
    PipelineSummaryPersistenceError,
)
from janus.registry import SourceRegistry, load_registry
from janus.runtime import (
    BatchExecutionInterrupted,
    BatchExecutionPreflightError,
    BatchExecutor,
    SourceExecutionService,
    SourceExecutor,
)
from janus.utils.environment import (
    RuntimeLocation,
    load_environment_config,
    prepare_runtime,
)
from janus.utils.logging import StructuredLogger, build_structured_logger

ARGUMENT_ERROR = 2
OPERATIONAL_ERROR = 1
INTERRUPTED = 130

_UNSUPPORTED_BATCH_OPTIONS = {
    "source_id": "--source-id",
    "execute": "--execute",
    "run_id": "--run-id",
    "include_disabled": "--include-disabled",
    "bronze_table": "--bronze-table",
    "ingest_raw_to_bronze": "--ingest-raw-to-bronze",
    "with_spark": "--with-spark",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the batch-only parser; all options intentionally follow ``run-all``."""
    parser = argparse.ArgumentParser(
        prog="janus run-all",
        allow_abbrev=False,
        description=(
            "Run enabled sources once in deterministic dependency order. Filters select "
            "roots; required upstreams are included automatically and identified in the "
            "aggregate summary. A failed source skips only its dependents, while independent "
            "sources continue. This command does not schedule future runs."
        ),
    )
    parser.add_argument(
        "--environment",
        default="local",
        help="Environment profile name under conf/environments without the .yaml suffix.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
        help=(
            "Project root used to resolve conf/ and data/ paths. Defaults to "
            "JANUS_PROJECT_ROOT, the current JANUS project, or the installed project."
        ),
    )
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument(
        "--tag",
        action="append",
        default=[],
        help=(
            "Select enabled roots carrying this tag; repeat for an any-match filter. "
            "Required upstreams need not carry the tag."
        ),
    )
    selectors.add_argument(
        "--domain",
        action="append",
        default=[],
        help=(
            "Select enabled roots in this domain; repeat for an any-match filter. "
            "Required upstreams may belong to another domain."
        ),
    )
    parser.add_argument(
        "--pipeline-run-id",
        help=(
            "Optional identity for this batch. This is distinct from the single-source "
            "--run-id and must be new for each standalone run."
        ),
    )
    parser.add_argument(
        "--started-at",
        type=parse_started_at,
        help="Optional timezone-aware ISO-8601 timestamp used to make planning deterministic.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume each source's extraction progress when available. This does not mark an "
            "earlier pipeline successful or suppress required upstream execution."
        ),
    )
    _add_unsupported_legacy_options(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the argument surface owned by ``run-all``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    unsupported = [
        option
        for destination, option in _UNSUPPORTED_BATCH_OPTIONS.items()
        if hasattr(args, destination)
    ]
    if unsupported:
        parser.error(
            f"{', '.join(unsupported)} cannot be used with run-all; "
            "use the existing single-source command instead"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Plan, execute, persist, and print one complete batch aggregate."""
    args = parse_args(argv)
    project_root = args.project_root.resolve()

    try:
        prepared = _prepare_batch(args, project_root)
    except PermissionError as exc:
        print(format_runtime_permission_error(exc), file=sys.stderr)
        return ARGUMENT_ERROR
    except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return ARGUMENT_ERROR

    return _execute_batch(prepared, resume=args.resume)


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    config: dict[str, Any]
    resolved_paths: dict[str, RuntimeLocation]
    request: BatchPlanRequest
    registry: SourceRegistry
    plan: BatchPlan


def _prepare_batch(args: argparse.Namespace, project_root: Path) -> _PreparedBatch:
    config = load_environment_config(args.environment, project_root)
    selection = BatchSelection.create(tags=args.tag, domains=args.domain)
    request = BatchPlanRequest.create(
        environment=args.environment,
        project_root=project_root,
        pipeline_run_id=args.pipeline_run_id,
        planned_at=args.started_at,
        selection=selection,
        attributes={"resume": "true"} if args.resume else None,
    )
    registry = load_registry(project_root)
    plan = BatchPlanner().plan(request, registry=registry)
    resolved_paths = prepare_runtime(config, project_root)
    return _PreparedBatch(config, resolved_paths, request, registry, plan)


def _execute_batch(prepared: _PreparedBatch, *, resume: bool) -> int:
    config = prepared.config
    request = prepared.request
    plan = prepared.plan
    project_root = request.project_root

    try:
        logger = build_structured_logger(
            "janus.batch",
            level=config.get("runtime", {}).get("log_level", "INFO"),
        ).bind(
            environment=config.get("name", request.environment),
            project_root=str(project_root),
            pipeline_run_id=request.pipeline_run_id,
        )
        logger.info(
            "cli_batch_execution_requested",
            root_ids=plan.root_ids,
            included_upstream_ids=plan.included_upstream_ids,
            source_order=plan.source_ids,
            resume=resume,
        )
        runner = _build_batch_executor(logger)
        outcome = runner.execute(
            plan,
            prepared.registry,
            config,
            prepared.resolved_paths,
        )
    except PipelineSummaryPersistenceError as exc:
        _print_summary(exc.pipeline_outcome.to_summary())
        print(str(exc), file=sys.stderr)
        return OPERATIONAL_ERROR
    except BatchExecutionInterrupted as exc:
        partial = exc.partial
        print(
            f"{exc}; completed sources: {len(partial.source_outcomes)}, "
            f"pending sources: {len(partial.pending_source_ids)}",
            file=sys.stderr,
        )
        return INTERRUPTED
    except (BatchExecutionPreflightError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return ARGUMENT_ERROR
    except Exception as exc:
        print(f"Batch execution failed: {exc}", file=sys.stderr)
        return OPERATIONAL_ERROR

    _print_summary(outcome.to_summary())
    return 0 if outcome.is_successful else OPERATIONAL_ERROR


def _build_batch_executor(logger: StructuredLogger) -> BatchExecutor:
    """Construct the production runner behind an injectable CLI composition seam."""
    return BatchExecutor(
        source_execution=SourceExecutionService(executor=SourceExecutor(logger=logger))
    )


def _add_unsupported_legacy_options(parser: argparse.ArgumentParser) -> None:
    """Parse legacy flags solely so the batch command can reject them by name."""
    parser.add_argument("--source-id", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--execute", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    parser.add_argument("--run-id", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--bronze-table", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--ingest-raw-to-bronze",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--with-spark", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )


def _print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))
