"""The one-shot ``janus run-all`` batch command."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import yaml

from janus.cli.common import (
    default_project_root,
    format_runtime_permission_error,
    log_source_config_deprecations,
    parse_started_at,
)
from janus.observability.metrics import resolve_metrics_settings
from janus.observability.openlineage import resolve_openlineage_settings
from janus.orchestration import (
    BatchPlan,
    BatchPlanner,
    BatchPlanRequest,
    BatchSelection,
    PipelineSummaryPersistenceError,
)
from janus.orchestration.settings import resolve_orchestration_settings
from janus.registry import SourceRegistry, load_registry
from janus.runtime import (
    BatchExecutionInterrupted,
    BatchExecutionPreflightError,
    BatchExecutor,
)
from janus.runtime.parallel import ParallelBatchExecutor, build_batch_executor
from janus.utils.environment import (
    RuntimeLocation,
    load_environment_config,
    prepare_runtime,
    read_spark_write_defaults,
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


class _BatchArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        """Keep existing error usage byte-identical while documenting the new flag."""
        parallel = self._option_string_actions["--max-parallel"]
        help_text = parallel.help
        if "--max-parallel" not in message:
            parallel.help = argparse.SUPPRESS
        try:
            super().error(message)
        finally:
            parallel.help = help_text


def build_parser() -> argparse.ArgumentParser:
    """Build the batch-only parser; all options intentionally follow ``run-all``."""
    parser = _BatchArgumentParser(
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
    parser.add_argument(
        "--fail-on-budget-breach",
        action="store_true",
        help=(
            "Treat a run that exceeded its operations.max_duration_seconds as a failed "
            "attempt, skipping its dependents. Off by default: a slow run that produced "
            "correct data is a scheduling signal, not a data failure."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Run up to N independent sources at once, each in its own process with its own Spark "
            "driver when it needs one. Defaults to the profile's "
            "orchestration.max_parallel_default, else 1. Peak memory grows with "
            "N \u00d7 spark.driver.memory."
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

    logger = build_structured_logger(
        "janus.batch", level=prepared.config.get("runtime", {}).get("log_level", "INFO")
    ).bind(
        environment=prepared.config.get("name", prepared.request.environment),
        project_root=str(project_root),
        pipeline_run_id=prepared.request.pipeline_run_id,
    )
    available = os.process_cpu_count() or os.cpu_count() or 1
    if prepared.max_parallel > available:
        logger.warning(
            "max_parallel_exceeds_cpu_count",
            max_parallel=prepared.max_parallel,
            cpu_count=available,
        )
        print(
            f"warning: --max-parallel {prepared.max_parallel} exceeds the {available} CPUs "
            "available to this process; each worker may start its own Spark driver",
            file=sys.stderr,
        )
    return _execute_batch(prepared, resume=args.resume, logger=logger)


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    config: dict[str, Any]
    resolved_paths: dict[str, RuntimeLocation]
    request: BatchPlanRequest
    registry: SourceRegistry
    plan: BatchPlan
    max_parallel: int
    lock_wait_seconds: int


def _prepare_batch(args: argparse.Namespace, project_root: Path) -> _PreparedBatch:
    config = load_environment_config(args.environment, project_root)
    orchestration = resolve_orchestration_settings(config)
    max_parallel = (
        args.max_parallel if args.max_parallel is not None else orchestration.max_parallel_default
    )
    # Fail closed before registry reads without creating paths for an invalid registry.
    resolve_metrics_settings(config)
    selection = BatchSelection.create(tags=args.tag, domains=args.domain)
    request = BatchPlanRequest.create(
        environment=args.environment,
        project_root=project_root,
        pipeline_run_id=args.pipeline_run_id,
        planned_at=args.started_at,
        selection=selection,
        attributes={"resume": "true"} if args.resume else None,
        fail_on_budget_breach=args.fail_on_budget_breach,
    )
    registry = load_registry(project_root, write_defaults=read_spark_write_defaults(config))
    plan = BatchPlanner().plan(request, registry=registry)
    resolved_paths = prepare_runtime(config, project_root)
    resolve_openlineage_settings(config)
    resolve_metrics_settings(config)
    return _PreparedBatch(
        config,
        resolved_paths,
        request,
        registry,
        plan,
        max_parallel,
        orchestration.lock_wait_seconds,
    )


def _execute_batch(prepared: _PreparedBatch, *, resume: bool, logger: StructuredLogger) -> int:
    config = prepared.config
    plan = prepared.plan

    try:
        log_source_config_deprecations(logger, prepared.registry.sources)
        logger.info(
            "cli_batch_execution_requested",
            root_ids=plan.root_ids,
            included_upstream_ids=plan.included_upstream_ids,
            source_order=plan.source_ids,
            resume=resume,
        )
        runner = _build_batch_executor(
            logger,
            max_parallel=prepared.max_parallel,
            lock_wait_seconds=prepared.lock_wait_seconds,
        )
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


def _build_batch_executor(
    logger: StructuredLogger, *, max_parallel: int, lock_wait_seconds: int
) -> BatchExecutor | ParallelBatchExecutor:
    """Construct the production runner behind an injectable CLI composition seam."""
    return build_batch_executor(
        max_parallel=max_parallel, logger=logger, lock_wait_seconds=lock_wait_seconds
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer >= 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return parsed


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
