from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from janus.planner import Planner, PlannerError, PlanningRequest
from janus.registry import SourceNotFoundError
from janus.runtime import SourceExecutor, SparkSessionProvider
from janus.scripts import ingest_raw_to_bronze
from janus.utils.environment import build_spark_session, load_environment_config, prepare_runtime
from janus.utils.logging import build_structured_logger


def default_project_root() -> Path:
    env_project_root = os.getenv("JANUS_PROJECT_ROOT")
    if env_project_root:
        return Path(env_project_root)

    cwd = Path.cwd()
    if (cwd / "conf" / "environments").exists():
        return cwd

    return Path(__file__).resolve().parents[2]


def parse_started_at(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "started-at must be a valid ISO-8601 timestamp"
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("started-at must include a timezone offset")

    return parsed


def format_runtime_permission_error(exc: PermissionError) -> str:
    path = exc.filename or "<unknown>"
    message = (
        f"JANUS could not prepare the runtime path {path!r}. "
        "The active environment needs write access to the configured storage "
        "and Spark cache directories."
    )
    if str(path).startswith("/workspace/"):
        message += (
            " If you are running inside the local container, recreate it with "
            "`make down && make up` so the Docker/Podman user mapping is "
            "applied correctly."
        )
    return message


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate JANUS runtime configuration, plan one source, or execute one "
            "configured source through the framework runtime."
        )
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
        help="Project root used to resolve conf/ and data/ paths.",
    )
    parser.add_argument(
        "--with-spark",
        action="store_true",
        help="Create and stop a Spark session after validating the environment config.",
    )
    parser.add_argument(
        "--source-id",
        help="Configured source_id to plan from the source registry.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional explicit run identifier for the planned source execution.",
    )
    parser.add_argument(
        "--started-at",
        type=parse_started_at,
        help="Optional timezone-aware ISO-8601 timestamp used to make planning deterministic.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the selected source through planning, extraction, Spark normalization, "
            "bronze writes, validation, and metadata persistence."
        ),
    )
    parser.add_argument(
        "--ingest-raw-to-bronze",
        action="store_true",
        help=(
            "Skip extraction, rediscover the selected source raw artifacts already stored in "
            "the raw zone, and load them into a bronze table informed at the CLI."
        ),
    )
    parser.add_argument(
        "--bronze-table",
        help=(
            "Target bronze table name for --ingest-raw-to-bronze. Accepts either table_name "
            "or namespace.table_name."
        ),
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Allow planning or executing a source that is configured but disabled.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume extraction from the last recorded page instead of starting over. "
            "Reads the extraction_progress.json written by a previous partial run and "
            "re-uses already-fetched raw files. Has no effect if no progress file exists."
        ),
    )
    args = parser.parse_args(argv)

    if args.execute and not args.source_id:
        parser.error("--execute requires --source-id")
    if args.ingest_raw_to_bronze and not args.source_id:
        parser.error("--ingest-raw-to-bronze requires --source-id")
    if args.ingest_raw_to_bronze and not args.bronze_table:
        parser.error("--ingest-raw-to-bronze requires --bronze-table")
    if args.execute and args.ingest_raw_to_bronze:
        parser.error("--execute and --ingest-raw-to-bronze cannot be used together")
    if args.include_disabled and not args.source_id:
        parser.error("--include-disabled requires --source-id")

    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()

    try:
        config = load_environment_config(args.environment, project_root)
        resolved_paths = prepare_runtime(config, project_root)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except PermissionError as exc:
        print(format_runtime_permission_error(exc), file=sys.stderr)
        return 2

    summary = {
        "environment": config.get("name", args.environment),
        "project_root": str(project_root),
        "spark": {
            "app_name": config.get("spark", {}).get("app_name"),
            "master": config.get("spark", {}).get("master"),
            "config": config.get("spark", {}).get("config", {}),
        },
        "paths": {name: str(path) for name, path in resolved_paths.items()},
    }

    planned_run = None
    if args.source_id:
        try:
            run_attributes: dict[str, str] = {"trigger": "cli"}
            if args.resume:
                run_attributes["resume"] = "true"
            planned_run = Planner().plan(
                PlanningRequest.create(
                    source_id=args.source_id,
                    environment=args.environment,
                    project_root=project_root,
                    run_id=args.run_id,
                    started_at=args.started_at,
                    include_disabled=args.include_disabled,
                    attributes=run_attributes,
                )
            )
        except (FileNotFoundError, PlannerError, SourceNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        summary["planned_run"] = planned_run.to_summary()

    if args.execute:
        assert planned_run is not None
        execution_logger = build_structured_logger(
            "janus.execution",
            level=config.get("runtime", {}).get("log_level", "INFO"),
        ).bind(
            environment=config.get("name", args.environment),
            project_root=str(project_root),
        )
        execution_logger.info(
            "cli_execution_requested",
            source_id=args.source_id,
            include_disabled=args.include_disabled,
        )

        spark_provider = SparkSessionProvider(config, resolved_paths, execution_logger)
        try:
            executed_run = SourceExecutor(logger=execution_logger).execute(
                planned_run,
                spark_provider,
                config,
            )
            summary["executed_run"] = executed_run.to_summary()
            if spark_provider.was_started:
                summary["spark_session"] = spark_provider.session_info
        finally:
            # The executor already stops in its own finally; this is the backstop for
            # a session acquired outside that span.
            spark_provider.stop()

        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if executed_run.is_successful else 1

    if args.ingest_raw_to_bronze:
        assert planned_run is not None
        assert args.bronze_table is not None
        execution_logger = build_structured_logger(
            "janus.execution",
            level=config.get("runtime", {}).get("log_level", "INFO"),
        ).bind(
            environment=config.get("name", args.environment),
            project_root=str(project_root),
        )
        execution_logger.info(
            "cli_raw_to_bronze_requested",
            source_id=args.source_id,
            include_disabled=args.include_disabled,
            bronze_table=args.bronze_table,
        )

        try:
            execution_logger.info(
                "spark_session_starting",
                app_name=config.get("spark", {}).get("app_name"),
                master=config.get("spark", {}).get("master"),
            )
            spark = build_spark_session(config, resolved_paths)
        except Exception as exc:  # pragma: no cover - defensive entrypoint guard
            execution_logger.exception(
                "spark_session_failed",
                failure_reason=str(exc),
                error_type=type(exc).__name__,
            )
            print(str(exc), file=sys.stderr)
            return 1

        try:
            summary["spark_session"] = {
                "app_name": spark.sparkContext.appName,
                "master": spark.sparkContext.master,
            }
            execution_logger.info(
                "spark_session_started",
                app_name=spark.sparkContext.appName,
                master=spark.sparkContext.master,
            )
            raw_to_bronze_run = ingest_raw_to_bronze(
                planned_run,
                spark,
                config,
                bronze_table=args.bronze_table,
                logger=execution_logger,
            )
            summary["raw_to_bronze_run"] = raw_to_bronze_run.to_summary()
        finally:
            spark.stop()
            execution_logger.info("spark_session_stopped")

        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if raw_to_bronze_run.is_successful else 1

    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.with_spark:
        return 0

    spark = build_spark_session(config, resolved_paths)
    try:
        print(
            json.dumps(
                {
                    "spark_app_name": spark.sparkContext.appName,
                    "spark_master": spark.sparkContext.master,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
