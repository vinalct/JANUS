"""Contract authoring commands. Drafting is isolated from normal source execution."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import yaml

from janus.cli.common import default_project_root
from janus.cli.contract_drafting import (
    _DRAFT_REQUEST_ATTRIBUTE,
    _contract_document,
    _draft_properties,
    _drafted_from,
    _DraftInput,
    _profile_dataframe,
    _read_draft_input,
)
from janus.models import (
    ExecutionPlan,
    ExtractedArtifact,
    ExtractionResult,
    SourceConfig,
    load_data_contract,
)
from janus.planner import PlannedRun, Planner, PlannerError, PlanningRequest
from janus.readers import SparkDatasetReader
from janus.registry import SourceNotFoundError
from janus.scripts.checksums import _artifact_format_for_path
from janus.scripts.rehydrate import _build_extraction_result_from_raw
from janus.utils.environment import load_environment_config, materialize_runtime_paths
from janus.utils.storage import StorageLayout, normalize_relative_path

if TYPE_CHECKING:
    from janus.runtime.spark_lifecycle import SparkSessionProvider

_DRAFT_HELP = (
    "Inference reflects only the supplied artifacts. All-null fields and text sentinels such "
    "as 'N/A' can change inferred types; review a draft before activating it."
)


@dataclass(frozen=True, slots=True)
class DraftedContract:
    """Summary of one written contract draft."""

    source_id: str
    output_path: Path
    rows: int
    columns: tuple[str, ...]
    required: tuple[str, ...]
    drafted_from: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "out": str(self.output_path),
            "rows": self.rows,
            "columns": list(self.columns),
            "required": list(self.required),
            "drafted_from": self.drafted_from,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="janus contract",
        allow_abbrev=False,
        description="Inspect and author versioned data contract files.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser(
        "draft",
        prog="janus contract draft",
        allow_abbrev=False,
        description=(
            "Infer one schema from saved raw artifacts or local fixtures and write a draft "
            "contract. This command does not extract data or write to raw or bronze."
        ),
        epilog=_DRAFT_HELP,
    )
    draft.add_argument("--source-id", required=True, help="Configured source to plan.")
    inputs = draft.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--from-raw",
        metavar="RUN_ID",
        help="Read artifacts from this exact previous raw run.",
    )
    inputs.add_argument(
        "--from-fixture",
        action="append",
        metavar="PATH",
        help="Read a local fixture; repeat to combine pages or files.",
    )
    draft.add_argument(
        "--out",
        type=Path,
        help="Write the draft to a project-local path; an explicit path may be replaced.",
    )
    draft.add_argument(
        "--environment",
        default="local",
        help="Environment profile name under conf/environments without the .yaml suffix.",
    )
    draft.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
        help="Project root used to resolve conf/ and data/ paths.",
    )
    draft.add_argument(
        "--include-disabled",
        action="store_true",
        help="Allow planning a configured source that is disabled.",
    )
    return parser


class _DraftArgumentError(ValueError):
    """A parser, project, or source refusal that should exit with status 2."""


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    try:
        planned_run, environment_config, resolved_paths = _prepare_draft_command(args)
    except _DraftArgumentError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from janus.runtime.spark_lifecycle import SparkSessionProvider

    provider = SparkSessionProvider(environment_config, resolved_paths)
    try:
        drafted = draft_contract(
            planned_run,
            provider,
            source=planned_run.plan.source_config,
            environment_config=environment_config,
        )
    except Exception as exc:
        print(f"Contract draft failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(drafted.to_summary(), indent=2, sort_keys=True))
    return 0


def _prepare_draft_command(args: argparse.Namespace):
    project_root = args.project_root.resolve()
    try:
        output_path = (
            _resolve_output_path(project_root, args.out)
            if args.out is not None
            else None
        )
    except ValueError as exc:
        raise _DraftArgumentError(str(exc)) from exc

    input_request = {
        "mode": "raw" if args.from_raw is not None else "fixture",
        "raw_run_id": args.from_raw,
        "fixture_paths": [
            str(_resolve_fixture_path(project_root, value)) for value in (args.from_fixture or ())
        ],
        "output_path": str(output_path) if output_path is not None else "",
    }
    attributes = {_DRAFT_REQUEST_ATTRIBUTE: json.dumps(input_request, sort_keys=True)}

    try:
        environment_config = load_environment_config(args.environment, project_root)
        resolved_paths = materialize_runtime_paths(environment_config, project_root)
        planned_run = Planner().plan(
            PlanningRequest.create(
                source_id=args.source_id,
                environment=args.environment,
                project_root=project_root,
                include_disabled=args.include_disabled,
                attributes=attributes,
            )
        )
    except (
        KeyError,
        OSError,
        PlannerError,
        SourceNotFoundError,
        TypeError,
        yaml.YAMLError,
        ValueError,
    ) as exc:
        raise _DraftArgumentError(str(exc)) from exc

    source = planned_run.plan.source_config
    existing_contract_path = (
        planned_run.plan.data_contract.contract_path.resolve()
        if planned_run.plan.data_contract is not None
        else None
    )
    if output_path is None or existing_contract_path == output_path:
        declared = str(existing_contract_path or source.schema.contract)
        raise _DraftArgumentError(
            f"Source {source.source_id!r} already declares schema.contract at {declared}; "
            "pass --out to write the draft to a different project-local path."
        )

    if any(Path(value).resolve() == output_path for value in input_request["fixture_paths"]):
        raise _DraftArgumentError("--out must not overwrite one of the fixture input files")
    storage_layout = StorageLayout.from_environment_config(environment_config, project_root)
    try:
        _refuse_zone_output(output_path, planned_run, storage_layout)
    except ValueError as exc:
        raise _DraftArgumentError(str(exc)) from exc

    input_request["output_path"] = str(output_path)
    draft_context = planned_run.plan.run_context.with_attribute(
        _DRAFT_REQUEST_ATTRIBUTE,
        json.dumps(input_request, sort_keys=True),
    )
    planned_run = replace(
        planned_run,
        plan=replace(planned_run.plan, run_context=draft_context),
    )
    return planned_run, environment_config, resolved_paths


def draft_contract(
    planned_run: PlannedRun,
    provider: SparkSessionProvider,
    *,
    source: SourceConfig,
    environment_config: Mapping[str, Any],
) -> DraftedContract:
    """Infer a schema under one scoped Spark session, then write a validated draft."""
    plan = planned_run.plan
    if source.source_id != plan.source.source_id:
        raise ValueError("source must match the planned run")

    request = _read_draft_input(plan.run_context.attributes_as_dict())
    output_path = request.output_path
    temporary_raw_root: Path
    with TemporaryDirectory(prefix="janus-contract-draft-") as scratch:
        temporary_raw_root = Path(scratch).resolve()
        source_artifacts = _load_input_artifacts(
            planned_run,
            provider,
            environment_config,
            request,
            temporary_raw_root,
        )
        handoff_plan = _plan_with_temporary_raw_root(plan, temporary_raw_root)
        handoff = planned_run.strategy.build_normalization_handoff(
            handoff_plan,
            source_artifacts,
            hook=planned_run.hook,
        )
        if handoff.is_empty:
            raise ValueError("No artifacts remained after preparing the normalization handoff")

        with provider.scoped() as spark:
            dataframe = SparkDatasetReader().read_extraction_result(
                spark,
                handoff,
                format_name=handoff.single_artifact_format(),
                schema=None,
                options=source.spark.read_options,
            )
            profile = _profile_dataframe(dataframe)

        properties = _draft_properties(profile, source)
        table_name = _bronze_table_name(plan)
        drafted_from = _drafted_from(request, source_artifacts, profile.row_count)
        document = _contract_document(source, table_name, properties, drafted_from)
        serialized = yaml.safe_dump(
            document,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    load_data_contract(output_path)

    column_names = tuple(prop.name for prop in properties)
    return DraftedContract(
        source_id=source.source_id,
        output_path=output_path,
        rows=profile.row_count,
        columns=column_names,
        required=tuple(prop.name for prop in properties if prop.required),
        drafted_from=drafted_from,
    )


def _load_input_artifacts(
    planned_run: PlannedRun,
    provider: SparkSessionProvider,
    environment_config: Mapping[str, Any],
    request: _DraftInput,
    temporary_raw_root: Path,
) -> ExtractionResult:
    plan = planned_run.plan
    if request.mode == "raw":
        if request.raw_run_id is None:
            raise ValueError("--from-raw requires a run id")
        storage_layout = StorageLayout.from_environment_config(
            environment_config,
            plan.run_context.project_root,
        )
        raw_target = storage_layout.resolve_output(plan, "raw")
        resolved_plan = replace(
            plan,
            raw_output=replace(plan.raw_output, path=str(raw_target.resolved_path)),
        )
        return _build_extraction_result_from_raw(
            planned_run,
            resolved_plan,
            provider,
            storage_layout,
            raw_run_id=request.raw_run_id,
            temporary_raw_root=temporary_raw_root,
        )

    artifacts = tuple(
        ExtractedArtifact(
            path=str(path),
            format=(
                _artifact_format_for_path(
                    path,
                    fallback=plan.source_config.spark.input_format,
                )
                if planned_run.hook is not None
                else plan.source_config.spark.input_format
            ),
        )
        for path in request.fixture_paths
    )
    if not artifacts:
        raise ValueError("--from-fixture requires at least one fixture path")
    return ExtractionResult.from_plan(plan, artifacts)


def _plan_with_temporary_raw_root(
    plan: ExecutionPlan, temporary_raw_root: Path
) -> ExecutionPlan:
    return replace(
        plan,
        raw_output=replace(plan.raw_output, path=str(temporary_raw_root.resolve())),
    )


def _resolve_fixture_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def _resolve_output_path(project_root: Path, configured: Path) -> Path:
    root = project_root.resolve()
    if configured.is_absolute():
        candidate = configured.resolve()
    else:
        relative = normalize_relative_path(configured)
        candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"--out must resolve inside the project root {root}: {candidate}")
    return candidate


def _refuse_zone_output(
    output_path: Path,
    planned_run: PlannedRun,
    storage_layout: StorageLayout,
) -> None:
    for zone in ("raw", "bronze"):
        roots = (
            storage_layout.zone_path(zone).resolve(),
            storage_layout.resolve_output(planned_run.plan, zone).resolved_path.resolve(),
        )
        if any(output_path == root or output_path.is_relative_to(root) for root in roots):
            raise ValueError(f"--out must not write into the {zone} zone: {output_path}")


def _bronze_table_name(plan: ExecutionPlan) -> str:
    table_name = plan.bronze_output.table_name or plan.source.source_id
    relative = normalize_relative_path(Path(table_name))
    if len(relative.parts) != 1:
        raise ValueError(f"Bronze table name must be one path segment: {table_name!r}")
    return relative.as_posix()


__all__ = ["DraftedContract", "build_parser", "draft_contract", "main"]
