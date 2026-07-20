import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from janus.main import main
from janus.models import ExecutionPlan, ExtractionResult, RunContext, WriteResult
from janus.models.source_config import SourceConfig
from janus.planner import (
    HookCatalog,
    HookResolutionError,
    Planner,
    PlanningRequest,
    StrategyBinding,
    StrategyCatalog,
    StrategyResolutionError,
)
from janus.strategies.base import BaseStrategy, SourceHook

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RecordingStrategy(BaseStrategy):
    def __init__(self, family: str, label: str) -> None:
        self._family = family
        self.label = label
        self.plan_calls: list[tuple[str, str, bool]] = []

    @property
    def strategy_family(self) -> str:
        return self._family

    def plan(
        self,
        source_config: SourceConfig,
        run_context: RunContext,
        hook: SourceHook | None = None,
    ) -> ExecutionPlan:
        self.plan_calls.append(
            (source_config.source_id, source_config.strategy_variant, hook is not None)
        )
        plan = ExecutionPlan.from_source_config(source_config, run_context).with_note(
            f"strategy:{self.label}"
        )
        if hook is not None:
            return hook.on_plan(plan)
        return plan

    def extract(
        self,
        plan: ExecutionPlan,
        hook: SourceHook | None = None,
        *,
        spark=None,
    ) -> ExtractionResult:
        del plan
        del hook
        del spark
        raise NotImplementedError

    def build_normalization_handoff(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        hook: SourceHook | None = None,
    ) -> ExtractionResult:
        del plan
        del extraction_result
        del hook
        raise NotImplementedError

    def emit_metadata(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
        hook: SourceHook | None = None,
    ) -> dict[str, Any]:
        del plan
        del extraction_result
        del write_results
        del hook
        raise NotImplementedError


class PlanningHook(SourceHook):
    def on_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        return plan.with_note("hook:resolved")


def test_planner_builds_execution_plan_from_checked_in_source():
    planner = Planner()
    request = PlanningRequest.create(
        source_id="federal_open_data_example",
        environment="local",
        project_root=PROJECT_ROOT,
        run_id="run-20260408-001",
        started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
        attributes={"trigger": "unit-test"},
    )

    planned_run = planner.plan(request)

    assert planned_run.strategy.strategy_family == "api"
    assert planned_run.dispatch_path == "api.page_number_api"
    assert planned_run.plan.source.source_id == "federal_open_data_example"
    assert planned_run.plan.extraction_mode == "incremental"
    assert planned_run.plan.checkpoint_strategy == "max_value"
    assert planned_run.plan.checkpoint_field == "updated_at"
    assert planned_run.plan.notes == (
        "strategy_family:api",
        "strategy_variant:page_number_api",
        "dispatch:api.page_number_api",
    )
    assert planned_run.plan.run_context.attributes_as_dict() == {
        "source_id": "federal_open_data_example",
        "source_type": "api",
        "strategy": "api",
        "strategy_variant": "page_number_api",
        "trigger": "unit-test",
    }
    assert planned_run.pre_run_metadata_as_dict()["source_config_path"].endswith(
        "conf/sources/example/example_source.yaml"
    )
    assert planned_run.pre_run_metadata_as_dict()["raw_output_path"] == (
        "data/raw/example/federal_open_data_example"
    )


def test_planner_selects_strategy_binding_from_source_metadata(tmp_path):
    project_root = _create_project(
        tmp_path,
        _source_yaml("planner_variant_source", variant="offset_api"),
    )
    page_number_strategy = RecordingStrategy("api", "page-number")
    offset_strategy = RecordingStrategy("api", "offset")
    planner = Planner(
        strategy_catalog=StrategyCatalog(
            bindings=(
                StrategyBinding(
                    family="api",
                    variant="page_number_api",
                    strategy=page_number_strategy,
                ),
                StrategyBinding(
                    family="api",
                    variant="offset_api",
                    strategy=offset_strategy,
                ),
            )
        )
    )

    planned_run = planner.plan(
        PlanningRequest.create(
            source_id="planner_variant_source",
            environment="local",
            project_root=project_root,
            run_id="run-variant-001",
            started_at=datetime(2026, 4, 8, 12, 30, tzinfo=UTC),
        )
    )

    assert planned_run.strategy is offset_strategy
    assert page_number_strategy.plan_calls == []
    assert offset_strategy.plan_calls == [
        ("planner_variant_source", "offset_api", False),
    ]
    assert planned_run.plan.notes == ("strategy:offset", "dispatch:api.offset_api")


def test_planner_reports_missing_strategy_variant_with_actionable_error(tmp_path):
    project_root = _create_project(
        tmp_path,
        _source_yaml("planner_variant_source", variant="offset_api"),
    )
    planner = Planner(
        strategy_catalog=StrategyCatalog(
            bindings=(
                StrategyBinding(
                    family="api",
                    variant="page_number_api",
                    strategy=RecordingStrategy("api", "page-number"),
                ),
            )
        )
    )

    with pytest.raises(StrategyResolutionError) as exc_info:
        planner.plan(
            PlanningRequest.create(
                source_id="planner_variant_source",
                environment="local",
                project_root=project_root,
                run_id="run-variant-002",
                started_at=datetime(2026, 4, 8, 13, 0, tzinfo=UTC),
            )
        )

    message = str(exc_info.value)
    assert "family 'api' variant 'offset_api'" in message
    assert "source 'planner_variant_source'" in message
    assert "Registered variants for family 'api': page_number_api" in message


def test_planner_resolves_registered_source_hook(tmp_path):
    project_root = _create_project(
        tmp_path,
        _source_yaml(
            "hooked_source",
            variant="page_number_api",
            source_hook="example.hook",
        ),
    )
    strategy = RecordingStrategy("api", "hooked")
    planner = Planner(
        strategy_catalog=StrategyCatalog(
            bindings=(
                StrategyBinding(
                    family="api",
                    variant="page_number_api",
                    strategy=strategy,
                ),
            )
        ),
        hook_catalog=HookCatalog(hooks=(("example.hook", PlanningHook()),)),
    )

    planned_run = planner.plan(
        PlanningRequest.create(
            source_id="hooked_source",
            environment="local",
            project_root=project_root,
            run_id="run-hook-001",
            started_at=datetime(2026, 4, 8, 13, 30, tzinfo=UTC),
        )
    )

    assert planned_run.plan.notes == (
        "strategy:hooked",
        "hook:resolved",
        "dispatch:api.page_number_api",
    )
    assert strategy.plan_calls == [("hooked_source", "page_number_api", True)]
    assert planned_run.pre_run_metadata_as_dict()["hook_id"] == "example.hook"


def test_planner_reports_missing_hook_binding(tmp_path):
    project_root = _create_project(
        tmp_path,
        _source_yaml(
            "hooked_source",
            variant="page_number_api",
            source_hook="example.hook",
        ),
    )
    planner = Planner(
        strategy_catalog=StrategyCatalog(
            bindings=(
                StrategyBinding(
                    family="api",
                    variant="page_number_api",
                    strategy=RecordingStrategy("api", "hooked"),
                ),
            )
        )
    )

    with pytest.raises(HookResolutionError, match="requires hook 'example.hook'"):
        planner.plan(
            PlanningRequest.create(
                source_id="hooked_source",
                environment="local",
                project_root=project_root,
                run_id="run-hook-002",
                started_at=datetime(2026, 4, 8, 14, 0, tzinfo=UTC),
            )
        )


def test_planner_carries_request_input_contract_forward_without_changing_dispatch(tmp_path):
    source_yaml = _source_yaml("parameterized_source", variant="page_number_api").replace(
        "  auth:\n"
        "    type: none\n",
        "  request_inputs:\n"
        "    type: date_window\n"
        "    start: 2025-01-01\n"
        "    end: 2025-03-31\n"
        "    step: month\n"
        "  parameter_bindings:\n"
        "    mesAno:\n"
        "      from: request_input.window_end\n"
        "      format: \"%Y%m\"\n"
        "  auth:\n"
        "    type: none\n",
        1,
    )
    project_root = _create_project(tmp_path, source_yaml)

    planned_run = Planner().plan(
        PlanningRequest.create(
            source_id="parameterized_source",
            environment="local",
            project_root=project_root,
            run_id="run-parameterized-001",
            started_at=datetime(2026, 4, 8, 14, 15, tzinfo=UTC),
        )
    )

    assert planned_run.dispatch_path == "api.page_number_api"
    assert planned_run.plan.source_config.access.request_inputs.type == "date_window"
    assert planned_run.plan.source_config.access.request_inputs.start.isoformat() == "2025-01-01"
    assert planned_run.plan.source_config.access.request_inputs.end.isoformat() == "2025-03-31"
    assert planned_run.plan.source_config.access.parameter_bindings is not None
    assert (
        planned_run.plan.source_config.access.parameter_bindings["mesAno"].from_
        == "request_input.window_end"
    )
    assert planned_run.plan.notes == (
        "strategy_family:api",
        "strategy_variant:page_number_api",
        "dispatch:api.page_number_api",
    )


def test_planner_can_include_disabled_source_when_requested(tmp_path):
    project_root = _create_project(
        tmp_path,
        _source_yaml("disabled_cli_source", variant="page_number_api", enabled=False),
        include_environment=True,
    )

    planned_run = Planner().plan(
        PlanningRequest.create(
            source_id="disabled_cli_source",
            environment="local",
            project_root=project_root,
            run_id="run-disabled-001",
            started_at=datetime(2026, 4, 8, 14, 30, tzinfo=UTC),
            include_disabled=True,
        )
    )

    assert planned_run.plan.source.source_id == "disabled_cli_source"


def test_main_prints_planned_run_summary_for_one_source(tmp_path, capsys):
    project_root = _create_project(
        tmp_path,
        _source_yaml("cli_source", variant="page_number_api"),
        include_environment=True,
    )

    exit_code = main(
        [
            "--environment",
            "local",
            "--project-root",
            str(project_root),
            "--source-id",
            "cli_source",
            "--run-id",
            "run-cli-001",
            "--started-at",
            "2026-04-08T15:00:00+00:00",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["environment"] == "local"
    assert payload["planned_run"]["run"]["run_id"] == "run-cli-001"
    assert payload["planned_run"]["strategy"]["dispatch_path"] == "api.page_number_api"
    assert payload["planned_run"]["source"]["source_id"] == "cli_source"
    assert payload["paths"]["raw_dir"] == str(project_root / "data" / "raw")


def test_main_executes_source_through_framework_runtime(tmp_path, capsys, monkeypatch):
    project_root = _create_project(
        tmp_path,
        _source_yaml("cli_source", variant="page_number_api"),
        include_environment=True,
    )
    stopped = []
    built = []

    class FakeSparkSession:
        sparkContext = SimpleNamespace(appName="janus-exec-test", master="local[1]")

        def stop(self):
            stopped.append(True)

    class FakeExecutedRun:
        is_successful = True

        def to_summary(self):
            return {
                "status": "succeeded",
                "artifact_count": 1,
                "validation": {"is_successful": True},
            }

    class FakeSourceExecutor:
        def __init__(self, logger=None):
            self.logger = logger

        def execute(self, planned_run, spark_provider, environment_config):
            assert planned_run.plan.source.source_id == "cli_source"
            assert environment_config["name"] == "local"
            # The CLI hands over a provider, not a session: nothing is built until the
            # executor reaches its materialization boundary.
            assert built == []
            assert spark_provider.get().sparkContext.appName == "janus-exec-test"
            return FakeExecutedRun()

    def _build_session(config, paths):
        del config
        del paths
        built.append(True)
        return FakeSparkSession()

    monkeypatch.setattr("janus.runtime.spark_lifecycle.build_spark_session", _build_session)
    monkeypatch.setattr("janus.main.SourceExecutor", FakeSourceExecutor)

    exit_code = main(
        [
            "--environment",
            "local",
            "--project-root",
            str(project_root),
            "--source-id",
            "cli_source",
            "--run-id",
            "run-cli-exec-001",
            "--started-at",
            "2026-04-08T16:00:00+00:00",
            "--execute",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    log_events = [
        json.loads(line)["event"] for line in captured.err.splitlines() if line.strip()
    ]

    assert exit_code == 0
    assert "cli_execution_requested" in log_events
    assert "spark_session_started" in log_events
    assert "spark_session_stopped" in log_events
    assert payload["planned_run"]["run"]["run_id"] == "run-cli-exec-001"
    assert payload["executed_run"]["status"] == "succeeded"
    assert payload["spark_session"]["app_name"] == "janus-exec-test"
    assert stopped == [True]


def test_main_ingests_existing_raw_artifacts_into_requested_bronze_table(
    tmp_path,
    capsys,
    monkeypatch,
):
    project_root = _create_project(
        tmp_path,
        _source_yaml("cli_source", variant="page_number_api"),
        include_environment=True,
    )
    stopped = []

    class FakeSparkSession:
        sparkContext = SimpleNamespace(appName="janus-raw-to-bronze-test", master="local[1]")

        def stop(self):
            stopped.append(True)

    class FakeRawToBronzeRun:
        is_successful = True

        def to_summary(self):
            return {
                "status": "succeeded",
                "target_table": "curated.cli_source_reloaded",
                "raw_artifact_count": 2,
            }

    def fake_ingest_raw_to_bronze(planned_run, spark, environment_config, *, bronze_table, logger):
        assert planned_run.plan.source.source_id == "cli_source"
        assert spark.sparkContext.appName == "janus-raw-to-bronze-test"
        assert environment_config["name"] == "local"
        assert bronze_table == "curated.cli_source_reloaded"
        assert logger is not None
        return FakeRawToBronzeRun()

    monkeypatch.setattr("janus.main.build_spark_session", lambda config, paths: FakeSparkSession())
    monkeypatch.setattr("janus.main.ingest_raw_to_bronze", fake_ingest_raw_to_bronze)

    exit_code = main(
        [
            "--environment",
            "local",
            "--project-root",
            str(project_root),
            "--source-id",
            "cli_source",
            "--run-id",
            "run-cli-raw-to-bronze-001",
            "--started-at",
            "2026-04-08T16:30:00+00:00",
            "--ingest-raw-to-bronze",
            "--bronze-table",
            "curated.cli_source_reloaded",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    log_events = [
        json.loads(line)["event"] for line in captured.err.splitlines() if line.strip()
    ]

    assert exit_code == 0
    assert "cli_raw_to_bronze_requested" in log_events
    assert "spark_session_started" in log_events
    assert "spark_session_stopped" in log_events
    assert payload["planned_run"]["run"]["run_id"] == "run-cli-raw-to-bronze-001"
    assert payload["raw_to_bronze_run"]["status"] == "succeeded"
    assert payload["raw_to_bronze_run"]["target_table"] == "curated.cli_source_reloaded"
    assert payload["spark_session"]["app_name"] == "janus-raw-to-bronze-test"
    assert stopped == [True]


def _create_project(
    tmp_path: Path,
    source_yaml: str,
    *,
    include_environment: bool = False,
) -> Path:
    conf_dir = tmp_path / "conf"
    sources_dir = conf_dir / "sources"
    sources_dir.mkdir(parents=True)
    (conf_dir / "app.yaml").write_text(
        "registry:\n  sources_dir: conf/sources\n  file_pattern: \"*.yaml\"\n",
        encoding="utf-8",
    )
    (sources_dir / "source.yaml").write_text(source_yaml.lstrip(), encoding="utf-8")

    if include_environment:
        environments_dir = conf_dir / "environments"
        environments_dir.mkdir(parents=True)
        (environments_dir / "local.yaml").write_text(
            """
name: local

runtime:
  log_level: INFO

spark:
  app_name: janus-test
  master: local[1]
  warehouse_dir: data/metadata/spark-warehouse
  config: {}

storage:
  root_dir: data
  raw_dir: data/raw
  bronze_dir: data/bronze
  metadata_dir: data/metadata
""".lstrip(),
            encoding="utf-8",
        )

    return tmp_path


def _source_yaml(
    source_id: str,
    *,
    variant: str,
    source_hook: str | None = None,
    enabled: bool = True,
) -> str:
    source_hook_line = f"source_hook: {source_hook}\n" if source_hook else ""
    enabled_line = "true" if enabled else "false"
    return f"""
source_id: {source_id}
name: {source_id}
owner: janus
enabled: {enabled_line}
source_type: api
strategy: api
strategy_variant: {variant}
federation_level: federal
domain: example
public_access: true
{source_hook_line}
access:
  base_url: https://example.invalid
  path: /records
  method: GET
  format: json
  timeout_seconds: 30
  auth:
    type: none
  pagination:
{_pagination_yaml(variant)}
  rate_limit:
    requests_per_minute: 10
    concurrency: 1

extraction:
  mode: full_refresh
  retry:
    max_attempts: 3
    backoff_strategy: fixed
    backoff_seconds: 1

schema:
  mode: infer

spark:
  input_format: json
  write_mode: append

outputs:
  raw:
    path: data/raw/example/{source_id}
    format: json
  bronze:
    path: data/bronze/example/{source_id}
    format: iceberg
  metadata:
    path: data/metadata/example/{source_id}
    format: json

quality:
  allow_schema_evolution: true
"""


def _pagination_yaml(variant: str) -> str:
    if variant == "offset_api":
        return """
    type: offset
    offset_param: offset
    limit_param: limit
    page_size: 100
""".rstrip()

    return """
    type: page_number
    page_param: page
    size_param: page_size
    page_size: 100
""".rstrip()
