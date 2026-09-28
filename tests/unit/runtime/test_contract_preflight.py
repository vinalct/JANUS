"""The execution preflight: the live table checked against the contract before any request."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from test_spark_lifecycle_evidence import ArmedSpyProvider

from janus.lineage import RunObserver
from janus.models import ExecutionPlan, ExtractionResult, RunContext
from janus.models.data_contracts import VOCABULARY, DataContract, load_data_contract
from janus.normalizers import NORMALIZATION_METADATA_COLUMNS
from janus.planner import PlannedRun
from janus.quality import QualityGate, ValidationReportStore
from janus.registry import load_registry
from janus.runtime import SourceExecutor
from janus.utils.catalog_options import HADOOP_CATALOG_TYPE
from janus.utils.catalog_properties import (
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
)
from janus.utils.logging import build_structured_logger
from janus.utils.storage import StorageLayout
from tests.support.spark_sessions import sqlite_catalog_target

pytestmark = pytest.mark.xfail(strict=True, reason="red until implementation finishes")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
SOURCE_ID = "federal_open_data_example"
IDENTIFIER = "bronze_example.federal_open_data_example"
STARTED_AT = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

ICEBERG_SPELLINGS = {
    "boolean": "boolean",
    "integer": "int",
    "long": "long",
    "float": "float",
    "double": "double",
    "decimal(p,s)": "decimal(18, 2)",
    "string": "string",
    "binary": "binary",
    "date": "date",
    "timestamp": "timestamp",
    "timestamptz": "timestamptz",
}


# ── the stand-in catalog ─────────────────────────────────────────────────────


class NoSuchTableError(Exception):
    pass


class NoSuchNamespaceError(Exception):
    pass


class IcebergTypeText:
    """What the preflight reads from a PyIceberg type: its ``str()``."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


@dataclass
class IcebergField:
    name: str
    field_type: IcebergTypeText
    required: bool = False


@dataclass
class StandInTable:
    fields: tuple[IcebergField, ...]
    properties: dict[str, str] = field(default_factory=dict)

    def schema(self) -> Any:
        return types.SimpleNamespace(fields=self.fields)


@dataclass
class StandInCatalog:
    table: StandInTable | None = None
    error: Exception | None = None
    loaded: list[str] = field(default_factory=list)

    def load_table(self, identifier: str) -> StandInTable:
        self.loaded.append(identifier)
        if self.error is not None:
            raise self.error
        assert self.table is not None
        return self.table


def _install_pyiceberg(monkeypatch: pytest.MonkeyPatch, catalog: StandInCatalog) -> list[Any]:
    """Shadow ``pyiceberg`` with stand-ins; return the recorded ``load_catalog`` calls."""
    calls: list[Any] = []

    def load_catalog(name: str, **properties: str) -> StandInCatalog:
        calls.append((name, properties))
        return catalog

    package = types.ModuleType("pyiceberg")
    package.__path__ = []  
    catalog_module = types.ModuleType("pyiceberg.catalog")
    catalog_module.load_catalog = load_catalog  
    exceptions_module = types.ModuleType("pyiceberg.exceptions")
    exceptions_module.NoSuchTableError = NoSuchTableError  
    exceptions_module.NoSuchNamespaceError = NoSuchNamespaceError  
    package.catalog = catalog_module  
    package.exceptions = exceptions_module 
    monkeypatch.setitem(sys.modules, "pyiceberg", package)
    monkeypatch.setitem(sys.modules, "pyiceberg.catalog", catalog_module)
    monkeypatch.setitem(sys.modules, "pyiceberg.exceptions", exceptions_module)
    return calls


def _table_like(name: str, *, stamp: str | None = "1.0.0", extra=()) -> StandInTable:
    shape = load_data_contract(HOSTILE / f"{name}.yaml")
    iceberg = {entry.name: entry.iceberg_name for entry in VOCABULARY}
    columns = [
        IcebergField(
            prop.name,
            IcebergTypeText(
                prop.physical_type.replace(",", ", ")
                if prop.physical_type.startswith("decimal")
                else iceberg[prop.physical_type]
            ),
        )
        for prop in shape.schema.properties
    ]
    columns.extend(IcebergField(column, IcebergTypeText("string")) for column in extra)
    properties = {"janus.contract_version": stamp} if stamp is not None else {}
    return StandInTable(tuple(columns), properties)


# ── builders ─────────────────────────────────────────────────────────────────


def _contract(name: str) -> DataContract:
    return load_data_contract(HOSTILE / f"{name}.yaml")


def _live(name: str, *, stamp: str | None = "1.0.0") -> Any:
    from janus.runtime.contract_preflight import LiveTable
    from janus.writers.evolution import LiveColumn

    shape = _contract(name)
    return LiveTable(
        columns=tuple(
            LiveColumn(prop.name, prop.physical_type, False) for prop in shape.schema.properties
        ),
        recorded_contract_version=stamp,
    )


def _plan(tmp_path: Path, contract: DataContract) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID)
    run_context = RunContext.create(
        run_id="run-preflight-001",
        environment="local",
        project_root=tmp_path,
        started_at=STARTED_AT,
    )
    return ExecutionPlan.from_source_config(source_config, run_context, data_contract=contract)


def _catalog_config(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    target = sqlite_catalog_target(tmp_path / "catalog")
    return target.environment_config(), dict(target.resolved_paths)


def _events(stream: StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ── the pure decision ────────────────────────────────────────────────────────


def test_the_outcomes_the_attribute_and_the_budget_are_declared_once():
    from janus.runtime.contract_preflight import (
        DEFAULT_PREFLIGHT_BUDGET_SECONDS,
        PREFLIGHT_ATTRIBUTE,
        PREFLIGHT_OUTCOMES,
    )

    assert frozenset({"ok", "will_evolve", "refused", "table_missing", "catalog_unavailable"}) == (
        PREFLIGHT_OUTCOMES
    )
    assert PREFLIGHT_ATTRIBUTE == "contract_preflight_outcome"
    assert DEFAULT_PREFLIGHT_BUDGET_SECONDS == 5.0


@pytest.mark.parametrize(
    ("contract_name", "live_name", "catalog_error", "strategy", "outcome", "reason_fragment"),
    [
        ("base", None, "OperationalError", "insert", "catalog_unavailable", "OperationalError"),
        ("base", None, None, "insert", "table_missing", ""),
        ("base", "base", None, "insert", "ok", ""),
        ("base_plus_nullable", "base", None, "insert", "will_evolve", "added:note"),
        ("base_v2", "base", None, "replace_table", "will_evolve", "breaking_replace"),
        ("base", "base_int", None, "insert", "refused", "amount"),
        ("base_v2", "base", None, "insert", "refused", "label"),
    ],
    ids=[
        "catalog_unavailable",
        "table_missing",
        "ok",
        "will_evolve-addition",
        "will_evolve-breaking_replace",
        "refused-promotion_under_additive",
        "refused-major_bump_on_append",
    ],
)
def test_decide_preflight_reaches_every_outcome(
    contract_name, live_name, catalog_error, strategy, outcome, reason_fragment
):
    from janus.runtime.contract_preflight import decide_preflight

    result = decide_preflight(
        contract=_contract(contract_name),
        live=None if live_name is None else _live(live_name),
        catalog_error=catalog_error,
        write_strategy=strategy,
    )

    assert result.outcome == outcome
    assert reason_fragment in result.reason
    assert (result.plan is None) is (outcome in {"catalog_unavailable", "table_missing"})


def test_the_eight_normalization_columns_are_not_drift():
    from janus.runtime.contract_preflight import decide_preflight
    from janus.writers.evolution import LiveColumn

    live = _live("base")
    live = replace(
        live,
        columns=(
            *live.columns,
            *(LiveColumn(column, "string", False) for column in NORMALIZATION_METADATA_COLUMNS),
        ),
    )

    result = decide_preflight(
        contract=_contract("base"), live=live, catalog_error=None, write_strategy="insert"
    )

    assert result.outcome == "ok"


# ── the PyIceberg loader ─────────────────────────────────────────────────────


def test_the_loader_derives_the_catalog_once_and_maps_every_scalar(monkeypatch, tmp_path):
    from janus.runtime.contract_preflight import load_live_table

    fields = tuple(
        IcebergField(f"c_{index}", IcebergTypeText(spelling), required=index == 0)
        for index, spelling in enumerate(ICEBERG_SPELLINGS.values())
    )
    catalog = StandInCatalog(StandInTable(fields, {"janus.contract_version": "1.2.0"}))
    calls = _install_pyiceberg(monkeypatch, catalog)
    config, resolved_paths = _catalog_config(tmp_path)

    live = load_live_table(config, resolved_paths, IDENTIFIER)

    assert calls == [
        (
            derive_pyiceberg_catalog_name(config),
            derive_pyiceberg_catalog_properties(config, resolved_paths),
        )
    ]
    assert catalog.loaded == [IDENTIFIER]
    assert [column.physical_type for column in live.columns] == [
        "decimal(18,2)" if name == "decimal(p,s)" else name for name in ICEBERG_SPELLINGS
    ]
    assert live.columns[0].required is True
    assert live.recorded_contract_version == "1.2.0"


@pytest.mark.parametrize("missing", [NoSuchTableError, NoSuchNamespaceError])
def test_a_missing_table_or_namespace_is_no_table(monkeypatch, tmp_path, missing):
    from janus.runtime.contract_preflight import load_live_table

    _install_pyiceberg(monkeypatch, StandInCatalog(error=missing(IDENTIFIER)))
    config, resolved_paths = _catalog_config(tmp_path)

    assert load_live_table(config, resolved_paths, IDENTIFIER) is None


def test_an_unstamped_table_records_no_version(monkeypatch, tmp_path):
    from janus.runtime.contract_preflight import load_live_table

    _install_pyiceberg(monkeypatch, StandInCatalog(_table_like("base", stamp=None)))
    config, resolved_paths = _catalog_config(tmp_path)

    live = load_live_table(config, resolved_paths, IDENTIFIER)

    assert live.recorded_contract_version is None


# ── the bounded run ──────────────────────────────────────────────────────────


def _run(tmp_path: Path, contract_name: str, loader: Any, **options: Any) -> Any:
    from janus.runtime.contract_preflight import run_contract_preflight

    config, resolved_paths = _catalog_config(tmp_path)
    return run_contract_preflight(
        _plan(tmp_path, _contract(contract_name)),
        config,
        resolved_paths,
        identifier=IDENTIFIER,
        logger=build_structured_logger("janus.tests.preflight", stream=StringIO()),
        loader=loader,
        **options,
    )


def test_a_missing_table_is_a_first_write(tmp_path):
    result = _run(tmp_path, "base", lambda *args, **kwargs: None)

    assert result.outcome == "table_missing"


def test_a_loader_that_raises_is_an_unavailable_catalog(tmp_path):
    def raising(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("catalog database is locked")

    result = _run(tmp_path, "base", raising)

    assert result.outcome == "catalog_unavailable"
    assert "RuntimeError" in result.reason


def test_a_hanging_catalog_is_cut_off_by_the_budget(tmp_path):
    budget = 0.5

    def hanging(*args: Any, **kwargs: Any) -> Any:
        time.sleep(budget + 3)

    started = time.monotonic()
    result = _run(tmp_path, "base", hanging, budget_seconds=budget)
    elapsed = time.monotonic() - started

    assert result.outcome == "catalog_unavailable"
    assert elapsed < budget + 0.5
    assert result.duration_seconds < budget + 0.5


def test_the_pre_migration_catalog_is_unavailable_not_a_crash(tmp_path):
    """D-19: PyIceberg implements no such catalog; the derivation's named error is the reason."""
    from janus.runtime.contract_preflight import load_live_table, run_contract_preflight

    config = {
        "spark": {
            "iceberg": {
                "catalog_type": HADOOP_CATALOG_TYPE,
                "catalog_name": "janus",
                "warehouse_dir": str(tmp_path / "warehouse"),
            }
        }
    }

    result = run_contract_preflight(
        _plan(tmp_path, _contract("base")),
        config,
        {},
        identifier=IDENTIFIER,
        logger=build_structured_logger("janus.tests.preflight", stream=StringIO()),
        loader=load_live_table,
    )

    assert result.outcome == "catalog_unavailable"
    assert re.search(r"hadoop.?catalog.?unrepresentable", result.reason, flags=re.IGNORECASE)


def test_the_reason_never_carries_the_catalog_uri(tmp_path):
    def leaking(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("could not open postgresql://janus:hunter22@db.internal/iceberg")

    result = _run(tmp_path, "base", leaking)

    assert "hunter22" not in result.reason


# ── enforcement ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("outcome", ["refused", "catalog_unavailable"])
def test_strict_fails_closed(outcome):
    from janus.quality.contract_checks import ContractEnforcementError
    from janus.runtime.contract_preflight import (
        ContractPreflightError,
        PreflightResult,
        enforce_preflight,
    )

    stream = StringIO()
    result = PreflightResult(
        outcome=outcome, reason="amount: retyped", plan=None, duration_seconds=0.1
    )

    with pytest.raises(ContractPreflightError) as raised:
        enforce_preflight(
            result,
            enforcement="strict",
            logger=build_structured_logger("janus.tests.preflight", stream=stream),
        )

    assert isinstance(raised.value, ContractEnforcementError)
    assert raised.value.failure_stage == "contract_preflight"
    assert f"preflight {outcome}" in str(raised.value)


@pytest.mark.parametrize("outcome", ["refused", "catalog_unavailable"])
def test_lenient_warns_once_and_proceeds(outcome):
    from janus.runtime.contract_preflight import PreflightResult, enforce_preflight

    stream = StringIO()
    result = PreflightResult(
        outcome=outcome, reason="amount: retyped", plan=None, duration_seconds=0.1
    )

    enforce_preflight(
        result,
        enforcement="lenient",
        logger=build_structured_logger("janus.tests.preflight", stream=stream),
    )

    warnings = [event for event in _events(stream) if event["level"] == "WARNING"]
    assert [event["event"] for event in warnings] == ["contract_preflight_warning"]


@pytest.mark.parametrize("outcome", ["ok", "will_evolve", "table_missing"])
def test_the_outcomes_that_may_proceed_never_raise_in_strict(outcome):
    from janus.runtime.contract_preflight import PreflightResult, enforce_preflight

    stream = StringIO()
    result = PreflightResult(outcome=outcome, reason="", plan=None, duration_seconds=0.1)

    enforce_preflight(
        result,
        enforcement="strict",
        logger=build_structured_logger("janus.tests.preflight", stream=stream),
    )

    assert not [event for event in _events(stream) if event["level"] == "WARNING"]


# ── the executor runs it before extraction, session-free ─────────────────────


@dataclass(slots=True)
class RecordingStrategy:
    calls: list[str]

    @property
    def strategy_family(self) -> str:
        return "api"

    def plan(self, source_config, run_context, hook=None):
        raise NotImplementedError

    def extract(self, plan, hook=None, *, spark=None):
        del hook, spark
        self.calls.append("extract")
        return ExtractionResult.from_plan(plan, artifacts=(), records_extracted=0)

    def build_normalization_handoff(self, plan, extraction_result, hook=None):
        del plan, hook
        return extraction_result

    def emit_metadata(self, plan, extraction_result, write_results=(), hook=None):
        del plan, extraction_result, write_results, hook
        return {}


def _executor(tmp_path: Path, loader: Any) -> SourceExecutor:
    return SourceExecutor(
        quality_gate=QualityGate(ValidationReportStore()),
        observer=RunObserver(),
        storage_layout_resolver=lambda plan, config: StorageLayout.from_environment_config(
            {
                "storage": {
                    "root_dir": str(tmp_path / "data"),
                    "raw_dir": str(tmp_path / "data" / "raw"),
                    "bronze_dir": str(tmp_path / "data" / "bronze"),
                    "metadata_dir": str(tmp_path / "data" / "metadata"),
                }
            },
            tmp_path,
        ),
        preflight_loader=loader,
    )


def _refusing_loader(*args: Any, **kwargs: Any) -> Any:
    from janus.runtime.contract_preflight import LiveTable
    from janus.writers.evolution import LiveColumn

    shape = _contract("base_int")
    return LiveTable(
        columns=tuple(LiveColumn(p.name, p.physical_type, False) for p in shape.schema.properties),
        recorded_contract_version="1.0.0",
    )


def test_a_strict_refusal_fails_the_run_before_extraction_and_starts_no_session(tmp_path):
    calls: list[str] = []
    planned_run = PlannedRun(
        plan=_plan(tmp_path, _contract("base")), strategy=RecordingStrategy(calls), hook=None
    )
    provider = ArmedSpyProvider()  # never armed: any get() fails the run with an AssertionError

    run = _executor(tmp_path, _refusing_loader).execute(planned_run, provider, {})

    assert run.status == "failed"
    assert run.error_type == "ContractPreflightError"
    assert run.failure_stage == "contract_preflight"
    assert "extract" not in calls
    assert run.write_results == ()
    assert provider.was_started is False
    metadata = json.loads(run.run_metadata_path.read_text(encoding="utf-8"))
    assert metadata["failure_stage"] == "contract_preflight"
    assert metadata["run_attributes"]["contract_preflight_outcome"] == "refused"


def test_a_lenient_refusal_warns_and_the_run_proceeds(tmp_path):
    calls: list[str] = []
    planned_run = PlannedRun(
        plan=_plan(tmp_path, _contract("base_lenient")),
        strategy=RecordingStrategy(calls),
        hook=None,
    )

    run = _executor(tmp_path, _refusing_loader).execute(planned_run, ArmedSpyProvider(), {})

    assert "extract" in calls
    assert run.status == "succeeded", run.failure_reason
    metadata = json.loads(run.run_metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_attributes"]["contract_preflight_outcome"] == "refused"


# ── engine-free import ───────────────────────────────────────────────────────


def test_importing_the_preflight_module_loads_no_engine():
    import_paths = (str(PROJECT_ROOT / "src"), *(entry for entry in sys.path if entry))
    command = (
        "import sys; "
        f"sys.path[:0] = {import_paths!r}; "
        "import janus.runtime.contract_preflight; "
        "assert not {'pyspark', 'pyiceberg', 'pyarrow'} & sys.modules.keys()"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
