"""Session-free comparison of the declared contract with the live Iceberg table."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from janus.models import ExecutionPlan, resolve_bronze_write_intent
from janus.models.data_contracts import DataContract, physical_type_from_iceberg_name
from janus.quality import ContractEnforcementError
from janus.utils.catalog_properties import (
    derive_pyiceberg_catalog_name,
    derive_pyiceberg_catalog_properties,
)
from janus.utils.environment import RuntimeLocation
from janus.utils.logging import StructuredLogger
from janus.writers.evolution import EvolutionPlan, LiveColumn, plan_schema_evolution

if TYPE_CHECKING:
    from janus.lineage import RunObserver
    from janus.planner import PlannedRun


PREFLIGHT_OUTCOMES = frozenset(
    {"ok", "will_evolve", "refused", "table_missing", "catalog_unavailable"}
)
PREFLIGHT_ATTRIBUTE = "contract_preflight_outcome"
DEFAULT_PREFLIGHT_BUDGET_SECONDS = 5.0
_LOG = logging.getLogger(__name__)
_ERROR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,99}$")
_MAX_REASON_LENGTH = 500


@dataclass(frozen=True, slots=True)
class LiveTable:
    columns: tuple[LiveColumn, ...]
    recorded_contract_version: str | None


@dataclass(frozen=True, slots=True)
class PreflightResult:
    outcome: str
    reason: str
    plan: EvolutionPlan | None
    duration_seconds: float
    contract_id: str | None = None
    contract_version: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in PREFLIGHT_OUTCOMES:
            raise ValueError(f"invalid preflight outcome: {self.outcome}")


def _safe_reason(value: str) -> str:
    one_line = " ".join(value.split())
    if len(one_line) > _MAX_REASON_LENGTH:
        return one_line[: _MAX_REASON_LENGTH - 1] + "…"
    return one_line


def decide_preflight(
    *,
    contract: DataContract,
    live: LiveTable | None,
    catalog_error: str | None,
    write_strategy: str,
) -> PreflightResult:
    """Classify the catalog result using only plain values and the pure evolution planner."""
    if catalog_error is not None:
        error_type = catalog_error if _ERROR_TYPE.fullmatch(catalog_error) else "CatalogError"
        return PreflightResult("catalog_unavailable", error_type, None, 0.0)
    if live is None:
        return PreflightResult("table_missing", "bronze table does not exist", None, 0.0)

    evolution = plan_schema_evolution(
        contract=contract,
        live_columns=live.columns,
        recorded_contract_version=live.recorded_contract_version,
        write_strategy=write_strategy,
    )
    if evolution.outcome == "noop":
        return PreflightResult("ok", evolution.reason, evolution, 0.0)
    if evolution.outcome in {"evolve", "breaking_replace"}:
        return PreflightResult("will_evolve", _safe_reason(evolution.render()), evolution, 0.0)
    refused = ", ".join(f"{item.column}: {item.kind}" for item in evolution.refusals)
    return PreflightResult("refused", _safe_reason(refused or evolution.reason), evolution, 0.0)


@dataclass(frozen=True, slots=True)
class _EngineDependencies:
    load_catalog: Callable[..., Any]
    no_such_table: type[Exception]
    no_such_namespace: type[Exception]


def _engine_dependencies() -> _EngineDependencies:
    from pyiceberg.catalog import load_catalog
    from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError

    return _EngineDependencies(load_catalog, NoSuchTableError, NoSuchNamespaceError)


def load_live_table(
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    identifier: str,
    *,
    engine: _EngineDependencies | None = None,
) -> LiveTable | None:
    """Read one Iceberg table without starting Spark or importing PyIceberg at module import."""
    properties = derive_pyiceberg_catalog_properties(dict(config), resolved_paths)
    catalog_name = derive_pyiceberg_catalog_name(dict(config))

    if properties.get("type") == "sql":
        properties["init_catalog_tables"] = "false"
    dependencies = engine or _engine_dependencies()
    catalog = dependencies.load_catalog(catalog_name, **properties)
    try:
        if not _catalog_initialized(catalog):
            return None
        try:
            table = catalog.load_table(identifier)
        except (dependencies.no_such_table, dependencies.no_such_namespace):
            return None
        return LiveTable(
            columns=tuple(
                LiveColumn(field.name, _iceberg_physical_type(field.field_type), field.required)
                for field in table.schema().fields
            ),
            recorded_contract_version=table.properties.get("janus.contract_version"),
        )
    finally:
        close = getattr(catalog, "close", None)
        if callable(close):
            close()


def _catalog_initialized(catalog: Any) -> bool:
    """A reachable SQL catalog without its metadata table is a first write.

    REST catalogs have no SQL engine. Inspection errors propagate so an
    inaccessible database still fails closed, rather than looking empty.
    """
    sql_engine = getattr(catalog, "engine", None)
    if sql_engine is None:
        return True
    from sqlalchemy import inspect

    return bool(inspect(sql_engine).has_table("iceberg_tables"))


def _iceberg_physical_type(field_type: Any) -> str:
    """Map scalars and containers through the contract vocabulary."""
    return physical_type_from_iceberg_name(str(field_type))


def run_contract_preflight(
    plan: ExecutionPlan,
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    identifier: str,
    logger: StructuredLogger | None,
    budget_seconds: float = DEFAULT_PREFLIGHT_BUDGET_SECONDS,
    loader: Callable[..., LiveTable | None] = load_live_table,
) -> PreflightResult:
    """Bound the entire catalog read and decision; turn catalog failures into an outcome."""
    del logger  # The caller logs the one finished event, including the outcome.
    started = time.monotonic()
    contract = plan.data_contract
    if plan.bronze_output.format != "iceberg":
        return PreflightResult("ok", "bronze is not an iceberg table", None, 0.0)
    if contract is None:
        # The materializer raises MissingContractError before any bronze write.
        return PreflightResult("ok", "no contract attached to plan", None, 0.0)

    completed: list[PreflightResult] = []
    errors: list[str] = []

    def worker() -> None:
        try:
            live = loader(config, resolved_paths, identifier)
            completed.append(
                decide_preflight(
                    contract=contract,
                    live=live,
                    catalog_error=None,
                    write_strategy=resolve_bronze_write_intent(plan).strategy,
                )
            )
        except Exception as exc:
            # Exception messages can contain connection URIs and credentials.
            errors.append(type(exc).__name__)

    if budget_seconds <= 0:
        errors.append("InvalidPreflightBudget")
    else:
        try:
            thread = threading.Thread(target=worker, name="janus-contract-preflight", daemon=True)
            thread.start()
            thread.join(budget_seconds)
            if thread.is_alive():
                errors.append("PreflightTimeoutError")
        except Exception as exc:
            errors.append(type(exc).__name__)

    duration = time.monotonic() - started
    if errors:
        result = decide_preflight(
            contract=contract, live=None, catalog_error=errors[0], write_strategy="insert"
        )
    elif completed:
        result = completed[0]
    else:
        result = decide_preflight(
            contract=contract,
            live=None,
            catalog_error="PreflightWorkerExitedWithoutResult",
            write_strategy="insert",
        )
    return PreflightResult(
        result.outcome,
        result.reason,
        result.plan,
        duration,
        contract.id,
        contract.version,
    )


def start_observed_preflight(
    planned_run: PlannedRun,
    observer: RunObserver,
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    identifier: str,
    logger: StructuredLogger | None,
    loader: Callable[..., LiveTable | None] = load_live_table,
) -> tuple[PlannedRun, PreflightResult]:
    """Start observation and record preflight before the caller enforces its result.

    The caller retains the annotated plan on refusal so terminal metadata carries
    the preflight outcome even when extraction never starts.
    """
    plan = planned_run.plan
    observer.start_run(plan)
    if logger is not None:
        logger.info("run_observation_started")
    result = run_contract_preflight(
        plan, config, resolved_paths, identifier=identifier, logger=logger, loader=loader
    )
    plan = replace(
        plan, run_context=plan.run_context.with_attribute(PREFLIGHT_ATTRIBUTE, result.outcome)
    )
    if logger is not None:
        logger.info(
            "contract_preflight_finished",
            outcome=result.outcome,
            reason=result.reason,
            duration_seconds=result.duration_seconds,
        )
    return replace(planned_run, plan=plan), result


class ContractPreflightError(ContractEnforcementError):
    failure_stage = "contract_preflight"

    def __init__(self, result: PreflightResult) -> None:
        self.result = result
        contract = (
            f"{result.contract_id} v{result.contract_version}: "
            if result.contract_id is not None and result.contract_version is not None
            else ""
        )
        super().__init__(f"{contract}preflight {result.outcome} — {result.reason}")


def enforce_preflight(
    result: PreflightResult, *, enforcement: str, logger: StructuredLogger | None
) -> None:
    if result.outcome in {"refused", "catalog_unavailable"}:
        if enforcement == "strict":
            raise ContractPreflightError(result)
        if logger is not None:
            logger.warning(
                "contract_preflight_warning", outcome=result.outcome, reason=result.reason
            )
        else:
            _LOG.warning(
                "contract_preflight_warning outcome=%s reason=%s", result.outcome, result.reason
            )
    elif logger is not None:
        logger.info("contract_preflight_ok", outcome=result.outcome, reason=result.reason)
