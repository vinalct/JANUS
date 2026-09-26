"""Best-effort, session-free appends to the queryable runs table."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

from janus.observability.records import RunRecord
from janus.observability.runs_table import (
    DEFAULT_RUNS_TABLE_IDENTIFIER,
    RUNS_TABLE_PARTITION_SPEC,
    RUNS_TABLE_SCHEMA,
    IcebergType,
    RunsTableColumn,
    RunsTableTarget,
    resolve_runs_table,
)
from janus.utils.catalog_properties import (
    HadoopCatalogUnrepresentableError,
    derive_pyiceberg_catalog_properties,
)
from janus.utils.environment import RuntimeLocation
from janus.utils.logging import StructuredLogger

DEFAULT_APPEND_TIMEOUT_SECONDS = 5.0
_UNRESOLVED_TABLE_IDENTIFIER = DEFAULT_RUNS_TABLE_IDENTIFIER
_PARTITION_FIELD_ID_START = 1000
_LOG = logging.getLogger(__name__)


SchemaFieldSignature = tuple[int, str, str, bool, int | None, bool | None]
_SCHEMA_FIELD_WITHOUT_NESTED_IDS_LENGTH = 4
_SCHEMA_FIELD_WITH_ELEMENT_ID_LENGTH = 5
_SCHEMA_FIELD_SIGNATURE_LENGTH = 6
SchemaFieldInput = (
    tuple[int, str, str, bool] | tuple[int, str, str, bool, int | None] | SchemaFieldSignature
)


class IcebergAppendReason(StrEnum):
    """Stable reasons reported by the runs-table sink."""

    LIVE_SCHEMA_MISMATCH = "live_schema_does_not_match_declaration"
    LIVE_SCHEMA_MISMATCH_AFTER_EVOLUTION = "live_schema_does_not_match_declaration_after_evolution"
    SCHEMA_EVOLUTION_FAILED = "schema_evolution_failed"


class IcebergAppendOutcome(StrEnum):
    """The three outcomes a caller can expose without consulting the catalog."""

    EMITTED = "emitted"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IcebergAppendResult:
    """Observable outcome of one best-effort runs-table append."""

    outcome: IcebergAppendOutcome
    table_identifier: str
    reason: str | None = None
    step: str | None = None
    exception_type: str | None = None

    @property
    def emitted(self) -> bool:
        return self.outcome is IcebergAppendOutcome.EMITTED


class _WarningLogger(Protocol):
    def warning(self, event: str, **fields: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class AdditivePlan:
    """Nullable declared columns that may be added to a live schema."""

    columns: tuple[SchemaFieldSignature, ...]


@dataclass(frozen=True, slots=True)
class _CatalogRequest:
    target: RunsTableTarget
    properties: dict[str, str]


@dataclass(frozen=True, slots=True)
class _EngineDependencies:
    load_catalog: Callable[..., Any]
    pyarrow: Any
    schema_type: Any
    nested_field_type: Any
    partition_field_type: Any
    partition_spec_type: Any
    day_transform_type: Any
    primitive_types: Mapping[IcebergType, Any]
    list_type: Any
    namespace_already_exists: type[Exception]
    table_already_exists: type[Exception]


def append_run_record(
    record: RunRecord,
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
    *,
    logger: StructuredLogger | _WarningLogger | None = None,
    timeout_seconds: float = DEFAULT_APPEND_TIMEOUT_SECONDS,
) -> IcebergAppendResult:
    """Append ``record`` once, returning rather than raising on every degradation path."""
    started_at = time.monotonic()
    request, preparation_result = _prepare_request(config, resolved_paths)
    if preparation_result is not None:
        return _report(preparation_result, logger)
    if request is None:
        return _report(
            _failed(
                _UNRESOLVED_TABLE_IDENTIFIER,
                step="catalog_properties",
                exception_type="RequestPreparationError",
            ),
            logger,
        )

    try:
        timeout_is_valid = math.isfinite(timeout_seconds) and timeout_seconds > 0
    except Exception:
        timeout_is_valid = False
    if not timeout_is_valid:
        return _report(
            _failed(
                request.target.identifier,
                step="budget",
                exception_type="InvalidAppendTimeout",
            ),
            logger,
        )

    remaining = timeout_seconds - (time.monotonic() - started_at)
    if remaining <= 0:
        return _report(_timed_out(request.target.identifier), logger)

    results: list[IcebergAppendResult] = []
    try:
        worker = threading.Thread(
            target=_run_worker,
            args=(results, record, request),
            name="janus-runs-table-append",
            daemon=True,
        )
        worker.start()
        worker.join(remaining)
    except Exception as exc:
        return _report(
            _failed(
                request.target.identifier,
                step="worker",
                exception_type=type(exc).__name__,
            ),
            logger,
        )

    if worker.is_alive():
        return _report(_timed_out(request.target.identifier), logger)
    if not results:
        return _report(
            _failed(
                request.target.identifier,
                step="worker",
                exception_type="WorkerExitedWithoutResult",
            ),
            logger,
        )
    return _report(results[0], logger)


def _prepare_request(
    config: Mapping[str, Any],
    resolved_paths: Mapping[str, RuntimeLocation],
) -> tuple[_CatalogRequest | None, IcebergAppendResult | None]:
    identifier = _UNRESOLVED_TABLE_IDENTIFIER
    try:
        config_copy = dict(config)
        target = resolve_runs_table(config_copy)
        identifier = target.identifier
        properties = derive_pyiceberg_catalog_properties(config_copy, resolved_paths)
    except HadoopCatalogUnrepresentableError as exc:
        return None, _skipped(
            identifier,
            reason="hadoop_catalog_unrepresentable",
            step="catalog_properties",
            exception_type=type(exc).__name__,
        )
    except Exception as exc:
        return None, _failed(
            identifier,
            step="catalog_properties",
            exception_type=type(exc).__name__,
        )
    return _CatalogRequest(target=target, properties=properties), None


def _run_worker(
    results: list[IcebergAppendResult],
    record: RunRecord,
    request: _CatalogRequest,
) -> None:
    results.append(_append_unbounded(record, request))


def _append_unbounded(
    record: RunRecord,
    request: _CatalogRequest,
) -> IcebergAppendResult:
    identifier = request.target.identifier
    try:
        dependencies = _load_engine_dependencies()
    except ImportError as exc:
        return _skipped(
            identifier,
            reason="pyiceberg_or_pyarrow_unavailable",
            step="dependency_import",
            exception_type=type(exc).__name__,
        )
    except Exception as exc:
        return _failed(
            identifier,
            step="dependency_import",
            exception_type=type(exc).__name__,
        )

    try:
        catalog = dependencies.load_catalog(
            request.target.catalog_name,
            **request.properties,
        )
    except Exception as exc:
        return _failed(identifier, step="catalog_load", exception_type=type(exc).__name__)

    namespace_result = _ensure_namespace(catalog, request.target, dependencies)
    if namespace_result is not None:
        return namespace_result

    table, table_result = _ensure_table(catalog, request.target, dependencies)
    if table_result is not None:
        return table_result
    if table is None:
        return _failed(
            identifier,
            step="table_load",
            exception_type="TableBootstrapReturnedNoTable",
        )

    live_schema, schema_result = _validate_or_evolve_schema(
        catalog, table, identifier, dependencies
    )
    if schema_result is not None:
        return schema_result
    assert live_schema is not None

    try:
        arrow_table = dependencies.pyarrow.Table.from_pylist(
            [record.to_dict()],
            schema=live_schema.as_arrow(),
        )
    except Exception as exc:
        return _failed(identifier, step="arrow_conversion", exception_type=type(exc).__name__)

    try:
        table.append(arrow_table)
    except Exception as exc:
        return _failed(identifier, step="append", exception_type=type(exc).__name__)
    return IcebergAppendResult(IcebergAppendOutcome.EMITTED, identifier)


def _validate_or_evolve_schema(
    catalog: Any,
    table: Any,
    identifier: str,
    dependencies: _EngineDependencies,
) -> tuple[Any | None, IcebergAppendResult | None]:
    try:
        declared_schema = _declared_schema(dependencies)
        live_schema = table.schema()
        live_fields = _schema_field_signatures(live_schema)
        declared_fields = _schema_field_signatures(declared_schema)
        if live_fields == declared_fields:
            return live_schema, None

        plan = _plan_additive_evolution(live_fields, declared_fields)
        if plan is None:
            return None, _skipped(
                identifier,
                reason=IcebergAppendReason.LIVE_SCHEMA_MISMATCH.value,
                step="schema_validation",
                exception_type="RunsTableSchemaMismatch",
            )
        if not plan.columns:
            return live_schema, None

        declared_columns = {column.field_id: column for column in RUNS_TABLE_SCHEMA}
        try:
            with table.update_schema() as update:
                for (
                    field_id,
                    name,
                    _type_name,
                    required,
                    _element_id,
                    _element_required,
                ) in plan.columns:
                    column = declared_columns[field_id]
                    update.add_column(
                        name,
                        _iceberg_column_type(column, dependencies),
                        required=required,
                    )
            table = catalog.load_table(identifier)
            live_schema = table.schema()
        except Exception as exc:
            return None, _failed(
                identifier,
                step="schema_evolution",
                exception_type=type(exc).__name__,
                reason=IcebergAppendReason.SCHEMA_EVOLUTION_FAILED.value,
            )

        if _schema_field_signatures(live_schema) != declared_fields:
            return None, _skipped(
                identifier,
                reason=IcebergAppendReason.LIVE_SCHEMA_MISMATCH_AFTER_EVOLUTION.value,
                step="schema_validation",
                exception_type="RunsTableSchemaMismatch",
            )
        return live_schema, None
    except Exception as exc:
        return None, _failed(
            identifier,
            step="schema_validation",
            exception_type=type(exc).__name__,
        )


def _ensure_namespace(
    catalog: Any,
    target: RunsTableTarget,
    dependencies: _EngineDependencies,
) -> IcebergAppendResult | None:
    try:
        catalog.create_namespace(target.namespace)
    except dependencies.namespace_already_exists:
        return None
    except Exception as exc:
        return _failed(
            target.identifier,
            step="namespace_create",
            exception_type=type(exc).__name__,
        )
    return None


def _ensure_table(
    catalog: Any,
    target: RunsTableTarget,
    dependencies: _EngineDependencies,
) -> tuple[Any | None, IcebergAppendResult | None]:
    try:
        schema = _declared_schema(dependencies, _creation_columns())
        partition_spec = _declared_partition_spec(dependencies)
        table = catalog.create_table(
            target.identifier,
            schema=schema,
            partition_spec=partition_spec,
        )
    except dependencies.table_already_exists:
        try:
            table = catalog.load_table(target.identifier)
        except Exception as exc:
            return None, _failed(
                target.identifier,
                step="table_load",
                exception_type=type(exc).__name__,
            )
    except Exception as exc:
        return None, _failed(
            target.identifier,
            step="table_create",
            exception_type=type(exc).__name__,
        )
    return table, None


def _creation_columns(
    columns: Sequence[RunsTableColumn] = RUNS_TABLE_SCHEMA,
) -> tuple[RunsTableColumn, ...]:
    """The longest declared prefix a fresh ``create_table`` numbers exactly as declared."""
    for size in range(len(columns), 0, -1):
        prefix = tuple(columns[:size])
        element_ids = [column.element_id for column in prefix if column.element_id is not None]
        if all(column.field_id == index for index, column in enumerate(prefix, 1)) and (
            element_ids == list(range(size + 1, size + 1 + len(element_ids)))
        ):
            return prefix
    return tuple(columns)


def _declared_schema(
    dependencies: _EngineDependencies,
    columns: Sequence[RunsTableColumn] = RUNS_TABLE_SCHEMA,
) -> Any:
    fields = []
    for column in columns:
        fields.append(
            dependencies.nested_field_type(
                field_id=column.field_id,
                name=column.name,
                field_type=_iceberg_column_type(column, dependencies),
                required=column.required,
            )
        )
    return dependencies.schema_type(*fields)


def _iceberg_column_type(column: RunsTableColumn, dependencies: _EngineDependencies) -> Any:
    if column.iceberg_type is IcebergType.STRING_LIST:
        return dependencies.list_type(
            element_id=column.element_id,
            element=dependencies.primitive_types[IcebergType.STRING](),
            element_required=True,
        )
    return dependencies.primitive_types[column.iceberg_type]()


def _schema_field_signatures(schema: Any) -> tuple[SchemaFieldSignature, ...]:
    """Return the field identity needed to decide whether a change is additive."""
    return tuple(
        (
            field.field_id,
            field.name,
            str(field.field_type),
            field.required,
            getattr(field.field_type, "element_id", None),
            getattr(field.field_type, "element_required", None),
        )
        for field in schema.fields
    )


def _plan_additive_evolution(
    live: Sequence[SchemaFieldInput],
    declared: Sequence[SchemaFieldInput],
) -> AdditivePlan | None:
    """Plan only newer nullable IDs from 4/5/6-item host tuples; reject conflicts."""
    live_fields = tuple(_normalize_schema_field(field) for field in live)
    declared_fields = tuple(_normalize_schema_field(field) for field in declared)
    live_by_id = {field[0]: field for field in live_fields}
    declared_by_id = {field[0]: field for field in declared_fields}
    live_by_name = {field[1]: field for field in live_fields}
    declared_by_name = {field[1]: field for field in declared_fields}
    if (
        len(live_by_id) != len(live)
        or len(declared_by_id) != len(declared)
        or len(live_by_name) != len(live)
        or len(declared_by_name) != len(declared)
    ):
        return None

    for field in live_fields:
        expected = declared_by_id.get(field[0])
        same_name = declared_by_name.get(field[1])
        if expected != field or same_name != field:
            return None

    missing = tuple(field for field in declared_fields if field[0] not in live_by_id)
    highest_live_id = max(
        (max(field[0], field[4] or 0) for field in live_fields),
        default=0,
    )
    if any(field[3] or field[0] <= highest_live_id for field in missing):
        return None
    return AdditivePlan(columns=missing)


def _normalize_schema_field(field: SchemaFieldInput) -> SchemaFieldSignature:
    values = cast(tuple[Any, ...], field)
    if len(values) not in {
        _SCHEMA_FIELD_WITHOUT_NESTED_IDS_LENGTH,
        _SCHEMA_FIELD_WITH_ELEMENT_ID_LENGTH,
        _SCHEMA_FIELD_SIGNATURE_LENGTH,
    }:
        raise ValueError("schema field signatures must contain four, five, or six items")
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4] if len(values) > _SCHEMA_FIELD_WITHOUT_NESTED_IDS_LENGTH else None,
        values[5] if len(values) > _SCHEMA_FIELD_WITH_ELEMENT_ID_LENGTH else None,
    )


def _declared_partition_spec(dependencies: _EngineDependencies) -> Any:
    source_ids = {column.name: column.field_id for column in RUNS_TABLE_SCHEMA}
    fields = (
        dependencies.partition_field_type(
            source_id=source_ids[field.source_column],
            field_id=_PARTITION_FIELD_ID_START + index,
            transform=dependencies.day_transform_type(),
            name=field.name,
        )
        for index, field in enumerate(RUNS_TABLE_PARTITION_SPEC)
    )
    return dependencies.partition_spec_type(*fields)


def _load_engine_dependencies() -> _EngineDependencies:
    """Import both engines only at the point where an append actually needs them."""
    import pyarrow
    from pyiceberg.catalog import load_catalog
    from pyiceberg.exceptions import NamespaceAlreadyExistsError, TableAlreadyExistsError
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import DayTransform
    from pyiceberg.types import (
        BooleanType,
        DoubleType,
        IntegerType,
        ListType,
        LongType,
        NestedField,
        StringType,
        TimestamptzType,
    )

    return _EngineDependencies(
        load_catalog=load_catalog,
        pyarrow=pyarrow,
        schema_type=Schema,
        nested_field_type=NestedField,
        partition_field_type=PartitionField,
        partition_spec_type=PartitionSpec,
        day_transform_type=DayTransform,
        primitive_types={
            IcebergType.STRING: StringType,
            IcebergType.INTEGER: IntegerType,
            IcebergType.LONG: LongType,
            IcebergType.DOUBLE: DoubleType,
            IcebergType.BOOLEAN: BooleanType,
            IcebergType.TIMESTAMPTZ: TimestamptzType,
        },
        list_type=ListType,
        namespace_already_exists=NamespaceAlreadyExistsError,
        table_already_exists=TableAlreadyExistsError,
    )


def _timed_out(identifier: str) -> IcebergAppendResult:
    return _failed(identifier, step="budget", exception_type="AppendTimeoutError")


def _failed(
    identifier: str,
    *,
    step: str,
    exception_type: str,
    reason: str | None = None,
) -> IcebergAppendResult:
    return IcebergAppendResult(
        IcebergAppendOutcome.FAILED,
        identifier,
        reason=reason or f"{step}_failed",
        step=step,
        exception_type=exception_type,
    )


def _skipped(
    identifier: str,
    *,
    reason: str,
    step: str,
    exception_type: str,
) -> IcebergAppendResult:
    return IcebergAppendResult(
        IcebergAppendOutcome.SKIPPED,
        identifier,
        reason=reason,
        step=step,
        exception_type=exception_type,
    )


def _report(
    result: IcebergAppendResult,
    logger: StructuredLogger | _WarningLogger | None,
) -> IcebergAppendResult:
    if result.emitted:
        return result
    fields = {
        "outcome": result.outcome,
        "reason": result.reason,
        "step": result.step,
        "exception_type": result.exception_type,
        "table_identifier": result.table_identifier,
    }
    try:
        if logger is None:
            _LOG.warning("runs_table_append_degraded", extra={"event_fields": fields})
        else:
            logger.warning("runs_table_append_degraded", **fields)
    except Exception:
        pass
    return result
