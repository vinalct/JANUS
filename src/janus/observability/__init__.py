"""Additive, best-effort run observability: the queryable projection of one run."""

from janus.observability.iceberg_sink import (
    DEFAULT_APPEND_TIMEOUT_SECONDS,
    IcebergAppendOutcome,
    IcebergAppendResult,
    append_run_record,
)
from janus.observability.records import RunEvidencePaths, RunRecord
from janus.observability.runs_table import (
    DEFAULT_RUNS_TABLE_IDENTIFIER,
    RUNS_TABLE,
    RUNS_TABLE_PARTITION_SPEC,
    RUNS_TABLE_SCHEMA,
    IcebergType,
    RunsTableColumn,
    RunsTableContractError,
    RunsTableDeclaration,
    RunsTablePartitionField,
    RunsTableTarget,
    resolve_runs_table,
    resolve_runs_table_identifier,
)
from janus.observability.vocabulary import (
    MAX_FAILURE_REASON_LENGTH,
    QUALITY_NOT_RUN,
    RUN_RECORD_SCHEMA_VERSION,
    RUN_RECORD_STATUSES,
    SUPPORTED_QUALITY_OUTCOMES,
)

__all__ = [
    "DEFAULT_APPEND_TIMEOUT_SECONDS",
    "DEFAULT_EMISSION_TIMEOUT_SECONDS",
    "DEFAULT_RUNS_TABLE_IDENTIFIER",
    "MAX_FAILURE_REASON_LENGTH",
    "QUALITY_NOT_RUN",
    "RUNS_TABLE",
    "RUNS_TABLE_PARTITION_SPEC",
    "RUNS_TABLE_SCHEMA",
    "RUN_RECORD_SCHEMA_VERSION",
    "RUN_RECORD_STATUSES",
    "SUPPORTED_QUALITY_OUTCOMES",
    "GuardedRunEventEmitter",
    "IcebergAppendOutcome",
    "IcebergAppendResult",
    "IcebergType",
    "RunEmissionOutcome",
    "RunEmissionResult",
    "RunEvidencePaths",
    "RunRecord",
    "RunsTableColumn",
    "RunsTableContractError",
    "RunsTableDeclaration",
    "RunsTablePartitionField",
    "RunsTableTarget",
    "append_run_record",
    "build_run_event_emitter",
    "latest_run_emission",
    "resolve_runs_table",
    "resolve_runs_table_identifier",
    "wire_run_event_emitter",
]


def __getattr__(name: str):
    """Load guarded emission only when its public API is requested."""
    if name in {
        "DEFAULT_EMISSION_TIMEOUT_SECONDS",
        "GuardedRunEventEmitter",
        "RunEmissionOutcome",
        "RunEmissionResult",
        "build_run_event_emitter",
        "latest_run_emission",
        "wire_run_event_emitter",
    }:
        from janus.observability.emission import (
            DEFAULT_EMISSION_TIMEOUT_SECONDS,
            GuardedRunEventEmitter,
            RunEmissionOutcome,
            RunEmissionResult,
            build_run_event_emitter,
            latest_run_emission,
            wire_run_event_emitter,
        )

        exports = {
            "DEFAULT_EMISSION_TIMEOUT_SECONDS": DEFAULT_EMISSION_TIMEOUT_SECONDS,
            "GuardedRunEventEmitter": GuardedRunEventEmitter,
            "RunEmissionOutcome": RunEmissionOutcome,
            "RunEmissionResult": RunEmissionResult,
            "build_run_event_emitter": build_run_event_emitter,
            "latest_run_emission": latest_run_emission,
            "wire_run_event_emitter": wire_run_event_emitter,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
