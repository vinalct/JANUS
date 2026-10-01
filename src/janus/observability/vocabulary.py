"""Closed vocabularies, bounds and the schema version of the ``metadata.runs`` row."""

from __future__ import annotations

RUN_RECORD_SCHEMA_VERSION = 3
RUN_RECORD_STATUSES = frozenset({"failed", "succeeded"})
SUPPORTED_QUALITY_OUTCOMES = frozenset({"failed", "not_run", "passed"})
QUALITY_NOT_RUN = "not_run"
MAX_FAILURE_REASON_LENGTH = 2000
PIPELINE_RUN_ID_ATTRIBUTE = "pipeline_run_id"
PIPELINE_ATTEMPT_ATTRIBUTE = "pipeline_attempt"
TRIGGER_ATTRIBUTE = "trigger"
BRONZE_ZONE = "bronze"

PREFLIGHT_ATTRIBUTE_NAME = "contract_preflight_outcome"
PREFLIGHT_OUTCOMES = frozenset(
    {"ok", "will_evolve", "refused", "table_missing", "catalog_unavailable"}
)
SCHEMA_EVOLUTION_METADATA_KEY = "schema_evolution"
SCHEMA_EVOLUTION_NONE = "none"
MALFORMED_ROWS_CHECK_PHASE = "data"
MALFORMED_ROWS_CHECK_NAME = "malformed_rows"
MALFORMED_ROWS_COUNT_DETAIL = "count"
