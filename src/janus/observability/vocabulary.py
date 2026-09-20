"""Closed vocabularies, bounds and the schema version of the ``metadata.runs`` row."""

from __future__ import annotations

RUN_RECORD_SCHEMA_VERSION = 1
RUN_RECORD_STATUSES = frozenset({"failed", "succeeded"})
SUPPORTED_QUALITY_OUTCOMES = frozenset({"failed", "not_run", "passed"})
QUALITY_NOT_RUN = "not_run"
MAX_FAILURE_REASON_LENGTH = 2000
PIPELINE_RUN_ID_ATTRIBUTE = "pipeline_run_id"
PIPELINE_ATTEMPT_ATTRIBUTE = "pipeline_attempt"
TRIGGER_ATTRIBUTE = "trigger"
BRONZE_ZONE = "bronze"
