"""Metadata dictionaries the API strategy attaches to a run.

Everything here builds a ``dict[str, str]`` for ``ExtractionResult.metadata``, for a
dead-letter record, or for a log line. Grouped by what they *produce* rather than by who
calls them — ``_speculation_metadata`` belongs beside its siblings even though the only
caller is the concurrent pagination path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from janus.models import (
    CombinedRequestInputsConfig,
    ExecutionPlan,
    IcebergRowsRequestInputsConfig,
)
from janus.strategies.http import ApiRequest


def _request_input_metadata(plan: ExecutionPlan, request_input_count: int) -> dict[str, str]:
    parameter_bindings = plan.source_config.access.parameter_bindings or {}
    request_inputs = plan.source_config.access.request_inputs
    metadata = {
        "request_input_type": request_inputs.type,
        "request_input_count": str(request_input_count),
    }
    if parameter_bindings:
        metadata["bound_parameter_names"] = ",".join(sorted(parameter_bindings))
    if request_inputs.type == "iceberg_rows" and isinstance(
        request_inputs, IcebergRowsRequestInputsConfig
    ):
        metadata["upstream_namespace"] = request_inputs.namespace
        metadata["upstream_table_name"] = request_inputs.table_name
        metadata["upstream_column_names"] = ",".join(
            sorted(
                {
                    str(column).strip()
                    for column in request_inputs.columns.values()
                    if str(column).strip()
                }
            )
        )
    if request_inputs.type == "combined" and isinstance(
        request_inputs, CombinedRequestInputsConfig
    ):
        for sub_ri in request_inputs.inputs:
            if isinstance(sub_ri, IcebergRowsRequestInputsConfig):
                metadata["upstream_namespace"] = sub_ri.namespace
                metadata["upstream_table_name"] = sub_ri.table_name
                metadata["upstream_column_names"] = ",".join(
                    sorted(
                        {
                            str(column).strip()
                            for column in sub_ri.columns.values()
                            if str(column).strip()
                        }
                    )
                )
                break
    return metadata


def _request_input_dead_letter_metadata(
    *,
    request_input: Mapping[str, Any] | None,
    request_input_index: int,
    request_input_count: int,
    request: ApiRequest,
) -> dict[str, str]:
    metadata = {
        "request_input_index": str(request_input_index),
        "request_input_count": str(request_input_count),
        "request_url": request.full_url(),
    }
    field_names = _request_input_field_names(request_input)
    if field_names:
        metadata["request_input_field_names"] = ",".join(field_names)
    return metadata


def _request_input_field_names(
    request_input: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if not request_input:
        return ()
    return tuple(sorted(str(field_name) for field_name in request_input))

def _speculation_metadata(
    *,
    concurrent_pagination_used: bool,
    speculative_requests: int,
    discarded_requests: int,
    past_end_terminated_count: int,
    past_end_status: int | None,
    lookahead_ceiling_source: str | None,
    total_records_reported: int | None,
) -> dict[str, str]:
    """Build the concurrency-only metadata block, empty for a purely sequential run."""
    if not concurrent_pagination_used:
        return {}
    metadata = {
        "speculative_request_count": str(speculative_requests),
        "speculative_discarded_count": str(discarded_requests),
        "past_end_terminated_count": str(past_end_terminated_count),
        "lookahead_ceiling_source": lookahead_ceiling_source or "none",
    }
    if past_end_status is not None:
        metadata["past_end_status"] = str(past_end_status)
    if total_records_reported is not None:
        metadata["total_records_reported"] = str(total_records_reported)
    return metadata
