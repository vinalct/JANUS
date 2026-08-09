"""Per-run metadata the catalog strategy attaches, plus the per-input request binding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from janus.strategies.api.request_inputs import resolve_parameter_bindings
from janus.strategies.http import ApiRequest, split_path_and_query_params


def _apply_per_input_params(
    base_request: ApiRequest,
    parameter_bindings: Any,
    request_input: dict[str, Any] | None,
    *,
    checkpoint_value: str | None = None,
) -> ApiRequest:
    """Apply per-request-input parameter bindings to the base request."""
    if not parameter_bindings:
        return base_request
    bound_params = resolve_parameter_bindings(
        parameter_bindings,
        request_input=request_input,
        checkpoint_value=checkpoint_value,
    )
    if not bound_params:
        return base_request
    path_params, query_params = split_path_and_query_params(base_request.url, bound_params)
    request = base_request
    if path_params:
        request = request.with_url(base_request.url.format_map(path_params))
    if query_params:
        request = request.with_params(query_params)
    return request


def _catalog_request_input_dead_letter_metadata(
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
    if request_input:
        metadata["request_input_field_names"] = ",".join(sorted(str(key) for key in request_input))
    return metadata
