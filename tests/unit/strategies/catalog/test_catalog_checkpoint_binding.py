"""
regression test: catalog per-input parameter binding must support checkpoint_value.
"""
from __future__ import annotations

import pytest

from janus.models import ParameterBinding
from janus.strategies.api import ApiRequest
from janus.strategies.api.request_inputs import ApiParameterBindingError
from janus.strategies.catalog.core import _apply_per_input_params

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_REQUEST = ApiRequest(
    method="GET",
    url="https://example.invalid/catalog",
    timeout_seconds=30,
)

_CHECKPOINT_BINDING = {"desde": ParameterBinding(from_="checkpoint_value")}


# ---------------------------------------------------------------------------
#  checkpoint_value is silently dropped by _apply_per_input_params
# ---------------------------------------------------------------------------


def test_catalog_per_input_binding_receives_checkpoint_value():
    """
    A catalog source using 'from: checkpoint_value' should receive the active
    checkpoint value as a query parameter, consistent with API strategy behaviour.
    """
    request_input = {"window_start": "2026-01-01"}
    checkpoint_value = "2026-04-10T00:00:00Z"

    updated = _apply_per_input_params(
        _BASE_REQUEST,
        _CHECKPOINT_BINDING,
        request_input,
        checkpoint_value=checkpoint_value,
    )

    assert updated.params_as_dict().get("desde") == checkpoint_value


def test_catalog_per_input_binding_without_checkpoint_value_raises_parameter_binding_error():
    """
    When checkpoint_value is unavailable, a 'from: checkpoint_value' binding must
    raise ApiParameterBindingError.
    """
    with pytest.raises(ApiParameterBindingError):
        _apply_per_input_params(
            _BASE_REQUEST,
            _CHECKPOINT_BINDING,
            request_input={"window_start": "2026-01-01"},
        )
