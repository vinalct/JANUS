"""Exception hierarchy for the API extraction strategy.

Separate from ``core.py`` so the mechanics modules (``speculation``, ``requests``,
``pagination_loop``) can raise these without importing the strategy façade — which
imports them. Mirrors ``strategies/http/errors.py``.
"""

from __future__ import annotations

from janus.strategies.http import ApiResponse, HttpStrategyError, response_body_excerpt
from janus.utils.logging import redact_url


class ApiStrategyError(HttpStrategyError):
    """Base failure for API strategy execution."""


class ApiResponseError(ApiStrategyError):
    """Raised when an API call finished with a non-success response.

    Carries a bounded excerpt of the response body. The status and URL alone say what
    happened but never why, and this error is what the dead-letter entry records — so
    anything the body explained was previously lost at the moment it mattered most.
    """

    def __init__(self, response: ApiResponse) -> None:
        self.response = response
        self.body_excerpt = response_body_excerpt(response)
        message = (
            f"API request failed with status {response.status_code} for "
            f"{redact_url(response.request.full_url())}"
        )
        if self.body_excerpt is not None:
            message = f"{message}: {self.body_excerpt}"
        super().__init__(message)


class ApiPastEndConflictError(ApiStrategyError):
    """Raised when a past-end status is contradicted by a later page that returned records."""

    def __init__(self, response: ApiResponse, *, conflicting_request_index: int) -> None:
        self.response = response
        self.conflicting_request_index = conflicting_request_index
        message = (
            f"API returned past-end status {response.status_code} for "
            f"{redact_url(response.request.full_url())}, but request index "
            f"{conflicting_request_index} returned records"
        )
        super().__init__(message)


class ApiPayloadError(ApiStrategyError):
    """Raised when the configured payload format cannot be decoded."""
