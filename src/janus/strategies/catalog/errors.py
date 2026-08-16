"""Exception hierarchy for the catalog extraction strategy.

Separate from ``core.py`` so the mechanics modules can raise these without importing the
strategy façade — which imports them. Mirrors ``strategies/api/errors.py``.
"""

from __future__ import annotations

from janus.strategies.http import ApiResponse, HttpStrategyError
from janus.utils.logging import redact_url


class CatalogStrategyError(HttpStrategyError):
    """Base failure for metadata/catalog strategy execution."""


class CatalogResponseError(CatalogStrategyError):
    """Raised when a catalog request finished with a non-success response."""

    def __init__(self, response: ApiResponse) -> None:
        self.response = response
        message = (
            f"Catalog request failed with status {response.status_code} for "
            f"{redact_url(response.request.full_url())}"
        )
        super().__init__(message)


class CatalogPayloadError(CatalogStrategyError):
    """Raised when a catalog payload cannot be decoded or traversed safely."""
