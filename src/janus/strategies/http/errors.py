"""Shared error base for the behavioral HTTP layer.

The api, catalog, and file strategies keep their own concrete error hierarchies
(so every ``except ApiStrategyError`` site and import path is undisturbed); they
only gain a common ancestor here. The shared retry loop raises the family error
supplied to it via ``RetryErrorPolicy``. This base is what lets a caller catch
"any HTTP-layer failure" without knowing the family.
"""

from __future__ import annotations


class HttpStrategyError(RuntimeError):
    """Base for retry-exhaustion / transport failures raised by the shared HTTP layer."""
