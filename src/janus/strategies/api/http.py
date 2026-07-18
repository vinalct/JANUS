"""Deprecated location: the shared HTTP transport moved to janus.strategies.http.

This shim re-exports the public names for one release; import from
janus.strategies.http instead.
"""

from janus.strategies.http.transport import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiTransport,
    ApiTransportError,
    AuthResolutionError,
    UrllibApiTransport,
    inject_auth,
)

__all__ = [
    "ApiClient",
    "ApiRequest",
    "ApiResponse",
    "ApiTransport",
    "ApiTransportError",
    "AuthResolutionError",
    "UrllibApiTransport",
    "inject_auth",
]
