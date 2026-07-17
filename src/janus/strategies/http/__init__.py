"""Shared HTTP extraction layer for the api, catalog, and file strategies.

The single home for the behavioral HTTP layer built on the reusable transport:
request pacing (throttle), the retry loop, payload decoding, and URL/param/
checkpoint binding. This package seeds it with the transport primitives all three
families already share; later tasks add the throttle, retry loop, decoder, and
binding helpers here so the import idiom stays uniform.
"""

from janus.strategies.http.errors import HttpStrategyError
from janus.strategies.http.payload import (
    ALL_PAYLOAD_FORMATS,
    PayloadDecodeError,
    decode_payload,
)
from janus.strategies.http.retry import (
    RETRYABLE_STATUS_CODES,
    RetryErrorPolicy,
    send_with_retries,
)
from janus.strategies.http.throttle import HttpRequestThrottle
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
    "ALL_PAYLOAD_FORMATS",
    "RETRYABLE_STATUS_CODES",
    "ApiClient",
    "ApiRequest",
    "ApiResponse",
    "ApiTransport",
    "ApiTransportError",
    "AuthResolutionError",
    "HttpRequestThrottle",
    "HttpStrategyError",
    "PayloadDecodeError",
    "RetryErrorPolicy",
    "UrllibApiTransport",
    "decode_payload",
    "inject_auth",
    "send_with_retries",
]
