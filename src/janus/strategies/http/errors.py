"""Shared error base for the behavioral HTTP layer.

The api, catalog, and file strategies keep their own concrete error hierarchies
(so every ``except ApiStrategyError`` site and import path is undisturbed); they
only gain a common ancestor here. The shared retry loop raises the family error
supplied to it via ``RetryErrorPolicy``. This base is what lets a caller catch
"any HTTP-layer failure" without knowing the family.
"""

from __future__ import annotations

from janus.strategies.http.transport import ApiResponse

#: How much of a failed response body is carried into the error message.
#: Long enough for the error documents these APIs actually return, short enough that a stray
#: HTML page or a truncated dump cannot flood a log line or a dead-letter entry.
RESPONSE_BODY_EXCERPT_LIMIT = 500


class HttpStrategyError(RuntimeError):
    """Base for retry-exhaustion / transport failures raised by the shared HTTP layer."""


def response_body_excerpt(
    response: ApiResponse,
    *,
    limit: int = RESPONSE_BODY_EXCERPT_LIMIT,
) -> str | None:
    """Return a single-line, bounded rendering of a failed response body.

    Every family's status error used to report only the code and the URL, which is exactly
    the information that does *not* explain the failure: an unexpected ``400`` from a
    paginated API says what happened but never why, and the body holding the reason was
    dropped before anything could log it. Returns ``None`` for an empty body so callers can
    omit the clause entirely rather than print an empty one.

    Decoding is deliberately lossy (``errors="replace"``): an error document that is not
    valid UTF-8 is still worth reading, and a diagnostic helper must not raise.
    """

    if not response.body:
        return None

    text = response.body.decode("utf-8", errors="replace")
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}… ({len(response.body)} bytes total)"
