"""One payload decoder for the JSON-shaped HTTP strategy families.

Collapses the api and catalog ``_decode_payload`` copies into a single format
dispatch. The file strategy is deliberately absent: it handles bytes/archives
and never JSON-decodes (PRD FR-3, §7 key decision 3 — over-generalization is a
named risk).

The two copies differed in exactly two ways, both kept as explicit parameters
here rather than a silent fork:

* which formats a family accepts (api takes ``binary|text|json|jsonl``; catalog
  is contractually json-only) — ``allowed_formats``;
* the message texts, including the api-has-format / catalog-doesn't asymmetry in
  the decode-failure message — ``family_label`` and ``failure_label``.

``decode_payload`` raises :class:`PayloadDecodeError`; each family keeps a thin
``_decode_payload`` wrapper that re-raises it as its own payload error, so every
``except ApiPayloadError`` / ``except CatalogPayloadError`` site (including the
shared retry loop's ``payload_error_types``) keeps working unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from janus.strategies.http.errors import HttpStrategyError
from janus.strategies.http.transport import ApiResponse
from janus.utils.logging import redact_url

ALL_PAYLOAD_FORMATS = frozenset({"binary", "text", "json", "jsonl"})


class PayloadDecodeError(HttpStrategyError):
    """Raised by :func:`decode_payload`; families wrap it into their payload error."""


def decode_payload(
    format_name: str,
    response: ApiResponse,
    *,
    allowed_formats: frozenset[str] | None = None,
    family_label: str,
    failure_label: str | None = None,
) -> Any:
    """Decode ``response`` according to ``format_name``.

    ``allowed_formats`` restricts which formats this family accepts (catalog
    passes ``frozenset({"json"})``; api leaves it ``None`` for all four). An
    unaccepted format is rejected before any decode attempt, matching both copies.

    Message texts stay family-specific: ``family_label`` fills the unsupported
    message (``"API"`` / ``"catalog"``) and ``failure_label`` fills the decode
    failure message — ``None`` uses ``format_name`` (api), catalog passes
    ``"catalog"`` so its message carries no format token.

    Decode failures catch ``(UnicodeDecodeError, ValueError)`` — a strict
    superset of both copies (``json.JSONDecodeError`` subclasses ``ValueError``).
    """
    permitted = ALL_PAYLOAD_FORMATS if allowed_formats is None else allowed_formats
    if format_name not in permitted:
        raise PayloadDecodeError(f"Unsupported {family_label} payload format: {format_name}")

    try:
        if format_name == "binary":
            return response.body
        if format_name == "text":
            return response.text()
        if format_name == "json":
            return response.json()
        if format_name == "jsonl":
            text = response.text()
            if not text.strip():
                return []
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeDecodeError, ValueError) as exc:
        noun = format_name if failure_label is None else failure_label
        raise PayloadDecodeError(
            f"Failed to decode {noun} payload from {redact_url(response.request.full_url())}"
        ) from exc

    raise PayloadDecodeError(f"Unsupported {family_label} payload format: {format_name}")
