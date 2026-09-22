"""Value-based secret scrubbing for remote response diagnostics."""

from __future__ import annotations

from typing import NoReturn
from urllib.parse import quote, quote_plus

from janus.utils.logging import REDACTED_VALUE

MIN_SECRET_LENGTH = 8


class SecretScrubber:
    """Hold resolved credential forms and remove them from diagnostic content.

    Values shorter than eight characters are deliberately ignored: very short values
    create unreadable error excerpts by matching ordinary text. Registration also stores
    URL-encoded forms because query credentials can be echoed either decoded or as sent.
    The held values cannot be iterated, rendered, compared, or pickled through this API.
    """

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: tuple[str, ...] = ()

    def register(self, *values: str) -> None:
        """Register non-empty credential forms, longest first and without duplicates."""
        registered = set(self._values)
        for value in values:
            if not value.strip() or len(value) < MIN_SECRET_LENGTH:
                continue
            registered.add(value)
            registered.add(quote(value, safe=""))
            registered.add(quote_plus(value, safe=""))
        self._values = tuple(sorted(registered, key=lambda value: (-len(value), value)))

    def scrub_bytes(self, body: bytes) -> bytes:
        """Replace every registered byte representation without changing no-op bodies."""
        scrubbed = body
        replacement = REDACTED_VALUE.encode("utf-8")
        for value in self._values:
            encoded = value.encode("utf-8")
            if encoded in scrubbed:
                scrubbed = scrubbed.replace(encoded, replacement)
        return scrubbed

    def scrub_text(self, text: str) -> str:
        """Replace every registered text representation without exposing held values."""
        scrubbed = text
        for value in self._values:
            if value in scrubbed:
                scrubbed = scrubbed.replace(value, REDACTED_VALUE)
        return scrubbed

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"SecretScrubber(values={len(self)})"

    __str__ = __repr__

    def __reduce__(self) -> NoReturn:
        raise TypeError("SecretScrubber cannot be pickled")
