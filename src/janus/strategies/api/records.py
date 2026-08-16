"""Turning one decoded API payload into the records it carries.

Three leaf helpers with no dependency on the strategy: the default "where do the records
live in this payload" discovery, and the dotted-path lookup / string coercion pair the
checkpoint resolver uses to read a field out of one record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_RECORD_KEYS = ("records", "items", "results", "data", "value")


def _default_records_from_payload(payload: Any) -> Sequence[Any]:
    if payload is None:
        return ()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return payload
    if isinstance(payload, Mapping):
        for key in DEFAULT_RECORD_KEYS:
            nested = payload.get(key)
            if isinstance(nested, list):
                return nested
        return (payload,)
    return ()


def _lookup_field(record: Mapping[str, Any], field_path: str) -> Any:
    current: Any = record
    for segment in field_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    return rendered
