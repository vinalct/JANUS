"""Pure Prometheus 0.0.4 gauge/counter exposition, without sample timestamps.

The parser checks the wire grammar independently of the renderer. JANUS requires
TYPE metadata and rejects trailing label commas and timestamp suffixes, even
though the general exposition format allows them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from janus.observability.metrics.catalogue import ALLOWED_LABELS, MetricType

if TYPE_CHECKING:
    from janus.observability.metrics.transport import MetricSample

_NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
_SAMPLE = re.compile(rf"({_NAME})(?:\{{(.*)\}})?[ \t]+([^ \t]+)[ \t]*")
_METADATA = re.compile(rf"#[ \t]+(HELP|TYPE)[ \t]+({_NAME})(?:[ \t]+(.*))?")
_LABEL_PAIR = re.compile(
    r'[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*=[ \t]*"((?:[^"\\\n]|\\[\\n"])*)"[ \t]*'
)
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_SPECIAL_VALUES = {"NaN": float("nan"), "+Inf": float("inf"), "-Inf": -float("inf")}


@dataclass(frozen=True, slots=True, eq=False)
class ParsedSample:
    """Wire fields only; exposition does not carry projection or batch metadata."""

    name: str
    metric_type: MetricType
    help_text: str
    labels: tuple[tuple[str, str], ...]
    value: int | float

    def _identity(self) -> tuple[object, ...]:
        # A wire NaN is stable across round trips, despite IEEE numeric equality.
        value = "NaN" if math.isnan(self.value) else self.value
        return self.name, self.metric_type, self.help_text, self.labels, value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedSample):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())


def _unescape(value: str, *, quoted: bool) -> str:
    escapes = {"\\": "\\", "n": "\n"}
    if quoted:
        escapes['"'] = '"'
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\\":
            index += 1
            if index == len(value) or value[index] not in escapes:
                raise ValueError("invalid exposition escape")
            character = escapes[value[index]]
        result.append(character)
        index += 1
    return "".join(result)


def _parse_labels(block: str | None) -> tuple[tuple[str, str], ...]:
    if block is None or not block.strip():
        return ()
    labels: dict[str, str] = {}
    position = 0
    while position < len(block):
        pair = _LABEL_PAIR.match(block, position)
        if pair is None:
            raise ValueError("malformed label block or unescaped quote")
        name, value = pair.groups()
        if name in labels:
            raise ValueError("duplicate label name")
        labels[name] = _unescape(value, quoted=True)
        position = pair.end()
        if position == len(block):
            break
        if block[position] != "," or not block[position + 1 :].strip():
            raise ValueError("malformed label block or trailing comma")
        position += 1
    return tuple(sorted(labels.items()))


def _parse_metadata(
    line: str,
    helps: dict[str, str],
    types: dict[str, MetricType],
    sampled_names: set[str],
) -> None:
    metadata = _METADATA.fullmatch(line)
    if metadata is None:
        raise ValueError("malformed HELP or TYPE metadata")
    directive, name, value = metadata.groups()
    if name in sampled_names:
        raise ValueError("metadata must precede the first sample")
    if directive == "HELP":
        if name in helps:
            raise ValueError("duplicate HELP metadata")
        helps[name] = _unescape(value or "", quoted=False)
    else:
        if name in types:
            raise ValueError("duplicate TYPE metadata")
        try:
            types[name] = MetricType(value)
        except ValueError as exc:
            raise ValueError("TYPE must be gauge or counter") from exc


def _parse_sample(line: str, helps: dict[str, str], types: dict[str, MetricType]) -> ParsedSample:
    match = _SAMPLE.fullmatch(line)
    if match is None:
        raise ValueError("malformed sample or missing value; timestamps are not supported")
    name, block, number = match.groups()
    if name not in types:
        raise ValueError("sample requires preceding TYPE metadata")
    labels = _parse_labels(block)
    if number in _SPECIAL_VALUES:
        value = _SPECIAL_VALUES[number]
    elif _NUMBER.fullmatch(number):
        value = int(number) if re.fullmatch(r"[+-]?[0-9]+", number) else float(number)
    else:
        raise ValueError("invalid sample value")
    return ParsedSample(name, types[name], helps.get(name, ""), labels, value)


def parse_exposition(text: str) -> tuple[ParsedSample, ...]:
    """Parse JANUS's strict gauge/counter subset of Prometheus text exposition."""
    if text and not text.endswith("\n"):
        raise ValueError("exposition must end with a newline")
    helps: dict[str, str] = {}
    types: dict[str, MetricType] = {}
    sampled_names: set[str] = set()
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    samples: list[ParsedSample] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip(" \t")
        if not line:
            continue
        if line.startswith("#"):
            if re.match(r"#[ \t]*(?:HELP|TYPE)(?:[ \t]|$)", line):
                _parse_metadata(line, helps, types, sampled_names)
            continue
        sample = _parse_sample(line, helps, types)
        identity = (sample.name, sample.labels)
        if identity in seen:
            raise ValueError("duplicate metric name and labels")
        if samples and samples[-1].name != sample.name and sample.name in sampled_names:
            raise ValueError("samples must be grouped by metric name")
        seen.add(identity)
        sampled_names.add(sample.name)
        samples.append(sample)
    return tuple(samples)


def _escape(value: str, *, quoted: bool) -> str:
    escaped = value.replace("\\", "\\\\").replace("\n", "\\n")
    return escaped.replace('"', '\\"') if quoted else escaped


def _render_value(value: int | float) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("metric values must be numeric")
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(value)


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    values: dict[str, str] = {}
    for name, value in labels:
        if name not in ALLOWED_LABELS or name in values:
            raise ValueError("metric labels must be allowed and unique")
        if not isinstance(value, str):
            raise TypeError("metric label values must be text")
        values[name] = value
    if not values:
        return ""
    pairs = [f'{name}="{_escape(value, quoted=True)}"' for name, value in sorted(values.items())]
    return "{" + ",".join(pairs) + "}"


def render_exposition(samples: Sequence[MetricSample | ParsedSample]) -> str:
    """Render deterministic Prometheus text exposition, version 0.0.4."""
    groups: dict[str, list[tuple[str, str]]] = {}
    metadata: dict[str, tuple[MetricType, str]] = {}
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        if sample.value is None:
            continue
        if re.fullmatch(_NAME, sample.name) is None or not isinstance(
            sample.metric_type, MetricType
        ):
            raise ValueError("invalid metric name or type")
        labels = _render_labels(sample.labels)
        identity = (sample.name, labels)
        if identity in seen:
            raise ValueError("duplicate metric name and labels")
        seen.add(identity)
        descriptor = (sample.metric_type, sample.help_text)
        if sample.name in metadata and metadata[sample.name] != descriptor:
            raise ValueError("metric samples must share type and help metadata")
        metadata[sample.name] = descriptor
        groups.setdefault(sample.name, []).append((labels, _render_value(sample.value)))

    lines: list[str] = []
    for name, group in sorted(groups.items()):
        metric_type, help_text = metadata[name]
        lines.append(f"# HELP {name} {_escape(help_text, quoted=False)}")
        lines.append(f"# TYPE {name} {metric_type.value}")
        lines.extend(f"{name}{labels} {value}" for labels, value in sorted(group))
    return "\n".join(lines) + "\n" if lines else ""
