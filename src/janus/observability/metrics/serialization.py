"""Shared metric label validation and scope-aware NDJSON batch rendering."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from janus.observability.metrics.catalogue import ALLOWED_LABELS, pipeline_metrics

if TYPE_CHECKING:
    from janus.observability.metrics.transport import MetricSample


def emitted_at(sample: MetricSample) -> datetime:
    value = sample.emitted_at
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("a metrics batch requires an aware emitted_at from its record")
    return value.astimezone(UTC)


def label_values(sample: MetricSample) -> dict[str, str]:
    labels: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in sample.labels:
        if name not in ALLOWED_LABELS or name not in sample.declaration.labels:
            raise ValueError("metric label is not declared or allowed")
        if name in seen:
            raise ValueError("metric labels must not contain duplicates")
        seen.add(name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError("metric label values must be text")
        if value.strip():
            labels[name] = value.strip()
    return labels


def pipeline_labels(samples: Sequence[MetricSample]) -> dict[str, str]:
    """Require one environment/trigger aggregate, with only declared status variations."""
    common = {key: value for key, value in label_values(samples[0]).items() if key != "status"}
    if set(common) != {"environment", "trigger"}:
        raise ValueError("pipeline metrics require environment and trigger only")
    declarations = pipeline_metrics()
    for sample in samples:
        labels = label_values(sample)
        if (
            sample.declaration not in declarations
            or set(labels) != set(sample.declaration.labels)
            or {key: value for key, value in labels.items() if key != "status"} != common
            or ("status" in labels and labels["status"] not in sample.declaration.source_fields)
        ):
            raise ValueError("pipeline samples must share an aggregate and declared statuses")
    return common


def serialize_batch(samples: Sequence[MetricSample]) -> bytes:
    """Keep run batches unchanged; pipeline samples add their status label locally."""
    first = samples[0]
    timestamp = emitted_at(first)
    labels = pipeline_labels(samples) if first.scope == "pipeline" else label_values(first)
    rendered = []
    for sample in samples:
        values = label_values(sample)
        if (
            emitted_at(sample) != timestamp
            or sample.scope != first.scope
            or (first.scope == "run" and values != labels)
        ):
            raise ValueError("metrics samples must share their batch time, scope and labels")
        if isinstance(sample.value, bool) or not isinstance(sample.value, int | float):
            raise TypeError("metric values must be numeric")
        item: dict[str, Any] = {
            "name": sample.name,
            "type": sample.metric_type.value,
            "value": sample.value,
        }
        if first.scope == "pipeline" and "status" in values:
            item["labels"] = {"status": values["status"]}
        rendered.append(item)
    batch = {
        "emitted_at": timestamp.isoformat(),
        "scope": first.scope,
        "labels": labels,
        "samples": rendered,
    }
    return json.dumps(batch, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
