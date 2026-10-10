import json
import os
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from tests.support.metrics_order23 import NOW, WarningSpy, record, sink

SINK_RED = pytest.mark.xfail(
    strict=True,
    reason="metrics sink warning and degradation paths are absent",
)
UDP_RED = pytest.mark.xfail(strict=True, reason="bounded StatsD UDP transport is absent")
SCHEMA = Path(__file__).resolve().parents[3] / "fixtures/observability/metrics_ndjson.schema.json"


def _validator():
    data = SCHEMA.read_bytes()
    assert sha256(data).hexdigest() == SCHEMA.with_suffix(".json.sha256").read_text().strip()
    schema = json.loads(data)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _file_samples(root, **overrides):
    from janus.observability.metrics.catalogue import label_values, run_metrics
    from janus.observability.metrics.transport import MetricSample

    projected = record(root, **overrides)
    labels = tuple(sorted(label_values(projected).items()))
    declarations = {metric.name: metric for metric in run_metrics()}
    return (
        MetricSample(
            declaration=declarations["janus_run_duration_seconds"],
            labels=labels,
            value=projected.duration_seconds,
            emitted_at=projected.emitted_at,
        ),
        MetricSample(
            declaration=declarations["janus_run_total"],
            labels=labels,
            value=1,
            emitted_at=projected.emitted_at,
        ),
    )


def _file_transport(root):
    from janus.observability.metrics.transport import FileMetricsTransport

    return FileMetricsTransport(root / "metrics")


def _textfile_samples(root, **overrides):
    from janus.observability.metrics.catalogue import run_metrics
    from janus.observability.metrics.transport import MetricSample

    if overrides.get("status") == "failed":
        overrides.setdefault("failure_reason", "fixture failure")
        overrides.setdefault("failure_reason_truncated", False)
        overrides.setdefault("failure_reason_length", len(overrides["failure_reason"]))
    batch = _file_samples(root, **overrides)
    projected = record(root, **overrides)
    declarations = {metric.name: metric for metric in run_metrics()}
    names = ["janus_last_run_timestamp_seconds"]
    if projected.status == "succeeded":
        names.append("janus_last_success_timestamp_seconds")
    return (
        *batch,
        *(
            MetricSample(
                declarations[name],
                batch[0].labels,
                projected.ended_at.timestamp(),
                projected.emitted_at,
            )
            for name in names
        ),
    )


def _textfile_transport(root, *, directory="metrics", logger=None):
    from janus.observability.metrics.transport import TextfileMetricsTransport

    return TextfileMetricsTransport(root / directory, logger=logger)


def test_ndjson_schema_detector_accepts_a_clean_batch_and_matches_catalogue_labels():
    from janus.observability.metrics.catalogue import ALLOWED_LABELS

    schema = _validator().schema
    declared = set(schema["properties"]["labels"]["properties"])
    assert declared and ALLOWED_LABELS
    assert declared == set(ALLOWED_LABELS)
    _validator().validate(
        {
            "emitted_at": "2026-10-09T12:00:00+00:00",
            "scope": "run",
            "labels": {"source_id": "fixture.api", "status": "succeeded"},
            "samples": [{"name": "janus_run_duration_seconds", "type": "gauge", "value": 5.0}],
        }
    )


@pytest.mark.parametrize("violation", ["run_id", "unknown-field", "invalid-type", "empty-label"])
def test_ndjson_schema_detector_rejects_deliberate_violations(violation):
    from jsonschema import ValidationError

    from janus.observability.metrics.catalogue import ALLOWED_LABELS

    assert ALLOWED_LABELS
    payload = {
        "emitted_at": "2026-10-09T12:00:00+00:00",
        "scope": "run",
        "labels": {"source_id": "fixture.api"},
        "samples": [{"name": "janus_run_duration_seconds", "type": "gauge", "value": 5.0}],
    }
    if violation == "run_id":
        payload["labels"]["run_id"] = "unbounded-id"
    elif violation == "unknown-field":
        payload["credential"] = "secret"
    elif violation == "invalid-type":
        payload["samples"][0]["type"] = "histogram"
    else:
        payload["labels"]["source_id"] = ""
    with pytest.raises(ValidationError):
        _validator().validate(payload)


def test_disabled_transport_returns_skipped_without_touching_disk(tmp_path):
    from janus.observability.metrics.transport import (
        DisabledMetricsTransport,
        MetricsEmissionOutcome,
    )

    result = DisabledMetricsTransport().send(_file_samples(tmp_path), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.SKIPPED
    assert result.sample_count == 0
    assert not list(tmp_path.rglob("*"))


def test_file_output_is_one_schema_valid_line_per_run(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    destination = _file_transport(tmp_path)
    for duration in (5.0, 7.5):
        result = destination.send(
            _file_samples(tmp_path, duration_seconds=duration), budget_seconds=1
        )
        assert result.outcome is MetricsEmissionOutcome.EMITTED
        assert result.sample_count == 2
    paths = list((tmp_path / "metrics").glob("*.ndjson"))
    assert len(paths) == 1
    raw = paths[0].read_bytes()
    assert raw.endswith(b"\n")
    lines = raw.splitlines()
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        _validator().validate(payload)
        assert payload["scope"] == "run"
        assert payload["emitted_at"].startswith("2026-10-09T12:00:00")
        assert "run_id" not in payload["labels"]
        assert "trigger" not in payload["labels"], "absent labels must be omitted"


def test_file_append_uses_one_write_and_o_append(tmp_path, monkeypatch):
    destination = _file_transport(tmp_path)
    opened, written = [], []
    real_open, real_write = os.open, os.write

    def open_spy(path, flags, *args, **kwargs):
        opened.append(flags)
        return real_open(path, flags, *args, **kwargs)

    def write_spy(descriptor, content):
        written.append(content)
        return real_write(descriptor, content)

    monkeypatch.setattr(os, "open", open_spy)
    monkeypatch.setattr(os, "write", write_spy)
    destination.send(_file_samples(tmp_path), budget_seconds=1)
    assert len(written) == 1
    assert opened and all(flags & os.O_APPEND for flags in opened)
    _validator().validate(json.loads(written[0]))


def test_concurrent_file_writers_produce_intact_distinct_batches(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    destination = _file_transport(tmp_path)
    barrier = threading.Barrier(10)

    def write(index):
        barrier.wait(timeout=5)
        return destination.send(
            _file_samples(tmp_path, duration_seconds=float(index)), budget_seconds=1
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(write, range(40)))
    assert len(results) == 40
    assert all(result.outcome is MetricsEmissionOutcome.EMITTED for result in results)
    paths = list((tmp_path / "metrics").glob("*.ndjson"))
    assert len(paths) == 1
    lines = paths[0].read_bytes().splitlines()
    assert len(lines) == 40
    durations = set()
    for line in lines:
        payload = json.loads(line)
        _validator().validate(payload)
        durations.add(
            next(
                sample["value"]
                for sample in payload["samples"]
                if sample["name"] == "janus_run_duration_seconds"
            )
        )
    assert durations == set(range(40))


@pytest.mark.parametrize(
    ("reason", "detail", "reportable"),
    [
        ("transport_not_configured", None, False),
        ("transport_not_configured", "invalid profile", True),
        ("transport_profile_error", None, True),
    ],
)
def test_disabled_transport_reports_only_configuration_problems(reason, detail, reportable):
    from janus.observability.metrics.transport import DisabledMetricsTransport

    destination = DisabledMetricsTransport(reason=reason, detail=detail)
    assert destination.reportable is reportable
    result = destination.send((), budget_seconds=0)
    assert result.to_summary() == {
        "outcome": "skipped",
        "transport": "disabled",
        "sample_count": 0,
        "reason": reason,
        "step": "transport_selection",
    }


def test_disabled_transport_does_not_inspect_the_samples():
    from janus.observability.metrics.transport import DisabledMetricsTransport

    class UnreadableSamples:
        def __len__(self):
            raise AssertionError("disabled delivery must not inspect samples")

        def __getitem__(self, index):
            raise AssertionError("disabled delivery must not read samples")

    assert DisabledMetricsTransport().send(UnreadableSamples(), budget_seconds=0).sample_count == 0


def test_file_day_comes_from_each_records_emitted_at(tmp_path):
    destination = _file_transport(tmp_path)
    for emitted_at in (NOW - timedelta(days=1), NOW):
        result = destination.send(_file_samples(tmp_path, emitted_at=emitted_at), budget_seconds=1)
        assert result.emitted
        assert Path(result.target).name == f"metrics-{emitted_at.date().isoformat()}.ndjson"
        payload = json.loads(Path(result.target).read_bytes())
        assert payload["emitted_at"] == emitted_at.isoformat()
    assert len(list(destination.directory.glob("*.ndjson"))) == 2


def test_file_day_and_batch_timestamp_are_normalized_to_utc(tmp_path):
    emitted_at = datetime(2026, 10, 8, 23, 30, tzinfo=timezone(timedelta(hours=-3)))
    result = _file_transport(tmp_path).send(
        _file_samples(tmp_path, emitted_at=emitted_at), budget_seconds=1
    )
    assert Path(result.target).name == "metrics-2026-10-09.ndjson"
    assert json.loads(Path(result.target).read_bytes())["emitted_at"] == (
        emitted_at.astimezone(UTC).isoformat()
    )


def test_file_omits_empty_and_absent_labels_and_escapes_values(tmp_path):
    original = _file_samples(tmp_path)
    escaped = tuple(
        (name, 'quoted "value"\nline' if name == "source_id" else value)
        for name, value in original[0].labels
    )
    labels = (*escaped, ("trigger", "   "))
    batch = tuple(replace(sample, labels=labels) for sample in original)
    result = _file_transport(tmp_path).send(batch, budget_seconds=1)
    lines = Path(result.target).read_bytes().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    _validator().validate(payload)
    assert payload["labels"]["source_id"] == 'quoted "value"\nline'
    assert "trigger" not in payload["labels"]
    assert all("emitted_at" not in sample for sample in payload["samples"])


@pytest.mark.parametrize("written", [0, 1])
def test_short_file_write_fails_once_and_closes_the_descriptor(tmp_path, monkeypatch, written):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    calls, closed = [], []
    real_close = os.close

    def short_write(descriptor, content):
        calls.append((descriptor, content))
        return written

    def close_spy(descriptor):
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "write", short_write)
    monkeypatch.setattr(os, "close", close_spy)
    result = _file_transport(tmp_path).send(_file_samples(tmp_path), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert result.reason == "partial_write"
    assert result.step == "append"
    assert result.exception_type == "PartialWriteError"
    assert result.sample_count == 0
    assert len(calls) == 1
    assert closed == [calls[0][0]]


def test_unwritable_file_open_fails_at_append(tmp_path, monkeypatch):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    def denied(*args, **kwargs):
        raise PermissionError("private directory")

    monkeypatch.setattr(os, "open", denied)
    result = _file_transport(tmp_path).send(_file_samples(tmp_path), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert result.step == "append"
    assert result.reason == "append_failed"
    assert result.exception_type == "PermissionError"
    assert "private directory" not in repr(result.to_summary())


def test_failing_file_mkdir_is_a_directory_create_failure(tmp_path, monkeypatch):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    def denied(*args, **kwargs):
        raise PermissionError("private directory")

    monkeypatch.setattr(Path, "mkdir", denied)
    result = _file_transport(tmp_path).send(_file_samples(tmp_path), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert result.step == "directory_create"
    assert result.reason == "directory_create_failed"
    assert result.exception_type == "PermissionError"
    assert not list(tmp_path.rglob("*"))


def test_a_file_blocking_the_metrics_directory_is_reported_not_raised(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    (tmp_path / "metrics").write_bytes(b"not a directory")
    result = _file_transport(tmp_path).send(_file_samples(tmp_path), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert result.step == "directory_create"
    assert result.exception_type == "FileExistsError"


@pytest.mark.parametrize("value", [object(), True, float("nan"), float("inf"), -float("inf")])
def test_undeserialisable_sample_is_a_serialization_failure(tmp_path, value):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    batch = (replace(_file_samples(tmp_path)[0], value=value),)
    result = _file_transport(tmp_path).send(batch, budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert result.step == "serialization"
    assert result.reason == "serialization_failed"
    assert result.sample_count == 0
    assert not list(tmp_path.rglob("*"))


@pytest.mark.parametrize("violation", ["time", "scope", "labels", "unknown-label", "duplicate"])
def test_inconsistent_or_unbounded_batch_is_a_serialization_failure(tmp_path, violation):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    first, second = _file_samples(tmp_path)
    if violation == "time":
        second = replace(second, emitted_at=NOW + timedelta(seconds=1))
    elif violation == "scope":
        second = replace(second, declaration=replace(second.declaration, scope="pipeline"))
    elif violation == "labels":
        second = replace(second, labels=(("source_id", "different-source"),))
    elif violation == "unknown-label":
        second = replace(second, labels=(*second.labels, ("run_id", "unbounded-id")))
    else:
        second = replace(second, labels=(*second.labels, second.labels[0]))
    result = _file_transport(tmp_path).send((first, second), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert result.step == "serialization"
    assert not list(tmp_path.rglob("*"))


@pytest.mark.parametrize("emitted_at", [None, datetime(2026, 10, 9)])
def test_missing_or_naive_batch_time_is_a_serialization_failure(tmp_path, emitted_at):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    batch = tuple(replace(sample, emitted_at=emitted_at) for sample in _file_samples(tmp_path))
    result = _file_transport(tmp_path).send(batch, budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert result.step == "serialization"
    assert not list(tmp_path.rglob("*"))


def test_empty_file_batch_is_skipped_without_touching_disk(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    result = _file_transport(tmp_path).send((), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.SKIPPED
    assert result.reason == "no_samples"
    assert result.sample_count == 0
    assert not list(tmp_path.rglob("*"))


def test_metrics_and_openlineage_share_the_only_o_append_implementation():
    import inspect

    import janus.observability as observability
    from janus.observability.metrics import transport as metrics_transport
    from janus.observability.openlineage import transport as lineage_transport

    root = Path(inspect.getfile(observability)).parent
    modules = sorted(root.rglob("*.py"))
    assert modules
    mentions = [
        path.relative_to(root).as_posix()
        for path in modules
        for _ in re.finditer(r"\bos\.O_APPEND\b", path.read_text(encoding="utf-8"))
    ]
    assert mentions == ["openlineage/transport.py"]
    assert metrics_transport._append_line is lineage_transport._append_line


def test_textfile_round_trips_name_type_help_labels_and_value(tmp_path):
    from janus.observability.metrics.exposition import parse_exposition, render_exposition
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    expected = _textfile_samples(tmp_path)
    destination = _textfile_transport(tmp_path, directory="metrics/textfile")
    assert destination.send(expected, budget_seconds=1).outcome is MetricsEmissionOutcome.EMITTED
    paths = list((tmp_path / "metrics/textfile").glob("*.prom"))
    assert len(paths) == 1
    text = paths[0].read_text()
    assert text.endswith("\n")
    actual = parse_exposition(text)
    assert actual
    assert render_exposition(actual) == render_exposition(expected)
    assert parse_exposition(render_exposition(actual)) == actual
    # Literal expectations supplement round-trip equality between renderer and parser.
    duration = next(sample for sample in actual if sample.name == "janus_run_duration_seconds")
    assert duration.value == 5.0
    assert str(duration.metric_type) == "gauge"
    assert duration.help_text.strip()
    assert dict(duration.labels)["source_id"] == "fixture.api"


@pytest.mark.parametrize(
    "text",
    [
        'janus_run_duration_seconds{source_id="fixture.api"} 5\n',
        "# TYPE janus_run_duration_seconds gauge\n"
        "janus_run_duration_seconds 5\njanus_run_duration_seconds 6\n",
        "# TYPE janus_run_duration_seconds gauge\n"
        'janus_run_duration_seconds{source_id="bad"quote"} 5\n',
        "# TYPE janus_run_duration_seconds gauge\n# TYPE janus_run_duration_seconds gauge\n"
        "janus_run_duration_seconds 5\n",
        '# TYPE janus_run_duration_seconds gauge\njanus_run_duration_seconds{source_id="x",} 5\n',
        '# TYPE janus_run_duration_seconds gauge\njanus_run_duration_seconds{source_id="x"}\n',
    ],
    ids=[
        "missing-type",
        "duplicate-sample",
        "unescaped-quote",
        "duplicate-type",
        "bad-labels",
        "missing-value",
    ],
)
def test_exposition_detector_rejects_deliberate_violations(text):
    from janus.observability.metrics.exposition import parse_exposition

    text = "# HELP janus_run_duration_seconds Duration of the run.\n" + text
    with pytest.raises(ValueError):
        parse_exposition(text)


def test_exposition_detector_accepts_independently_written_clean_input():
    from janus.observability.metrics.exposition import parse_exposition

    text = (
        "# HELP janus_run_duration_seconds Duration of the run.\n"
        "# TYPE janus_run_duration_seconds gauge\n"
        'janus_run_duration_seconds{source_id="fixture.api",status="succeeded"} 5.0\n'
    )
    parsed = parse_exposition(text)
    assert len(parsed) == 1
    assert parsed[0].name == "janus_run_duration_seconds"
    assert parsed[0].value == 5.0
    assert parsed[0].help_text == "Duration of the run."
    assert str(parsed[0].metric_type) == "gauge"
    assert dict(parsed[0].labels) == {"source_id": "fixture.api", "status": "succeeded"}


def test_atomic_textfile_rewrite_never_exposes_partial_bytes_to_a_reader(tmp_path):
    destination = _textfile_transport(tmp_path)
    original = _textfile_samples(tmp_path)
    changed = _textfile_samples(tmp_path, duration_seconds=9.0)
    assert destination.send(original, budget_seconds=1).emitted
    paths = list((tmp_path / "metrics").glob("*.prom"))
    assert len(paths) == 1
    path = paths[0]
    before = path.read_bytes()
    assert destination.send(changed, budget_seconds=1).emitted
    after = path.read_bytes()
    assert before and after and before != after
    ready, stop = threading.Event(), threading.Event()
    reads, errors = [], []

    def read():
        try:
            while not stop.is_set():
                reads.append(path.read_bytes())
                ready.set()
        except Exception as exc:
            errors.append(exc)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert ready.wait(2), "the reader must actually observe the target"
        for index in range(100):
            assert destination.send(original if index % 2 else changed, budget_seconds=1).emitted
    finally:
        stop.set()
        reader.join(timeout=2)
    assert not reader.is_alive()
    assert not errors
    assert reads and set(reads) <= {before, after}
    assert len(reads) > 1


def test_failed_run_preserves_the_last_success_timestamp(tmp_path):
    from janus.observability.metrics.exposition import parse_exposition

    destination = _textfile_transport(tmp_path)
    succeeded = record(tmp_path)
    assert destination.send(_textfile_samples(tmp_path), budget_seconds=1).emitted
    failed_samples = _textfile_samples(
        tmp_path,
        status="failed",
        started_at=NOW,
        ended_at=NOW + timedelta(hours=1),
        emitted_at=NOW + timedelta(hours=1),
        duration_seconds=3600.0,
        failure_reason="fixture failure",
        failure_reason_truncated=False,
        failure_reason_length=15,
        error_type="RuntimeError",
    )
    result = destination.send(failed_samples, budget_seconds=1)
    assert result.emitted and result.sample_count == 4
    paths = list((tmp_path / "metrics").glob("*.prom"))
    assert len(paths) == 1
    parsed = parse_exposition(paths[0].read_text())
    success = [sample for sample in parsed if sample.name == "janus_last_success_timestamp_seconds"]
    assert len(success) == 1
    assert success[0].value == succeeded.ended_at.timestamp()
    last_run = next(
        sample for sample in parsed if sample.name == "janus_last_run_timestamp_seconds"
    )
    assert last_run.value == (NOW + timedelta(hours=1)).timestamp()
    assert last_run.value > success[0].value
    assert dict(last_run.labels)["status"] == "failed"
    assert dict(success[0].labels)["status"] == "succeeded"


@pytest.mark.parametrize("source_id", ["../escape", "../../outside/source", "a/b", "..\\escape"])
def test_textfile_filename_sanitizes_hostile_source_ids_without_escaping(tmp_path, source_id):
    from janus.utils.storage import sanitize_identifier_segment

    destination = _textfile_transport(tmp_path)
    result = destination.send(_textfile_samples(tmp_path, source_id=source_id), budget_seconds=1)
    path = Path(result.target)
    assert result.emitted
    assert path.parent == destination.directory
    assert path.name == f"{sanitize_identifier_segment(source_id)}.prom"
    assert path.resolve().is_relative_to(destination.directory.resolve())


def test_textfile_replaces_last_run_and_keeps_one_file_per_source(tmp_path):
    from janus.observability.metrics.exposition import parse_exposition

    destination = _textfile_transport(tmp_path)
    first = destination.send(_textfile_samples(tmp_path), budget_seconds=1)
    assert first.emitted
    assert destination.send(
        _textfile_samples(tmp_path, duration_seconds=9.0), budget_seconds=1
    ).emitted
    assert destination.send(
        _textfile_samples(tmp_path, source_id="other.source"), budget_seconds=1
    ).emitted
    paths = sorted(destination.directory.glob("*.prom"))
    assert [path.name for path in paths] == ["fixture_api.prom", "other_source.prom"]
    parsed = parse_exposition(Path(first.target).read_text())
    duration = [sample for sample in parsed if sample.name == "janus_run_duration_seconds"]
    assert len(duration) == 1 and duration[0].value == 9.0


def test_textfile_temp_is_in_target_directory_and_readable_before_replace(tmp_path, monkeypatch):
    destination = _textfile_transport(tmp_path)
    replacements = []
    real_replace = os.replace

    def replace_spy(temporary, target):
        assert temporary.parent == target.parent == destination.directory
        assert temporary.name.startswith(f".{target.name}.") and temporary.suffix == ".tmp"
        assert temporary.stat().st_mode & 0o777 == 0o644
        replacements.append((temporary, target))
        real_replace(temporary, target)

    monkeypatch.setattr(os, "replace", replace_spy)
    previous_umask = os.umask(0o077)
    try:
        result = destination.send(_textfile_samples(tmp_path), budget_seconds=1)
    finally:
        os.umask(previous_umask)
    assert result.emitted
    assert len(replacements) == 1
    assert Path(result.target).stat().st_mode & 0o777 == 0o644
    assert not list(destination.directory.glob("*.tmp"))


def test_textfile_failure_without_prior_success_has_no_success_placeholder(tmp_path):
    from janus.observability.metrics.exposition import parse_exposition

    result = _textfile_transport(tmp_path).send(
        _textfile_samples(tmp_path, status="failed"), budget_seconds=1
    )
    assert result.emitted and result.sample_count == 3
    parsed = parse_exposition(Path(result.target).read_text())
    assert not any(sample.name == "janus_last_success_timestamp_seconds" for sample in parsed)


def test_textfile_keeps_success_across_repeated_failures_then_refreshes_it(tmp_path):
    from janus.observability.metrics.exposition import parse_exposition

    destination = _textfile_transport(tmp_path)
    result = destination.send(_textfile_samples(tmp_path), budget_seconds=1)
    for hours in (1, 2, 3):
        result = destination.send(
            _textfile_samples(tmp_path, status="failed", ended_at=NOW + timedelta(hours=hours)),
            budget_seconds=1,
        )
        assert result.emitted
        parsed = parse_exposition(Path(result.target).read_text())
        success = next(
            item for item in parsed if item.name == "janus_last_success_timestamp_seconds"
        )
        assert success.value == NOW.timestamp()
    result = destination.send(
        _textfile_samples(tmp_path, ended_at=NOW + timedelta(hours=4)), budget_seconds=1
    )
    parsed = parse_exposition(Path(result.target).read_text())
    success = [item for item in parsed if item.name == "janus_last_success_timestamp_seconds"]
    assert len(success) == 1 and success[0].value == (NOW + timedelta(hours=4)).timestamp()


@pytest.mark.parametrize("content", [b"corrupt token=secret-token\n", b"\xff\n"])
def test_textfile_discards_corrupt_existing_content_with_one_safe_warning(tmp_path, content):
    from janus.observability.metrics.exposition import parse_exposition

    warnings = WarningSpy()
    destination = _textfile_transport(tmp_path, logger=warnings)
    destination.directory.mkdir()
    path = destination.path_for({"source_id": "fixture.api"})
    path.write_bytes(content)
    result = destination.send(_textfile_samples(tmp_path, status="failed"), budget_seconds=1)
    assert result.emitted
    assert len(warnings.warnings) == 1
    assert warnings.warnings[0][0] == "metrics_textfile_existing_discarded"
    assert warnings.warnings[0][1]["step"] == "read_existing"
    assert "secret-token" not in repr(warnings.warnings)
    assert parse_exposition(path.read_text())


def test_textfile_default_warning_is_safe_and_raising_logger_cannot_break_rewrite(tmp_path, caplog):
    destination = _textfile_transport(tmp_path)
    destination.directory.mkdir()
    path = destination.path_for({"source_id": "fixture.api"})
    path.write_text("malformed credential=secret-token\n")
    assert destination.send(_textfile_samples(tmp_path, status="failed"), budget_seconds=1).emitted
    assert len(caplog.records) == 1
    assert "secret-token" not in repr(caplog.records[0].__dict__)

    class BrokenLogger:
        def warning(self, *args, **kwargs):
            raise RuntimeError("logger unavailable")

    destination = _textfile_transport(tmp_path, logger=BrokenLogger())
    path.write_text("malformed\n")
    assert destination.send(_textfile_samples(tmp_path, status="failed"), budget_seconds=1).emitted


@pytest.mark.parametrize(
    "step", ["render", "read_existing", "directory_create", "write_temp", "replace"]
)
def test_each_textfile_failure_step_returns_safe_failure_and_preserves_the_previous_file(
    tmp_path, monkeypatch, step
):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    destination = _textfile_transport(tmp_path)
    original = destination.send(_textfile_samples(tmp_path), budget_seconds=1)
    path = Path(original.target)
    before = path.read_bytes()
    batch = _textfile_samples(tmp_path, status="failed")
    calls = []

    def denied(*args, **kwargs):
        calls.append(args)
        raise PermissionError("https://user:credential@private.invalid?token=secret-token")

    if step == "render":
        monkeypatch.setattr("janus.observability.metrics.transport.render_exposition", denied)
    elif step == "read_existing":
        monkeypatch.setattr(Path, "read_text", denied)
    elif step == "directory_create":
        monkeypatch.setattr(Path, "mkdir", denied)
    elif step == "write_temp":
        monkeypatch.setattr(Path, "write_text", denied)
    else:
        monkeypatch.setattr(os, "replace", denied)
    result = destination.send(batch, budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert result.step == step and result.reason == f"{step}_failed"
    assert result.exception_type == "PermissionError" and result.sample_count == 0
    assert len(calls) == 1
    assert path.read_bytes() == before
    assert not list(destination.directory.glob("*.tmp"))
    assert not any(
        secret in repr(result.to_summary())
        for secret in ("credential", "private.invalid", "secret-token")
    )


@pytest.mark.parametrize("failure", ["short-write", "partial-write", "chmod"])
def test_textfile_temp_failure_cleans_up_and_does_not_replace(tmp_path, monkeypatch, failure):
    destination = _textfile_transport(tmp_path)
    original = destination.send(_textfile_samples(tmp_path), budget_seconds=1)
    path = Path(original.target)
    before = path.read_bytes()
    real_write = Path.write_text

    def broken_write(temporary, payload, **kwargs):
        real_write(temporary, payload[:8], **kwargs)
        if failure == "partial-write":
            raise OSError("disk full")
        return 8

    def broken_chmod(*args, **kwargs):
        raise PermissionError("chmod denied")

    if failure == "chmod":
        monkeypatch.setattr(Path, "chmod", broken_chmod)
    else:
        monkeypatch.setattr(Path, "write_text", broken_write)
    result = destination.send(_textfile_samples(tmp_path), budget_seconds=1)
    assert not result.emitted and result.step == "write_temp"
    assert path.read_bytes() == before
    assert not list(destination.directory.glob("*.tmp"))


@pytest.mark.parametrize(
    "violation", ["different-source", "pipeline", "unknown-label", "false-success"]
)
def test_textfile_rejects_mixed_sources_pipeline_samples_and_invalid_labels(tmp_path, violation):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    batch = _textfile_samples(tmp_path)
    first = batch[0]
    if violation == "different-source":
        first = replace(first, labels=(("source_id", "other"),))
    elif violation == "pipeline":
        first = replace(first, declaration=replace(first.declaration, scope="pipeline"))
    elif violation == "unknown-label":
        first = replace(first, labels=(*first.labels, ("run_id", "unbounded")))
    else:
        batch = tuple(
            replace(
                item,
                labels=tuple(
                    (key, "failed" if key == "status" else value) for key, value in item.labels
                ),
            )
            for item in batch
        )
        first = batch[0]
    result = _textfile_transport(tmp_path).send((first, *batch[1:]), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.FAILED and result.step == "render"
    assert not list(tmp_path.rglob("*"))


def test_empty_or_absent_textfile_batch_is_skipped_without_touching_disk(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    destination = _textfile_transport(tmp_path)
    for batch in ((), tuple(replace(item, value=None) for item in _textfile_samples(tmp_path))):
        result = destination.send(batch, budget_seconds=1)
        assert result.outcome is MetricsEmissionOutcome.SKIPPED and result.reason == "no_samples"
    assert not list(tmp_path.rglob("*"))


@pytest.mark.parametrize("source_id", ["", "  ", "..", "/../../"])
def test_textfile_rejects_source_ids_without_a_safe_identifier(tmp_path, source_id):
    batch = tuple(
        replace(
            sample,
            labels=tuple(
                (key, source_id if key == "source_id" else value) for key, value in sample.labels
            ),
        )
        for sample in _textfile_samples(tmp_path)
    )
    result = _textfile_transport(tmp_path).send(batch, budget_seconds=1)
    assert not result.emitted and result.step == "render"
    assert not list(tmp_path.rglob("*"))


def test_textfile_omits_absent_and_empty_labels_and_keeps_measured_zero(tmp_path):
    from janus.observability.metrics.exposition import parse_exposition

    batch = _textfile_samples(tmp_path, duration_seconds=0.0)
    batch = tuple(replace(item, labels=(*item.labels, ("trigger", "  "))) for item in batch)
    result = _textfile_transport(tmp_path).send(
        (*batch, replace(batch[0], value=None)), budget_seconds=1
    )
    assert result.emitted and result.sample_count == len(batch)
    parsed = parse_exposition(Path(result.target).read_text())
    assert all("trigger" not in dict(item.labels) for item in parsed)
    assert next(item for item in parsed if item.name == "janus_run_duration_seconds").value == 0


@pytest.mark.parametrize("violation", ["source", "unbounded-label", "type", "multiple-successes"])
def test_textfile_discards_invalid_retained_success_samples(tmp_path, violation):
    warnings = WarningSpy()
    destination = _textfile_transport(tmp_path, logger=warnings)
    destination.directory.mkdir()
    path = destination.path_for({"source_id": "fixture.api"})
    name = "janus_last_success_timestamp_seconds"
    metric_type = "counter" if violation == "type" else "gauge"
    source = "different" if violation == "source" else "fixture.api"
    labels = f'source_id="{source}"'
    if violation == "unbounded-label":
        labels += ',run_id="unbounded"'
    text = f"# HELP {name} Last success.\n# TYPE {name} {metric_type}\n{name}{{{labels}}} 123.0\n"
    if violation == "multiple-successes":
        text += f'{name}{{source_id="fixture.api",status="succeeded"}} 456.0\n'
    path.write_text(text)
    result = destination.send(_textfile_samples(tmp_path, status="failed"), budget_seconds=1)
    assert result.emitted and result.sample_count == 3
    assert name not in path.read_text()
    assert len(warnings.warnings) == 1


@UDP_RED
def test_statsd_sends_one_valid_datagram_batch_to_loopback(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        destination = sink(tmp_path, "statsd", host="127.0.0.1", port=receiver.getsockname()[1])
        result = destination.emit(record(tmp_path), budget_seconds=1)
        assert result.outcome is MetricsEmissionOutcome.EMITTED
        payload, _ = receiver.recvfrom(65535)
        lines = payload.decode().splitlines()
        assert lines
        parsed = [
            re.fullmatch(r"(janus\.[a-zA-Z0-9_]+):(-?\d+(?:\.\d+)?)\|([gc])(?:\|#(.+))?", line)
            for line in lines
        ]
        assert all(parsed)
        assert {match[3] for match in parsed} == {"g", "c"}
        assert any(
            match[1] == "janus.run_duration_seconds" and float(match[2]) == 5 for match in parsed
        )
        for match in parsed:
            tags = match[4].split(",") if match[4] else []
            assert tags == sorted(tags)
            assert all(tag.split(":", 1)[1] for tag in tags)
            assert all(not tag.startswith("run_id:") for tag in tags)
        receiver.settimeout(0.05)
        with pytest.raises(TimeoutError):
            receiver.recvfrom(65535)


@UDP_RED
def test_unreachable_statsd_send_fails_once_without_retry_or_secret_warning(tmp_path, monkeypatch):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    warnings = WarningSpy()
    destination = sink(tmp_path, "statsd", host="127.0.0.1", port=8125, logger=warnings)
    calls = []

    class UnreachableSocket:
        def settimeout(self, timeout):
            assert 0 < timeout <= 0.5

        def sendto(self, payload, address):
            calls.append(address)
            raise OSError("https://user:credential@private.invalid?token=secret-token")

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: UnreachableSocket())
    result = destination.emit(record(tmp_path), budget_seconds=1, logger=warnings)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert len(calls) == len(warnings.warnings) == 1
    assert not any(
        secret in repr(warnings.warnings)
        for secret in ("credential", "private.invalid", "secret-token")
    )


@pytest.mark.parametrize(
    "kind", [pytest.param("file", marks=SINK_RED), pytest.param("textfile", marks=SINK_RED)]
)
def test_unwritable_directory_is_a_failed_result_and_one_warning(kind, tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    (tmp_path / "blocked").write_bytes(b"a file cannot be a directory")
    warnings = WarningSpy()
    destination = sink(tmp_path, kind, path="blocked/metrics", logger=warnings)
    result = destination.emit(record(tmp_path), budget_seconds=1, logger=warnings)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert len(warnings.warnings) == 1


@SINK_RED
def test_raising_file_write_is_reported_without_exception_or_secret(tmp_path, monkeypatch):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    warnings = WarningSpy()
    destination = sink(tmp_path, "file", path="metrics", logger=warnings)

    def raise_write(*args):
        raise OSError("https://user:credential@private.invalid?token=secret-token")

    monkeypatch.setattr(os, "write", raise_write)
    result = destination.emit(record(tmp_path), budget_seconds=1, logger=warnings)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert len(warnings.warnings) == 1
    assert not any(
        secret in repr(warnings.warnings)
        for secret in ("credential", "private.invalid", "secret-token")
    )


@pytest.mark.parametrize(
    "kind",
    [
        "file",
        "textfile",
        pytest.param("statsd", marks=UDP_RED),
    ],
)
def test_transport_output_cannot_leak_failure_credentials(kind, tmp_path):
    secret = "https://user:credential@private.invalid/records?token=secret-token"
    failed = record(
        tmp_path,
        status="failed",
        failure_reason=secret,
        failure_reason_truncated=False,
        failure_reason_length=len(secret),
        error_type="OSError",
    )
    options = {"host": "127.0.0.1", "port": 8125} if kind == "statsd" else {"path": "metrics"}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        if kind == "statsd":
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(1)
            options["port"] = receiver.getsockname()[1]
        if kind == "file":
            result = _file_transport(tmp_path).send(
                _file_samples(
                    tmp_path,
                    status="failed",
                    failure_reason=secret,
                    failure_reason_truncated=False,
                    failure_reason_length=len(secret),
                    error_type="OSError",
                ),
                budget_seconds=1,
            )
        elif kind == "textfile":
            result = _textfile_transport(tmp_path).send(
                _textfile_samples(
                    tmp_path,
                    status="failed",
                    failure_reason=secret,
                    failure_reason_truncated=False,
                    failure_reason_length=len(secret),
                    error_type="OSError",
                ),
                budget_seconds=1,
            )
        else:
            destination = sink(tmp_path, kind, **options)
            result = destination.emit(failed, budget_seconds=1)
        output = (
            receiver.recvfrom(65535)[0]
            if kind == "statsd"
            else b"".join(
                path.read_bytes() for path in (tmp_path / "metrics").rglob("*") if path.is_file()
            )
        )
    assert output
    combined = output.decode() + repr(result.to_summary())
    assert not any(value in combined for value in ("credential", "private.invalid", "secret-token"))
