import json
import os
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from tests.support.metrics_order23 import NOW, WarningSpy, record, samples, sink

FILE_RED = pytest.mark.xfail(
    strict=True,
    reason="disabled and append-only NDJSON metrics transports are absent",
)
TEXT_RED = pytest.mark.xfail(
    strict=True,
    reason="strict exposition parser and atomic textfile transport are absent",
)
UDP_RED = pytest.mark.xfail(
    strict=True, reason="bounded StatsD UDP transport is absent"
)
SCHEMA = Path(__file__).resolve().parents[3] / "fixtures/observability/metrics_ndjson.schema.json"


def _validator():
    data = SCHEMA.read_bytes()
    assert sha256(data).hexdigest() == SCHEMA.with_suffix(".json.sha256").read_text().strip()
    schema = json.loads(data)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


@FILE_RED
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


@FILE_RED
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


@FILE_RED
def test_disabled_transport_returns_skipped_without_touching_disk(tmp_path):
    from janus.observability.metrics.transport import (
        DisabledMetricsTransport,
        MetricsEmissionOutcome,
    )

    result = DisabledMetricsTransport().send(samples(tmp_path), budget_seconds=1)
    assert result.outcome is MetricsEmissionOutcome.SKIPPED
    assert not list(tmp_path.rglob("*"))


@FILE_RED
def test_file_output_is_one_schema_valid_line_per_run(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    destination = sink(tmp_path, "file", path="metrics")
    for duration in (5.0, 7.5):
        result = destination.emit(record(tmp_path, duration_seconds=duration), budget_seconds=1)
        assert result.outcome is MetricsEmissionOutcome.EMITTED
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


@FILE_RED
def test_file_append_uses_one_write_and_o_append(tmp_path, monkeypatch):
    destination = sink(tmp_path, "file", path="metrics")
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
    destination.emit(record(tmp_path), budget_seconds=1)
    assert len(written) == 1
    assert opened and all(flags & os.O_APPEND for flags in opened)
    _validator().validate(json.loads(written[0]))


@FILE_RED
def test_concurrent_file_writers_produce_intact_distinct_batches(tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    destination = sink(tmp_path, "file", path="metrics")

    def write(index):
        return destination.emit(record(tmp_path, duration_seconds=float(index)), budget_seconds=1)

    with ThreadPoolExecutor(max_workers=8) as pool:
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


@TEXT_RED
def test_textfile_round_trips_name_type_help_labels_and_value(tmp_path):
    from janus.observability.metrics.exposition import parse_exposition, render_exposition
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    expected = samples(tmp_path)
    destination = sink(tmp_path, "textfile", path="metrics/textfile")
    assert (
        destination.emit(record(tmp_path), budget_seconds=1).outcome
        is MetricsEmissionOutcome.EMITTED
    )
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


@TEXT_RED
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


@TEXT_RED
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


@TEXT_RED
def test_atomic_textfile_rewrite_never_exposes_partial_bytes_to_a_reader(tmp_path):
    destination = sink(tmp_path, "textfile", path="metrics")
    original, changed = record(tmp_path), record(tmp_path, duration_seconds=9.0)
    destination.emit(original, budget_seconds=1)
    paths = list((tmp_path / "metrics").glob("*.prom"))
    assert len(paths) == 1
    path = paths[0]
    before = path.read_bytes()
    destination.emit(changed, budget_seconds=1)
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
            destination.emit(original if index % 2 else changed, budget_seconds=1)
    finally:
        stop.set()
        reader.join(timeout=2)
    assert not reader.is_alive()
    assert not errors
    assert reads and set(reads) <= {before, after}
    assert len(reads) > 1


@TEXT_RED
def test_failed_run_preserves_the_last_success_timestamp(tmp_path):
    from janus.observability.metrics.exposition import parse_exposition

    destination = sink(tmp_path, "textfile", path="metrics")
    succeeded = record(tmp_path)
    destination.emit(succeeded, budget_seconds=1)
    failed = record(
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
    destination.emit(failed, budget_seconds=1)
    paths = list((tmp_path / "metrics").glob("*.prom"))
    assert len(paths) == 1
    parsed = parse_exposition(paths[0].read_text())
    success = [sample for sample in parsed if sample.name == "janus_last_success_timestamp_seconds"]
    assert len(success) == 1
    assert success[0].value == succeeded.ended_at.timestamp()
    last_run = next(
        sample for sample in parsed if sample.name == "janus_last_run_timestamp_seconds"
    )
    assert last_run.value == failed.ended_at.timestamp()
    assert last_run.value > success[0].value


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


@FILE_RED
@pytest.mark.parametrize("kind", ["file", "textfile"])
def test_unwritable_directory_is_a_failed_result_and_one_warning(kind, tmp_path):
    from janus.observability.metrics.transport import MetricsEmissionOutcome

    (tmp_path / "blocked").write_bytes(b"a file cannot be a directory")
    warnings = WarningSpy()
    destination = sink(tmp_path, kind, path="blocked/metrics", logger=warnings)
    result = destination.emit(record(tmp_path), budget_seconds=1, logger=warnings)
    assert result.outcome is MetricsEmissionOutcome.FAILED
    assert len(warnings.warnings) == 1


@FILE_RED
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
        pytest.param("file", marks=FILE_RED),
        pytest.param("textfile", marks=TEXT_RED),
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
