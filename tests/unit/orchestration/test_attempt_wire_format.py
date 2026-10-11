import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from janus.orchestration.results import FailureDetails, SourceAttempt
from janus.orchestration.timing import ExecutionTiming

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="attempt, failure and timing decoders are absent",
)

INSTANT = datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)


def make_attempt(shape="succeeded"):
    phase = {
        "failed-returned": "execution",
        "failed-raised": "execution",
        "cleanup": "cleanup",
        "result-translation": "result_translation",
        "lock": "lock",
        "snapshot-drift": "snapshot_drift",
        "worker-crash": "worker_crash",
        "bound": "execution",
    }.get(shape)
    return SourceAttempt(
        source_id="A",
        run_id="run-a",
        attempt=1,
        status="failed" if phase else "succeeded",
        timing=ExecutionTiming(INSTANT, INSTANT, 0),
        evidence={"nested": [{"id": "1", "authorization": "secret"}]}
        if shape == "redacted"
        else {},
        failure=FailureDetails(
            phase, "FixtureFailure", "x" * 2000 if shape == "bound" else "failed"
        )
        if phase
        else None,
    )


@pytest.mark.parametrize(
    "shape",
    [
        "succeeded",
        "failed-returned",
        "failed-raised",
        "cleanup",
        "result-translation",
        "lock",
        "snapshot-drift",
        "worker-crash",
        "redacted",
        "bound",
    ],
)
def test_every_attempt_shape_round_trips_byte_identically(shape):
    attempt = make_attempt(shape)
    wire = json.dumps(attempt.to_summary(), sort_keys=True)
    restored = SourceAttempt.from_summary(json.loads(wire))
    assert restored == attempt
    assert json.dumps(restored.to_summary(), sort_keys=True) == wire
    if shape == "redacted":
        assert "secret" not in wire
    if shape == "bound":
        assert len(restored.failure.reason) == 2000


def test_failure_and_zero_duration_microsecond_timing_round_trip():
    failure = FailureDetails("lock", "SourceLockedError", "held by pid 123")
    timing = ExecutionTiming(INSTANT, INSTANT, 0)
    assert FailureDetails.from_summary(failure.to_summary()) == failure
    assert ExecutionTiming.from_summary(timing.to_summary()) == timing
    assert ExecutionTiming.from_summary(timing.to_summary()).started_at.microsecond == 123456


@pytest.mark.parametrize(
    "record,field",
    [
        (FailureDetails("lock", "SourceLockedError", "held"), "phase"),
        (FailureDetails("lock", "SourceLockedError", "held"), "error_type"),
        (FailureDetails("lock", "SourceLockedError", "held"), "reason"),
        (ExecutionTiming(INSTANT, INSTANT, 0), "started_at"),
        (ExecutionTiming(INSTANT, INSTANT, 0), "ended_at"),
        (ExecutionTiming(INSTANT, INSTANT, 0), "duration_seconds"),
    ],
)
def test_component_decoder_names_missing_field(record, field):
    data = record.to_summary()
    data.pop(field)
    with pytest.raises(ValueError, match=field):
        type(record).from_summary(data)


@pytest.mark.parametrize(
    "field", ["source_id", "run_id", "attempt", "status", "timing", "evidence"]
)
def test_missing_wire_field_names_the_field(field):
    data = make_attempt().to_summary()
    data.pop(field)
    with pytest.raises(ValueError, match=field):
        SourceAttempt.from_summary(data)


@pytest.mark.parametrize(
    "field,value", [("status", "pending"), ("started_at", "2026-01-01T12:00:00")]
)
def test_malformed_wire_value_names_the_field(field, value):
    data = make_attempt().to_summary()
    target = data["timing"] if field == "started_at" else data
    target[field] = value
    with pytest.raises(ValueError, match=field):
        SourceAttempt.from_summary(data)


def test_decoder_has_room_below_results_soft_ceiling():
    assert callable(getattr(SourceAttempt, "from_summary", None))
    path = Path(__file__).resolve().parents[3] / "src/janus/orchestration/results.py"
    assert len(path.read_text().splitlines()) < 600
