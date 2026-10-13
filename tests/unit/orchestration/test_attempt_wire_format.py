import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from janus.orchestration.plans import MAX_FAILURE_REASON_LENGTH
from janus.orchestration.results import FailureDetails, SourceAttempt
from janus.orchestration.summary_values import _bounded_reason, freeze_evidence
from janus.orchestration.timing import ExecutionTiming
from janus.utils.logging import REDACTED_VALUE

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


@pytest.mark.parametrize("duration", [0, 0.0, 0.1 + 0.2])
@pytest.mark.parametrize("offset", [UTC, timezone(timedelta(hours=-3))])
def test_attempt_preserves_duration_representation_and_timestamp_offset(duration, offset):
    instant = INSTANT.astimezone(offset)
    attempt = replace(make_attempt(), timing=ExecutionTiming(instant, instant, duration))
    wire = json.dumps(attempt.to_summary(), sort_keys=True)
    restored = SourceAttempt.from_summary(json.loads(wire))
    assert restored == attempt
    assert json.dumps(restored.to_summary(), sort_keys=True) == wire


@pytest.mark.parametrize("suffix", ["+00:00", "Z"])
def test_timing_accepts_utc_forms_and_renders_the_existing_canonical_form(suffix):
    canonical = ExecutionTiming(INSTANT, INSTANT, 0.1 + 0.2).to_summary()
    data = dict(canonical)
    for name in ("started_at", "ended_at"):
        data[name] = data[name].replace("+00:00", suffix)
    restored = ExecutionTiming.from_summary(data)
    assert restored.to_summary() == canonical
    assert restored.started_at.microsecond == INSTANT.microsecond


@pytest.mark.parametrize("length", [MAX_FAILURE_REASON_LENGTH, MAX_FAILURE_REASON_LENGTH + 1])
def test_failure_bounding_and_nested_evidence_freezing_are_idempotent(length):
    reason = "x" * length
    failure = FailureDetails("execution", "FixtureFailure", reason)
    evidence = {
        "nested": [{"failure_reason": reason, "token": "secret-token"}],
        "credential": {"password": "secret-password"},
        "response_body": "private-response",
    }
    attempt = replace(make_attempt(), status="failed", failure=failure, evidence=evidence)
    wire = json.dumps(attempt.to_summary(), sort_keys=True)
    restored = SourceAttempt.from_summary(json.loads(wire))
    assert restored == attempt
    assert json.dumps(restored.to_summary(), sort_keys=True) == wire
    assert _bounded_reason(failure.reason) == failure.reason
    assert freeze_evidence(attempt.evidence) == attempt.evidence
    frozen = restored.evidence
    assert frozen["nested"][0]["failure_reason"] == failure.reason
    assert frozen["nested"][0]["token"] == REDACTED_VALUE
    assert frozen["credential"] == frozen["response_body"] == REDACTED_VALUE
    expected = reason if length == MAX_FAILURE_REASON_LENGTH else reason[:-1] + "… (truncated)"
    assert failure.reason == expected
    assert "secret-token" not in wire and "private-response" not in wire


@pytest.mark.parametrize("field", ["attempted", "failure"])
def test_missing_additional_required_attempt_field_names_the_field(field):
    data = make_attempt("lock").to_summary()
    data.pop(field)
    with pytest.raises(ValueError, match=field):
        SourceAttempt.from_summary(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempted", False),
        ("attempted", 1),
        ("attempted", "true"),
        ("status", "skipped"),
        ("status", []),
        ("source_id", None),
        ("source_id", ""),
        ("run_id", 7),
        ("run_id", ""),
        ("attempt", True),
        ("attempt", 0),
        ("attempt", "1"),
        ("attempt", 1.5),
        ("timing", None),
        ("timing", []),
        ("evidence", None),
        ("evidence", []),
        ("evidence", {"nested": {"value": object()}}),
        ("evidence", {"nested": float("nan")}),
        ("failure", False),
        ("failure", []),
        ("failure", {"phase": "execution", "error_type": "Failure", "reason": "failed"}),
    ],
)
def test_malformed_attempt_fields_raise_value_error(field, value):
    data = make_attempt().to_summary()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        SourceAttempt.from_summary(data)


@pytest.mark.parametrize(
    "record",
    [
        make_attempt(),
        FailureDetails("lock", "Failure", "held"),
        ExecutionTiming(INSTANT, INSTANT, 0),
    ],
)
def test_decoders_reject_unknown_fields(record):
    data = record.to_summary()
    data["unexpected"] = "ignored?"
    with pytest.raises(ValueError, match="unexpected"):
        type(record).from_summary(data)


@pytest.mark.parametrize("field", ["started_at", "ended_at"])
@pytest.mark.parametrize("value", ["not-a-timestamp", "2026-01-01T12:00:00", 3, False, {}])
def test_timing_decoder_rejects_malformed_timestamps(field, value):
    data = ExecutionTiming(INSTANT, INSTANT, 0).to_summary()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        ExecutionTiming.from_summary(data)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "0", None, {}, 10**400])
def test_timing_decoder_rejects_malformed_duration(value):
    data = ExecutionTiming(INSTANT, INSTANT, 0).to_summary()
    data["duration_seconds"] = value
    with pytest.raises(ValueError, match="duration_seconds"):
        ExecutionTiming.from_summary(data)


def test_nonexecuted_timing_round_trips_but_is_not_an_attempt():
    timing = ExecutionTiming.not_executed()
    assert ExecutionTiming.from_summary(timing.to_summary()) == timing
    data = make_attempt().to_summary()
    data["timing"] = timing.to_summary()
    with pytest.raises(ValueError, match="timing"):
        SourceAttempt.from_summary(data)


@pytest.mark.parametrize("field", ["started_at", "ended_at"])
def test_timing_decoder_rejects_a_single_absent_timestamp(field):
    data = ExecutionTiming(INSTANT, INSTANT, 0).to_summary()
    data[field] = None
    with pytest.raises(ValueError, match=field):
        ExecutionTiming.from_summary(data)


def test_nonexecuted_timing_requires_zero_duration():
    data = ExecutionTiming.not_executed().to_summary()
    data["duration_seconds"] = 1
    with pytest.raises(ValueError, match="duration_seconds"):
        ExecutionTiming.from_summary(data)


@pytest.mark.parametrize("field", ["phase", "error_type", "reason"])
@pytest.mark.parametrize("value", [None, False, 1, [], " "])
def test_failure_decoder_rejects_malformed_strings(field, value):
    data = FailureDetails("lock", "Failure", "held").to_summary()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        FailureDetails.from_summary(data)


@pytest.mark.parametrize(
    "record",
    [
        make_attempt(),
        FailureDetails("lock", "Failure", "held"),
        ExecutionTiming(INSTANT, INSTANT, 0),
    ],
)
@pytest.mark.parametrize("value", [None, [], "summary", 1])
def test_decoders_reject_non_objects(record, value):
    with pytest.raises(ValueError, match="summary"):
        type(record).from_summary(value)
