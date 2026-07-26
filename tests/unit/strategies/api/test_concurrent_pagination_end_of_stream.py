"""Characterization of concurrent pagination at the end of the stream."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from conftest import (
    PageScript,
    RecordingProgressStore,
    ScriptedPageTransport,
    build_concurrent_plan,
    build_concurrent_strategy,
)

from janus.strategies.api.core import (
    CONCURRENCY_ONLY_METADATA_KEYS,
    ApiPastEndConflictError,
    ApiResponseError,
)

RECORDS_PAGE_1 = ({"id": "1"}, {"id": "2"})
RECORDS_PAGE_2 = ({"id": "3"}, {"id": "4"})
RECORDS_PAGE_3 = ({"id": "5"}, {"id": "6"})


def test_empty_two_hundred_past_end_completes(tmp_path):
    """An empty 200 past the last page ends the stream cleanly — today's precondition."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="empty_two_hundred_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1, latency_seconds=0.05),
            2: PageScript(records=RECORDS_PAGE_2, latency_seconds=0.05),
            3: PageScript(records=({"id": "5"},), latency_seconds=0.05),
        },
        default_script=PageScript(status_code=200, records=(), latency_seconds=0.05),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    assert [Path(artifact.path).name for artifact in result.artifacts] == [
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
    ]
    assert result.records_extracted == 5
    assert result.metadata_as_dict()["dead_letter_count"] == "0"
    assert not strategy.dead_letter_store.path(plan).exists()
    assert transport.max_active_requests >= 2


def test_genuine_server_error_still_dead_letters(tmp_path):
    """A real 500 exhausts the retries, dead-letters the input, and fails the run."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="server_error_source",
        page_size=2,
        concurrency=3,
        retry_max_attempts=2,
        dead_letter_max_items=1,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(status_code=500),
        },
        default_script=PageScript(status_code=200, records=()),
    )
    sleeps: list[float] = []
    strategy = build_concurrent_strategy(tmp_path, transport, sleeper=sleeps.append)

    with pytest.raises(ApiResponseError):
        strategy.extract(plan)

    # One request input means can_continue_after_dead_letter() is false, so the run
    # raises even though dead_letter_max_items allows one entry.
    dead_letter_path = strategy.dead_letter_store.path(plan)
    dead_letter_payload = json.loads(dead_letter_path.read_text(encoding="utf-8"))
    entry = dead_letter_payload["entries"][0]
    assert entry["item_key"] == "__none__"
    assert entry["item_type"] == "request_input"
    assert entry["error_type"] == "ApiResponseError"
    assert transport.requested_keys.count(2) == 2
    assert sleeps == [1.0]


def test_genuine_server_error_dead_letters_one_input_and_continues(tmp_path):
    """With several request inputs the failing one is skipped and the run completes."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="server_error_windowed_source",
        page_size=2,
        concurrency=3,
        retry_max_attempts=2,
        request_inputs={
            "type": "date_window",
            "start": date(2025, 1, 1),
            "end": date(2025, 2, 28),
            "step": "month",
        },
        parameter_bindings={"mesAno": {"from": "request_input.window_end", "format": "%Y%m"}},
        dead_letter_max_items=1,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=({"id": "3"},)),
        },
        default_script=PageScript(status_code=200, records=()),
        scope_param="mesAno",
        scoped_scripts={"202502": {2: PageScript(status_code=500)}},
    )
    strategy = build_concurrent_strategy(tmp_path, transport, sleeper=lambda seconds: None)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert [Path(artifact.path).parent.name for artifact in result.artifacts] == [
        "request-input-000001",
        "request-input-000001",
    ]
    assert result.records_extracted == 3
    assert metadata["dead_letter_count"] == "1"
    assert metadata["dead_letter_skipped_count"] == "1"

    dead_letter_payload = json.loads(
        strategy.dead_letter_store.path(plan).read_text(encoding="utf-8")
    )
    assert dead_letter_payload["entries"][0]["item_key"] == (
        "window_end=2025-02-28|window_start=2025-02-01"
    )


def test_past_end_status_ends_stream_without_dead_letter(tmp_path):
    """A 404 past the last page must read as end-of-stream, not as a failed input."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_page_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
            3: PageScript(records=RECORDS_PAGE_3),
        },
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert [Path(artifact.path).name for artifact in result.artifacts] == [
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
    ]
    assert result.records_extracted == 6
    assert metadata["dead_letter_count"] == "0"
    assert metadata["past_end_status"] == "404"
    assert not strategy.dead_letter_store.path(plan).exists()
    assert transport.requested_keys.count(4) == 1


def test_past_end_status_ends_offset_stream_without_dead_letter(tmp_path):
    """The same contract on the other concurrency-capable paginator: offset + 416."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_offset_source",
        variant="offset_api",
        pagination_type="offset",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        {
            0: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
            4: PageScript(records=RECORDS_PAGE_3),
        },
        key_param="offset",
        default_script=PageScript(status_code=416),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert [Path(artifact.path).name for artifact in result.artifacts] == [
        "offset-00000000.json",
        "offset-00000002.json",
        "offset-00000004.json",
    ]
    assert result.records_extracted == 6
    assert metadata["dead_letter_count"] == "0"
    assert metadata["past_end_status"] == "416"
    assert not strategy.dead_letter_store.path(plan).exists()
    assert transport.requested_keys.count(6) == 1


def test_past_end_status_on_first_page_still_fails(tmp_path):
    """A past-end status on the *first* page was never a guess — it is a broken endpoint."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="first_page_past_end_source",
        page_size=2,
        concurrency=3,
        request_inputs={
            "type": "date_window",
            "start": date(2025, 1, 1),
            "end": date(2025, 2, 28),
            "step": "month",
        },
        parameter_bindings={"mesAno": {"from": "request_input.window_end", "format": "%Y%m"}},
        dead_letter_max_items=1,
    )
    transport = ScriptedPageTransport(
        {1: PageScript(records=RECORDS_PAGE_1)},
        default_script=PageScript(status_code=200, records=()),
        scope_param="mesAno",
        scoped_scripts={"202501": {1: PageScript(status_code=404)}},
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert metadata["dead_letter_count"] == "1"
    assert metadata["past_end_terminated_count"] == "0"
    assert "past_end_status" not in metadata

    entry = json.loads(strategy.dead_letter_store.path(plan).read_text(encoding="utf-8"))[
        "entries"
    ][0]
    assert entry["error_type"] == "ApiResponseError"
    assert "page=1" in entry["error_message"]


def test_past_end_conflict_raises(tmp_path):
    """A later page with records contradicts the past-end read, so the input must fail loudly."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_conflict_source",
        page_size=2,
        concurrency=4,
        dead_letter_max_items=1,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
            3: PageScript(status_code=404, latency_seconds=0.3),
            4: PageScript(records=RECORDS_PAGE_3),
        },
        default_script=PageScript(status_code=200, records=()),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    with pytest.raises(ApiPastEndConflictError) as raised:
        strategy.extract(plan)

    assert raised.value.conflicting_request_index == 4

    entry = json.loads(strategy.dead_letter_store.path(plan).read_text(encoding="utf-8"))[
        "entries"
    ][0]
    assert entry["item_type"] == "request_input"
    assert entry["error_type"] == "ApiPastEndConflictError"


def test_end_of_stream_cancels_outstanding_speculation(tmp_path):
    """Detecting the end stops submission and discards everything still in flight."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="cancel_speculation_source",
        page_size=2,
        concurrency=4,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1, latency_seconds=0.05),
            2: PageScript(records=RECORDS_PAGE_2, latency_seconds=0.05),
        },
        default_script=PageScript(status_code=404, latency_seconds=0.05),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    # Page 3 is the past-end page. Speculation is structurally bounded to the window that was
    # already in flight when it committed — never an unbounded walk past the end. The exact
    # zero-over-fetch bound arrives with the total-count ceiling.
    assert max(transport.requested_keys) <= 3 + 4 - 1
    assert int(metadata["speculative_discarded_count"]) > 0
    assert int(metadata["speculative_request_count"]) > 0
    assert metadata["past_end_terminated_count"] == "1"


def test_past_end_page_writes_no_artifact(tmp_path):
    """The past-end error body is not data: no raw artifact, no sidecar, no record."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_no_artifact_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
            3: PageScript(records=RECORDS_PAGE_3),
        },
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    raw_pages = sorted(path.name for path in tmp_path.rglob("page-*.json"))
    assert raw_pages == ["page-0001.json", "page-0002.json", "page-0003.json"]
    assert not list(tmp_path.rglob("page-0004.json"))
    assert result.metadata_as_dict()["request_count"] == "3"


def test_past_end_does_not_advance_the_checkpoint(tmp_path):
    """The checkpoint reflects committed pages only — a past-end call is not a page."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_checkpoint_source",
        page_size=2,
        concurrency=3,
        checkpoint_field="updated_at",
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(
                records=(
                    {"id": "1", "updated_at": "2025-01-01"},
                    {"id": "2", "updated_at": "2025-01-02"},
                )
            ),
            2: PageScript(
                records=(
                    {"id": "3", "updated_at": "2025-01-03"},
                    {"id": "4", "updated_at": "2025-01-04"},
                )
            ),
        },
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    assert result.checkpoint_value == "2025-01-04"


def test_progress_store_has_no_entry_for_the_past_end_page(tmp_path):
    """A resume must never rediscover a page that has no artifact."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_progress_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
        },
        default_script=PageScript(status_code=404),
    )
    progress_store = RecordingProgressStore()
    strategy = build_concurrent_strategy(tmp_path, transport, progress_store=progress_store)

    strategy.extract(plan)

    assert progress_store.saved_request_indexes == [1, 2]


def test_artifacts_committed_in_request_index_order(tmp_path):
    """Completion order is not commit order: artifacts stay ordered by request index."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_ordering_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1, latency_seconds=0.15),
            2: PageScript(records=RECORDS_PAGE_2, latency_seconds=0.10),
            3: PageScript(records=RECORDS_PAGE_3, latency_seconds=0.01),
        },
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    assert [Path(artifact.path).name for artifact in result.artifacts] == [
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
    ]
    assert result.records_extracted == 6


def test_second_request_input_runs_after_a_past_end_first_input(tmp_path):
    """End-of-stream ends one request input, never the run — this is AC-1's core claim."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_two_windows_source",
        page_size=2,
        concurrency=3,
        request_inputs={
            "type": "date_window",
            "start": date(2025, 1, 1),
            "end": date(2025, 2, 28),
            "step": "month",
        },
        parameter_bindings={"mesAno": {"from": "request_input.window_end", "format": "%Y%m"}},
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
        },
        default_script=PageScript(status_code=200, records=()),
        scope_param="mesAno",
        scoped_scripts={"202501": {3: PageScript(status_code=404)}},
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    # Window 1 ends on the 404, which writes nothing; window 2 ends on an empty 200, which is
    # a normal (if empty) page and keeps its artifact. Only the first is an *inferred* end.
    assert [Path(artifact.path).name for artifact in result.artifacts] == [
        "page-0001.json",
        "page-0002.json",
        "page-0001.json",
        "page-0002.json",
        "page-0003.json",
    ]
    assert [Path(artifact.path).parent.name for artifact in result.artifacts] == [
        "request-input-000001",
        "request-input-000001",
        "request-input-000002",
        "request-input-000002",
        "request-input-000002",
    ]
    assert metadata["dead_letter_count"] == "0"
    assert metadata["past_end_terminated_count"] == "1"
    assert not strategy.dead_letter_store.path(plan).exists()


def test_past_end_metadata_is_reported(tmp_path):
    """An operator must be able to see that this run's end-of-stream was inferred."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_metadata_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
            3: PageScript(records=RECORDS_PAGE_3),
        },
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    metadata = strategy.extract(plan).metadata_as_dict()

    assert metadata["past_end_terminated_count"] == "1"
    assert metadata["past_end_status"] == "404"
    assert metadata["pagination_concurrency"] == "3"
    assert set(metadata) >= CONCURRENCY_ONLY_METADATA_KEYS


def test_empty_past_end_set_restores_raising(tmp_path):
    """Declaring no past-end statuses opts out: the 404 is a failure again."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_opt_out_source",
        page_size=2,
        concurrency=3,
        past_end_status_codes=[],
        dead_letter_max_items=1,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
            3: PageScript(records=RECORDS_PAGE_3),
        },
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    with pytest.raises(ApiResponseError):
        strategy.extract(plan)

    entry = json.loads(strategy.dead_letter_store.path(plan).read_text(encoding="utf-8"))[
        "entries"
    ][0]
    assert entry["error_type"] == "ApiResponseError"
