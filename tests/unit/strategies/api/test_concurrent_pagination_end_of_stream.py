"""Characterization of concurrent pagination at the end of the stream."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from conftest import (
    PageScript,
    ScriptedPageTransport,
    build_concurrent_plan,
    build_concurrent_strategy,
)

from janus.strategies.api.core import ApiResponseError

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


@pytest.mark.xfail(
    reason="order-08: a past-end 404 dead-letters the whole request input today",
    strict=True,
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


@pytest.mark.xfail(
    reason="order-08: a past-end 416 dead-letters the whole request input today",
    strict=True,
)
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
