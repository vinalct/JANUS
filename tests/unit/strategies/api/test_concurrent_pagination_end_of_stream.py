"""Characterization of concurrent pagination at the end of the stream."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

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
    ApiHook,
    ApiPastEndConflictError,
    ApiResponseError,
)
from janus.utils.logging import StructuredLogger

RECORDS_PAGE_1 = ({"id": "1"}, {"id": "2"})
RECORDS_PAGE_2 = ({"id": "3"}, {"id": "4"})
RECORDS_PAGE_3 = ({"id": "5"}, {"id": "6"})


def full_pages(
    count: int,
    *,
    extra_payload: Mapping[int, Mapping[str, Any]] | Mapping[str, Any] | None = None,
    latency_seconds: float = 0.0,
) -> dict[int, PageScript]:
    """Script ``count`` full pages of two records each, keyed by page number."""
    scripts: dict[int, PageScript] = {}
    for page in range(1, count + 1):
        if extra_payload is None:
            page_extra: Mapping[str, Any] = {}
        elif all(isinstance(key, int) for key in extra_payload):
            page_extra = extra_payload.get(page, {})
        else:
            page_extra = extra_payload 
        scripts[page] = PageScript(
            records=({"id": str(page * 2 - 1)}, {"id": str(page * 2)}),
            extra_payload=page_extra,
            latency_seconds=latency_seconds,
        )
    return scripts


def requested_keys_for_scope(
    transport: ScriptedPageTransport,
    scope_param: str,
    scope_value: str,
) -> list[int]:
    """Return the pagination keys requested under one request input."""
    keys = []
    for request in transport.requests:
        query = parse_qs(urlsplit(request.full_url()).query)
        if query.get(scope_param) == [scope_value]:
            keys.append(int(query["page"][0]))
    return keys


class TotalRecordsHook(ApiHook):
    """Source-local escape hatch: the API reports its size somewhere generic discovery misses."""

    def __init__(self, total_records: Any) -> None:
        self._total_records = total_records

    def resolve_total_records(self, plan, request, response, payload) -> Any:
        del plan, request, response, payload
        return self._total_records


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
    assert metadata["lookahead_ceiling_source"] == "none"
    assert set(metadata) >= CONCURRENCY_ONLY_METADATA_KEYS - {"total_records_reported"}


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


# ---------------------------------------------------------------------------
# Bounded over-fetch: the total-count ceiling (AC-3)
#
# Six full pages of two records, page 7 past the end, concurrency 3. The reported total of 12
# puts the last useful request index at 6, so the tail is capped: the loop stops guessing at 6
# and only ever issues page 7 because page 6 came back full — the one confirming request that
# keeps a stale total from truncating the run. Uncapped, the same script speculates through
# pages 8 and 9 before the past-end read lands.
# ---------------------------------------------------------------------------

CAPPED_PAGE_COUNT = 6
CAPPED_TOTAL_RECORDS = 12
CAPPED_REQUEST_KEYS = [1, 2, 3, 4, 5, 6, 7]


def test_resolve_total_records_defaults_to_none(tmp_path):
    """The hook point is an escape hatch: unopted sources keep generic payload discovery."""
    plan = build_concurrent_plan(tmp_path, source_id="hook_default_source")

    assert ApiHook().resolve_total_records(plan, None, None, {"total": 12}) is None


def test_total_count_caps_lookahead_to_the_last_page_plus_one(tmp_path):
    """A payload total bounds speculation: nothing beyond the confirming page is requested."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="total_count_cap_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        full_pages(
            CAPPED_PAGE_COUNT,
            extra_payload={"total": CAPPED_TOTAL_RECORDS},
            latency_seconds=0.05,
        ),
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert sorted(transport.requested_keys) == CAPPED_REQUEST_KEYS
    assert len(result.artifacts) == CAPPED_PAGE_COUNT
    assert result.records_extracted == CAPPED_TOTAL_RECORDS
    assert metadata["dead_letter_count"] == "0"
    assert metadata["lookahead_ceiling_source"] == "payload"
    assert metadata["total_records_reported"] == str(CAPPED_TOTAL_RECORDS)
    # The cap must not disable concurrency for the pages that are known to exist.
    assert transport.max_active_requests >= 2


def test_total_count_from_nested_meta_is_used(tmp_path):
    """Generic discovery reaches into meta/metadata/pagination containers."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="nested_total_count_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        full_pages(
            CAPPED_PAGE_COUNT,
            extra_payload={"meta": {"total": CAPPED_TOTAL_RECORDS}},
        ),
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    assert sorted(transport.requested_keys) == CAPPED_REQUEST_KEYS
    assert len(result.artifacts) == CAPPED_PAGE_COUNT


def test_explicit_total_count_field_wins(tmp_path):
    """The configured dotted path beats a misleading root hint key."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="explicit_total_field_source",
        page_size=2,
        concurrency=3,
        total_count_field="meta.total",
    )
    transport = ScriptedPageTransport(
        full_pages(
            CAPPED_PAGE_COUNT,
            extra_payload={"total": 999, "meta": {"total": CAPPED_TOTAL_RECORDS}},
        ),
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    assert sorted(transport.requested_keys) == CAPPED_REQUEST_KEYS
    assert result.records_extracted == CAPPED_TOTAL_RECORDS


def test_hook_total_records_overrides_payload(tmp_path):
    """The source-local hook outranks generic discovery."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="hook_total_records_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        full_pages(CAPPED_PAGE_COUNT, extra_payload={"total": 999}),
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan, TotalRecordsHook(CAPPED_TOTAL_RECORDS))
    metadata = result.metadata_as_dict()

    assert sorted(transport.requested_keys) == CAPPED_REQUEST_KEYS
    assert metadata["lookahead_ceiling_source"] == "hook"
    assert metadata["total_records_reported"] == str(CAPPED_TOTAL_RECORDS)


def test_invalid_hook_total_falls_back_to_payload(tmp_path, caplog):
    """A hook must not be able to cap a run by returning nonsense."""
    caplog.set_level(logging.WARNING, logger="janus.test.total_records")
    plan = build_concurrent_plan(
        tmp_path,
        source_id="invalid_hook_total_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        full_pages(CAPPED_PAGE_COUNT, extra_payload={"total": CAPPED_TOTAL_RECORDS}),
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(
        tmp_path,
        transport,
        logger=StructuredLogger(logger=logging.getLogger("janus.test.total_records")),
    )

    result = strategy.extract(plan, TotalRecordsHook(-3))
    metadata = result.metadata_as_dict()

    assert "api_total_records_invalid" in caplog.text
    assert sorted(transport.requested_keys) == CAPPED_REQUEST_KEYS
    assert metadata["lookahead_ceiling_source"] == "payload"
    assert metadata["total_records_reported"] == str(CAPPED_TOTAL_RECORDS)


def test_shrinking_total_does_not_truncate(tmp_path):
    """A total that collapses mid-stream costs nothing: the ceiling only ever rises."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="shrinking_total_source",
        page_size=2,
        concurrency=3,
    )
    transport = ScriptedPageTransport(
        full_pages(
            CAPPED_PAGE_COUNT,
            extra_payload={page: {"total": 1000 if page == 1 else 2} for page in range(1, 7)},
            latency_seconds=0.05,
        ),
        default_script=PageScript(status_code=404, latency_seconds=0.05),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    assert len(result.artifacts) == CAPPED_PAGE_COUNT
    assert result.records_extracted == CAPPED_TOTAL_RECORDS
    assert transport.max_active_requests >= 2


def test_growing_total_extends_the_ceiling(tmp_path):
    """A total that grows mid-stream raises the cap instead of stranding the extra pages."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="growing_total_source",
        page_size=2,
        concurrency=2,
    )
    transport = ScriptedPageTransport(
        full_pages(
            CAPPED_PAGE_COUNT,
            extra_payload={page: {"total": 6 if page < 3 else 12} for page in range(1, 7)},
        ),
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    assert len(result.artifacts) == CAPPED_PAGE_COUNT
    assert result.records_extracted == CAPPED_TOTAL_RECORDS
    assert sorted(transport.requested_keys) == CAPPED_REQUEST_KEYS


def test_lookahead_above_ceiling_degrades_to_sequential_not_stop(tmp_path):
    """A lying total costs parallelism at the tail — never completeness."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="lying_total_source",
        page_size=2,
        concurrency=4,
    )
    transport = ScriptedPageTransport(
        # "There are 4 records" — but six full pages keep coming.
        full_pages(CAPPED_PAGE_COUNT, extra_payload={"total": 4}),
        default_script=PageScript(status_code=404),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    assert len(result.artifacts) == CAPPED_PAGE_COUNT
    assert result.records_extracted == CAPPED_TOTAL_RECORDS
    # Every page above the ceiling is still fetched, one at a time, exactly once.
    for key in (5, 6, 7):
        assert transport.requested_keys.count(key) == 1


def test_overfetch_bounded_without_total_count(tmp_path):
    """No total means the structural bound: the in-flight window, and nothing beyond it."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="no_total_count_source",
        page_size=2,
        concurrency=4,
    )
    transport = ScriptedPageTransport(
        {
            1: PageScript(records=RECORDS_PAGE_1),
            2: PageScript(records=RECORDS_PAGE_2),
            3: PageScript(records=RECORDS_PAGE_3),
        },
        default_script=PageScript(status_code=200, records=()),
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)

    end_of_stream_key = 4
    assert len(result.artifacts) == end_of_stream_key
    assert max(transport.requested_keys) <= end_of_stream_key + 4 - 1
    assert len(set(transport.requested_keys)) == len(transport.requested_keys)


def test_past_end_page_is_requested_at_most_once(tmp_path):
    """Even uncapped, the page that proves the end is never retried or re-submitted."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="past_end_once_source",
        page_size=2,
        concurrency=4,
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

    assert transport.requested_keys.count(4) == 1
    assert result.metadata_as_dict()["past_end_status"] == "404"


def test_ceiling_is_per_request_input(tmp_path):
    """Each request input caps on its own total; nothing leaks across inputs."""
    plan = build_concurrent_plan(
        tmp_path,
        source_id="per_input_ceiling_source",
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
        {},
        default_script=PageScript(status_code=404),
        scope_param="mesAno",
        scoped_scripts={
            # January claims a thousand records and stops after two pages: an uncapped tail.
            "202501": full_pages(2, extra_payload={"total": 1000}),
            # February tells the truth, so its tail is capped at page 3 + the confirming 404.
            "202502": full_pages(3, extra_payload={"total": 6}),
        },
    )
    strategy = build_concurrent_strategy(tmp_path, transport)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert [Path(artifact.path).parent.name for artifact in result.artifacts] == [
        "request-input-000001",
        "request-input-000001",
        "request-input-000002",
        "request-input-000002",
        "request-input-000002",
    ]
    assert result.records_extracted == 10
    assert metadata["dead_letter_count"] == "0"
    assert metadata["past_end_terminated_count"] == "2"
    # February's own total bounds February, even though January ran with a ceiling of 500.
    assert sorted(requested_keys_for_scope(transport, "mesAno", "202502")) == [1, 2, 3, 4]
