"""AC-1: the golden remains fixed while downloads complete out of candidate order."""

from __future__ import annotations

import concurrent.futures
import importlib
import json
import logging
import threading
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from janus.strategies.files import FILE_CONCURRENCY_ONLY_METADATA_KEYS, FileDownloadError, FileHook
from janus.utils.logging import StructuredLogger, redact_url
from tests.support.file_downloads import (
    FileRunFingerprint,
    ScriptedFileServer,
    UrlResponse,
    build_archive_package_fixture,
    build_cnpj_fixture,
    build_flat_files_fixture,
    fingerprint_file_run,
)

BUILDERS = {
    "cnpj_ten_archives": build_cnpj_fixture,
    "flat_ten_csvs": build_flat_files_fixture,
    "archive_package": build_archive_package_fixture,
}
GOLDEN = (
    Path(__file__).resolve().parents[3] / "fixtures/files/file_downloads_sequential_golden.json"
)
CONCURRENCY_KEYS = frozenset(
    {
        "download_concurrency",
        "concurrent_downloads_used",
        "max_in_flight_observed",
    }
)


def _run(fixture, run_id, concurrency):
    strategy = fixture.build_strategy()
    plan = fixture.build_plan(run_id, concurrency=concurrency)
    result = strategy.extract(plan)
    return fingerprint_file_run(strategy, plan, result), result.metadata_as_dict()


def _assert_fields_equal(actual, expected):
    for field in fields(FileRunFingerprint):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


@pytest.mark.parametrize("concurrency", [2, 4, 10])
@pytest.mark.parametrize("scenario", BUILDERS)
def test_concurrent_outputs_equal_sequential_and_frozen_golden(tmp_path, scenario, concurrency):
    # Missing mechanics are an intended red cause; never skip their import. Import before
    # arming gates, so today's sequential loop cannot spend five seconds at each gate.
    window = importlib.import_module("janus.strategies.files.download_window")
    assert callable(window.OrderedDownloadWindow)
    seq = BUILDERS[scenario](tmp_path / "seq")
    con = BUILDERS[scenario](tmp_path / "con")
    sequential, _ = _run(seq, "seq", 1)
    golden = json.loads(GOLDEN.read_text("utf-8"))[scenario]["fingerprint"]
    _assert_fields_equal(sequential, FileRunFingerprint.from_json(json.dumps(golden)))

    # Only reverse the initial dispatch window: a chain spanning more than capacity
    # would deadlock an otherwise correct bounded implementation.
    con.server.reverse_completion(con.urls[: min(concurrency, len(con.urls))])
    concurrent, _ = _run(con, "con", concurrency)
    _assert_fields_equal(concurrent, sequential)
    answers = [
        answer
        for answer in con.server.completions
        if answer.method == "GET" and answer.url in {redact_url(url) for url in con.urls}
    ]
    assert all(answer.thread_name.startswith("janus-file") for answer in answers)
    if len(con.urls) > 1:
        first = next(answer for answer in answers if answer.url == redact_url(con.urls[0]))
        assert any(
            answer.sequence < first.sequence and answer.url != first.url for answer in answers
        ), "fixture no longer forces out-of-order answers"


@pytest.mark.parametrize("concurrency", [2, 4, 10])
@pytest.mark.parametrize("scenario", BUILDERS)
def test_metadata_keys_and_values_are_exclusive_to_concurrent_runs(tmp_path, scenario, concurrency):
    assert FILE_CONCURRENCY_ONLY_METADATA_KEYS == CONCURRENCY_KEYS
    seq = BUILDERS[scenario](tmp_path / "seq")
    con = BUILDERS[scenario](tmp_path / "con")
    _, seq_metadata = _run(seq, "seq", 1)
    _, con_metadata = _run(con, "con", concurrency)
    assert set(con_metadata) - set(seq_metadata) == FILE_CONCURRENCY_ONLY_METADATA_KEYS
    assert set(seq_metadata) - set(con_metadata) == set()
    assert not FILE_CONCURRENCY_ONLY_METADATA_KEYS.intersection(seq_metadata)
    assert not {"request_count", "retry_count", "attempt_count"}.intersection(con_metadata)
    maximum = min(concurrency, len(con.urls))
    assert con_metadata["download_concurrency"] == str(concurrency)
    assert con_metadata["max_in_flight_observed"] == str(maximum)
    assert con_metadata["concurrent_downloads_used"] == str(maximum > 1).lower()


@pytest.mark.parametrize("scenario", BUILDERS)
def test_sequential_path_never_constructs_a_pool(tmp_path, monkeypatch, scenario):
    def forbidden_pool(*args, **kwargs):
        pytest.fail("concurrency=1 constructed a ThreadPoolExecutor")

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "__init__", forbidden_pool)
    try:
        window = importlib.import_module("janus.strategies.files.download_window")
    except ModuleNotFoundError as exc:
        assert exc.name == "janus.strategies.files.download_window"
    else:
        monkeypatch.setattr(window.ThreadPoolExecutor, "__init__", forbidden_pool)
    fixture = BUILDERS[scenario](tmp_path)
    _run(fixture, "seq", 1)
    assert fixture.server.requests
    assert all(request.thread_ident == threading.get_ident() for request in fixture.server.requests)
    assert fixture.server.opened == fixture.server.closed == 1


def test_one_eligible_candidate_reports_one_slot_without_overlap(tmp_path):
    sequential, _ = _run(build_cnpj_fixture(tmp_path / "seq", archive_count=1), "seq", 1)
    fixture = build_cnpj_fixture(tmp_path / "con", archive_count=1)
    concurrent, metadata = _run(fixture, "con", 4)
    _assert_fields_equal(concurrent, sequential)
    assert metadata["max_in_flight_observed"] == "1"
    assert metadata["concurrent_downloads_used"] == "false"
    assert metadata["download_concurrency"] == "4"
    assert any(request.thread_name.startswith("janus-file") for request in fixture.server.requests)


def test_concurrency_metadata_constant_is_immutable_and_exported():
    from janus.strategies import files
    from janus.strategies.files import core

    assert isinstance(FILE_CONCURRENCY_ONLY_METADATA_KEYS, frozenset)
    assert FILE_CONCURRENCY_ONLY_METADATA_KEYS is core.FILE_CONCURRENCY_ONLY_METADATA_KEYS
    assert "FILE_CONCURRENCY_ONLY_METADATA_KEYS" in files.__all__
    assert "FILE_CONCURRENCY_ONLY_METADATA_KEYS" in core.__all__


@pytest.mark.parametrize("concurrency", [1, 4])
@pytest.mark.parametrize("scenario", BUILDERS)
def test_concurrency_metadata_travels_through_strategy_emit_metadata(
    tmp_path, scenario, concurrency,
):
    fixture = BUILDERS[scenario](tmp_path)
    strategy = fixture.build_strategy()
    plan = fixture.build_plan("metadata", concurrency=concurrency)
    result = strategy.extract(plan)
    emitted = strategy.emit_metadata(plan, result)
    expected = (
        {
            "download_concurrency": "4",
            "max_in_flight_observed": str(min(4, len(fixture.urls))),
            "concurrent_downloads_used": str(len(fixture.urls) > 1).lower(),
        }
        if concurrency > 1 else {}
    )
    assert {key: emitted[key] for key in FILE_CONCURRENCY_ONLY_METADATA_KEYS if key in emitted} == (
        expected
    )
    assert not {"request_count", "retry_count", "attempt_count"}.intersection(emitted)


def _with_query_token(fixture, token):
    """Put credentials on the share and discovered URLs without changing the golden fixture."""
    query = f"?token={token}"
    fixture.source_mapping["access"]["url"] += query
    routes = {
        (method, url if method == "PROPFIND" else url + query): answers
        for (method, url), answers in fixture.server.routes.items()
    }

    class QueryTokenHook(FileHook):
        def discovered_files(self, plan, files):
            # WebDAV builds public URLs from paths, dropping the share query. Attach
            # it via the supported hook so the log's candidate URL exercises redaction.
            return tuple(replace(file, location=file.location + query) for file in files)

    return replace(
        fixture, server=ScriptedFileServer(routes),
        archive_urls=tuple(url + query for url in fixture.urls),
    ), QueryTokenHook()


def _event_records(caplog, event):
    return [record for record in caplog.records if record.getMessage() == event]


def _assert_download_events_on_loop_thread(caplog):
    records = [
        record for record in caplog.records if record.getMessage().startswith("file_download_")
    ]
    assert records
    assert all(record.thread == threading.get_ident() for record in records)


def test_concurrent_download_events_are_ordered_bounded_and_redacted(tmp_path, caplog):
    token = "fixture-query-credential"
    fixture, hook = _with_query_token(build_cnpj_fixture(tmp_path, retrying={2: 1}), token)
    fixture.server.reverse_completion(fixture.urls[:4])
    strategy = fixture.build_strategy(logger=StructuredLogger(logging.getLogger(__name__)))
    plan = fixture.build_plan("logs", concurrency=4)
    with caplog.at_level(logging.INFO, logger=__name__):
        result = strategy.extract(plan, hook=hook)

    dispatched = _event_records(caplog, "file_download_dispatched")
    completed = _event_records(caplog, "file_download_completed")
    indices = list(range(1, 11))
    assert [record.event_fields["file_index"] for record in dispatched] == indices
    assert [record.event_fields["file_index"] for record in completed] == indices
    assert not _event_records(caplog, "file_download_drained")
    assert [record.event_fields["in_flight"] for record in dispatched] == [1, 2, 3, *[4] * 7]
    assert [record.event_fields["in_flight"] for record in completed] == [*[4] * 7, 3, 2, 1]
    for index, record in enumerate(dispatched, start=1):
        payload = record.event_fields
        assert payload["selected_file_count"] == 10
        assert payload["download_concurrency"] == 4
        assert payload["filename"] == f"Empresas{index - 1}.zip"
        assert payload["source_location"] == redact_url(fixture.urls[index - 1])
        assert token not in payload["source_location"]
        assert "token=***REDACTED***" in payload["source_location"]
        assert "fixture-share" not in payload["source_location"]
    for index, record in enumerate(completed, start=1):
        payload = record.event_fields
        assert payload["outcome"] == "downloaded"
        assert payload["attempts_used"] == (2 if index == 2 else 1)
        assert payload["payload_size_bytes"] == len(
            fixture.server.routes[("GET", fixture.urls[index - 1])][-1].body
        )
        assert payload["download_seconds"] >= 0
        assert payload["waited_seconds"] >= 0
        assert "error_type" not in payload
    finished = _event_records(caplog, "file_extraction_finished")
    assert len(finished) == 1
    assert finished[0].event_fields["download_concurrency"] == 4
    assert finished[0].event_fields["max_in_flight_observed"] == 4
    assert result.metadata_as_dict()["concurrent_downloads_used"] == "true"
    _assert_download_events_on_loop_thread(caplog)


@pytest.mark.parametrize(
    "failures,budget,stop,drained_indices",
    [((1,), 0, 1, [2, 3, 4]), ((3, 6, 7), 1, 6, [7, 8, 9]), ((10,), 0, 10, [])],
    ids=["early-exceed", "middle-exceed-with-drained-failure", "late-exceed"],
)
def test_budget_exceeded_events_report_failed_and_drained_outcomes(
    tmp_path, caplog, failures, budget, stop, drained_indices,
):
    failure = UrlResponse(500, b"scripted download failure")
    fixture = build_cnpj_fixture(
        tmp_path, failing=dict.fromkeys(failures, failure), dead_letter_max_items=budget,
    )
    fixture.server.reverse_completion(fixture.urls[:4])
    strategy = fixture.build_strategy(logger=StructuredLogger(logging.getLogger(__name__)))
    plan = fixture.build_plan("failure-logs", concurrency=4)
    with caplog.at_level(logging.INFO, logger=__name__), pytest.raises(FileDownloadError):
        strategy.extract(plan)

    dispatched = _event_records(caplog, "file_download_dispatched")
    completed = _event_records(caplog, "file_download_completed")
    drained = _event_records(caplog, "file_download_drained")
    assert [record.event_fields["file_index"] for record in dispatched] == list(
        range(1, min(10, stop + 3) + 1)
    )
    assert [record.event_fields["file_index"] for record in completed] == list(range(1, stop + 1))
    assert [record.event_fields["file_index"] for record in drained] == drained_indices
    assert all(1 <= record.event_fields["in_flight"] <= 4 for record in dispatched + completed)
    for record in completed:
        payload = record.event_fields
        if payload["file_index"] in failures:
            assert payload["outcome"] == "failed"
            assert payload["attempts_used"] is None
            assert payload["payload_size_bytes"] is None
            assert payload["error_type"] == "FileDownloadError"
        else:
            assert payload["outcome"] == "downloaded"
            assert "error_type" not in payload
        assert payload["download_seconds"] >= 0
        assert payload["waited_seconds"] >= 0
    for record in drained:
        payload = record.event_fields
        expected = {"file_index": payload["file_index"], "discarded": True}
        if payload["file_index"] in failures:
            expected.update(outcome="failed", error_type="FileDownloadError")
        else:
            expected["outcome"] = "downloaded"
        assert {key: payload[key] for key in expected} == expected
        assert ("error_type" in payload) == (payload["file_index"] in failures)
    assert not _event_records(caplog, "file_extraction_finished")
    _assert_download_events_on_loop_thread(caplog)


def test_sequential_events_keep_their_existing_fields(tmp_path, caplog):
    fixture = build_cnpj_fixture(tmp_path)
    strategy = fixture.build_strategy(logger=StructuredLogger(logging.getLogger(__name__)))
    plan = fixture.build_plan("sequential-logs", concurrency=1)
    with caplog.at_level(logging.INFO, logger=__name__):
        strategy.extract(plan)
    assert not any(record.getMessage().startswith("file_download_") for record in caplog.records)
    finished = _event_records(caplog, "file_extraction_finished")
    assert len(finished) == 1
    assert set(finished[0].event_fields) == {
        "run_id", "source_id", "strategy_family", "discovered_file_count", "selected_file_count",
        "load_count", "attempt_count", "retry_count", "persisted_file_count",
        "archive_member_count",
        "checksum_verified_count", "skipped_file_count", "normalization_candidate_count",
        "artifact_count", "checkpoint_value", "dead_letter_count", "dead_letter_skipped_count",
    }


def test_completed_event_measures_only_the_wait_around_take(tmp_path, caplog, monkeypatch):
    from janus.strategies.files import concurrent_extraction

    calls = []
    times = iter((10.0, 12.5))

    def monotonic():
        calls.append("clock")
        return next(times)

    original_take = concurrent_extraction.OrderedDownloadWindow.take

    def take(self, index):
        calls.append("take")
        return original_take(self, index)

    monkeypatch.setattr(concurrent_extraction, "time", SimpleNamespace(monotonic=monotonic))
    monkeypatch.setattr(concurrent_extraction.OrderedDownloadWindow, "take", take)
    fixture = build_cnpj_fixture(tmp_path, archive_count=1)
    strategy = fixture.build_strategy(logger=StructuredLogger(logging.getLogger(__name__)))
    with caplog.at_level(logging.INFO, logger=__name__):
        strategy.extract(fixture.build_plan("timing", concurrency=4))
    assert calls == ["clock", "take", "clock"]
    completed = _event_records(caplog, "file_download_completed")
    assert len(completed) == 1
    assert completed[0].event_fields["waited_seconds"] == 2.5
