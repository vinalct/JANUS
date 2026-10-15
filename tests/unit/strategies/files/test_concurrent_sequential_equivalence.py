"""AC-1: the golden remains fixed while downloads complete out of candidate order."""

from __future__ import annotations

import concurrent.futures
import importlib
import json
import threading
from dataclasses import fields
from pathlib import Path

import pytest

from janus.utils.logging import redact_url
from tests.support.file_downloads import (
    FileRunFingerprint,
    build_archive_package_fixture,
    build_cnpj_fixture,
    build_flat_files_fixture,
    fingerprint_file_run,
)

TASK07 = pytest.mark.xfail(strict=True, reason="concurrency-only metadata")
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


@TASK07
@pytest.mark.parametrize("concurrency", [2, 4, 10])
@pytest.mark.parametrize("scenario", BUILDERS)
def test_metadata_keys_and_values_are_exclusive_to_concurrent_runs(tmp_path, scenario, concurrency):
    from janus.strategies.files.core import FILE_CONCURRENCY_ONLY_METADATA_KEYS

    assert FILE_CONCURRENCY_ONLY_METADATA_KEYS == CONCURRENCY_KEYS
    seq = BUILDERS[scenario](tmp_path / "seq")
    con = BUILDERS[scenario](tmp_path / "con")
    _, seq_metadata = _run(seq, "seq", 1)
    _, con_metadata = _run(con, "con", concurrency)
    assert set(con_metadata) - set(seq_metadata) == FILE_CONCURRENCY_ONLY_METADATA_KEYS
    assert set(seq_metadata) - set(con_metadata) == set()
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


@TASK07
def test_one_eligible_candidate_reports_one_slot_without_overlap(tmp_path):
    sequential, _ = _run(build_cnpj_fixture(tmp_path / "seq", archive_count=1), "seq", 1)
    fixture = build_cnpj_fixture(tmp_path / "con", archive_count=1)
    concurrent, metadata = _run(fixture, "con", 4)
    _assert_fields_equal(concurrent, sequential)
    assert metadata["max_in_flight_observed"] == "1"
    assert metadata["concurrent_downloads_used"] == "false"
    assert metadata["download_concurrency"] == "4"
    assert any(request.thread_name.startswith("janus-file") for request in fixture.server.requests)
