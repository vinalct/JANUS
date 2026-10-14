"""AC-3: budgets are consumed in candidate order; drained outcomes leave no state."""

from __future__ import annotations

import importlib
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from janus.checkpoints import DeadLetterStore
from janus.strategies.files.core import FileHook
from janus.utils.logging import redact_url
from tests.support.file_downloads import (
    ScriptedFileServer,
    UrlResponse,
    build_cnpj_fixture,
    fingerprint_file_run,
)

TASK06 = pytest.mark.xfail(
    strict=True, reason="concurrent budget and drain semantics"
)
pytestmark = TASK06
FAILURE = UrlResponse(500, b"scripted download failure")


def run_leg(fixture, concurrency, *, run_id=None, resume=False, hook=None, **strategy_options):
    """Capture both successful outputs and a failed run's on-disk aftermath."""
    strategy = fixture.build_strategy(**strategy_options)
    plan = fixture.build_plan(
        run_id or ("seq" if concurrency == 1 else "con"), concurrency=concurrency, resume=resume
    )
    result, error = None, None
    try:
        result = strategy.extract(plan, hook=hook)
    except Exception as exc:
        error = exc
    fingerprint = fingerprint_file_run(strategy, plan, result)
    return SimpleNamespace(
        fixture=fixture,
        strategy=strategy,
        plan=plan,
        result=result,
        error=error,
        fingerprint=fingerprint,
    )


def download_requests(leg):
    urls = {redact_url(url) for url in leg.fixture.urls}
    return [
        request
        for request in leg.fixture.server.requests
        if request.method == "GET" and request.url in urls
    ]


def assert_worker_downloads(leg):
    requests = download_requests(leg)
    assert requests, "thread proof did not exercise any download"
    assert all(request.thread_ident != threading.get_ident() for request in requests), (
        "concurrency > 1 still downloads on the loop thread"
    )
    assert all(request.thread_name.startswith("janus-file") for request in requests)


def assert_no_staging(fixture):
    assert not list(fixture.storage_layout.project_root.rglob(".staging")), (
        "uncommitted download staging survived the run"
    )


def dead_letter_indices(leg):
    state = leg.fingerprint.dead_letters
    return [int(entry["metadata"]["file_index"]) for entry in state["entries"]] if state else []


def committed_archives(leg):
    return [path for path in leg.fingerprint.raw_tree if path.endswith(".zip")]


def seed_dead_letters(fixture, indices):
    plan = fixture.build_plan("seed")
    store = DeadLetterStore()
    for index in indices:
        store.record(
            plan,
            item_key=f"remote:{fixture.urls[index - 1]}",
            item_type="file_candidate",
            error=RuntimeError("carried-over failure"),
            metadata={"file_index": str(index)},
        )


def _arm_initial_disorder(fixture):
    try:
        window = importlib.import_module("janus.strategies.files.download_window")
    except ModuleNotFoundError as exc:
        assert exc.name == "janus.strategies.files.download_window"
        return None
    # A missing window still fails the worker proof on today's baseline. Once it
    # exists, force the initial window to answer backwards, without timed sleeps.
    fixture.server.reverse_completion(fixture.urls[:4])
    return window.OrderedDownloadWindow


@pytest.fixture(autouse=True)
def staged_downloads(monkeypatch):
    # Small real ZIPs still exercise StagedWrite and drain cleanup, without giant fixtures.
    import janus.strategies.files.download as download
    import janus.strategies.files.download_loop as loop

    monkeypatch.setattr(download, "SPOOL_THRESHOLD_BYTES", 32)
    monkeypatch.setattr(loop, "SPOOL_THRESHOLD_BYTES", 32)


@pytest.mark.parametrize("failure_index", [1, 5, 10], ids=["early", "middle", "late"])
def test_one_failure_continues_and_is_recorded_at_its_candidate_index(tmp_path, failure_index):
    legs = []
    for label, concurrency in (("seq", 1), ("con", 4)):
        fixture = build_cnpj_fixture(tmp_path / label, failing={failure_index: FAILURE})
        if concurrency > 1:
            _arm_initial_disorder(fixture)
        legs.append(run_leg(fixture, concurrency))
    for leg in legs:
        assert leg.error is None
        assert leg.result.metadata_as_dict()["persisted_file_count"] == "9"
        assert len(committed_archives(leg)) == 9
        assert dead_letter_indices(leg) == [failure_index]
        assert_no_staging(leg.fixture)
    assert legs[0].fingerprint == legs[1].fingerprint
    assert_worker_downloads(legs[1])


@pytest.mark.parametrize(
    "failures,budget,stop,committed",
    [
        ((3, 6), 1, 6, 4),
        ((1,), 0, 1, 0),
        ((10,), 0, 10, 9),
        ((3, 6, 7), 1, 6, 4),
    ],
    ids=["two-failures", "early-exceed", "late-exceed", "drained-failure"],
)
def test_budget_exceed_drains_without_committing_or_dead_lettering_tail(
    tmp_path,
    monkeypatch,
    failures,
    budget,
    stop,
    committed,
):
    con_fixture = build_cnpj_fixture(
        tmp_path / "con", failing=dict.fromkeys(failures, FAILURE), dead_letter_max_items=budget
    )
    consumed_at, submitted = [], []
    window_class = _arm_initial_disorder(con_fixture)
    if window_class is not None:
        if stop == 6:
            # Candidate 7 (including its possible failure) finishes before the
            # budget-exceeding 6; it must be discarded without a dead letter.
            con_fixture.server.release_after(con_fixture.urls[5], after=con_fixture.urls[6])
        original_submit = window_class.submit

        def submit(self, index, download):
            with con_fixture.server._lock:
                con_fixture.server._sequence += 1
                submitted.append((index, con_fixture.server._sequence))
            return original_submit(self, index, download)

        monkeypatch.setattr(window_class, "submit", submit)
    original_record = DeadLetterStore.record

    def record(self, plan, **kwargs):
        state = original_record(self, plan, **kwargs)
        if plan.run_context.run_id == "con" and int(kwargs["metadata"]["file_index"]) == stop:
            with con_fixture.server._lock:
                con_fixture.server._sequence += 1
                consumed_at.append(con_fixture.server._sequence)
        return state

    monkeypatch.setattr(DeadLetterStore, "record", record)
    seq_fixture = build_cnpj_fixture(
        tmp_path / "seq", failing=dict.fromkeys(failures, FAILURE), dead_letter_max_items=budget
    )
    seq, con = run_leg(seq_fixture, 1), run_leg(con_fixture, 4)
    assert seq.error is not None and con.error is not None
    assert type(seq.error) is type(con.error)
    assert str(seq.error) == str(con.error)
    assert seq.fingerprint == con.fingerprint
    assert dead_letter_indices(con) == [index for index in failures if index <= stop]
    assert len(committed_archives(seq)) == len(committed_archives(con)) == committed
    assert_no_staging(con_fixture)
    assert len(consumed_at) == 1
    by_url = {redact_url(url): index for index, url in enumerate(con_fixture.urls, 1)}
    requests = download_requests(con)
    assert all(by_url[request.url] <= min(10, stop + 4 - 1) for request in requests)
    # Submission happens on the loop thread. A submitted worker can begin its HTTP
    # request after consumption; request-start timestamps cannot prove dispatch order.
    dispatched = submitted or [(by_url[request.url], request.sequence) for request in requests]
    assert all(index <= min(10, stop + 4 - 1) for index, _ in dispatched)
    assert all(sequence < consumed_at[0] for _, sequence in dispatched), (
        "a candidate was dispatched after the budget was exceeded"
    )
    # D-4: any speculative tail was already dispatched, answered, then discarded.
    answered = {answer.url for answer in con_fixture.server.completions}
    assert all(request.url in answered for request in requests)
    if stop == 1:
        assert {by_url[request.url] for request in requests} == {1, 2, 3, 4}
    assert_worker_downloads(con)


@pytest.mark.parametrize("budget", [0, 5])
def test_a_single_candidate_failure_always_aborts(tmp_path, budget):
    legs = [
        run_leg(
            build_cnpj_fixture(
                tmp_path / label,
                archive_count=1,
                failing={1: FAILURE},
                dead_letter_max_items=budget,
            ),
            n,
        )
        for label, n in (("seq", 1), ("con", 4))
    ]
    seq, con = legs
    assert type(seq.error) is type(con.error) and seq.error is not None
    assert str(seq.error) == str(con.error)
    assert seq.fingerprint == con.fingerprint
    assert dead_letter_indices(con) == [1]
    assert committed_archives(con) == []
    assert_no_staging(con.fixture)
    assert_worker_downloads(con)


def test_duplicate_dead_letter_key_is_checked_again_at_consumption(tmp_path, monkeypatch):
    from janus.strategies.files.download import FileDownloader

    class DuplicateHook(FileHook):
        def discovered_files(self, plan, files):
            duplicate = replace(files[1], filename="EmpresasDuplicate.zip")
            return (files[0], files[1], files[2], duplicate, *files[4:])

    first_failed = {run_id: threading.Event() for run_id in ("seq", "con")}
    original = FileDownloader.load_payload

    def load(self, plan, file, **kwargs):
        gate = first_failed[plan.run_context.run_id]
        if file.filename == "EmpresasDuplicate.zip":
            assert gate.wait(5), "duplicate started before its earlier copy failed"
        try:
            return original(self, plan, file, **kwargs)
        finally:
            if file.filename == "Empresas1.zip":
                gate.set()

    monkeypatch.setattr(FileDownloader, "load_payload", load)
    legs = []
    for label, concurrency in (("seq", 1), ("con", 4)):
        fixture = build_cnpj_fixture(tmp_path / label)
        routes = dict(fixture.server.routes)
        key = ("GET", fixture.urls[1])
        # The later duplicate succeeds, so missing the consumption-time skip would
        # commit an extra archive. A second identical failure could mask that bug.
        routes[key] = [FAILURE, routes[key][0]]
        fixture = replace(fixture, server=ScriptedFileServer(routes))
        legs.append(run_leg(fixture, concurrency, hook=DuplicateHook()))
    seq, con = legs
    assert seq.error is con.error is None
    assert seq.fingerprint == con.fingerprint
    assert dead_letter_indices(con) == [2]
    assert con.result.metadata_as_dict()["dead_letter_skipped_count"] == "2"
    assert_no_staging(con.fixture)
    assert (
        sum(request.url == redact_url(seq.fixture.urls[1]) for request in download_requests(seq))
        == 1
    )
    assert (
        sum(request.url == redact_url(con.fixture.urls[1]) for request in download_requests(con))
        == 2
    )
    assert_worker_downloads(con)


def test_resume_prefilters_carried_over_dead_letters(tmp_path):
    legs = []
    for label, concurrency in (("seq", 1), ("con", 4)):
        fixture = build_cnpj_fixture(tmp_path / label)
        seed_dead_letters(fixture, [2, 8])
        leg = run_leg(fixture, concurrency, resume=True)
        assert leg.error is None
        requested = {request.url for request in download_requests(leg)}
        assert not requested & {redact_url(fixture.urls[index - 1]) for index in (2, 8)}
        assert len(requested) == 8
        assert leg.result.metadata_as_dict()["dead_letter_skipped_count"] == "2"
        legs.append(leg)
    assert legs[0].fingerprint == legs[1].fingerprint
    assert_worker_downloads(legs[1])


def test_loop_thread_interrupt_drains_and_closes_every_client(tmp_path):
    class InterruptHook(FileHook):
        def resolve_version(self, plan, discovered_file):
            assert threading.get_ident() == main_thread
            if discovered_file.filename == "Empresas2.zip":
                raise KeyboardInterrupt("interrupt at candidate 3")

    main_thread = threading.get_ident()
    legs = []
    for label, concurrency in (("seq", 1), ("con", 4)):
        fixture = build_cnpj_fixture(tmp_path / label)
        strategy = fixture.build_strategy()
        plan = fixture.build_plan(label, concurrency=concurrency)
        with pytest.raises(KeyboardInterrupt, match="interrupt at candidate 3"):
            strategy.extract(plan, hook=InterruptHook())
        assert_no_staging(fixture)
        assert fixture.server.closed == fixture.server.opened
        leg = SimpleNamespace(
            fixture=fixture, fingerprint=fingerprint_file_run(strategy, plan, None)
        )
        assert len(committed_archives(leg)) == 2
        legs.append(leg)
    assert legs[0].fingerprint == legs[1].fingerprint
    assert_worker_downloads(legs[1])
