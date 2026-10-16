"""AC-6/AC-7: sequential characterization and concurrent counterparts."""

from __future__ import annotations

import http.client
import json
import logging
import threading
import zipfile
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from pathlib import Path
from random import Random
from urllib.request import HTTPRedirectHandler

import pytest

from janus.checkpoints import CheckpointStore, DeadLetterStore
from janus.strategies.files.core import ArchiveExtractionError, FileDownloadError, FileHook
from janus.strategies.files.download import FileDownloader
from janus.strategies.http import (
    ApiClient,
    ApiNonRetryableTransportError,
    ApiResponseTooLargeError,
    UrllibApiTransport,
)
from janus.utils.logging import StructuredLogger, redact_url
from janus.writers import RawArtifactWriter, StagedWrite
from tests.support.file_downloads import (
    FileRunSpec,
    ScriptedFileServer,
    UrlResponse,
    build_cnpj_fixture,
    build_flat_files_fixture,
    fingerprint_file_run,
    run_twice,
    serve_file_downloads,
    serve_redirect_origins,
)
from tests.support.file_downloads_fixtures import deterministic_zip
from tests.unit.strategies.files.test_concurrent_dead_letter_order import (
    assert_no_staging,
    assert_worker_downloads,
    committed_archives,
    dead_letter_indices,
    download_requests,
    run_leg,
)

CAP_CONCURRENCIES = [1, 4]
RESUME_CONCURRENCIES = [1, 4]
FAILURE = UrlResponse(500, b"scripted download failure")


@pytest.fixture(autouse=True)
def small_spool(monkeypatch):
    import janus.strategies.files.download as download
    import janus.strategies.files.download_loop as loop

    monkeypatch.setattr(download, "SPOOL_THRESHOLD_BYTES", 32)
    monkeypatch.setattr(loop, "SPOOL_THRESHOLD_BYTES", 32)


def _finish(leg, concurrency):
    assert_no_staging(leg.fixture)
    assert leg.fixture.server.closed == leg.fixture.server.opened
    if concurrency > 1:
        assert_worker_downloads(leg)


@pytest.fixture
def dead_letter_records(monkeypatch):
    records = []
    loop_ident = threading.get_ident()
    original = DeadLetterStore.record

    def record(self, plan, **kwargs):
        assert threading.get_ident() == loop_ident, "dead letter recorded outside the loop thread"
        records.append((kwargs["metadata"]["file_index"], kwargs["error"]))
        return original(self, plan, **kwargs)

    monkeypatch.setattr(DeadLetterStore, "record", record)
    return records


def _download_errors(monkeypatch):
    errors = []
    original = FileDownloader.load_payload

    def load(self, plan, discovered_file, **kwargs):
        try:
            return original(self, plan, discovered_file, **kwargs)
        except FileDownloadError as exc:
            errors.append((discovered_file.filename, exc, threading.current_thread()))
            raise

    monkeypatch.setattr(FileDownloader, "load_payload", load)
    return errors


def _reference_fixture(fixture, root):
    reference = build_cnpj_fixture(root)
    mapping = deepcopy(fixture.source_mapping)
    mapping["outputs"] = reference.source_mapping["outputs"]
    return replace(
        reference,
        source_mapping=mapping,
        archive_urls=fixture.urls,
        server=ScriptedFileServer(fixture.server.routes),
    )


def _assert_sequential_equivalence(leg, root, **strategy_options):
    reference = run_leg(_reference_fixture(leg.fixture, root), 1, **strategy_options)
    assert type(leg.error) is type(reference.error)
    assert str(leg.error) == str(reference.error)
    assert leg.fingerprint == reference.fingerprint
    _finish(reference, 1)


def _cap_fixture(root, cap, *, budget=5):
    fixture = build_cnpj_fixture(root, dead_letter_max_items=budget)
    routes = dict(fixture.server.routes)
    oversized = deterministic_zip({"cap.csv": Random(25).randbytes(cap + 1)})
    assert len(oversized) > cap
    assert all(len(routes[("GET", url)][0].body) < cap for url in fixture.urls)
    routes[("GET", fixture.urls[5])] = [UrlResponse(200, oversized)]
    fixture.source_mapping["access"]["limits"] = {"max_payload_bytes": cap}
    return replace(fixture, server=ScriptedFileServer(routes))


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
@pytest.mark.parametrize("budget", [5, 0], ids=["continue", "exceed-budget"])
def test_payload_cap_dead_letters_only_the_capped_candidate(
    tmp_path, monkeypatch, concurrency, budget, dead_letter_records, record_property,
):
    errors = _download_errors(monkeypatch)
    discarded = []
    original_write = StagedWrite.write_from

    def write(self, stream):
        try:
            return original_write(self, stream)
        except ApiResponseTooLargeError:
            discarded.append(self.path)
            assert not self.path.exists(), "capped staging file survived worker cleanup"
            raise

    monkeypatch.setattr(StagedWrite, "write_from", write)
    fixture = _cap_fixture(tmp_path, 8192, budget=budget)
    leg = run_leg(fixture, concurrency)
    assert dead_letter_indices(leg) == [6]
    entry = leg.fingerprint.dead_letters["entries"][0]
    assert "max_payload_bytes=" in entry["error_message"]
    assert entry["error_type"] == "FileDownloadError"
    assert entry["item_key"] == f"remote:{fixture.urls[5]}"
    assert entry["metadata"]["file_index"] == "6"
    assert len(errors) == len(discarded) == 1
    filename, error, thread = errors[0]
    assert filename == "Empresas5.zip"
    assert dead_letter_records == [("6", error)]
    if concurrency > 1:
        assert thread.ident != threading.get_ident()
        assert thread.name.startswith("janus-file")
    else:
        assert thread.ident == threading.get_ident()
    record_property("cap_error_thread", thread.name)
    record_property("dead_letter_thread", threading.current_thread().name)
    if budget:
        assert leg.error is None
        assert leg.result.metadata_as_dict()["persisted_file_count"] == "9"
        assert len(committed_archives(leg)) == 9
    else:
        assert leg.error is error
        assert len(committed_archives(leg)) == 5
        requests = download_requests(leg)
        expected = fixture.urls[:9 if concurrency > 1 else 6]
        assert {request.url for request in requests} == {redact_url(url) for url in expected}
        answered = {answer.url for answer in fixture.server.completions}
        assert all(request.url in answered for request in requests)
    _finish(leg, concurrency)
    _assert_sequential_equivalence(leg, tmp_path / "reference")


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
@pytest.mark.parametrize("limit", ["member", "total", "ratio"])
def test_archive_caps_apply_to_one_candidate_after_original_archive_commit(
    tmp_path, monkeypatch, concurrency, limit, dead_letter_records, record_property,
):
    import janus.strategies.files.archives as archives

    checks, opened = [], []
    original_check = archives._check_archive_limits
    original_open = zipfile.ZipFile.open
    loop_ident = threading.get_ident()

    def check(*args, **kwargs):
        assert threading.get_ident() == loop_ident, "archive cap checked outside the loop thread"
        try:
            return original_check(*args, **kwargs)
        except ArchiveExtractionError as exc:
            checks.append(exc)
            raise

    def open_member(self, name, *args, **kwargs):
        if self.mode == "r":
            opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(archives, "_check_archive_limits", check)
    monkeypatch.setattr(zipfile.ZipFile, "open", open_member)
    fixture = build_cnpj_fixture(tmp_path)
    member = b"a" * 32768
    limits = {
        "member": {"max_archive_member_bytes": 8192, "max_archive_total_bytes": 65536},
        "total": {"max_archive_member_bytes": 8192, "max_archive_total_bytes": 8192},
        "ratio": {"max_archive_ratio": 50},
    }
    members = (
        {"a.csv": b"a" * 5000, "b.csv": b"b" * 5000} if limit == "total" else {"a.csv": member}
    )
    routes = dict(fixture.server.routes)
    routes[("GET", fixture.urls[3])] = [UrlResponse(200, deterministic_zip(members))]
    fixture = replace(fixture, server=ScriptedFileServer(routes))
    fixture.source_mapping["access"]["limits"] = limits[limit]
    leg = run_leg(fixture, concurrency)
    assert leg.error is None
    assert dead_letter_indices(leg) == [4]
    entry = leg.fingerprint.dead_letters["entries"][0]
    assert entry["error_type"] == "ArchiveExtractionError"
    assert entry["item_key"] == f"remote:{fixture.urls[3]}"
    assert len(checks) == 1
    assert dead_letter_records == [("4", checks[0])]
    assert not members.keys() & set(opened), "capped archive member was opened"
    assert leg.result.metadata_as_dict()["persisted_file_count"] == "9"
    assert (
        len(committed_archives(leg)) == 10
    )  # Persist precedes expansion in the sequential contract.
    assert "downloads/current/Empresas3.zip" in committed_archives(leg)
    assert not [path for path in leg.fingerprint.raw_tree if "extracted/current/Empresas3/" in path]
    _finish(leg, concurrency)
    record_property("archive_cap_thread", threading.current_thread().name)
    record_property("rejected_members_opened", 0)
    _assert_sequential_equivalence(leg, tmp_path / "reference")


def _listing():
    return UrlResponse(
        200, "".join(f'<a href="Empresas{i}.zip">archive</a>' for i in range(10)).encode()
    )


def _bodies(fixture):
    return {
        f"/Empresas{index}.zip": fixture.server.routes[("GET", url)][0]
        for index, url in enumerate(fixture.urls)
    }


def _real_fixture(fixture, base):
    fixture.source_mapping["access"].update(url=f"{base}/", link_resolver="html_links")
    urls = tuple(f"{base}/Empresas{index}.zip" for index in range(10))
    return replace(fixture, archive_urls=urls)


def _stream_threads(monkeypatch):
    identities = []
    original = ApiClient.stream

    @wraps(original)
    def stream(self, request):
        identities.append(threading.current_thread())
        return original(self, request)

    monkeypatch.setattr(ApiClient, "stream", stream)
    return identities


def _assert_real_workers(identities, concurrency):
    assert identities
    if concurrency > 1:
        assert all(thread.ident != threading.get_ident() for thread in identities), (
            "real transport downloads still ran on the loop thread"
        )
        assert all(thread.name.startswith("janus-file") for thread in identities)
    else:
        assert all(thread.ident == threading.get_ident() for thread in identities)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_declared_length_is_refused_before_body_read_or_staging(
    tmp_path, monkeypatch, concurrency, dead_letter_records,
):
    fixture = build_cnpj_fixture(tmp_path)
    bodies = _bodies(fixture)
    reads, staging = [], []
    errors = _download_errors(monkeypatch)
    original_read = http.client.HTTPResponse.read
    original_stage = RawArtifactWriter.begin_staged_write

    def read(self, *args, **kwargs):
        if self.getheader("X-Cap-Probe"):
            reads.append(True)
        return original_read(self, *args, **kwargs)

    download_context = threading.local()
    original_load = FileDownloader.load_payload

    def load(self, plan, discovered_file, **kwargs):
        download_context.filename = discovered_file.filename
        try:
            return original_load(self, plan, discovered_file, **kwargs)
        finally:
            del download_context.filename

    def stage(self, *args, **kwargs):
        staging.append(download_context.filename)
        return original_stage(self, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPResponse, "read", read)
    monkeypatch.setattr(RawArtifactWriter, "begin_staged_write", stage)
    monkeypatch.setattr(FileDownloader, "load_payload", load)
    identities = _stream_threads(monkeypatch)

    def answer(path, headers):
        if path == "/":
            return _listing()
        if path == "/Empresas2.zip":
            return UrlResponse(
                200, b"short", (("Content-Length", "999999"), ("X-Cap-Probe", "true"))
            )
        return bodies[path]

    with serve_file_downloads(answer) as base:
        fixture = _real_fixture(fixture, base)
        fixture.source_mapping["access"]["limits"] = {"max_payload_bytes": 8192}
        leg = run_leg(fixture, concurrency, transport_factory=UrllibApiTransport)
        assert leg.error is None
        assert dead_letter_indices(leg) == [3]
        entry = leg.fingerprint.dead_letters["entries"][0]
        assert "max_payload_bytes=" in entry["error_message"]
        assert entry["item_key"] == f"remote:{fixture.urls[2]}"
        assert entry["error_type"] == "FileDownloadError"
        assert len(errors) == 1
        assert dead_letter_records == [("3", errors[0][1])]
        assert leg.result.metadata_as_dict()["persisted_file_count"] == "9"
        assert len(committed_archives(leg)) == 9
        assert reads == [] and len(staging) == 9
        assert set(staging) == {f"Empresas{index}.zip" for index in range(10) if index != 2}
        _assert_real_workers(identities, concurrency)
        _assert_sequential_equivalence(
            leg, tmp_path / "reference", transport_factory=UrllibApiTransport,
        )
    assert_no_staging(fixture)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_cross_origin_redirects_strip_auth_and_preserve_archive_bytes(
    tmp_path, monkeypatch, concurrency, record_property,
):
    fixture = build_cnpj_fixture(tmp_path)
    bodies = _bodies(fixture)
    identities = _stream_threads(monkeypatch)
    monkeypatch.setenv("ORDER25_TEST_KEY", "fixture-secret")

    with serve_redirect_origins(bodies, _listing(), require_overlap=concurrency > 1) as server:
        fixture = _real_fixture(fixture, server.origin_url)
        fixture.source_mapping["access"]["auth"] = {
            "type": "header_token",
            "header_name": "X-Api-Key",
            "env_var": "ORDER25_TEST_KEY",
        }
        leg = run_leg(fixture, concurrency, transport_factory=UrllibApiTransport)
        if concurrency > 1:
            assert server.max_in_flight >= 2, "redirected downloads did not overlap"
        assert leg.error is None
        assert len(server.origin_headers) == len(server.destination_headers) == 10
        assert all(
            headers.get("x-api-key") == "fixture-secret" for headers in server.origin_headers
        )
        assert all(
            "x-api-key" not in headers and "authorization" not in headers
            for headers in server.destination_headers
        )
        assert len(committed_archives(leg)) == 10
        for artifact in leg.result.artifacts:
            path = Path(artifact.path)
            if path.suffix == ".zip":
                assert path.read_bytes() == bodies[f"/{path.name}"].body
        _assert_real_workers(identities, concurrency)
        record_property("redirect_destination_peak", server.max_in_flight)
        record_property("redirect_credentials_leaked", 0)
        _assert_sequential_equivalence(
            leg, tmp_path / "reference", transport_factory=UrllibApiTransport,
        )
    assert_no_staging(fixture)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_redirect_hop_limit_dead_letters_only_that_candidate(
    tmp_path, monkeypatch, concurrency, dead_letter_records,
):
    fixture = build_cnpj_fixture(tmp_path)
    bodies = _bodies(fixture)
    identities = _stream_threads(monkeypatch)
    errors = _download_errors(monkeypatch)
    hops = []

    def answer(path, headers):
        if path == "/":
            return _listing()
        if path.startswith("/Empresas1.zip"):
            hops.append(path)
            hop = int(path.split("?")[-1]) if "?" in path else 0
            return UrlResponse(302, b"", (("Location", f"/Empresas1.zip?{hop + 1}"),))
        return bodies[path]

    with serve_file_downloads(answer) as base:
        fixture = _real_fixture(fixture, base)
        fixture.source_mapping["access"]["limits"] = {"max_redirects": 2}
        leg = run_leg(fixture, concurrency, transport_factory=UrllibApiTransport)
        assert leg.error is None
        assert dead_letter_indices(leg) == [2]
        entry = leg.fingerprint.dead_letters["entries"][0]
        assert "max_redirects" in entry["error_message"]
        assert entry["item_key"] == f"remote:{fixture.urls[1]}"
        assert len(errors) == 1
        assert isinstance(errors[0][1].__cause__, ApiNonRetryableTransportError)
        assert dead_letter_records == [("2", errors[0][1])]
        assert hops == ["/Empresas1.zip", "/Empresas1.zip?1", "/Empresas1.zip?2"]
        assert len(committed_archives(leg)) == 9
        _assert_real_workers(identities, concurrency)
        _assert_sequential_equivalence(
            leg, tmp_path / "reference", transport_factory=UrllibApiTransport,
        )
    assert_no_staging(fixture)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_each_real_opener_uses_the_existing_redirect_policy(
    tmp_path, monkeypatch, concurrency, record_property,
):
    import janus.strategies.http.transport as transport

    fixture = build_cnpj_fixture(tmp_path)
    bodies = _bodies(fixture)
    handlers, openers, opener_threads = [], [], []
    original = transport._build_opener

    def build(context, *, redirect_handler):
        handlers.append(redirect_handler)
        opener_threads.append(threading.current_thread())
        opener = original(context, redirect_handler=redirect_handler)
        openers.append(opener)
        return opener

    monkeypatch.setattr(transport, "_build_opener", build)
    identities = _stream_threads(monkeypatch)

    def answer(path, headers):
        return _listing() if path == "/" else bodies[path]

    with serve_file_downloads(answer) as base:
        leg = run_leg(
            _real_fixture(fixture, base), concurrency, transport_factory=UrllibApiTransport
        )
        assert leg.error is None
        assert all(isinstance(handler, transport.JanusRedirectHandler) for handler in handlers)
        assert len(handlers) == (11 if concurrency > 1 else 1)
        for opener in openers:
            redirect_handlers = [
                handler for handler in opener.handlers
                if isinstance(handler, HTTPRedirectHandler)
            ]
            assert len(redirect_handlers) == 1
            assert isinstance(redirect_handlers[0], transport.JanusRedirectHandler)
        assert opener_threads[0].ident == threading.get_ident()
        if concurrency > 1:
            assert all(thread.name.startswith("janus-file") for thread in opener_threads[1:])
        record_property("janus_redirect_openers", len(openers))
        _assert_real_workers(identities, concurrency)
        _assert_sequential_equivalence(
            leg, tmp_path / "reference", transport_factory=UrllibApiTransport,
        )
    assert_no_staging(leg.fixture)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_hostile_listing_keeps_the_same_allowed_candidates(tmp_path, concurrency):
    fixture = build_flat_files_fixture(tmp_path)
    listing = Path(__file__).resolve().parents[3] / "fixtures/files/hostile_listing.html"
    urls = (
        "https://EXAMPLE.GOV.BR/dados/upper.csv",
        "https://example.gov.br/dados/dados_2023.csv",
        "https://example.gov.br/dados/dados_2024.csv",
        "https://example.gov.br/dados/secret.csv?token=abc",
    )
    routes = {("GET", "https://example.gov.br/dados/"): [UrlResponse(200, listing.read_bytes())]}
    routes.update({("GET", url): [UrlResponse(200, b"id\n1\n")] for url in urls})
    fixture.source_mapping["access"]["url"] = "https://example.gov.br/dados/"
    fixture.source_mapping["access"]["remote_file_pattern"] = None
    fixture = replace(fixture, server=ScriptedFileServer(routes), archive_urls=urls)
    leg = run_leg(fixture, concurrency)
    assert leg.error is None
    assert {request.url for request in download_requests(leg)} == {redact_url(url) for url in urls}
    assert leg.result.metadata_as_dict()["discovered_file_count"] == "4"
    assert leg.fingerprint.dead_letters is None
    assert len(fixture.server.requests) == 5
    assert fixture.server.requests[0].thread_ident == threading.get_ident()
    _finish(leg, concurrency)
    _assert_sequential_equivalence(leg, tmp_path / "reference")


def _failing_routes(fixture, indices):
    routes = dict(fixture.server.routes)
    for index in indices:
        routes[("GET", fixture.urls[index - 1])] = [FAILURE]
    return routes


def _request_indices(leg):
    indices = {redact_url(url): index for index, url in enumerate(leg.fixture.urls, 1)}
    return sorted(
        indices[request.url] for request in leg.requests
        if request.method == "GET" and request.url in indices
    )


def _state_documents(leg):
    documents = {}
    for name, payload in leg.state_files:
        document = json.loads(payload)
        # Keep run identity, filenames and all state values; only wall-clock writes vary.
        document.pop("updated_at", None)
        document.pop("recorded_at", None)
        for entry in document.get("entries", ()):
            entry.pop("recorded_at", None)
        documents[name] = document
    return documents


def _assert_two_run_equivalence(actual, reference, concurrency):
    for leg, sequential in zip(actual, reference, strict=True):
        assert type(leg.error) is type(sequential.error)
        assert str(leg.error) == str(sequential.error)
        assert leg.fingerprint == sequential.fingerprint
        assert _state_documents(leg) == _state_documents(sequential)
        assert leg.staging_paths == sequential.staging_paths == ()
        _finish(leg, concurrency)
        _finish(sequential, 1)


def _assert_resume_counters(leg, *, persisted, skipped=0, dead=0, dead_skipped=0):
    assert leg.error is None
    assert leg.plan.run_context.attributes_as_dict()["resume"] == "true"
    metadata = leg.result.metadata_as_dict()
    assert metadata["persisted_file_count"] == str(persisted)
    assert metadata["skipped_file_count"] == str(skipped)
    assert metadata["dead_letter_count"] == str(dead)
    assert metadata["dead_letter_skipped_count"] == str(dead_skipped)
    assert metadata["selected_file_count"] == "10"
    assert leg.result.records_extracted == persisted
    assert all(
        f"/run_id={leg.plan.run_context.run_id}/" in artifact.path
        and Path(artifact.path).is_file() for artifact in leg.result.artifacts
    )


def _record_resume_counters(leg, record_property):
    record_property("resume_counters", json.dumps({
        key: value for key, value in leg.fingerprint.extraction_metadata.items()
        if key.endswith("_count")
    }, sort_keys=True))
    record_property("requested_candidates", json.dumps(_request_indices(leg)))
    record_property("dead_letter_indices", json.dumps(dead_letter_indices(leg)))
    record_property("committed_archives", json.dumps(committed_archives(leg)))


def _carry_over_runs(root, concurrency, *, exceed_budget=False, logger=None):
    fixture = build_cnpj_fixture(root)
    return run_twice(
        fixture, concurrency=concurrency,
        first=FileRunSpec(
            routes=_failing_routes(fixture, (3, 7)),
            reverse_completion_indices=(1, 2, 3, 4),
        ),
        second=FileRunSpec(
            resume=True, dead_letter_max_items=2 if exceed_budget else 5,
            routes=_failing_routes(fixture, (5,)) if exceed_budget else None,
            expected_error=FileDownloadError if exceed_budget else None,
            reverse_completion_indices=(1, 2, 4, 5), logger=logger,
        ),
    )


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_resume_carries_over_dead_letters_and_skips_their_requests(
    tmp_path, concurrency, record_property,
):
    first, leg = runs = _carry_over_runs(tmp_path, concurrency)
    assert dead_letter_indices(first) == dead_letter_indices(leg) == [3, 7]
    assert _request_indices(first) == list(range(1, 11))
    assert _request_indices(leg) == [1, 2, 4, 5, 6, 8, 9, 10]
    assert first.result.metadata_as_dict()["persisted_file_count"] == "8"
    _assert_resume_counters(leg, persisted=8, dead=2, dead_skipped=2)
    assert len(committed_archives(leg)) == 8
    # Loading and skipping carried-over keys must preserve the actual state file bytes.
    assert first.state_files == leg.state_files
    assert first.plan.run_context.attributes_as_dict().get("resume") is None
    assert first.plan.metadata_output == leg.plan.metadata_output
    assert {first.plan.run_context.run_id, leg.plan.run_context.run_id} == {"first", "second"}
    assert fingerprint_file_run(first.strategy, first.plan, first.result) == first.fingerprint
    _assert_two_run_equivalence(runs, _carry_over_runs(tmp_path / "reference", 1), concurrency)
    _record_resume_counters(leg, record_property)


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_resume_without_prior_dead_letters_behaves_like_a_fresh_run(
    tmp_path, concurrency, record_property,
):
    fixture = build_cnpj_fixture(tmp_path)
    plan = fixture.build_plan("fresh", concurrency=concurrency, resume=True)
    assert not DeadLetterStore().path(plan).exists()
    leg = run_leg(fixture, concurrency, run_id="fresh", resume=True)
    reference = run_leg(build_cnpj_fixture(tmp_path / "reference"), 1, run_id="fresh")
    _assert_resume_counters(leg, persisted=10)
    assert leg.fingerprint == reference.fingerprint
    assert len(committed_archives(leg)) == len(download_requests(leg)) == 10
    assert leg.fingerprint.dead_letters is None
    assert not DeadLetterStore().path(leg.plan).exists()
    _finish(leg, concurrency)
    _finish(reference, 1)
    record_property(
        "resume_counters", json.dumps(leg.fingerprint.extraction_metadata, sort_keys=True),
    )


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_resume_budget_includes_the_carried_over_entries(
    tmp_path, concurrency, caplog, record_property, dead_letter_records,
):
    logger = StructuredLogger(logging.getLogger(__name__))
    with caplog.at_level(logging.INFO, logger=__name__):
        first, leg = runs = _carry_over_runs(
            tmp_path, concurrency, exceed_budget=True, logger=logger,
        )
    assert isinstance(leg.error, FileDownloadError)
    assert [index for index, _ in dead_letter_records] == ["3", "7", "5"]
    assert dead_letter_records[-1][1] is leg.error
    assert dead_letter_indices(first) == [3, 7]
    assert dead_letter_indices(leg) == [3, 7, 5]
    assert dict(first.state_files)["dead_letters/current.json"] != (
        dict(leg.state_files)["dead_letters/current.json"]
    )
    assert leg.fingerprint.dead_letters["entries"][:2] == first.fingerprint.dead_letters["entries"]
    assert {Path(path).name for path in committed_archives(leg)} == {
        "Empresas0.zip", "Empresas1.zip", "Empresas3.zip",
    }
    requested = [1, 2, 4, 5, 6, 8, 9] if concurrency > 1 else [1, 2, 4, 5]
    assert _request_indices(leg) == requested
    assert {request.url for request in leg.requests} == {answer.url for answer in leg.completions}
    drained = [
        record.event_fields["file_index"] for record in caplog.records
        if record.getMessage() == "file_download_drained"
    ]
    # D-4: carried-over key 7 never occupies a slot; the eligible tail is 6, 8, 9.
    assert drained == ([6, 8, 9] if concurrency > 1 else [])
    assert first.fingerprint.raw_tree == fingerprint_file_run(
        first.strategy, first.plan, first.result,
    ).raw_tree
    reference = _carry_over_runs(tmp_path / "reference", 1, exceed_budget=True)
    _assert_two_run_equivalence(runs, reference, concurrency)
    _record_resume_counters(leg, record_property)


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_interrupt_then_resume_redownloads_successes_under_the_new_run_prefix(
    tmp_path, concurrency, record_property,
):
    loop_thread = threading.get_ident()

    class InterruptHook(FileHook):
        def resolve_version(self, plan, discovered_file):
            assert threading.get_ident() == loop_thread
            if discovered_file.filename == "Empresas3.zip":
                raise KeyboardInterrupt("candidate 4")

    def interrupted_runs(root, workers):
        return run_twice(
            build_cnpj_fixture(root), concurrency=workers,
            first=FileRunSpec(
                hook=InterruptHook(), expected_error=KeyboardInterrupt,
                reverse_completion_indices=(1, 2, 3, 4),
            ),
            second=FileRunSpec(resume=True, reverse_completion_indices=(1, 2, 3, 4)),
        )

    first, leg = runs = interrupted_runs(tmp_path, concurrency)
    assert str(first.error) == "candidate 4"
    assert first.staging_paths == ()
    assert {Path(path).name for path in committed_archives(first)} == {
        "Empresas0.zip", "Empresas1.zip", "Empresas2.zip",
    }
    raw_root = first.fixture.storage_layout.resolve_output(first.plan, "raw").resolved_path
    zip_paths = list(raw_root.rglob("*.zip"))
    assert len(zip_paths) == 13
    assert sum("/run_id=first/" in str(path) for path in zip_paths) == 3
    assert sum("/run_id=second/" in str(path) for path in zip_paths) == 10
    # D-9: resume has no persisted-candidate rediscovery. All ten are requested again,
    # including archives 1..3 committed before the loop-thread interrupt at candidate 4.
    assert _request_indices(leg) == list(range(1, 11))
    _assert_resume_counters(leg, persisted=10)
    assert len(committed_archives(leg)) == 10
    assert first.fingerprint.dead_letters is leg.fingerprint.dead_letters is None
    assert first.state_files == leg.state_files == ()
    assert fingerprint_file_run(first.strategy, first.plan, None) == first.fingerprint
    _assert_two_run_equivalence(runs, interrupted_runs(tmp_path / "reference", 1), concurrency)
    _record_resume_counters(leg, record_property)


class MonthlyVersionHook(FileHook):
    def discovered_files(self, plan, files):
        # Discovery's publication version is 2027; the source hook resolves the data
        # vintage only after download. This exercises the *second* checkpoint gate.
        return tuple(
            replace(file, version=f"2027-{index:02d}") for index, file in enumerate(files, 1)
        )

    def resolve_version(self, plan, discovered_file):
        return f"2026-{int(Path(discovered_file.filename).stem.split('-')[-1]) + 1:02d}"


def _versioned_fixture(root):
    fixture = build_flat_files_fixture(root)
    fixture.source_mapping["strategy_variant"] = "versioned_file"
    fixture.source_mapping["extraction"].update(
        mode="incremental", checkpoint_strategy="max_value", checkpoint_field="publication_version"
    )
    plan = fixture.build_plan("checkpoint")
    CheckpointStore().save(plan, "2026-06", updated_at=plan.run_context.started_at)
    return fixture


def _checkpoint_runs(root, concurrency, with_dead_letter, *, logger=None):
    fixture = _versioned_fixture(root)
    return run_twice(
        fixture, concurrency=concurrency,
        first=FileRunSpec(
            hook=MonthlyVersionHook(),
            routes=_failing_routes(fixture, (8,)) if with_dead_letter else None,
            reverse_completion_indices=(1, 2, 3, 4),
        ),
        second=FileRunSpec(
            resume=True, hook=MonthlyVersionHook(), logger=logger,
            reverse_completion_indices=(1, 2, 3, 4),
        ),
    )


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
@pytest.mark.parametrize(
    "with_dead_letter", [False, True], ids=["checkpoint", "checkpoint-and-dead-letter"]
)
def test_resume_checkpoint_gate_discards_downloads_with_old_resolved_versions(
    tmp_path, concurrency, with_dead_letter, caplog, record_property,
):
    logger = StructuredLogger(logging.getLogger(__name__))
    with caplog.at_level(logging.INFO, logger=__name__):
        first, leg = runs = _checkpoint_runs(tmp_path, concurrency, with_dead_letter, logger=logger)
    persisted = 3 if with_dead_letter else 4
    _assert_resume_counters(
        leg, persisted=persisted, skipped=6,
        dead=int(with_dead_letter), dead_skipped=int(with_dead_letter),
    )
    assert _request_indices(first) == list(range(1, 11))
    assert _request_indices(leg) == [
        index for index in range(1, 11) if not (with_dead_letter and index == 8)
    ]
    assert dead_letter_indices(first) == dead_letter_indices(leg) == (
        [8] if with_dead_letter else []
    )
    skipped = [
        record for record in caplog.records if record.getMessage() == "file_candidate_skipped"
    ]
    assert [record.event_fields["file_index"] for record in skipped] == list(range(1, 7))
    assert [record.event_fields["resolved_version"] for record in skipped] == [
        f"2026-{month:02d}" for month in range(1, 7)
    ]
    assert all(record.event_fields["reason"] == "checkpoint_not_newer" for record in skipped)
    assert all(record.event_fields["checkpoint_value"] == "2026-06" for record in skipped)
    assert all(record.thread == threading.get_ident() for record in skipped)
    expected_versions = {"2026-07", "2026-09", "2026-10"}
    if not with_dead_letter:
        expected_versions.add("2026-08")
    assert {
        artifact.path.split("/downloads/")[1].split("/")[0]
        for artifact in leg.result.artifacts
    } == expected_versions
    assert leg.result.checkpoint_value == "2026-10"
    assert CheckpointStore().load(leg.plan).checkpoint_value == "2026-06"
    # A strategy extraction cannot advance the seeded checkpoint; no executor ran.
    assert first.state_files == leg.state_files
    assert "checkpoints/current.json" in dict(leg.state_files)
    reference = _checkpoint_runs(tmp_path / "reference", 1, with_dead_letter)
    _assert_two_run_equivalence(runs, reference, concurrency)
    _record_resume_counters(leg, record_property)


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_known_checkpoint_versions_are_prefiltered_during_discovery(tmp_path, concurrency):
    class KnownVersions(MonthlyVersionHook):
        def discovered_files(self, plan, files):
            return tuple(
                replace(file, version=f"2026-{index:02d}") for index, file in enumerate(files, 1)
            )

    leg = run_leg(_versioned_fixture(tmp_path), concurrency, resume=True, hook=KnownVersions())
    assert leg.error is None
    assert len(download_requests(leg)) == 4
    assert leg.result.metadata_as_dict()["selected_file_count"] == "4"
    assert leg.result.metadata_as_dict()["skipped_file_count"] == "0"
    reference = run_leg(
        _versioned_fixture(tmp_path / "reference"), 1, resume=True, hook=KnownVersions()
    )
    assert leg.fingerprint == reference.fingerprint
    _finish(leg, concurrency)
