"""AC-6/AC-7: sequential characterization and concurrent counterparts."""

from __future__ import annotations

import http.client
import threading
from contextlib import contextmanager, suppress
from dataclasses import replace
from functools import wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from janus.checkpoints import CheckpointStore
from janus.strategies.files.core import FileHook
from janus.strategies.http import ApiClient, UrllibApiTransport
from janus.utils.logging import redact_url
from janus.writers import RawArtifactWriter
from tests.support.file_downloads import (
    ScriptedFileServer,
    UrlResponse,
    build_cnpj_fixture,
    build_flat_files_fixture,
    fingerprint_file_run,
)
from tests.support.file_downloads_fixtures import deterministic_zip
from tests.unit.strategies.files.test_concurrent_dead_letter_order import (
    assert_no_staging,
    assert_worker_downloads,
    committed_archives,
    dead_letter_indices,
    download_requests,
    run_leg,
    seed_dead_letters,
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


def _cap_fixture(root, cap, *, budget=5):
    fixture = build_cnpj_fixture(root, dead_letter_max_items=budget)
    routes = dict(fixture.server.routes)
    routes[("GET", fixture.urls[5])] = [UrlResponse(200, b"x" * (cap + 1))]
    fixture.source_mapping["access"]["limits"] = {"max_payload_bytes": cap}
    return replace(fixture, server=ScriptedFileServer(routes))


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
@pytest.mark.parametrize("budget", [5, 0], ids=["continue", "exceed-budget"])
def test_payload_cap_dead_letters_only_the_capped_candidate(tmp_path, concurrency, budget):
    fixture = _cap_fixture(tmp_path, 8192, budget=budget)
    leg = run_leg(fixture, concurrency)
    assert dead_letter_indices(leg) == [6]
    entry = leg.fingerprint.dead_letters["entries"][0]
    assert "max_payload_bytes=" in entry["error_message"]
    assert entry["error_type"] == "FileDownloadError"
    if budget:
        assert leg.error is None
        assert leg.result.metadata_as_dict()["persisted_file_count"] == "9"
        assert len(committed_archives(leg)) == 9
    else:
        assert leg.error is not None
        reference = run_leg(_cap_fixture(tmp_path / "reference", 8192, budget=0), 1)
        assert type(leg.error) is type(reference.error)
        assert str(leg.error) == str(reference.error)
        assert leg.fingerprint == reference.fingerprint
        assert len(committed_archives(leg)) == 5
    _finish(leg, concurrency)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
@pytest.mark.parametrize("limit", ["member", "total", "ratio"])
def test_archive_caps_apply_to_one_candidate_after_original_archive_commit(
    tmp_path, concurrency, limit
):
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
    assert leg.fingerprint.dead_letters["entries"][0]["error_type"] == "ArchiveExtractionError"
    assert leg.result.metadata_as_dict()["persisted_file_count"] == "9"
    assert (
        len(committed_archives(leg)) == 10
    )  # Persist precedes expansion in the sequential contract.
    assert not [path for path in leg.fingerprint.raw_tree if "extracted/current/Empresas3/" in path]
    _finish(leg, concurrency)


@contextmanager
def _serve(answer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            response = answer(
                self.path, {key.lower(): value for key, value in self.headers.items()}
            )
            self.send_response(response.status_code)
            headers = dict(response.headers)
            headers.setdefault("Content-Length", str(len(response.body)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            # Declared-length rejection closes before the body is read.
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(response.body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


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
        identities.append(threading.get_ident())
        return original(self, request)

    monkeypatch.setattr(ApiClient, "stream", stream)
    return identities


def _assert_real_workers(identities, concurrency):
    assert identities
    if concurrency > 1:
        assert all(ident != threading.get_ident() for ident in identities), (
            "real transport downloads still ran on the loop thread"
        )


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_declared_length_is_refused_before_body_read_or_staging(tmp_path, monkeypatch, concurrency):
    fixture = build_cnpj_fixture(tmp_path)
    bodies = _bodies(fixture)
    reads, staging = [], []
    original_read = http.client.HTTPResponse.read
    original_stage = RawArtifactWriter.begin_staged_write

    def read(self, *args, **kwargs):
        if self.getheader("X-Cap-Probe"):
            reads.append(True)
        return original_read(self, *args, **kwargs)

    def stage(self, *args, **kwargs):
        staging.append(True)
        return original_stage(self, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPResponse, "read", read)
    monkeypatch.setattr(RawArtifactWriter, "begin_staged_write", stage)
    identities = _stream_threads(monkeypatch)

    def answer(path, headers):
        if path == "/":
            return _listing()
        if path == "/Empresas2.zip":
            return UrlResponse(
                200, b"short", (("Content-Length", "999999"), ("X-Cap-Probe", "true"))
            )
        return bodies[path]

    with _serve(answer) as base:
        fixture = _real_fixture(fixture, base)
        fixture.source_mapping["access"]["limits"] = {"max_payload_bytes": 8192}
        leg = run_leg(fixture, concurrency, transport_factory=UrllibApiTransport)
    assert leg.error is None
    assert dead_letter_indices(leg) == [3]
    assert "max_payload_bytes=" in leg.fingerprint.dead_letters["entries"][0]["error_message"]
    assert leg.result.metadata_as_dict()["persisted_file_count"] == "9"
    assert reads == [] and len(staging) == 9
    assert_no_staging(fixture)
    _assert_real_workers(identities, concurrency)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_cross_origin_redirects_strip_auth_and_preserve_archive_bytes(
    tmp_path, monkeypatch, concurrency
):
    fixture = build_cnpj_fixture(tmp_path)
    bodies = _bodies(fixture)
    received, active, peaks = [], set(), []
    lock, overlap = threading.Lock(), threading.Event()
    identities = _stream_threads(monkeypatch)
    monkeypatch.setenv("ORDER25_TEST_KEY", "fixture-secret")

    def destination(path, headers):
        with lock:
            received.append(headers)
            active.add(path)
            peaks.append(len(active))
            if len(active) >= 2:
                overlap.set()
            wait_for_overlap = concurrency > 1 and not overlap.is_set()
        try:
            # The first arriving request can be any initial candidate; waiting only
            # on index 1 deadlocks if its peers have already answered.
            if wait_for_overlap:
                assert overlap.wait(5), "redirected downloads did not overlap"
            return bodies[path]
        finally:
            with lock:
                active.remove(path)

    with _serve(destination) as destination_base:

        def origin(path, headers):
            if path == "/":
                return _listing()
            return UrlResponse(302, b"", (("Location", destination_base + path),))

        with _serve(origin) as base:
            fixture = _real_fixture(fixture, base)
            fixture.source_mapping["access"]["auth"] = {
                "type": "header_token",
                "header_name": "X-Api-Key",
                "env_var": "ORDER25_TEST_KEY",
            }
            leg = run_leg(fixture, concurrency, transport_factory=UrllibApiTransport)
    if concurrency > 1:
        assert max(peaks) >= 2, "redirected downloads did not overlap"
    assert leg.error is None
    assert len(received) == 10
    assert all(
        "x-api-key" not in headers and "authorization" not in headers for headers in received
    )
    assert len(committed_archives(leg)) == 10
    for artifact in leg.result.artifacts:
        path = Path(artifact.path)
        if path.suffix == ".zip":
            assert path.read_bytes() == bodies[f"/{path.name}"].body
    assert_no_staging(fixture)
    _assert_real_workers(identities, concurrency)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_redirect_hop_limit_dead_letters_only_that_candidate(tmp_path, monkeypatch, concurrency):
    fixture = build_cnpj_fixture(tmp_path)
    bodies = _bodies(fixture)
    identities = _stream_threads(monkeypatch)

    def answer(path, headers):
        if path == "/":
            return _listing()
        if path.startswith("/Empresas1.zip"):
            hop = int(path.split("?")[-1]) if "?" in path else 0
            return UrlResponse(302, b"", (("Location", f"/Empresas1.zip?{hop + 1}"),))
        return bodies[path]

    with _serve(answer) as base:
        fixture = _real_fixture(fixture, base)
        fixture.source_mapping["access"]["limits"] = {"max_redirects": 2}
        leg = run_leg(fixture, concurrency, transport_factory=UrllibApiTransport)
    assert leg.error is None
    assert dead_letter_indices(leg) == [2]
    assert "max_redirects" in leg.fingerprint.dead_letters["entries"][0]["error_message"]
    assert len(committed_archives(leg)) == 9
    assert_no_staging(fixture)
    _assert_real_workers(identities, concurrency)


@pytest.mark.parametrize("concurrency", CAP_CONCURRENCIES)
def test_each_real_opener_uses_the_existing_redirect_policy(tmp_path, monkeypatch, concurrency):
    import janus.strategies.http.transport as transport

    fixture = build_cnpj_fixture(tmp_path)
    bodies = _bodies(fixture)
    handlers = []
    original = transport._build_opener

    def build(context, *, redirect_handler):
        handlers.append(redirect_handler)
        return original(context, redirect_handler=redirect_handler)

    monkeypatch.setattr(transport, "_build_opener", build)
    identities = _stream_threads(monkeypatch)
    with _serve(lambda path, headers: _listing() if path == "/" else bodies[path]) as base:
        leg = run_leg(
            _real_fixture(fixture, base), concurrency, transport_factory=UrllibApiTransport
        )
    assert leg.error is None
    assert all(isinstance(handler, transport.JanusRedirectHandler) for handler in handlers)
    assert len(handlers) == (11 if concurrency > 1 else 1)
    _assert_real_workers(identities, concurrency)


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
    assert fixture.server.requests[0].thread_ident == threading.get_ident()
    _finish(leg, concurrency)


def _carry_over(fixture, concurrency):
    first = run_leg(fixture, concurrency, run_id="first")
    assert first.error is None
    assert dead_letter_indices(first) == [3, 7]
    healthy = build_cnpj_fixture(fixture.storage_layout.project_root)
    return replace(fixture, server=healthy.server), first


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_resume_carries_over_dead_letters_and_skips_their_requests(tmp_path, concurrency):
    fixture, _ = _carry_over(
        build_cnpj_fixture(tmp_path, failing={3: FAILURE, 7: FAILURE}), concurrency
    )
    leg = run_leg(fixture, concurrency, run_id="second", resume=True)
    assert leg.error is None
    assert dead_letter_indices(leg) == [3, 7]
    assert len(download_requests(leg)) == 8
    assert not {redact_url(fixture.urls[index - 1]) for index in (3, 7)} & {
        request.url for request in download_requests(leg)
    }
    metadata = leg.result.metadata_as_dict()
    assert metadata["dead_letter_count"] == metadata["dead_letter_skipped_count"] == "2"
    assert metadata["persisted_file_count"] == "8"
    reference_fixture, _ = _carry_over(
        build_cnpj_fixture(tmp_path / "reference", failing={3: FAILURE, 7: FAILURE}), 1
    )
    reference = run_leg(reference_fixture, 1, run_id="second", resume=True)
    assert leg.fingerprint == reference.fingerprint
    _finish(leg, concurrency)


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_resume_without_prior_dead_letters_behaves_like_a_fresh_run(tmp_path, concurrency):
    leg = run_leg(build_cnpj_fixture(tmp_path), concurrency, resume=True)
    reference = run_leg(build_cnpj_fixture(tmp_path / "reference"), 1)
    assert leg.error is reference.error is None
    assert leg.fingerprint == reference.fingerprint
    assert len(download_requests(leg)) == 10
    _finish(leg, concurrency)


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_resume_budget_includes_the_carried_over_entries(tmp_path, concurrency):
    fixture, _ = _carry_over(
        build_cnpj_fixture(tmp_path, failing={3: FAILURE, 7: FAILURE}), concurrency
    )
    routes = dict(fixture.server.routes)
    routes[("GET", fixture.urls[4])] = [FAILURE]
    fixture = replace(fixture, server=ScriptedFileServer(routes))
    fixture.source_mapping["extraction"]["dead_letter_max_items"] = 2
    leg = run_leg(fixture, concurrency, run_id="second", resume=True)
    assert leg.error is not None
    assert dead_letter_indices(leg) == [3, 7, 5]
    assert len(committed_archives(leg)) == 3
    assert all(
        Path(path).name in {"Empresas0.zip", "Empresas1.zip", "Empresas3.zip"}
        for path in committed_archives(leg)
    )
    reference_fixture = build_cnpj_fixture(tmp_path / "reference", failing={5: FAILURE})
    reference_fixture.source_mapping["extraction"]["dead_letter_max_items"] = 2
    # Same carried-over entries and error text as the first run above.
    first_reference = run_leg(
        build_cnpj_fixture(tmp_path / "reference", failing={3: FAILURE, 7: FAILURE}),
        1,
        run_id="first",
    )
    assert first_reference.error is None
    reference = run_leg(reference_fixture, 1, run_id="second", resume=True)
    assert type(leg.error) is type(reference.error)
    assert str(leg.error) == str(reference.error)
    assert leg.fingerprint == reference.fingerprint
    _finish(leg, concurrency)


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
def test_interrupt_then_resume_redownloads_successes_under_the_new_run_prefix(
    tmp_path, concurrency
):
    class InterruptHook(FileHook):
        def resolve_version(self, plan, discovered_file):
            if discovered_file.filename == "Empresas3.zip":
                raise KeyboardInterrupt("candidate 4")

    fixture = build_cnpj_fixture(tmp_path)
    strategy = fixture.build_strategy()
    first_plan = fixture.build_plan("first", concurrency=concurrency)
    with pytest.raises(KeyboardInterrupt, match="candidate 4"):
        strategy.extract(first_plan, hook=InterruptHook())
    before = fingerprint_file_run(strategy, first_plan, None)
    assert len([path for path in before.raw_tree if path.endswith(".zip")]) == 3
    assert_no_staging(fixture)
    assert fixture.server.opened == fixture.server.closed
    fixture.server.requests.clear()
    leg = run_leg(fixture, concurrency, run_id="second", resume=True)
    # D-9: resume has no persisted-candidate rediscovery. All successes are requested again.
    assert len(download_requests(leg)) == 10
    assert leg.error is None
    assert len(committed_archives(leg)) == 10
    assert leg.fingerprint.dead_letters == before.dead_letters
    assert all("run_id=second" in artifact.path for artifact in leg.result.artifacts)
    assert fingerprint_file_run(strategy, first_plan, None) == before
    reference_fixture = build_cnpj_fixture(tmp_path / "reference")
    reference_strategy = reference_fixture.build_strategy()
    reference_plan = reference_fixture.build_plan("first")
    with pytest.raises(KeyboardInterrupt, match="candidate 4"):
        reference_strategy.extract(reference_plan, hook=InterruptHook())
    assert fingerprint_file_run(reference_strategy, reference_plan, None) == before
    reference = run_leg(reference_fixture, 1, run_id="second", resume=True)
    assert leg.fingerprint == reference.fingerprint
    _finish(leg, concurrency)


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
    CheckpointStore().save(fixture.build_plan("checkpoint"), "2026-06")
    return fixture


@pytest.mark.parametrize("concurrency", RESUME_CONCURRENCIES)
@pytest.mark.parametrize(
    "with_dead_letter", [False, True], ids=["checkpoint", "checkpoint-and-dead-letter"]
)
def test_resume_checkpoint_gate_discards_downloads_with_old_resolved_versions(
    tmp_path,
    concurrency,
    with_dead_letter,
):
    fixture = _versioned_fixture(tmp_path)
    if with_dead_letter:
        seed_dead_letters(fixture, [8])
    leg = run_leg(fixture, concurrency, resume=True, hook=MonthlyVersionHook())
    assert leg.error is None
    metadata = leg.result.metadata_as_dict()
    assert metadata["skipped_file_count"] == "6"
    assert metadata["persisted_file_count"] == str(3 if with_dead_letter else 4)
    assert metadata["dead_letter_skipped_count"] == str(int(with_dead_letter))
    assert len(download_requests(leg)) == (9 if with_dead_letter else 10)
    assert leg.result.checkpoint_value == "2026-10"
    assert CheckpointStore().load(leg.plan).checkpoint_value == "2026-06"
    reference_fixture = _versioned_fixture(tmp_path / "reference")
    if with_dead_letter:
        seed_dead_letters(reference_fixture, [8])
    reference = run_leg(reference_fixture, 1, resume=True, hook=MonthlyVersionHook())
    assert leg.fingerprint == reference.fingerprint
    _finish(leg, concurrency)


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
