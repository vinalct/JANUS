"""Validate the download harness and its sequential extraction reference."""

from __future__ import annotations

import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import tests.support.file_downloads as harness
from janus.strategies.http import ApiClient, ApiRequest, ApiResponseTooLargeError, ApiTransportError
from janus.utils.logging import redact_url
from tests.support.file_downloads import (
    GOLDEN_SCENARIOS,
    FileRunFingerprint,
    ScriptedFileServer,
    UrlResponse,
    build_archive_package_fixture,
    build_cnpj_fixture,
    build_flat_files_fixture,
    fingerprint_file_run,
    run_golden_scenario,
)
from tests.support.file_downloads_fixtures import WEBDAV_URL, ZIP_TIME, deterministic_zip

GOLDEN_PATH = (
    Path(__file__).resolve().parents[3] / "fixtures/files/file_downloads_sequential_golden.json"
)
URL = "https://files.download.test/a.csv"
OTHER_URL = "https://files.download.test/b.csv"


def _request(url: str = URL, *, method: str = "GET", **kwargs) -> ApiRequest:
    return ApiRequest(method=method, url=url, timeout_seconds=5, **kwargs)


def test_unscripted_url_and_method_fail_loudly():
    server = ScriptedFileServer({("GET", URL): [UrlResponse(200, b"a")]})
    with ApiClient(server.transport_factory()()) as client:
        for request in (_request(OTHER_URL), _request(method="PROPFIND")):
            with pytest.raises(AssertionError, match=request.url):
                client.send(request)


def test_attempt_lists_are_per_url_and_repeat_the_last_answer():
    server = ScriptedFileServer({
        ("GET", URL): [UrlResponse(503, b"retry"), UrlResponse(200, b"ok")],
        ("GET", OTHER_URL): [UrlResponse(400, b"other")],
        ("PROPFIND", URL): [UrlResponse(207, b"listing")],
    })
    with ApiClient(server.transport_factory()()) as client:
        assert client.send(_request()).body == b"retry"
        assert client.send(_request(OTHER_URL)).body == b"other"
        assert client.send(_request(method="PROPFIND")).body == b"listing"
        assert client.send(_request()).body == b"ok"
        assert client.send(_request()).body == b"ok"
    assert [r.sequence for r in server.requests] == [1, 3, 5, 7, 9]
    assert all(r.thread_ident == threading.get_ident() for r in server.requests)
    assert all(r.thread_name == threading.current_thread().name for r in server.requests)


def test_routes_use_the_full_url_but_record_redacted_urls():
    request = _request(params=(("token", "secret"), ("part", "1")), sensitive_params=("token",))
    server = ScriptedFileServer({("GET", request.full_url()): [UrlResponse(200, b"ok")]})
    with ApiClient(server.transport_factory()()) as client:
        client.send(request)
    assert server.requests[0].url == request.redacted_url()
    assert "secret" not in server.requests[0].url


def test_reverse_completion_orders_ten_worker_answers_without_sleeps():
    urls = tuple(f"https://files.download.test/part-{i}.csv" for i in range(10))
    server = ScriptedFileServer({("GET", url): [UrlResponse(200, url.encode())] for url in urls})
    server.reverse_completion(urls)

    def download(url):
        with ApiClient(server.transport_factory()()) as client:
            return client.send(_request(url)).body

    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="file-download-harness") as pool:
        futures = [pool.submit(download, url) for url in urls]
        assert [future.result() for future in futures] == [url.encode() for url in urls]
    completions = sorted(server.completions, key=lambda r: r.sequence)
    assert [r.url for r in completions] == list(reversed(urls))
    assert len({r.thread_ident for r in server.requests}) == 10
    assert all(r.thread_name.startswith("file-download-harness") for r in server.requests)
    assert server.opened == server.closed == 10


def test_exception_answers_release_dependents_and_advance_attempts():
    failure = ApiTransportError("scripted connection failure")
    server = ScriptedFileServer({
        ("GET", URL): [failure, UrlResponse(200, b"recovered")],
        ("GET", OTHER_URL): [UrlResponse(200, b"other")],
    })
    server.release_after(OTHER_URL, after=URL)
    with ApiClient(server.transport_factory()()) as client:
        with pytest.raises(ApiTransportError) as caught:
            client.send(_request())
        assert caught.value is failure
        assert server.gates[OTHER_URL].is_set()
        assert client.send(_request(OTHER_URL)).body == b"other"
        assert client.send(_request()).body == b"recovered"


def test_release_after_handles_an_already_answered_url():
    server = ScriptedFileServer({("GET", URL): [UrlResponse(200, b"ok")]})
    with ApiClient(server.transport_factory()()) as client:
        client.send(_request())
    server.release_after(OTHER_URL, after=URL)
    assert server.gates[OTHER_URL].is_set()


def test_gate_timeout_is_a_test_failure(monkeypatch):
    monkeypatch.setattr(harness, "GATE_TIMEOUT_SECONDS", 0)
    server = ScriptedFileServer(
        {("GET", URL): [UrlResponse(200, b"ok")]}, gates={URL: threading.Event()},
    )
    with (
        ApiClient(server.transport_factory()()) as client,
        pytest.raises(AssertionError, match="Timed out.*a.csv"),
    ):
        client.send(_request())


def test_factory_returns_distinct_transports_and_close_is_terminal():
    server = ScriptedFileServer({("GET", URL): [UrlResponse(200, b"ok")]})
    factory = server.transport_factory()
    first, second = factory(), factory()
    assert first is not second
    with pytest.raises(AssertionError, match="not open"):
        first.send(_request())
    first.open()
    second.open()
    first.close()
    first.close()
    with pytest.raises(AssertionError, match="closed"):
        first.send(_request())
    with pytest.raises(AssertionError, match="reopen"):
        first.open()
    assert second.send(_request()).body == b"ok"
    second.close()
    assert server.opened == server.closed == 2


def test_send_only_fake_exercises_the_real_stream_cap():
    server = ScriptedFileServer({("GET", URL): [UrlResponse(200, b"oversized")]})
    transport = server.transport_factory()()
    assert not hasattr(transport, "stream")
    with ApiClient(transport) as client:
        response = client.stream(_request(max_payload_bytes=4))
        try:
            with pytest.raises(ApiResponseTooLargeError):
                response.body.read()
        finally:
            response.close()


def test_cnpj_archives_are_deterministic_distinct_and_latin1(tmp_path):
    first = build_cnpj_fixture(tmp_path / "first")
    second = build_cnpj_fixture(tmp_path / "second")
    assert first.archive_urls == second.archive_urls
    sizes = set()
    for index, url in enumerate(first.archive_urls):
        body = first.server.routes[("GET", url)][0].body
        assert body == second.server.routes[("GET", url)][0].body
        sizes.add(len(body))
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            member = archive.infolist()[0]
            assert member.filename == f"K3241.K03200Y{index}.D30513.EMPRECSV"
            assert member.date_time == ZIP_TIME
            rows = archive.read(member)
            assert b";" in rows
            assert any(byte > 127 for byte in rows)
            assert len(rows.decode("iso-8859-1").splitlines()) == 3 + index
            assert not rows.startswith(b"cnpj_basico")
    assert len(sizes) == 10
    plan = first.build_plan("discovery")
    strategy = first.build_strategy()
    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()
    assert metadata["discovered_file_count"] == metadata["selected_file_count"] == "10"
    assert first.server.requests[0].method == "PROPFIND"
    assert first.server.requests[0].url == WEBDAV_URL
    assert [r.url for r in first.server.requests[1:]] == [redact_url(u) for u in first.urls]


def test_retrying_candidate_recovers_using_its_own_route(tmp_path):
    fixture = build_cnpj_fixture(tmp_path, retrying={5: 2})
    plan = fixture.build_plan("retry")
    result = fixture.build_strategy().extract(plan)
    assert result.metadata_as_dict()["persisted_file_count"] == "10"
    assert result.metadata_as_dict()["dead_letter_count"] == "0"
    assert sum(r.url == redact_url(fixture.urls[4]) for r in fixture.server.requests) == 3


@pytest.mark.parametrize("failure", [False, True])
def test_fingerprint_normalizes_run_ids_and_temporary_roots(tmp_path, failure):
    fingerprints = []
    for label in ("first", "second"):
        fixture = build_cnpj_fixture(
            tmp_path / label, failing={5: UrlResponse(500, b"failure")} if failure else None,
        )
        strategy = fixture.build_strategy()
        plan = fixture.build_plan(label)
        result = strategy.extract(plan)
        observed = fingerprint_file_run(strategy, plan, result)
        fingerprints.append(observed)
        other_plan = fixture.build_plan(f"{label}-other-run")
        assert fingerprint_file_run(strategy, other_plan, strategy.extract(other_plan)) == observed
        assert fixture.server.opened == fixture.server.closed == 2
    assert fingerprints[0] == fingerprints[1]
    assert FileRunFingerprint.from_json(fingerprints[0].to_json()) == fingerprints[0]
    if failure:
        assert fingerprints[0].extraction_metadata["dead_letter_path"] == (
            "dead_letters/current.json"
        )
        entry = fingerprints[0].dead_letters["entries"][0]
        assert entry["metadata"]["file_index"] == "5"
        assert entry["item_key"].endswith("Empresas4.zip")
        assert "recorded_at" not in entry


def test_fingerprint_detects_a_one_byte_change_in_an_archive_member(tmp_path):
    fixture = build_cnpj_fixture(tmp_path)
    strategy = fixture.build_strategy()
    first_plan = fixture.build_plan("before")
    before = fingerprint_file_run(strategy, first_plan, strategy.extract(first_plan))
    url = fixture.urls[0]
    answer = fixture.server.routes[("GET", url)][0]
    with zipfile.ZipFile(io.BytesIO(answer.body)) as archive:
        name = archive.namelist()[0]
        original = archive.read(name)
    changed = original.replace(b"0", b"1", 1)
    assert len(changed) == len(original)
    assert sum(a != b for a, b in zip(changed, original, strict=True)) == 1
    fixture.server.routes[("GET", url)] = (
        replace(answer, body=deterministic_zip({name: changed})),
    )
    second_plan = fixture.build_plan("after")
    after = fingerprint_file_run(strategy, second_plan, strategy.extract(second_plan))
    assert before != after
    assert before.artifact_paths == after.artifact_paths
    assert before.artifact_bytes_sha256[:2] != after.artifact_bytes_sha256[:2]
    assert before.sidecars != after.sidecars


def test_fingerprint_retains_staging_sidecars_order_and_stable_metadata(tmp_path):
    fixture = build_flat_files_fixture(tmp_path)
    plan = fixture.build_plan("outputs")
    strategy = fixture.build_strategy()
    result = strategy.extract(plan)
    before = fingerprint_file_run(strategy, plan, result)
    metadata_changed = result.with_metadata("unexpected_stable_key", "observable")
    assert fingerprint_file_run(strategy, plan, metadata_changed) != before
    for key in harness._concurrency_metadata_keys():
        metadata_changed = result.with_metadata(key, "4")
        assert fingerprint_file_run(strategy, plan, metadata_changed) == before
    reversed_result = replace(result, artifacts=tuple(reversed(result.artifacts)))
    reversed_fingerprint = fingerprint_file_run(strategy, plan, reversed_result)
    assert reversed_fingerprint.artifact_paths == tuple(reversed(before.artifact_paths))
    assert reversed_fingerprint.handoff != before.handoff
    raw_root = Path(result.artifacts[0].path).parents[2]
    staging = raw_root / ".staging" / "leftover.partial"
    staging.parent.mkdir()
    staging.write_bytes(b"uncommitted")
    after = fingerprint_file_run(strategy, plan, result)
    assert ".staging/leftover.partial" in after.raw_tree
    assert after != before
    staging.unlink()
    staging.parent.rmdir()
    sidecar = Path(result.artifacts[0].path + ".sha256")
    sidecar.write_text("changed checksum\n", encoding="utf-8")
    assert fingerprint_file_run(strategy, plan, result).sidecars != before.sidecars


def test_archive_package_filters_one_of_three_members(tmp_path):
    fixture = build_archive_package_fixture(tmp_path)
    plan = fixture.build_plan("archive-package")
    strategy = fixture.build_strategy()
    result = strategy.extract(plan)
    observed = fingerprint_file_run(strategy, plan, result)
    assert observed.artifact_formats == ("binary", "csv", "csv")
    assert observed.records_extracted == 2
    assert observed.handoff["normalization_artifact_count"] == "2"
    assert not any("leia-me" in path for path in observed.raw_tree)
    body = fixture.server.routes[("GET", fixture.urls[0])][0].body
    with zipfile.ZipFile(io.BytesIO(body)) as zip_:
        assert len(zip_.infolist()) == 3


def test_plan_parameters_and_resume_match_the_cli_contract(tmp_path):
    fixture = build_cnpj_fixture(tmp_path)
    plan = fixture.build_plan("resume", concurrency=4, dead_letter_max_items=1, resume=True)
    assert plan.source_config.access.rate_limit.concurrency == 4
    assert plan.source_config.extraction.dead_letter_max_items == 1
    assert plan.run_context.attributes_as_dict() == {"resume": "true"}
    assert fixture.source_mapping["access"]["rate_limit"]["concurrency"] == 1


@pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS)
def test_fresh_sequential_run_equals_the_unmodified_baseline_golden(tmp_path, scenario):
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert set(golden) == set(GOLDEN_SCENARIOS)
    observed = run_golden_scenario(tmp_path, scenario)
    assert observed == golden[scenario]
    assert not any(".staging/" in path for path in observed["fingerprint"]["raw_tree"])
    restored = FileRunFingerprint.from_json(json.dumps(golden[scenario]["fingerprint"]))
    assert json.loads(restored.to_json()) == observed["fingerprint"]


def test_budget_failure_records_order_and_only_the_sequential_prefix(tmp_path):
    observed = run_golden_scenario(tmp_path, "cnpj_failure_exceeds_budget")
    assert observed["exception"]["type"] == "FileDownloadError"
    fingerprint = observed["fingerprint"]
    entries = fingerprint["dead_letters"]["entries"]
    assert [e["metadata"]["file_index"] for e in entries] == ["3", "6"]
    downloads = [path for path in fingerprint["raw_tree"] if path.endswith(".zip")]
    assert downloads == [f"downloads/current/Empresas{i}.zip" for i in (0, 1, 3, 4)]
    assert len(fingerprint["raw_files_sha256"]) == 16
    assert fingerprint["handoff"] is None
    assert fingerprint["records_extracted"] is None


def test_generator_requires_explicit_force_to_overwrite(tmp_path, capsys):
    path = tmp_path / "golden.json"
    path.write_text("existing reference", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        harness.main(["--write-golden", str(path)])
    assert caught.value.code == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == "existing reference"
    assert harness.main(["--write-golden", str(path), "--force"]) == 0
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
        GOLDEN_PATH.read_text("utf-8")
    )
