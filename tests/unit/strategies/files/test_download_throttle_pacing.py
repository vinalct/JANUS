"""AC-2 (D-10): pacing is unchanged when downloads overlap.

The PRD says "no six consecutive request timestamps span less than a minute".
At six requests per minute, six correctly spaced starts span 50 seconds. D-10
restates the contract as slots at least 10 seconds apart, at most six starts in
each half-open 60-second window, and seven consecutive starts spanning at least
60 seconds. Retries consume slots through the same throttle as every download.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise

import pytest

from janus.strategies.files.extraction import FileExtractionContext
from janus.strategies.http import (
    ApiRequest,
    ApiTransportError,
    HttpRequestThrottle,
    UrllibApiTransport,
)
from janus.utils.logging import redact_url
from tests.support.file_downloads import build_cnpj_fixture

CONCURRENCIES = [1, 3]


def _measure_slots(tmp_path, monkeypatch, concurrency):
    slots, built = [], []
    spy_lock = threading.Lock()
    original_wait = HttpRequestThrottle.wait_for_turn
    original_build = FileExtractionContext.build.__func__

    def wait_for_turn(self):
        # Hold a test-owned lock across the call and read: production sleeps outside
        # its lock, so another caller could otherwise advance _next_allowed_at.
        # Sleeper recordings miss the zero-delay first slot and also include retry
        # backoff: FileStrategy injects the same sleeper into both paths.
        with spy_lock:
            original_wait(self)
            slots.append(
                (
                    id(self),
                    threading.current_thread().name,
                    self._next_allowed_at - 60 / self.requests_per_minute,
                )
            )

    def build(cls, *args, **kwargs):
        context = original_build(cls, *args, **kwargs)
        built.append(id(context.throttle))
        return context

    fixture = build_cnpj_fixture(tmp_path, retrying={4: 1})
    fixture.source_mapping["access"]["rate_limit"]["requests_per_minute"] = 6
    if concurrency > 1:
        # All three initial downloads must enter the transport before the first
        # completes, making the multiple-worker assertion independent of scheduling.
        fixture.server.reverse_completion(fixture.urls[:concurrency])
    with monkeypatch.context() as spies:
        spies.setattr(HttpRequestThrottle, "wait_for_turn", wait_for_turn)
        spies.setattr(FileExtractionContext, "build", classmethod(build))
        result = fixture.build_strategy(clock=lambda: 0.0, sleeper=lambda seconds: None).extract(
            fixture.build_plan("paced", concurrency=concurrency),
        )
    assert result.metadata_as_dict()["persisted_file_count"] == "10"
    assert fixture.server.opened == fixture.server.closed
    assert fixture.source_mapping["extraction"]["retry"]["max_attempts"] == 2
    retry_responses = fixture.server.routes[("GET", fixture.urls[3])]
    assert [response.status_code for response in retry_responses] == [503, 200]
    return fixture, slots, built


def _assert_worker_slots(slots, concurrency):
    if concurrency > 1:
        names = {name for _, name, _ in slots}
        assert all(name.startswith("janus-file") for name in names), (
            "concurrent download slots were granted on the loop thread"
        )
        assert len(names) >= 2, "pacing must be exercised by multiple download workers"


@pytest.mark.parametrize("concurrency", CONCURRENCIES)
def test_slots_obey_rate_limit_and_match_sequential_retry_schedule(
    tmp_path, monkeypatch, concurrency
):
    fixture, recorded, _ = _measure_slots(tmp_path / "candidate", monkeypatch, concurrency)
    slots = sorted(slot for _, _, slot in recorded)
    assert slots == list(range(0, 101, 10))
    assert len(slots) == 11
    assert all(right - left >= 10 for left, right in pairwise(slots))
    assert all(sum(start <= slot < start + 60 for slot in slots) <= 6 for start in slots)
    assert all(slots[index + 6] - slots[index] >= 60 for index in range(len(slots) - 6))
    _, sequential, _ = _measure_slots(tmp_path / "seq", monkeypatch, 1)
    assert slots == sorted(slot for _, _, slot in sequential)
    assert fixture.server.opened == (11 if concurrency > 1 else 1)
    _assert_worker_slots(recorded, concurrency)


@pytest.mark.parametrize("concurrency", CONCURRENCIES)
def test_all_attempts_use_the_throttle_created_by_context_build(tmp_path, monkeypatch, concurrency):
    _, slots, built = _measure_slots(tmp_path, monkeypatch, concurrency)
    assert len(slots) == 11
    assert len(built) == 1
    assert len({ident for ident, _, _ in slots}) == 1
    assert {ident for ident, _, _ in slots} == set(built)
    _assert_worker_slots(slots, concurrency)


@pytest.mark.parametrize("concurrency", CONCURRENCIES)
def test_transports_are_dedicated_to_each_concurrent_download(tmp_path, monkeypatch, concurrency):
    fixture, slots, _ = _measure_slots(tmp_path, monkeypatch, concurrency)
    server = fixture.server
    expected_clients = 11 if concurrency > 1 else 1
    assert server.opened == server.closed == expected_clients
    assert len(server.transports) == expected_clients
    assert len(server.transport_requests) == len(server.requests) == 12

    threads_by_transport = defaultdict(set)
    requests_by_transport = defaultdict(list)
    for (transport_id, thread_ident), request in zip(
        server.transport_requests, server.requests, strict=True,
    ):
        assert thread_ident == request.thread_ident
        threads_by_transport[transport_id].add(thread_ident)
        requests_by_transport[transport_id].append(request)
    assert all(len(threads) == 1 for threads in threads_by_transport.values())
    assert all(transport._closed for transport in server.transports.values())

    discovery_id = server.transport_requests[0][0]
    assert server.requests[0].method == "PROPFIND"
    assert threads_by_transport[discovery_id] == {threading.get_ident()}
    if concurrency > 1:
        assert len(requests_by_transport[discovery_id]) == 1
        download_transports = {
            ident: requests for ident, requests in requests_by_transport.items()
            if ident != discovery_id
        }
        assert len(download_transports) == 10
        assert {requests[0].url for requests in download_transports.values()} == {
            redact_url(url) for url in fixture.urls
        }
        for ident, requests in download_transports.items():
            assert threading.get_ident() not in threads_by_transport[ident]
            assert {request.method for request in requests} == {"GET"}
            assert len({request.url for request in requests}) == 1
            assert len(requests) == (2 if requests[0].url == redact_url(fixture.urls[3]) else 1)
    _assert_worker_slots(slots, concurrency)


@pytest.fixture
def loopback_url():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01),
            name="urllib-race-server", daemon=True,
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_shared_urllib_transport_close_can_interrupt_send(monkeypatch, loopback_url):
    """D-5: close can clear the cached opener after send initializes it.

    Events force this interleaving on two real threads, without scheduler jitter,
    in a bounded loop. Only open's scheduling is instrumented; send, close, opener
    construction and _open_response's failed-initialization check are real.
    """
    transport = UrllibApiTransport()
    request = ApiRequest(method="GET", url=loopback_url, timeout_seconds=1)
    opener_ready, closed = threading.Event(), threading.Event()
    send_threads, close_threads, errors = [], [], []
    original_open = UrllibApiTransport.open

    def open_then_pause(self):
        original_open(self)
        assert self.opener is not None
        send_threads.append(threading.get_ident())
        opener_ready.set()
        assert closed.wait(timeout=5), "closer did not clear the initialized opener"

    def close_after_open():
        try:
            assert opener_ready.wait(timeout=5), "sender did not initialize its opener"
            close_threads.append(threading.get_ident())
            transport.close()
        finally:
            closed.set()

    try:
        assert transport.send(request).body == b"ok"
        with (
            monkeypatch.context() as spies,
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="shared-urllib") as pool,
        ):
            spies.setattr(UrllibApiTransport, "open", open_then_pause)
            for _ in range(2):
                opener_ready.clear()
                closed.clear()
                send = pool.submit(transport.send, request)
                close = pool.submit(close_after_open)
                with pytest.raises(
                    ApiTransportError,
                    match="^API transport failed to initialize urllib opener$",
                ) as caught:
                    send.result(timeout=5)
                close.result(timeout=5)
                errors.append(str(caught.value))
                assert transport.opener is None
                assert send_threads[-1] != close_threads[-1]
        assert errors == ["API transport failed to initialize urllib opener"] * 2
        # Once the scheduling spy is removed, the same transport can open again.
        assert transport.send(request).body == b"ok"
    finally:
        transport.close()
