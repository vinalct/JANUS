from __future__ import annotations

import argparse
import json
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from typing import Any

from janus.lineage.persistence import MetadataZonePaths
from janus.models import ExecutionPlan, ExtractionResult
from janus.strategies.common import _raw_run_path_prefix
from janus.strategies.files import FileHook, FileStrategy
from janus.strategies.http import ApiRequest, ApiResponse, ApiTransport
from janus.utils.logging import StructuredLogger
from tests.support.file_downloads_fixtures import (
    CnpjFixture,
    build_archive_package_fixture,
    build_cnpj_fixture,
    build_flat_files_fixture,
)

__all__ = [
    "CnpjFixture",
    "FileRunCapture",
    "FileRunFingerprint",
    "FileRunSpec",
    "RecordedRequest",
    "RedirectFileServer",
    "ScriptedFileServer",
    "ScriptedFileTransport",
    "UrlResponse",
    "UrlScript",
    "build_archive_package_fixture",
    "build_cnpj_fixture",
    "build_flat_files_fixture",
    "fingerprint_file_run",
    "run_twice",
    "serve_file_downloads",
    "serve_redirect_origins",
]

GATE_TIMEOUT_SECONDS = 5
FIXTURE_TIME = datetime(2026, 9, 17, 12, tzinfo=UTC)
GOLDEN_SCENARIOS = (
    "cnpj_ten_archives",
    "cnpj_one_failure_continues",
    "cnpj_failure_exceeds_budget",
    "flat_ten_csvs",
    "archive_package",
)


@dataclass(frozen=True, slots=True)
class UrlResponse:
    status_code: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


type UrlScript = Sequence[UrlResponse | Exception]


@contextmanager
def serve_file_downloads(
    answer: Callable[[str, dict[str, str]], UrlResponse],
) -> Iterator[str]:
    """Serve real HTTP on loopback, propagating handler failures to the test thread."""
    errors: list[Exception] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                response = answer(
                    self.path, {key.lower(): value for key, value in self.headers.items()}
                )
            except Exception as exc:
                with lock:
                    errors.append(exc)
                response = UrlResponse(500, b"loopback handler failed")
            self.send_response(response.status_code)
            headers = dict(response.headers)
            headers.setdefault("Content-Length", str(len(response.body)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            # Declared-length rejection closes the response without reading its body.
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(response.body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(GATE_TIMEOUT_SECONDS)
        assert not thread.is_alive(), "loopback server did not shut down"
    if errors:
        raise AssertionError("loopback handler failed") from errors[0]


@dataclass(slots=True)
class RedirectFileServer:
    """The two loopback origins and the headers observed before and after redirecting."""

    origin_url: str = ""
    destination_url: str = ""
    origin_headers: list[dict[str, str]] = field(default_factory=list)
    destination_headers: list[dict[str, str]] = field(default_factory=list)
    max_in_flight: int = 0


@contextmanager
def serve_redirect_origins(
    bodies: Mapping[str, UrlResponse], listing: UrlResponse, *, require_overlap: bool = False,
) -> Iterator[RedirectFileServer]:
    """Redirect archive URLs to another port; events prove concurrent arrivals at B."""
    observed = RedirectFileServer()
    lock, overlap = threading.Lock(), threading.Event()
    active = 0

    def destination(path: str, headers: dict[str, str]) -> UrlResponse:
        nonlocal active
        with lock:
            observed.destination_headers.append(headers)
            active += 1
            observed.max_in_flight = max(observed.max_in_flight, active)
            if active >= 2:
                overlap.set()
        try:
            if require_overlap:
                assert overlap.wait(GATE_TIMEOUT_SECONDS), "redirected downloads did not overlap"
            return bodies[path]
        finally:
            with lock:
                active -= 1

    def origin(path: str, headers: dict[str, str]) -> UrlResponse:
        if path == "/":
            return listing
        assert path in bodies, f"Unscripted redirect URL: {path}"
        with lock:
            observed.origin_headers.append(headers)
        return UrlResponse(302, b"", (("Location", observed.destination_url + path),))

    with serve_file_downloads(destination) as destination_url:
        observed.destination_url = destination_url
        with serve_file_downloads(origin) as origin_url:
            observed.origin_url = origin_url
            yield observed


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    method: str
    url: str
    thread_ident: int
    thread_name: str
    sequence: int


@dataclass
class ScriptedFileServer:
    """One lock-guarded script shared by fresh transports; gates never hold the lock.

    Configure routes and gates before starting workers. ``requests`` records starts;
    ``completions`` records answers (including exceptions), with a shared sequence counter.
    """

    routes: Mapping[tuple[str, str], UrlScript]
    gates: Mapping[str, threading.Event] = field(default_factory=dict)
    requests: list[RecordedRequest] = field(default_factory=list, init=False)
    completions: list[RecordedRequest] = field(default_factory=list, init=False)
    transport_requests: list[tuple[int, int]] = field(default_factory=list, init=False)
    # Retain instances so Python cannot reuse an id and merge unrelated clients.
    transports: dict[int, ScriptedFileTransport] = field(default_factory=dict, init=False)
    opened: int = field(default=0, init=False)
    closed: int = field(default=0, init=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _attempts: dict[tuple[str, str], int] = field(default_factory=dict, init=False)
    _releases: dict[str, list[threading.Event]] = field(default_factory=dict, init=False)
    _answered: set[str] = field(default_factory=set, init=False)
    _sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.routes = {key: tuple(answers) for key, answers in self.routes.items()}
        self.gates = dict(self.gates)
        if any(not answers for answers in self.routes.values()):
            raise ValueError("Every scripted route needs at least one answer")

    def transport_factory(self) -> Callable[[], ApiTransport]:
        return lambda: ScriptedFileTransport(self)

    def release_after(self, url: str, *, after: str) -> None:
        """Hold ``url`` until ``after`` has answered, even if it answered already."""
        with self._lock:
            gate = self.gates.setdefault(url, threading.Event())
            if after in self._answered:
                gate.set()
            else:
                self._releases.setdefault(after, []).append(gate)

    def reverse_completion(self, urls: Sequence[str]) -> None:
        """Wire a reverse chain; callers must start every URL in the chain concurrently."""
        for url, after in pairwise(urls):
            self.release_after(url, after=after)

    def send(self, request: ApiRequest, transport: ScriptedFileTransport) -> ApiResponse:
        key = (request.method, request.full_url())
        with self._lock:
            if not transport._opened or transport._closed:
                raise AssertionError("Send on a transport that is not open or has been closed")
            if key not in self.routes:
                raise AssertionError(f"Unscripted {key[0]} URL: {key[1]}")
            attempt = self._attempts.get(key, 0)
            self._attempts[key] = attempt + 1
            answers = self.routes[key]
            answer = answers[min(attempt, len(answers) - 1)]
            self._sequence += 1
            recorded = RecordedRequest(
                request.method,
                request.redacted_url(),
                threading.get_ident(),
                threading.current_thread().name,
                self._sequence,
            )
            self.requests.append(recorded)
            self.transports[id(transport)] = transport
            self.transport_requests.append((id(transport), recorded.thread_ident))
            gate = self.gates.get(key[1])

        if gate is not None and not gate.wait(timeout=GATE_TIMEOUT_SECONDS):
            raise AssertionError(f"Timed out waiting for scripted URL gate: {recorded.url}")

        response = None if isinstance(answer, Exception) else ApiResponse(
            request=request,
            status_code=answer.status_code,
            body=answer.body,
            headers=answer.headers,
            received_at=FIXTURE_TIME,
        )
        with self._lock:
            self._sequence += 1
            self.completions.append(replace(recorded, sequence=self._sequence))
            self._answered.add(key[1])
            for dependent in self._releases.get(key[1], ()):
                dependent.set()
        if isinstance(answer, Exception):
            raise answer
        assert response is not None
        return response


class ScriptedFileTransport:
    """Send-only transport: ApiClient exercises its real capped-reader fallback."""

    def __init__(self, server: ScriptedFileServer) -> None:
        self.server = server
        self._opened = False
        self._closed = False

    def open(self) -> None:
        with self.server._lock:
            if self._closed:
                raise AssertionError("Cannot reopen a closed transport")
            if not self._opened:
                self._opened = True
                self.server.opened += 1

    def close(self) -> None:
        with self.server._lock:
            if self._opened and not self._closed:
                self.server.closed += 1
            self._closed = True

    def send(self, request: ApiRequest) -> ApiResponse:
        return self.server.send(request, self)


@dataclass(frozen=True, slots=True)
class FileRunFingerprint:
    """Consumer-visible outputs with only run identity and timestamps normalized.

    A failed extraction has no result: result-only fields are empty/None, while the
    complete raw tree, file hashes, sidecars and dead letters still describe its aftermath.
    """

    artifact_paths: tuple[str, ...]
    artifact_formats: tuple[str, ...]
    artifact_checksums: tuple[str | None, ...]
    artifact_bytes_sha256: tuple[str, ...]
    sidecars: tuple[tuple[str, str], ...]
    raw_tree: tuple[str, ...]
    raw_files_sha256: tuple[tuple[str, str], ...]
    handoff: dict[str, Any] | None
    dead_letters: dict[str, Any] | None
    records_extracted: int | None
    checkpoint_value: str | None
    extraction_metadata: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> FileRunFingerprint:
        data = json.loads(payload)
        for key in (
            "artifact_paths", "artifact_formats", "artifact_checksums",
            "artifact_bytes_sha256", "raw_tree",
        ):
            data[key] = tuple(data[key])
        for key in ("sidecars", "raw_files_sha256"):
            data[key] = tuple(tuple(pair) for pair in data[key])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class FileRunSpec:
    """Per-run answers and options; unexpected failures always propagate."""

    resume: bool = False
    dead_letter_max_items: int | None = None
    routes: Mapping[tuple[str, str], UrlScript] | None = None
    hook: FileHook | None = None
    logger: StructuredLogger | None = None
    expected_error: type[BaseException] | None = None
    reverse_completion_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class FileRunCapture:
    """Snapshot one run before the next can change its metadata or request log."""

    fixture: CnpjFixture
    strategy: FileStrategy
    plan: ExecutionPlan
    result: ExtractionResult | None
    error: BaseException | None
    fingerprint: FileRunFingerprint
    requests: tuple[RecordedRequest, ...]
    completions: tuple[RecordedRequest, ...]
    state_files: tuple[tuple[str, bytes], ...]
    staging_paths: tuple[str, ...]


def _capture_file_run(
    fixture: CnpjFixture, concurrency: int, run_id: str, spec: FileRunSpec,
) -> FileRunCapture:
    # Fresh servers isolate scripts/attempt counters, while both runs retain the
    # source's one metadata zone. Never clear a request log from a previous run.
    routes = fixture.server.routes if spec.routes is None else spec.routes
    server = ScriptedFileServer(routes)
    if concurrency > 1:
        server.reverse_completion([
            fixture.urls[index - 1] for index in spec.reverse_completion_indices
        ])
    fixture = replace(fixture, server=server)
    strategy = fixture.build_strategy(logger=spec.logger)
    plan = fixture.build_plan(
        run_id, concurrency=concurrency, resume=spec.resume,
        dead_letter_max_items=spec.dead_letter_max_items,
    )
    result, error = None, None
    try:
        result = strategy.extract(plan, hook=spec.hook)
    except BaseException as exc:
        if spec.expected_error is None or not isinstance(exc, spec.expected_error):
            raise
        error = exc
    if spec.expected_error is not None and error is None:
        raise AssertionError(f"Expected {spec.expected_error.__name__} in run {run_id}")
    metadata_root = MetadataZonePaths.from_plan(plan).base_dir
    raw_root = fixture.storage_layout.resolve_output(plan, "raw").resolved_path
    return FileRunCapture(
        fixture=fixture, strategy=strategy, plan=plan, result=result, error=error,
        fingerprint=fingerprint_file_run(strategy, plan, result),
        requests=tuple(server.requests), completions=tuple(server.completions),
        state_files=tuple(
            (path.relative_to(metadata_root).as_posix(), path.read_bytes())
            for path in sorted(metadata_root.rglob("*")) if path.is_file()
        ),
        staging_paths=tuple(
            path.relative_to(raw_root).as_posix() for path in sorted(raw_root.rglob(".staging"))
        ),
    )


def run_twice(
    fixture: CnpjFixture, *, concurrency: int, first: FileRunSpec, second: FileRunSpec,
) -> tuple[FileRunCapture, FileRunCapture]:
    """Extract twice into one metadata zone with distinct run-scoped raw prefixes.

    Each capture contains its fingerprint, immutable per-run request/completion logs
    and exact state-file bytes. Resume is explicit, using the CLI's ``resume=true``
    attribute through ``build_plan``; the helper never seeds or advances state.
    """
    first_run = _capture_file_run(fixture, concurrency, "first", first)
    second_run = _capture_file_run(fixture, concurrency, "second", second)
    return first_run, second_run


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _concurrency_metadata_keys() -> frozenset[str]:
    # Resolve the family's vocabulary at call time so this harness supports both
    # sequential and concurrent implementations without importing private mechanics.
    from janus.strategies.files import core

    return getattr(core, "FILE_CONCURRENCY_ONLY_METADATA_KEYS", frozenset({
        "download_concurrency", "concurrent_downloads_used", "max_in_flight_observed",
    }))


def fingerprint_file_run(
    strategy: FileStrategy, plan: ExecutionPlan, result: ExtractionResult | None,
) -> FileRunFingerprint:
    storage_layout = strategy.storage_layout_factory(plan)
    raw_root = storage_layout.resolve_output(plan, "raw").resolved_path
    prefix = _raw_run_path_prefix(plan)
    run_raw_root = raw_root / prefix if prefix is not None else raw_root
    files = sorted(path for path in run_raw_root.rglob("*") if path.is_file())
    artifacts = result.artifacts if result is not None else ()
    metadata = result.metadata_as_dict() if result is not None else {}
    metadata.pop("raw_path_prefix", None)
    for key in _concurrency_metadata_keys():
        metadata.pop(key, None)

    dead_letter_path = strategy.dead_letter_store.path(plan)
    # Retain this metadata key and its meaning, but make its absolute root comparable
    # across temporary fixture roots. No other metadata values are filtered.
    if "dead_letter_path" in metadata:
        metadata_root = Path(plan.metadata_output.path)
        if not metadata_root.is_absolute():
            metadata_root = plan.run_context.project_root / metadata_root
        metadata["dead_letter_path"] = Path(metadata["dead_letter_path"]).relative_to(
            metadata_root
        ).as_posix()
    dead_letters = None
    if dead_letter_path.exists():
        dead_letters = json.loads(dead_letter_path.read_text(encoding="utf-8"))
        for key in ("run_id", "updated_at"):
            dead_letters.pop(key, None)
        for entry in dead_letters["entries"]:
            entry.pop("recorded_at", None)

    handoff = None
    if result is not None:
        normalized = strategy.build_normalization_handoff(plan, result)
        handoff = {
            "artifact_paths": [
                Path(a.path).relative_to(run_raw_root).as_posix() for a in normalized.artifacts
            ],
            "artifact_formats": [a.format for a in normalized.artifacts],
            "normalization_artifact_count": normalized.metadata_as_dict()[
                "normalization_artifact_count"
            ],
        }
    return FileRunFingerprint(
        artifact_paths=tuple(Path(a.path).relative_to(run_raw_root).as_posix() for a in artifacts),
        artifact_formats=tuple(a.format for a in artifacts),
        artifact_checksums=tuple(a.checksum for a in artifacts),
        artifact_bytes_sha256=tuple(_file_sha256(Path(a.path)) for a in artifacts),
        sidecars=tuple(
            (path.relative_to(run_raw_root).as_posix(), path.read_text(encoding="utf-8"))
            for path in files if path.name.endswith(".sha256")
        ),
        raw_tree=tuple(path.relative_to(run_raw_root).as_posix() for path in files),
        raw_files_sha256=tuple(
            (path.relative_to(run_raw_root).as_posix(), _file_sha256(path)) for path in files
        ),
        handoff=handoff,
        dead_letters=dead_letters,
        records_extracted=result.records_extracted if result is not None else None,
        checkpoint_value=result.checkpoint_value if result is not None else None,
        extraction_metadata=metadata,
    )


def run_golden_scenario(tmp_path: Path, scenario: str) -> dict[str, Any]:
    """Run one of the five baseline scenarios; indices denote 1-based candidates."""
    failure = UrlResponse(500, b"scripted download failure")
    builders = {
        "cnpj_ten_archives": lambda: build_cnpj_fixture(tmp_path),
        "cnpj_one_failure_continues": lambda: build_cnpj_fixture(tmp_path, failing={5: failure}),
        "cnpj_failure_exceeds_budget": lambda: build_cnpj_fixture(
            tmp_path, failing={3: failure, 6: failure}, dead_letter_max_items=1,
        ),
        "flat_ten_csvs": lambda: build_flat_files_fixture(tmp_path),
        "archive_package": lambda: build_archive_package_fixture(tmp_path),
    }
    fixture = builders[scenario]()
    strategy = fixture.build_strategy()
    plan = fixture.build_plan(f"golden-{scenario}", concurrency=1)
    result = None
    exception = None
    try:
        result = strategy.extract(plan)
    except Exception as exc:
        if scenario != "cnpj_failure_exceeds_budget":
            raise
        exception = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "exception": exception,
        "fingerprint": json.loads(fingerprint_file_run(strategy, plan, result).to_json()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-golden", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.write_golden.exists() and not args.force:
        parser.error(f"Refusing to overwrite {args.write_golden}; use --force explicitly")
    with tempfile.TemporaryDirectory(prefix="janus-file-downloads-golden-") as tmp:
        golden = {
            name: run_golden_scenario(Path(tmp) / name, name) for name in GOLDEN_SCENARIOS
        }
    args.write_golden.parent.mkdir(parents=True, exist_ok=True)
    with args.write_golden.open("w" if args.force else "x", encoding="utf-8") as handle:
        handle.write(json.dumps(golden, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"Wrote {len(golden)} scenarios to {args.write_golden}")
    print(f"sha256={_file_sha256(args.write_golden)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
