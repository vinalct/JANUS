"""Nothing remote gets to decide how much memory JANUS spends."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

import janus.strategies.files.download as download_module
import janus.strategies.files.download_loop as download_loop_module
from janus.checkpoints import CheckpointStore
from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api import ApiResponse
from janus.strategies.files import (
    ArchiveExtractionError,
    FileDownloadError,
    FileHook,
    FileIntegrityError,
    FileStrategy,
)
from janus.utils.logging import build_structured_logger
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter
from tests.support.memory_probe import max_rss_bytes, measure_download_peak_rss

MIB = 1024 * 1024

#: NFR-6's bound: the writer spools at 64 MiB, so a download's peak RSS must be the spool
#: threshold plus a chunk — independent of payload size — not twice the payload.
SPOOL_THRESHOLD_BYTES = 64 * MIB
RSS_SLACK_BYTES = 32 * MIB
RSS_PROBE_PAYLOAD_BYTES = 256 * MIB


# ---------------------------------------------------------------------------
# Harness


@dataclass(frozen=True, slots=True)
class ResponseSpec:
    status_code: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


class FakeTransport:
    """Send-only, exactly like every other file-family fake."""

    def __init__(self, responses: list[ResponseSpec | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def send(self, request):
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("No fake responses remain for this transport")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ApiResponse(
            request=request,
            status_code=response.status_code,
            body=response.body,
            headers=tuple(sorted(response.headers.items())),
        )


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": "runtime",
                "raw_dir": "runtime/raw",
                "bronze_dir": "runtime/bronze",
                "metadata_dir": "runtime/metadata",
            }
        },
        tmp_path,
    )


def _build_source_config(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "static_file",
    access_path: Path | None = None,
    access_url: str | None = None,
    access_format: str = "csv",
    file_pattern: str | None = None,
    link_resolver: str | None = None,
    limits: dict[str, int] | None = None,
    dead_letter_max_items: int = 0,
    extraction_mode: str = "full_refresh",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
) -> SourceConfig:
    access: dict[str, Any] = {
        "method": "GET",
        "format": access_format,
        "timeout_seconds": 30,
        "auth": {"type": "none"},
        "pagination": {"type": "none"},
        "rate_limit": {"requests_per_minute": None, "concurrency": 1, "backoff_seconds": 5},
    }
    if access_path is not None:
        access["path"] = str(access_path)
    if access_url is not None:
        access["url"] = access_url
    if file_pattern is not None:
        access["file_pattern"] = file_pattern
    if link_resolver is not None:
        access["link_resolver"] = link_resolver
    if limits is not None:
        access["limits"] = dict(limits)

    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "file",
            "strategy": "file",
            "strategy_variant": variant,
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": access,
            "extraction": {
                "mode": extraction_mode,
                "checkpoint_field": checkpoint_field,
                "checkpoint_strategy": checkpoint_strategy,
                "dead_letter_max_items": dead_letter_max_items,
                "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "csv", "write_mode": "append"},
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "binary"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {
                "allow_schema_evolution": True,
                **({"unique_fields": ["id"]} if extraction_mode == "incremental" else {}),
            },
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _build_plan(tmp_path: Path, **kwargs) -> ExecutionPlan:
    source_config = _build_source_config(tmp_path, **kwargs)
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id=f"run-{source_config.source_id}",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        ),
    )


def _build_strategy(tmp_path: Path, responses: list[ResponseSpec | Exception] | None = None):
    transport = FakeTransport(responses or [])
    strategy = FileStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    return strategy, transport


def _raw_root(tmp_path: Path, plan: ExecutionPlan) -> Path:
    return Path(_storage_layout(tmp_path).resolve_output(plan, "raw").resolved_path)


def _extracted_files(raw_root: Path) -> list[Path]:
    return _files_below_raw_subdirectory(raw_root, "extracted")


def _downloaded_files(raw_root: Path) -> list[Path]:
    return _files_below_raw_subdirectory(raw_root, "downloads")


def _files_below_raw_subdirectory(raw_root: Path, directory_name: str) -> list[Path]:
    return [
        path
        for path in raw_root.rglob("*")
        if path.is_file() and directory_name in path.relative_to(raw_root).parts
    ]


# ---------------------------------------------------------------------------
# Crafted archives


def _zip_archive(path: Path, members: dict[str, bytes], *, compress: bool = True) -> Path:
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=mode) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _targz_archive(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _csv_bytes(size: int) -> bytes:
    """Compressible-but-not-absurd CSV filler of exactly ``size`` bytes."""
    row = b"id,name\n"
    block = b"".join(f"{index},name-{index}\n".encode() for index in range(256))
    body = (row + block * (size // len(block) + 1))[:size]
    return body


# ---------------------------------------------------------------------------
# FR-5 — the caps, checked before a member is opened

ARCHIVE_BUILDERS = (("zip", _zip_archive), ("targz", _targz_archive))


@pytest.mark.parametrize(
    ("kind", "build"), ARCHIVE_BUILDERS, ids=[row[0] for row in ARCHIVE_BUILDERS]
)
def test_a_member_over_the_member_cap_is_refused_before_decompression(tmp_path, kind, build):
    """The size is in the header; reading it costs nothing and must happen first."""
    suffix = "zip" if kind == "zip" else "tar.gz"
    archive = build(tmp_path / "fixtures" / f"package.{suffix}", {"big.csv": _csv_bytes(4 * MIB)})
    plan = _build_plan(
        tmp_path,
        source_id=f"member_cap_{kind}",
        variant="archive_package",
        access_path=archive,
        access_format="binary",
        limits={
            "max_archive_member_bytes": MIB,
            "max_archive_total_bytes": 64 * MIB,
            "max_archive_ratio": 10_000,
        },
    )
    strategy, _transport = _build_strategy(tmp_path)

    with pytest.raises(ArchiveExtractionError) as excinfo:
        strategy.extract(plan)

    message = str(excinfo.value)
    assert "big.csv" in message, f"the error must name the member: {message!r}"
    assert "max_archive_member_bytes" in message
    assert str(MIB) in message

    raw_root = _raw_root(tmp_path, plan)
    assert _extracted_files(raw_root) == [], "a refused member was written anyway"
    assert _downloaded_files(raw_root), (
        "the archive itself must still be persisted under downloads/, exactly as today"
    )


@pytest.mark.parametrize(
    ("kind", "build"), ARCHIVE_BUILDERS, ids=[row[0] for row in ARCHIVE_BUILDERS]
)
def test_the_selected_total_over_the_total_cap_is_refused(tmp_path, kind, build):
    """Four members that each clear the member cap can still exhaust the driver together."""
    suffix = "zip" if kind == "zip" else "tar.gz"
    archive = build(
        tmp_path / "fixtures" / f"package.{suffix}",
        {f"part-{index}.csv": _csv_bytes(MIB) for index in range(4)},
    )
    plan = _build_plan(
        tmp_path,
        source_id=f"total_cap_{kind}",
        variant="archive_package",
        access_path=archive,
        access_format="binary",
        limits={
            "max_archive_member_bytes": 2 * MIB,
            "max_archive_total_bytes": 2 * MIB,
            "max_archive_ratio": 10_000,
        },
    )
    strategy, _transport = _build_strategy(tmp_path)

    with pytest.raises(ArchiveExtractionError) as excinfo:
        strategy.extract(plan)

    message = str(excinfo.value)
    assert "max_archive_total_bytes" in message
    assert _extracted_files(_raw_root(tmp_path, plan)) == []


@pytest.mark.parametrize(
    ("kind", "build"), ARCHIVE_BUILDERS, ids=[row[0] for row in ARCHIVE_BUILDERS]
)
def test_an_over_ratio_archive_is_refused(tmp_path, kind, build):
    """8 MiB of zeros deflates roughly 1000:1 — the shape of every zip bomb ever written."""
    suffix = "zip" if kind == "zip" else "tar.gz"
    archive = build(tmp_path / "fixtures" / f"package.{suffix}", {"bomb.csv": b"\0" * (8 * MIB)})
    plan = _build_plan(
        tmp_path,
        source_id=f"ratio_cap_{kind}",
        variant="archive_package",
        access_path=archive,
        access_format="binary",
        limits={
            "max_archive_member_bytes": 64 * MIB,
            "max_archive_total_bytes": 128 * MIB,
            "max_archive_ratio": 200,
        },
    )
    strategy, _transport = _build_strategy(tmp_path)

    with pytest.raises(ArchiveExtractionError) as excinfo:
        strategy.extract(plan)

    message = str(excinfo.value)
    assert "max_archive_ratio" in message
    assert "200" in message
    assert _extracted_files(_raw_root(tmp_path, plan)) == []


def test_the_caps_are_checked_over_the_selection_not_the_whole_archive(tmp_path):
    """An unselected member is never opened, but a selected member costs its full size."""
    members = {"big.csv": _csv_bytes(4 * MIB), "small.csv": b"id,name\n1,alpha\n"}
    caps = {
        "max_archive_member_bytes": MIB,
        "max_archive_total_bytes": 64 * MIB,
        "max_archive_ratio": 10_000,
    }
    archive = _zip_archive(tmp_path / "fixtures" / "package.zip", members)

    selective_plan = _build_plan(
        tmp_path,
        source_id="caps_over_selection",
        variant="archive_package",
        access_path=archive,
        access_format="binary",
        file_pattern="small.csv",
        limits=caps,
    )
    strategy, _transport = _build_strategy(tmp_path)
    names = [Path(artifact.path).name for artifact in strategy.extract(selective_plan).artifacts]
    assert "small.csv" in names
    assert "big.csv" not in names

    everything_plan = _build_plan(
        tmp_path,
        source_id="caps_over_everything",
        variant="archive_package",
        access_path=archive,
        access_format="binary",
        limits=caps,
    )
    strategy, _transport = _build_strategy(tmp_path)
    with pytest.raises(ArchiveExtractionError):
        strategy.extract(everything_plan)


def test_archive_members_are_byte_identical_with_matching_checksums(tmp_path):
    """Green on arrival: a pin, not a red test."""
    members = {
        "nested/records.csv": _csv_bytes(64 * 1024),
        "nested/notes.csv": b"id,name\n1,alpha\n",
    }
    archive = _zip_archive(tmp_path / "fixtures" / "package.zip", members)
    plan = _build_plan(
        tmp_path,
        source_id="streaming_members",
        variant="archive_package",
        access_path=archive,
        access_format="binary",
        limits={"max_archive_member_bytes": 16 * MIB, "max_archive_total_bytes": 32 * MIB},
    )
    strategy, _transport = _build_strategy(tmp_path)

    result = strategy.extract(plan)

    by_name = {Path(artifact.path).name: artifact for artifact in result.artifacts}
    for name, payload in members.items():
        artifact = by_name[Path(name).name]
        assert Path(artifact.path).read_bytes() == payload
        assert artifact.checksum == sha256(payload).hexdigest()


def test_a_lying_member_header_is_still_capped_by_the_copy(tmp_path, monkeypatch):
    """A header is a claim. The writer's ``max_bytes`` is what turns it into a guarantee."""
    archive = _zip_archive(
        tmp_path / "fixtures" / "package.zip", {"honest.csv": _csv_bytes(4096)}
    )
    plan = _build_plan(
        tmp_path,
        source_id="lying_header",
        variant="archive_package",
        access_path=archive,
        access_format="binary",
        limits={"max_archive_member_bytes": 8192, "max_archive_total_bytes": 16384},
    )

    real_open = zipfile.ZipFile.open

    def _oversized_open(self, name, mode="r", pwd=None, *, force_zip64=False):
        del force_zip64
        if mode == "r" and not str(name).endswith("/"):
            return io.BytesIO(b"z" * (64 * 1024))
        return real_open(self, name, mode, pwd)

    monkeypatch.setattr(zipfile.ZipFile, "open", _oversized_open)
    strategy, _transport = _build_strategy(tmp_path)

    with pytest.raises(ArchiveExtractionError):
        strategy.extract(plan)

    assert _extracted_files(_raw_root(tmp_path, plan)) == []


# ---------------------------------------------------------------------------
# FR-4 — the download cap


def test_a_download_over_the_payload_cap_is_that_candidates_dead_letter(tmp_path):
    """AC-3's shape: a named failure the loop dead-letters, never an OOM and never a run abort."""
    body = b"y" * 4096
    plan = _build_plan(
        tmp_path,
        source_id="download_cap",
        access_url="https://example.gov.br/dados/big.csv",
        access_format="csv",
        limits={"max_payload_bytes": 1024},
        dead_letter_max_items=0,
    )
    strategy, _transport = _build_strategy(
        tmp_path, [ResponseSpec(200, body, {"Content-Type": "text/csv"})]
    )

    with pytest.raises(FileDownloadError) as excinfo:
        strategy.extract(plan)

    message = str(excinfo.value)
    assert "max_payload_bytes=1024" in message, f"the error must name the cap: {message!r}"
    assert "bytes" in message


def test_a_capped_candidate_is_dead_lettered_and_the_run_continues(tmp_path):
    """One oversized part must not cost the other 411 parts of a CNPJ run."""
    plan = _build_plan(
        tmp_path,
        source_id="download_cap_continues",
        access_url="https://example.gov.br/dados/",
        access_format="csv",
        link_resolver="html_links",
        limits={"max_payload_bytes": 1024},
        dead_letter_max_items=1,
    )
    listing = (
        b"<html><body>"
        b'<a href="https://example.gov.br/dados/big.csv">big</a>'
        b'<a href="https://example.gov.br/dados/small.csv">small</a>'
        b"</body></html>"
    )
    strategy, _transport = _build_strategy(
        tmp_path,
        [
            ResponseSpec(200, listing, {"Content-Type": "text/html"}),
            ResponseSpec(200, b"y" * 4096, {"Content-Type": "text/csv"}),
            ResponseSpec(200, b"id,name\n1,alpha\n", {"Content-Type": "text/csv"}),
        ],
    )

    result = strategy.extract(plan)

    metadata = result.metadata_as_dict()
    assert metadata["dead_letter_count"] == "1"
    dead_letters = json.loads(
        strategy.dead_letter_store.path(plan).read_text(encoding="utf-8")
    )
    assert "max_payload_bytes=1024" in dead_letters["entries"][0]["error_message"]

    raw_root = _raw_root(tmp_path, plan)
    staged = [path for path in raw_root.rglob("*") if ".staging" in path.parts]
    assert staged == [], "a capped candidate left a staged file behind"


# ---------------------------------------------------------------------------
# NFR-6 — bounded memory


def test_a_large_download_costs_one_spool_threshold_of_rss_not_twice_the_payload():
    with tempfile.TemporaryDirectory(prefix="janus-order-17-rss-") as scratch:
        baseline = max_rss_bytes()
        result = measure_download_peak_rss(Path(scratch), RSS_PROBE_PAYLOAD_BYTES)

    assert result.artifact_count == 1
    assert result.artifact_bytes == RSS_PROBE_PAYLOAD_BYTES
    assert result.delta_bytes < SPOOL_THRESHOLD_BYTES + RSS_SLACK_BYTES, (
        f"peak RSS grew by {result.delta_bytes:,} bytes for a "
        f"{RSS_PROBE_PAYLOAD_BYTES:,}-byte payload ({result.delta_ratio:.2f}x) — the download "
        "is still holding the payload in memory"
    )
    assert baseline > 0


# ---------------------------------------------------------------------------
# AC-4 — the small-file path does not change (green on arrival)


def test_a_small_remote_payload_is_persisted_exactly_as_it_is_today(tmp_path, monkeypatch):
    """Green on arrival: a pin, not a red test.

    Below the spool threshold the payload stays inline and ``write_bytes`` stays the
    persistence call, so raw path, bytes, sidecar and checksum are byte-identical. This is
    the half of AC-4 that a streaming rewrite is most likely to move without noticing.
    """
    payload = b"id,name\n1,alpha\n2,beta\n"
    plan = _build_plan(
        tmp_path,
        source_id="small_payload_unchanged",
        access_url="https://example.gov.br/dados/records.csv",
        access_format="csv",
    )
    strategy, _transport = _build_strategy(
        tmp_path, [ResponseSpec(200, payload, {"Content-Type": "text/csv"})]
    )
    write_bytes_calls: list[int] = []
    original_write_bytes = RawArtifactWriter.write_bytes

    def _record_write_bytes(self, plan, relative_path, body, **kwargs):
        write_bytes_calls.append(len(body))
        return original_write_bytes(self, plan, relative_path, body, **kwargs)

    monkeypatch.setattr(RawArtifactWriter, "write_bytes", _record_write_bytes)

    result = strategy.extract(plan)

    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    path = Path(artifact.path)
    assert path.read_bytes() == payload
    assert artifact.checksum == sha256(payload).hexdigest()
    relative_parts = path.relative_to(_raw_root(tmp_path, plan)).parts
    assert relative_parts[0] == "runs", relative_parts
    assert "downloads" in relative_parts, relative_parts
    sidecar = path.with_name(path.name + ".sha256")
    assert sidecar.read_text(encoding="utf-8") == f"{artifact.checksum}\n"
    assert write_bytes_calls == [len(payload)]


def _set_small_spool_threshold(monkeypatch, threshold: int = 1024) -> None:
    monkeypatch.setattr(download_module, "SPOOL_THRESHOLD_BYTES", threshold)
    monkeypatch.setattr(download_loop_module, "SPOOL_THRESHOLD_BYTES", threshold)


def _staging_entries(tmp_path: Path, plan: ExecutionPlan) -> list[Path]:
    raw_root = _raw_root(tmp_path, plan)
    return [path for path in raw_root.rglob("*") if ".staging" in path.parts]


def test_a_staged_payload_is_discarded_after_a_candidate_failure(tmp_path, monkeypatch):
    _set_small_spool_threshold(monkeypatch)
    payload = b"x" * 4096
    plan = _build_plan(
        tmp_path,
        source_id="staged_checksum_failure",
        access_url="https://example.gov.br/data.csv",
        access_format="csv",
    )
    strategy, _transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, payload, {"X-Checksum-Sha256": "deadbeef"})],
    )

    with pytest.raises(FileIntegrityError, match="Checksum mismatch"):
        strategy.extract(plan)

    assert _staging_entries(tmp_path, plan) == []
    assert _downloaded_files(_raw_root(tmp_path, plan)) == []


def test_a_large_local_file_persists_through_write_stream(tmp_path, monkeypatch):
    _set_small_spool_threshold(monkeypatch)
    payload = b"x" * 4096
    source = tmp_path / "fixtures" / "local.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    plan = _build_plan(
        tmp_path,
        source_id="large_local_stream",
        access_path=source,
        access_format="csv",
    )
    write_stream_calls: list[int | None] = []
    original_write_stream = RawArtifactWriter.write_stream

    def _record_write_stream(self, plan, relative_path, stream, **kwargs):
        write_stream_calls.append(kwargs.get("max_bytes"))
        return original_write_stream(self, plan, relative_path, stream, **kwargs)

    monkeypatch.setattr(RawArtifactWriter, "write_stream", _record_write_stream)
    strategy, _transport = _build_strategy(tmp_path)

    result = strategy.extract(plan)

    artifact = result.artifacts[0]
    assert Path(artifact.path).read_bytes() == payload
    assert artifact.checksum == sha256(payload).hexdigest()
    assert write_stream_calls == [len(payload)]


def test_a_committed_staged_archive_remains_openable_for_expansion(tmp_path, monkeypatch):
    _set_small_spool_threshold(monkeypatch)
    archive = io.BytesIO()
    member_payload = b"id,name\n" + b"1,alpha\n" * 512
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("nested/records.csv", member_payload)
    archive_payload = archive.getvalue()
    assert len(archive_payload) > 1024

    plan = _build_plan(
        tmp_path,
        source_id="staged_archive",
        variant="archive_package",
        access_url="https://example.gov.br/package.zip",
        access_format="binary",
    )
    strategy, _transport = _build_strategy(tmp_path, [ResponseSpec(200, archive_payload)])

    result = strategy.extract(plan)

    by_name = {Path(artifact.path).name: artifact for artifact in result.artifacts}
    assert Path(by_name["package.zip"].path).read_bytes() == archive_payload
    assert Path(by_name["records.csv"].path).read_bytes() == member_payload
    assert _staging_entries(tmp_path, plan) == []


def test_a_checkpoint_skip_discards_its_staged_payload(tmp_path, monkeypatch):
    _set_small_spool_threshold(monkeypatch)

    class _FixedVersionHook(FileHook):
        def resolve_version(self, plan, discovered_file):
            del plan
            del discovered_file
            return "2026-01"

    plan = _build_plan(
        tmp_path,
        source_id="staged_checkpoint_skip",
        variant="versioned_file",
        access_url="https://example.gov.br/data.csv",
        extraction_mode="incremental",
        checkpoint_field="publication_version",
        checkpoint_strategy="max_value",
    )
    CheckpointStore().save(plan, "2026-01")
    strategy, _transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, b"x" * 4096, {"Content-Type": "text/csv"})],
    )

    result = strategy.extract(plan, hook=_FixedVersionHook())

    assert result.artifacts == ()
    assert result.metadata_as_dict()["skipped_file_count"] == "1"
    assert _staging_entries(tmp_path, plan) == []


def test_an_overridden_hook_materializes_once_and_restages_its_result(tmp_path, monkeypatch):
    _set_small_spool_threshold(monkeypatch)
    source_payload = b"x" * 4096
    transformed_payload = b"id,name\n" + source_payload
    stream = StringIO()
    logger = build_structured_logger("janus.tests.file.materialized_hook", stream=stream)

    class _TransformingHook(FileHook):
        received: bytes | None = None

        def prepare_download(self, plan, discovered_file, payload, *, response=None):
            del plan
            del discovered_file
            del response
            self.received = payload
            return transformed_payload

    hook = _TransformingHook()
    plan = _build_plan(
        tmp_path,
        source_id="staged_hook",
        access_url="https://example.gov.br/data.csv",
        access_format="csv",
    )
    strategy, _transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, source_payload, {"Content-Type": "text/csv"})],
    )
    strategy.logger = logger

    result = strategy.extract(plan, hook=hook)

    assert hook.received == source_payload
    assert Path(result.artifacts[0].path).read_bytes() == transformed_payload
    warnings = [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if json.loads(line)["event"] == "file_payload_materialized_for_hook"
    ]
    assert len(warnings) == 1
    fields = warnings[0]["fields"]
    assert fields["hook"] == "_TransformingHook"
    assert fields["size_bytes"] == len(source_payload)
    assert fields["spool_threshold_bytes"] == 1024
    assert _staging_entries(tmp_path, plan) == []


def test_the_default_hook_is_not_called(tmp_path):
    hook = FileHook()

    def _unexpected_call(*args, **kwargs):
        del args
        del kwargs
        raise AssertionError("the default prepare_download hook was called")

    hook.prepare_download = _unexpected_call
    plan = _build_plan(
        tmp_path,
        source_id="default_hook",
        access_url="https://example.gov.br/data.csv",
    )
    strategy, _transport = _build_strategy(
        tmp_path,
        [ResponseSpec(200, b"id,name\n1,alpha\n", {"Content-Type": "text/csv"})],
    )

    result = strategy.extract(plan, hook=hook)

    assert len(result.artifacts) == 1
