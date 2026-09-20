"""Nothing remote gets to decide how much memory JANUS spends."""

from __future__ import annotations

import io
import tarfile
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api import ApiResponse
from janus.strategies.files import ArchiveExtractionError, FileDownloadError, FileStrategy
from janus.utils.storage import StorageLayout
from tests.support.memory_probe import max_rss_bytes, measure_download_peak_rss

RED_UNTIL_11 = pytest.mark.xfail(
    strict=True, reason="red until: the file family's streaming download path"
)
RED_UNTIL_12 = pytest.mark.xfail(
    strict=True, reason="red until: archive caps before decompression"
)

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
    limits: dict[str, int] | None = None,
    dead_letter_max_items: int = 0,
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
                "mode": "full_refresh",
                "checkpoint_strategy": "none",
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
            "quality": {"allow_schema_evolution": True},
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
    extracted = raw_root / "extracted"
    if not extracted.exists():
        return []
    return [path for path in extracted.rglob("*") if path.is_file()]


def _downloaded_files(raw_root: Path) -> list[Path]:
    downloads = raw_root / "downloads"
    if not downloads.exists():
        return []
    return [path for path in downloads.rglob("*") if path.is_file()]


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


@RED_UNTIL_12
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


@RED_UNTIL_12
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


@RED_UNTIL_12
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


@RED_UNTIL_12
def test_the_caps_are_checked_over_the_selection_not_the_whole_archive(tmp_path):
    """An unselected member is never opened, so it cannot cost anything — but a selected one does."""
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


@RED_UNTIL_12
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


@RED_UNTIL_11
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


@RED_UNTIL_11
def test_a_capped_candidate_is_dead_lettered_and_the_run_continues(tmp_path):
    """One oversized part must not cost the other 411 parts of a CNPJ run."""
    import json

    plan = _build_plan(
        tmp_path,
        source_id="download_cap_continues",
        access_url="https://example.gov.br/dados/",
        access_format="csv",
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


@RED_UNTIL_11
def test_a_large_download_costs_one_spool_threshold_of_rss_not_twice_the_payload():

    import tempfile

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


def test_a_small_remote_payload_is_persisted_exactly_as_it_is_today(tmp_path):
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
