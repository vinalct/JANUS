"""Hash on write, stage inside the zone, leave nothing behind."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

from janus.models import ExecutionPlan, RunContext
from janus.registry import load_registry
from janus.scripts.raw_to_bronze import _rediscover_raw_artifacts
from janus.utils.storage import StorageLayout
from janus.writers import (
    PARTIAL_SUFFIX_MARKER,
    SIDECAR_SUFFIX,
    STAGING_DIRNAME,
    RawArtifactWriter,
    RawWriteLimitError,
)
from janus.writers.raw import _write_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHUNK = 1024 * 1024
PAYLOAD_SIZES = (0, 1024, 3 * CHUNK + 1)

RELATIVE_PATH = "downloads/current/payload.bin"


def _deterministic_payload(size: int) -> bytes:
    """A reproducible, non-compressible-ish payload of exactly ``size`` bytes."""
    if size == 0:
        return b""
    block = bytes((index * 31 + 7) % 251 for index in range(4096))
    return (block * (size // len(block) + 1))[:size]


def _plan(tmp_path: Path, *, run_id: str = "run-write-stream-001") -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id=run_id,
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        ),
    )


def _layout(tmp_path: Path) -> StorageLayout:
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


def _writer(tmp_path: Path, **kwargs) -> RawArtifactWriter:
    return RawArtifactWriter(_layout(tmp_path), **kwargs)


def _sidecar_of(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _partials_under(directory: Path) -> list[Path]:
    return [path for path in directory.rglob("*") if PARTIAL_SUFFIX_MARKER in path.name]


# ---------------------------------------------------------------------------
# AC-4 — identical path, bytes, sidecar and checksum


@pytest.mark.parametrize("size", PAYLOAD_SIZES, ids=[f"{size}B" for size in PAYLOAD_SIZES])
def test_write_stream_matches_write_bytes_for_the_same_payload(tmp_path, size):
    """The differential AC-4 actually asks for: same path, same bytes, same digest."""
    payload = _deterministic_payload(size)

    reference = _writer(tmp_path / "reference").write_bytes(
        _plan(tmp_path / "reference"), RELATIVE_PATH, payload
    )
    streamed = _writer(tmp_path / "streamed").write_stream(
        _plan(tmp_path / "streamed"), RELATIVE_PATH, BytesIO(payload)
    )

    reference_path = Path(reference.artifact.path)
    streamed_path = Path(streamed.artifact.path)

    assert streamed_path.relative_to(tmp_path / "streamed") == reference_path.relative_to(
        tmp_path / "reference"
    )
    assert streamed_path.read_bytes() == payload
    assert streamed.artifact.checksum == reference.artifact.checksum == sha256(payload).hexdigest()
    assert _sidecar_of(streamed_path).read_text(encoding="utf-8") == (
        f"{reference.artifact.checksum}\n"
    )
    assert streamed.artifact.format == reference.artifact.format
    assert streamed.write_result.records_written == reference.write_result.records_written
    assert streamed.write_result.mode == reference.write_result.mode


# ---------------------------------------------------------------------------
# FR-4 — the temp file lives in the target directory, and never survives a failure


def test_the_staging_file_sits_in_the_target_directory(tmp_path):
    """Same directory means same filesystem, so ``os.replace`` is atomic - and never ``/tmp``.

    Spooling a partially downloaded 6 GiB CNPJ part onto the root filesystem is the other
    half of the reason.
    """
    plan = _plan(tmp_path)
    writer = _writer(tmp_path)
    target_dir = Path(writer.storage_layout.resolve_output(plan, "raw").resolved_path)
    observed: list[list[str]] = []

    class _InspectingStream(BytesIO):
        def read(self, amount: int | None = -1) -> bytes:
            observed.append(sorted(path.name for path in _partials_under(target_dir)))
            return super().read(amount)

    persisted = writer.write_stream(
        plan, RELATIVE_PATH, _InspectingStream(_deterministic_payload(3 * CHUNK + 1))
    )

    mid_copy = [entry for entry in observed if entry]
    assert mid_copy, "no partial file was visible at any point during the copy"
    assert all(len(entry) == 1 for entry in mid_copy), (
        f"more than one staging file existed at once: {mid_copy}"
    )
    assert _partials_under(target_dir) == [], "the staging file outlived the write"
    assert Path(persisted.artifact.path).is_file()


def test_an_interrupted_copy_leaves_no_artifact_no_sidecar_and_no_partial(tmp_path):
    """A half-written artifact that replay can find is worse than no artifact at all."""
    plan = _plan(tmp_path)
    writer = _writer(tmp_path)
    target_dir = Path(writer.storage_layout.resolve_output(plan, "raw").resolved_path)

    class _FailingStream:
        def __init__(self) -> None:
            self.reads = 0

        def read(self, amount: int | None = -1) -> bytes:
            self.reads += 1
            if self.reads > 2:
                raise OSError("connection reset mid-copy")
            return b"x" * CHUNK

    with pytest.raises(OSError, match="connection reset mid-copy"):
        writer.write_stream(plan, RELATIVE_PATH, _FailingStream())

    final = target_dir / RELATIVE_PATH
    assert not final.exists()
    assert not _sidecar_of(final).exists()
    assert _partials_under(target_dir) == []


def test_max_bytes_is_enforced_while_copying_and_leaves_nothing(tmp_path):
    """The writer's cap is what a lying archive header cannot talk its way past (FR-5)."""
    plan = _plan(tmp_path)
    writer = _writer(tmp_path)
    target_dir = Path(writer.storage_layout.resolve_output(plan, "raw").resolved_path)

    with pytest.raises(RawWriteLimitError) as excinfo:
        writer.write_stream(
            plan, RELATIVE_PATH, BytesIO(b"y" * 4096), max_bytes=1024
        )

    message = str(excinfo.value)
    assert "1024" in message, f"the message must name the cap; got {message!r}"
    assert "4096" in message or "bytes" in message
    assert not (target_dir / RELATIVE_PATH).exists()
    assert _partials_under(target_dir) == []


# ---------------------------------------------------------------------------
# FR-4 — modes


def test_ignore_mode_returns_the_existing_digest_without_consuming_the_stream(tmp_path):
    """``ignore`` means "already here": re-downloading it would defeat the mode."""
    plan = _plan(tmp_path)
    writer = _writer(tmp_path)
    first = writer.write_bytes(plan, RELATIVE_PATH, b"original bytes")

    class _ExplodingStream:
        def read(self, amount: int | None = -1) -> bytes:
            raise AssertionError("ignore mode consumed the stream")

    second = writer.write_stream(plan, RELATIVE_PATH, _ExplodingStream(), mode="ignore")

    assert second.artifact.checksum == first.artifact.checksum
    assert Path(second.artifact.path).read_bytes() == b"original bytes"


def test_append_mode_is_refused_for_a_streamed_artifact(tmp_path):
    """Appending a stream has no digest contract worth defining; ``write_json`` says so too."""
    plan = _plan(tmp_path)

    with pytest.raises(ValueError, match="append"):
        _writer(tmp_path).write_stream(plan, RELATIVE_PATH, BytesIO(b"x"), mode="append")


# ---------------------------------------------------------------------------
# FR-4 — the two-phase staged write 


def test_a_staged_write_commits_to_the_same_artifact_write_bytes_would_have_produced(tmp_path):
    """The file loop resolves ``downloads/<version>/`` only *after* the body is in hand."""
    payload = _deterministic_payload(3 * CHUNK + 1)

    reference = _writer(tmp_path / "reference").write_bytes(
        _plan(tmp_path / "reference"), RELATIVE_PATH, payload, metadata={"resolved_version": "v1"}
    )

    plan = _plan(tmp_path / "staged")
    writer = _writer(tmp_path / "staged")
    with writer.begin_staged_write(plan) as staged:
        staged.write_from(BytesIO(payload))
        assert staged.bytes_written == len(payload)
        assert staged.sha256_hex == sha256(payload).hexdigest()
        committed = staged.commit(RELATIVE_PATH, metadata={"resolved_version": "v1"})

    committed_path = Path(committed.artifact.path)
    assert committed_path.read_bytes() == payload
    assert committed.artifact.checksum == reference.artifact.checksum
    assert _sidecar_of(committed_path).read_text(encoding="utf-8") == (
        f"{reference.artifact.checksum}\n"
    )
    assert committed.write_result.metadata_as_dict() == reference.write_result.metadata_as_dict()


def test_a_staged_write_that_is_never_committed_leaves_nothing_behind(tmp_path):
    """A checkpoint skip and a dead letter both exit this way; the raw zone must not notice."""
    plan = _plan(tmp_path)
    writer = _writer(tmp_path)
    raw_root = Path(writer.storage_layout.resolve_output(plan, "raw").resolved_path)

    with writer.begin_staged_write(plan) as staged:
        staged.write_from(BytesIO(b"abandoned"))
        staged_path = staged.path

    assert not staged_path.exists()
    assert [path for path in raw_root.rglob("*") if path.is_file()] == []


def test_a_staged_write_commits_into_a_directory_that_does_not_exist_yet(tmp_path):
    """``downloads/<version>/`` is created by the commit, as ``write_bytes`` creates it today."""
    plan = _plan(tmp_path)
    writer = _writer(tmp_path)

    with writer.begin_staged_write(plan) as staged:
        staged.write_from(BytesIO(b"payload"))
        committed = staged.commit("downloads/2026-09/new.bin")

    assert Path(committed.artifact.path).read_bytes() == b"payload"


# ---------------------------------------------------------------------------
# FR-4 — replay never picks up staged or partial files


def test_replay_discovery_ignores_staged_and_partial_files(tmp_path):
    """A crashed run's leftovers are attributable, and never mistaken for another run's data."""
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        source_config,
        outputs=replace(
            source_config.outputs,
            raw=replace(source_config.outputs.raw, path=str(tmp_path / "raw")),
        ),
    )
    plan = ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id="run-replay-staging-001",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        ),
    )

    raw_root = Path(plan.raw_output.path)
    (raw_root / "pages").mkdir(parents=True, exist_ok=True)
    (raw_root / "pages" / "page-0001.json").write_text('{"page": 1}\n', encoding="utf-8")
    before = _rediscover_raw_artifacts(plan)

    (raw_root / STAGING_DIRNAME).mkdir(parents=True, exist_ok=True)
    (raw_root / STAGING_DIRNAME / "x.partial").write_bytes(b"half a download")
    (raw_root / "pages" / f".page-0002.json{PARTIAL_SUFFIX_MARKER}abc").write_bytes(b"half")

    assert _rediscover_raw_artifacts(plan) == before


# ---------------------------------------------------------------------------
# AC-4 — the digest equivalence hash-on-write must preserve (green on arrival)


@pytest.mark.parametrize("mode", ("overwrite", "ignore", "append"))
@pytest.mark.parametrize("size", PAYLOAD_SIZES, ids=[f"{size}B" for size in PAYLOAD_SIZES])
def test_write_bytes_digest_equals_sha256_of_the_file_on_disk(tmp_path, mode, size):
    """Green on arrival: a pin, not a red test."""

    payload = _deterministic_payload(size)
    path = tmp_path / "artifact.bin"
    if mode in {"ignore", "append"}:
        path.write_bytes(b"pre-existing ")

    digest, persisted_path = _write_bytes(path, payload, mode)

    assert persisted_path == path
    assert digest == sha256(path.read_bytes()).hexdigest()
