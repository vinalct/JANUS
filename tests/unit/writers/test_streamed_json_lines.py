"""Encoded JSONL chunks retain the raw writer's byte and atomicity contracts."""

from pathlib import Path

import pytest

from janus.writers import PARTIAL_SUFFIX_MARKER, RawArtifactWriter, RawWriteLimitError
from janus.writers.raw import iter_json_lines
from tests.fixtures.handoff.cases import build_raw_zone


@pytest.mark.parametrize("records", [[], [{"z": None, "a": "ação\nnext"}, [1, True], 7]])
def test_streamed_json_lines_match_existing_jsonl_bytes_and_sidecars(tmp_path, records):
    zone = build_raw_zone(tmp_path)
    writer = RawArtifactWriter(zone.layout)
    reference = writer.write_json_lines(zone.plan, "reference.jsonl", records)
    streamed = writer.write_stream(
        zone.plan,
        "streamed.jsonl",
        iter_json_lines(iter(records)),
        format_name="jsonl",
    )
    assert streamed.artifact.format == "jsonl"
    assert streamed.write_result.format == "jsonl"
    assert streamed.artifact.checksum == reference.artifact.checksum
    assert Path(streamed.artifact.path).read_bytes() == Path(reference.artifact.path).read_bytes()
    assert (
        Path(streamed.artifact.path + ".sha256").read_bytes()
        == Path(reference.artifact.path + ".sha256").read_bytes()
    )


@pytest.mark.parametrize("failure", ["iterator", "limit"])
def test_failed_chunk_stream_preserves_existing_artifact_and_removes_partial(tmp_path, failure):
    zone = build_raw_zone(tmp_path)
    writer = RawArtifactWriter(zone.layout)
    original = writer.write_json_lines(zone.plan, "current.jsonl", [{"id": "original"}])
    path = Path(original.artifact.path)
    before = (path.read_bytes(), Path(str(path) + ".sha256").read_bytes())

    def chunks():
        yield b'{"id": "replacement"}\n'
        if failure == "iterator":
            raise OSError("line generation failed")
        yield b"x" * 100

    expected = OSError if failure == "iterator" else RawWriteLimitError
    with pytest.raises(expected):
        writer.write_stream(zone.plan, "current.jsonl", chunks(), max_bytes=50, format_name="jsonl")
    assert (path.read_bytes(), Path(str(path) + ".sha256").read_bytes()) == before
    assert not list(path.parent.glob(f"*{PARTIAL_SUFFIX_MARKER}*"))


def test_empty_chunks_are_not_mistaken_for_end_of_stream(tmp_path):
    zone = build_raw_zone(tmp_path)
    writer = RawArtifactWriter(zone.layout)
    persisted = writer.write_stream(zone.plan, "chunks.bin", iter((b"first", b"", b"last")))
    assert Path(persisted.artifact.path).read_bytes() == b"firstlast"
