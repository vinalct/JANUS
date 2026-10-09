"""Handoff sizing uses filesystem metadata once and never guesses from a partial batch."""

from pathlib import Path
from unittest.mock import patch

import pytest

from janus.models import ExtractedArtifact
from janus.runtime.materialize import _estimated_batch_bytes


def artifact(path):
    return ExtractedArtifact(str(path), "jsonl")


def test_each_artifact_is_statted_once_without_reading_payloads(tmp_path):
    paths = [tmp_path / "one.jsonl", tmp_path / "two.jsonl"]
    for path, payload in zip(paths, (b"123", b"45678"), strict=True):
        path.write_bytes(payload)
    real_stat = Path.stat
    calls = []

    def stat(path, **kwargs):
        calls.append(path)
        return real_stat(path, **kwargs)

    with (
        patch.object(Path, "stat", stat),
        patch.object(Path, "open", side_effect=AssertionError("payload read")),
    ):
        assert _estimated_batch_bytes(tuple(map(artifact, paths))) == 8
    assert calls == paths


@pytest.mark.parametrize("error", [FileNotFoundError(), PermissionError(), OSError("stat failed")])
def test_any_stat_failure_discards_the_partial_sum(tmp_path, error):
    paths = [tmp_path / "one.jsonl", tmp_path / "two.jsonl"]
    paths[0].write_bytes(b"known size")
    real_stat = Path.stat

    def stat(path, **kwargs):
        if path == paths[1]:
            raise error
        return real_stat(path, **kwargs)

    with patch.object(Path, "stat", stat):
        assert _estimated_batch_bytes(tuple(map(artifact, paths))) is None


def test_deleted_artifact_makes_the_whole_batch_unknown(tmp_path):
    known = tmp_path / "known.jsonl"
    known.write_bytes(b"known size")
    assert _estimated_batch_bytes((artifact(known), artifact(tmp_path / "deleted"))) is None


@pytest.mark.parametrize("uri", ["s3://bucket/handoff.jsonl", "https://example.org/data.csv"])
def test_location_uri_is_not_statted(uri):
    with patch.object(Path, "stat", side_effect=AssertionError("URI stat")):
        assert _estimated_batch_bytes((artifact(uri),)) is None


def test_empty_batch_and_empty_file_have_known_zero_size(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()
    assert _estimated_batch_bytes(()) == 0
    assert _estimated_batch_bytes((artifact(path),)) == 0


def test_location_uri_discards_known_local_sizes(tmp_path):
    path = tmp_path / "known.jsonl"
    path.write_bytes(b"known size")
    assert _estimated_batch_bytes((artifact(path), artifact("s3://bucket/data"))) is None
