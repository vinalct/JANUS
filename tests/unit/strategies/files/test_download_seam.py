"""The network phase owns client lifetime and cannot reach completion hooks."""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path

import pytest

from janus.strategies.files.core import DiscoveredFile, FileHook
from janus.strategies.files.download import FileDownloader
from janus.strategies.files.download_loop import _download_candidate
from janus.strategies.files.extraction import FileExtractionContext
from janus.strategies.http import HttpRequestThrottle
from janus.writers import RawArtifactWriter
from tests.support.file_downloads import build_flat_files_fixture


class RaisingHook(FileHook):
    def __getattribute__(self, name):
        raise AssertionError(f"The download phase accessed FileHook.{name}")


@pytest.fixture
def candidate(tmp_path):
    fixture = build_flat_files_fixture(tmp_path, file_count=1)
    plan = fixture.build_plan("download-seam")
    context = FileExtractionContext(
        plan=plan,
        file_hook=RaisingHook(),
        storage_layout=fixture.storage_layout,
        checkpoint_state=None,
        downloader=FileDownloader(transport_factory=fixture.server.transport_factory()),
        throttle=HttpRequestThrottle(None, clock=lambda: 0.0, sleeper=lambda seconds: None),
        logger=None,
        raw_writer_factory=RawArtifactWriter,
    )
    discovered_file = DiscoveredFile(
        source_kind="remote", location=fixture.urls[0],
        filename=Path(fixture.urls[0]).name, format="csv",
    )
    return fixture, context, discovered_file, RawArtifactWriter(fixture.storage_layout)


def test_download_candidate_opens_and_closes_one_dedicated_client_per_call(candidate):
    fixture, context, discovered_file, raw_writer = candidate
    expected = fixture.server.routes[("GET", discovered_file.location)][0].body
    for calls in (1, 2):
        downloaded = _download_candidate(
            context.downloader, context.plan, discovered_file,
            client=None, throttle=context.throttle, logger=context.logger,
            raw_writer=raw_writer,
        )
        assert fixture.server.opened == fixture.server.closed == calls
        assert downloaded.payload.materialize() == expected
        assert downloaded.payload.sha256_hex == sha256(expected).hexdigest()
        assert downloaded.attempts_used == 1


@pytest.mark.parametrize("reuse_client", [False, True])
def test_download_candidate_takes_only_collaborators_and_never_calls_hooks(candidate, reuse_client):
    fixture, context, discovered_file, raw_writer = candidate
    client_scope = context.downloader.open_client() if reuse_client else nullcontext(None)
    with client_scope as client:
        downloaded = _download_candidate(
            context.downloader, context.plan, discovered_file,
            client=client, throttle=context.throttle, logger=context.logger,
            raw_writer=raw_writer,
        )
        assert downloaded.payload.size_bytes > 0
        assert downloaded.response is not None
        assert downloaded.response.status_code == 200
        assert fixture.server.opened == 1
        assert fixture.server.closed == (0 if reuse_client else 1)
    assert fixture.server.closed == 1
    raw_path = Path(context.plan.source_config.outputs.raw.path)
    assert not raw_path.exists()
