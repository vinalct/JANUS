"""Fetch one file's bytes, and prove they are the bytes the source advertised.

This is the file family's equivalent of ``requests.py`` in the API and catalog families: the
family-specific *composition* of the shared HTTP layer, never a second copy of it. The
transport, the throttle and the retry loop all come from :mod:`janus.strategies.http` —
``stream_with_retries`` owns attempt counting, backoff and status classification, and
**nothing here reimplements retry or backoff**. A local file needs none of it and is read
straight off disk. Small payloads stay inline so their persistence path is unchanged; large
payloads are staged inside the raw zone and hashed while they are copied.

Integrity is the other half. :func:`_validate_remote_payload` rejects an answer that is
plainly not the file (an HTML error page served with 200, or zero bytes where discovery saw a
size), and :func:`_expected_checksum` resolves the digest to compare against — from the hook,
a ``.sha256`` sidecar or a checksum response header. The comparison itself, and the
``FileIntegrityError`` it raises, belong to the caller in ``extraction.py``: a mismatch is one
candidate's failure, and it is the dead-letter policy there — not this module — that decides
whether the run continues.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO, RawIOBase
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast

from janus.models import ExecutionPlan
from janus.strategies.common import _freeze_string_mapping
from janus.strategies.http import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiResponseTooLargeError,
    ApiStreamedResponse,
    ApiTransport,
    HttpRequestThrottle,
    RetryErrorPolicy,
    UrllibApiTransport,
    inject_auth,
    response_body_excerpt,
    stream_with_retries,
)
from janus.utils.logging import StructuredLogger, redact_url
from janus.writers import (
    SPOOL_THRESHOLD_BYTES,
    PersistedArtifact,
    RawArtifactWriter,
    RawWriteLimitError,
    StagedWrite,
)

from .artifacts import _safe_filename
from .core import FileDownloadError, FileIntegrityError
from .formats import _filename_from_content_disposition

if TYPE_CHECKING:
    from .core import DiscoveredFile, FileHook

#: Checksum headers this family trusts, in preference order.
CHECKSUM_HEADER_CANDIDATES = ("x-checksum-sha256", "x-amz-checksum-sha256")
DOWNLOAD_READ_CHUNK_BYTES = 1024 * 1024

FileResponse = ApiResponse | ApiStreamedResponse


def _file_response_error(response: ApiResponse) -> FileDownloadError:
    """Build the family's status error, carrying whatever the server said about it.

    ``FileDownloadError`` holds no response object, so the bounded body excerpt goes into
    the message — the only place downstream logging and dead-lettering will read it.
    """
    message = (
        "File request failed with status "
        f"{response.status_code} for {redact_url(response.request.full_url())}"
    )
    excerpt = response_body_excerpt(response)
    if excerpt is not None:
        message = f"{message}: {excerpt}"
    return FileDownloadError(message)


#: Configuration of the *one* call this module makes into the shared retry loop.
_RETRY_POLICY = RetryErrorPolicy(
    transport_error_factory=FileDownloadError,
    response_error_factory=_file_response_error,
    retry_log_event="file_retry_scheduled",
)


class _PrefixThenStream(RawIOBase):
    """Read a bounded in-memory prefix before continuing with the response body."""

    def __init__(self, prefix: BinaryIO, tail: BinaryIO) -> None:
        self._prefix: BinaryIO | None = prefix
        self._tail = tail

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> bytes:
        if size == 0:
            return b""
        amount = -1 if size is None else size
        if self._prefix is not None:
            chunk = self._prefix.read(amount)
            if chunk:
                return chunk
            self._prefix = None
        return self._tail.read(-1 if size is None else size)


@dataclass(slots=True)
class DownloadedPayload:
    """One candidate's bytes, inline when small and disk-backed when large."""

    size_bytes: int
    sha256_hex: str
    inline: bytes | None = None
    staged: StagedWrite | None = None
    _path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        representations = sum(
            value is not None for value in (self.inline, self.staged, self._path)
        )
        if representations != 1:
            raise ValueError("downloaded payload must have exactly one byte representation")

    def open(self) -> BinaryIO:
        """Open a seekable reader over the candidate without materializing it."""
        if self.inline is not None:
            return BytesIO(self.inline)
        if self.staged is not None:
            return self.staged.open()
        if self._path is not None:
            return self._path.open("rb")
        raise RuntimeError("downloaded payload has no readable representation")

    def materialize(self) -> bytes:
        """Return the complete payload, primarily for the legacy hook/archive contracts."""
        if self.inline is not None:
            return self.inline
        with self.open() as stream:
            payload = stream.read(self.size_bytes + 1)
        if len(payload) != self.size_bytes:
            raise OSError(
                "disk-backed payload changed after download: "
                f"expected {self.size_bytes} bytes, read {len(payload)}"
            )
        return payload

    def persist(
        self,
        raw_writer: RawArtifactWriter,
        plan: ExecutionPlan,
        relative_path: str | Path,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> PersistedArtifact:
        """Persist at the resolved path, preserving the existing inline write path."""
        if self.inline is not None:
            return raw_writer.write_bytes(plan, relative_path, self.inline, metadata=metadata)

        if self.staged is not None:
            persisted = self.staged.commit(relative_path, metadata=metadata)
            self.staged = None
            self._path = Path(persisted.artifact.path)
            return persisted

        if self._path is None:
            raise RuntimeError("downloaded payload has no persistable representation")
        with self._path.open("rb") as stream:
            persisted = raw_writer.write_stream(
                plan,
                relative_path,
                stream,
                metadata=metadata,
                max_bytes=self.size_bytes,
            )
        self._path = Path(persisted.artifact.path)
        return persisted

    def discard(self) -> None:
        """Discard an uncommitted remote staging file; safe to call repeatedly."""
        if self.staged is not None:
            self.staged.discard()

    @classmethod
    def from_local_path(cls, path: Path) -> DownloadedPayload:
        """Load a small local file inline, or hash a large one without materializing it."""
        if path.stat().st_size <= SPOOL_THRESHOLD_BYTES:
            payload = path.read_bytes()
            return cls(
                size_bytes=len(payload),
                sha256_hex=sha256(payload).hexdigest(),
                inline=payload,
            )

        size_bytes, sha256_hex = _hash_local_path(path)
        return cls(size_bytes=size_bytes, sha256_hex=sha256_hex, _path=path)


@dataclass(frozen=True, slots=True)
class FileDownloader:
    """Fetches one discovered file end to end: build → send → hand back the bytes.

    Depends on exactly three of the strategy's collaborators, which is what makes it a small
    object rather than a bag of functions taking ``self``.
    """

    transport_factory: Callable[[], ApiTransport] = UrllibApiTransport
    sleeper: Callable[[float], None] = time.sleep
    env_reader: Callable[[str], str | None] = os.getenv

    @contextmanager
    def open_client(self) -> Iterator[ApiClient]:
        """Open the one client a file run uses for link resolution and every download.

        Yielding the client rather than holding it in the strategy is what lets the
        orchestration layer own that lifetime without owning the transport.
        """
        with ApiClient(self.transport_factory()) as client:
            yield client

    def load_payload(
        self,
        plan: ExecutionPlan,
        discovered_file: DiscoveredFile,
        *,
        client: ApiClient,
        throttle: HttpRequestThrottle,
        logger: StructuredLogger | None,
        raw_writer: RawArtifactWriter,
    ) -> tuple[DownloadedPayload, ApiStreamedResponse | None, int]:
        """Return one candidate's bounded payload, response metadata and attempts used."""
        if discovered_file.source_kind == "local":
            return DownloadedPayload.from_local_path(Path(discovered_file.location)), None, 1

        request = ApiRequest(
            method=plan.source_config.access.method,
            url=discovered_file.location,
            timeout_seconds=plan.source_config.access.timeout_seconds,
            headers=_freeze_string_mapping(plan.source_config.access.headers or {}),
            params=_freeze_string_mapping(plan.source_config.access.params or {}),
            max_payload_bytes=plan.source_config.access.limits.max_payload_bytes,
            max_redirects=plan.source_config.access.limits.max_redirects,
        )
        request = inject_auth(
            request,
            plan.source_config.access.auth,
            env_reader=self.resolve_env_var,
        )
        streamed, attempts_used = stream_with_retries(
            plan,
            client,
            request,
            throttle,
            logger,
            policy=_RETRY_POLICY,
            sleeper=self.sleeper,
        )
        try:
            payload = _load_streamed_payload(
                plan,
                request,
                streamed,
                raw_writer=raw_writer,
            )
        except ApiResponseTooLargeError as exc:
            raise FileDownloadError(str(exc)) from exc
        except RawWriteLimitError as exc:
            error = ApiResponseTooLargeError(
                request.full_url(),
                limit_bytes=exc.max_bytes,
                bytes_seen=exc.bytes_seen,
            )
            raise FileDownloadError(str(error)) from exc
        finally:
            streamed.close()
        return payload, streamed, attempts_used

    def resolve_env_var(self, name: str) -> str | None:
        value = self.env_reader(name)
        if value is not None:
            return value
        return os.getenv(name)


def _load_streamed_payload(
    plan: ExecutionPlan,
    request: ApiRequest,
    streamed: ApiStreamedResponse,
    *,
    raw_writer: RawArtifactWriter,
) -> DownloadedPayload:
    with BytesIO() as buffered:
        buffered_size = 0
        while buffered_size <= SPOOL_THRESHOLD_BYTES:
            remaining = SPOOL_THRESHOLD_BYTES + 1 - buffered_size
            chunk = streamed.body.read(min(DOWNLOAD_READ_CHUNK_BYTES, remaining))
            if not chunk:
                inline = buffered.getvalue()
                return DownloadedPayload(
                    size_bytes=len(inline),
                    sha256_hex=sha256(inline).hexdigest(),
                    inline=inline,
                )
            buffered.write(chunk)
            buffered_size += len(chunk)

        buffered.seek(0)
        staged = raw_writer.begin_staged_write(
            plan,
            max_bytes=request.max_payload_bytes,
        )
        staged.write_from(
            cast(
                BinaryIO,
                _PrefixThenStream(buffered, cast(BinaryIO, streamed.body)),
            )
        )
    return DownloadedPayload(
        size_bytes=staged.bytes_written,
        sha256_hex=staged.sha256_hex,
        staged=staged,
    )


def _payload_from_bytes(
    payload: bytes,
    plan: ExecutionPlan,
    raw_writer: RawArtifactWriter,
    *,
    max_bytes: int,
) -> DownloadedPayload:
    """Wrap hook output, staging it when it no longer fits the inline path."""
    size_bytes = len(payload)
    if size_bytes > max_bytes:
        raise FileDownloadError(
            "Prepared file payload exceeds "
            f"access.limits.max_payload_bytes={max_bytes}: {size_bytes} bytes seen"
        )
    if size_bytes <= SPOOL_THRESHOLD_BYTES:
        return DownloadedPayload(
            size_bytes=size_bytes,
            sha256_hex=sha256(payload).hexdigest(),
            inline=payload,
        )

    staged = raw_writer.begin_staged_write(plan, max_bytes=max_bytes)
    try:
        staged.write_from(BytesIO(payload))
    except RawWriteLimitError as exc:
        raise FileDownloadError(
            "Prepared file payload exceeds "
            f"access.limits.max_payload_bytes={exc.max_bytes}: "
            f"{exc.bytes_seen} bytes seen"
        ) from exc
    return DownloadedPayload(
        size_bytes=staged.bytes_written,
        sha256_hex=staged.sha256_hex,
        staged=staged,
    )


def _hash_local_path(path: Path) -> tuple[int, str]:
    digest = sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(DOWNLOAD_READ_CHUNK_BYTES):
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()


def _validate_remote_payload(
    discovered_file: DiscoveredFile,
    payload: DownloadedPayload,
    response: FileResponse | None,
) -> None:
    if discovered_file.source_kind != "remote" or response is None:
        return

    content_type = response.headers_as_dict().get("Content-Type", "")
    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type in {"text/html", "application/xhtml+xml"}:
        raise FileDownloadError(
            f"Expected a file payload for {discovered_file.filename!r} but received "
            f"{normalized_content_type or 'HTML'} from "
            f"{redact_url(response.request.full_url())}"
        )

    if (
        payload.size_bytes == 0
        and discovered_file.size_bytes is not None
        and discovered_file.size_bytes > 0
    ):
        raise FileDownloadError(
            f"Expected a non-empty payload for {discovered_file.filename!r} "
            f"({discovered_file.size_bytes} bytes discovered) but received 0 bytes"
        )


def _resolved_filename(filename: str, response: FileResponse | None) -> str:
    if response is None:
        return _safe_filename(filename)

    content_disposition = response.headers_as_dict().get("Content-Disposition")
    if isinstance(content_disposition, str):
        cd_name = _filename_from_content_disposition(content_disposition)
        if cd_name is not None:
            return _safe_filename(cd_name)
    return _safe_filename(filename)


def _resolve_version(
    plan: ExecutionPlan,
    discovered_file: DiscoveredFile,
    payload: DownloadedPayload,
    response: FileResponse | None,
    file_hook: FileHook | None,
) -> str:
    if file_hook is not None:
        hook_version = file_hook.resolve_version(plan, discovered_file)
        if hook_version is not None and hook_version.strip():
            return hook_version.strip()

    if discovered_file.version is not None:
        return discovered_file.version

    if response is not None:
        headers = {
            key.lower(): value.strip()
            for key, value in response.headers_as_dict().items()
            if isinstance(value, str) and value.strip()
        }
        etag = headers.get("etag")
        if etag:
            return etag.strip('"')
        last_modified = headers.get("last-modified")
        if last_modified:
            return last_modified

    if plan.source.strategy_variant == "static_file":
        return "current"
    return payload.sha256_hex


def _expected_checksum(
    plan: ExecutionPlan,
    discovered_file: DiscoveredFile,
    *,
    response: FileResponse | None,
    file_hook: FileHook | None,
) -> str | None:
    if file_hook is not None:
        hook_checksum = file_hook.expected_checksum(plan, discovered_file)
        if hook_checksum:
            return hook_checksum.strip()

    if discovered_file.source_kind == "local":
        sidecar = Path(discovered_file.location).with_name(f"{discovered_file.filename}.sha256")
        if sidecar.exists():
            return _read_checksum_sidecar(sidecar)

    if response is None:
        return None

    headers = {key.lower(): value for key, value in response.headers_as_dict().items()}
    for header_name in CHECKSUM_HEADER_CANDIDATES:
        value = headers.get(header_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_checksum_sidecar(path: Path) -> str:
    """Read the bare-hex digest ``RawArtifactWriter`` writes beside every raw artifact."""
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise FileIntegrityError(f"Checksum sidecar is empty: {path}")
    return content.split()[0]
