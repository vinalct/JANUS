"""Fetch one file's bytes, and prove they are the bytes the source advertised.

This is the file family's equivalent of ``requests.py`` in the API and catalog families: the
family-specific *composition* of the shared HTTP layer, never a second copy of it. The
transport, the throttle and the retry loop all come from :mod:`janus.strategies.http` —
``send_with_retries`` owns attempt counting, backoff and status classification, and
**nothing here reimplements retry or backoff**. A local file needs none of it and is read
straight off disk.

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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from janus.models import ExecutionPlan
from janus.strategies.common import _freeze_string_mapping
from janus.strategies.http import (
    ApiClient,
    ApiRequest,
    ApiResponse,
    ApiTransport,
    HttpRequestThrottle,
    RetryErrorPolicy,
    UrllibApiTransport,
    inject_auth,
    send_with_retries,
)
from janus.utils.logging import StructuredLogger, redact_url

from .artifacts import _safe_filename
from .core import FileDownloadError, FileIntegrityError
from .formats import _filename_from_content_disposition

if TYPE_CHECKING:
    from .core import DiscoveredFile, FileHook

#: Checksum headers this family trusts, in preference order.
CHECKSUM_HEADER_CANDIDATES = ("x-checksum-sha256", "x-amz-checksum-sha256")


def _file_response_error(response: ApiResponse) -> FileDownloadError:
    return FileDownloadError(
        "File request failed with status "
        f"{response.status_code} for {redact_url(response.request.full_url())}"
    )


#: Configuration of the *one* call this module makes into the shared retry loop.
_RETRY_POLICY = RetryErrorPolicy(
    transport_error_factory=FileDownloadError,
    response_error_factory=_file_response_error,
    retry_log_event="file_retry_scheduled",
)


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
    ) -> tuple[bytes, ApiResponse | None, int]:
        """Return one candidate's bytes, its response (``None`` locally) and attempts used."""
        if discovered_file.source_kind == "local":
            return Path(discovered_file.location).read_bytes(), None, 1

        request = ApiRequest(
            method=plan.source_config.access.method,
            url=discovered_file.location,
            timeout_seconds=plan.source_config.access.timeout_seconds,
            headers=_freeze_string_mapping(plan.source_config.access.headers or {}),
            params=_freeze_string_mapping(plan.source_config.access.params or {}),
        )
        request = inject_auth(
            request,
            plan.source_config.access.auth,
            env_reader=self.resolve_env_var,
        )
        response, _payload, attempts_used = send_with_retries(
            plan,
            client,
            request,
            throttle,
            logger,
            policy=_RETRY_POLICY,
            sleeper=self.sleeper,
        )
        return response.body, response, attempts_used

    def resolve_env_var(self, name: str) -> str | None:
        value = self.env_reader(name)
        if value is not None:
            return value
        return os.getenv(name)


def _validate_remote_payload(
    discovered_file: DiscoveredFile,
    payload: bytes,
    response: ApiResponse | None,
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
        not payload
        and discovered_file.size_bytes is not None
        and discovered_file.size_bytes > 0
    ):
        raise FileDownloadError(
            f"Expected a non-empty payload for {discovered_file.filename!r} "
            f"({discovered_file.size_bytes} bytes discovered) but received 0 bytes"
        )


def _resolved_filename(filename: str, response: ApiResponse | None) -> str:
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
    payload: bytes,
    response: ApiResponse | None,
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
    return sha256(payload).hexdigest()


def _expected_checksum(
    plan: ExecutionPlan,
    discovered_file: DiscoveredFile,
    *,
    response: ApiResponse | None,
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
