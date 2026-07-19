from __future__ import annotations

import base64
import contextlib
import logging
import re
from collections.abc import Sequence
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.error import URLError
from urllib.parse import quote, unquote, urljoin, urlsplit
from xml.etree import ElementTree

from janus.strategies.api import (
    ApiRequest,
    ApiTransport,
    ApiTransportError,
    AuthResolutionError,
)
from janus.strategies.files.core import DiscoveredFile
from janus.strategies.files.formats import (
    DIRECT_FILE_EXTENSIONS,
    _filename_from_content_disposition,
    _filename_from_url,
    _infer_format_name,
)
from janus.utils.logging import redact_url

_NEXTCLOUD_SHARE_PATTERN = re.compile(r"/index\.php/s/([^/?#]+)")
_CONTENT_TYPE_FORMAT_MAP: dict[str, str] = {
    "text/csv": "csv",
    "application/json": "json",
    "application/x-ndjson": "jsonl",
    "application/vnd.apache.parquet": "parquet",
    "application/octet-stream": "binary",
    "application/zip": "binary",
    "application/x-zip-compressed": "binary",
}

_DAV_NS = "DAV:"
_NEXTCLOUD_WEBDAV_PREFIX = "/public.php/webdav/"
_NEXTCLOUD_RECURSIVE_DEPTH = "infinity"
_LOGGER = logging.getLogger(__name__)
_TRANSPORT_EXCEPTIONS = (ApiTransportError, AuthResolutionError, URLError, OSError)


class LinkResolver(Protocol):
    """Protocol for URL-to-file resolvers used in the resolver chain."""

    def can_handle(self, url: str, formato: str | None) -> bool: ...

    def resolve(
        self, url: str, formato: str | None, transport: ApiTransport
    ) -> Sequence[DiscoveredFile]: ...


class DirectResolver:
    """Returns one DiscoveredFile for URLs with a known file extension — no HTTP call."""

    def can_handle(self, url: str, formato: str | None) -> bool:
        path = urlsplit(url).path.lower()
        return any(path.endswith(ext) for ext in DIRECT_FILE_EXTENSIONS)

    def resolve(
        self, url: str, formato: str | None, transport: ApiTransport
    ) -> Sequence[DiscoveredFile]:
        filename = _filename_from_url(url)
        fmt = _infer_format_name(filename, fallback=formato or "binary")
        return (DiscoveredFile(source_kind="remote", location=url, filename=filename, format=fmt),)


class RedirectResolver:
    """Issues a HEAD request; returns a DiscoveredFile if the response is not HTML."""

    def can_handle(self, url: str, formato: str | None) -> bool:
        return urlsplit(url).scheme in {"http", "https"}

    def resolve(
        self, url: str, formato: str | None, transport: ApiTransport
    ) -> Sequence[DiscoveredFile]:
        request = ApiRequest(method="HEAD", url=url, timeout_seconds=30)
        try:
            response = transport.send(request)
        except _TRANSPORT_EXCEPTIONS as exc:
            _log_resolver_diagnostic(self, url, "transport_error", error=exc)
            return ()

        if not (200 <= response.status_code < 400):
            _log_resolver_diagnostic(
                self,
                url,
                "http_status",
                status_code=response.status_code,
            )
            return ()

        content_type = response.headers_as_dict().get("Content-Type", "")
        if "text/html" in content_type:
            _log_resolver_diagnostic(self, url, "html_content_type")
            return ()

        content_disposition = response.headers_as_dict().get("Content-Disposition", "")
        filename = _filename_from_content_disposition(
            content_disposition
        ) or _filename_from_url(url)
        # Prefer filename-based format; only fall back to content-type for non-binary types
        filename_fmt = _infer_format_name(filename, fallback="")
        content_type_fmt = _format_from_content_type(content_type)
        fmt = (
            filename_fmt
            if filename_fmt and filename_fmt != "binary"
            else (content_type_fmt or filename_fmt or formato or "binary")
        )
        size_str = response.headers_as_dict().get("Content-Length", "")
        size_bytes = int(size_str) if size_str.isdigit() else None

        return (
            DiscoveredFile(
                source_kind="remote",
                location=url,
                filename=filename,
                format=fmt,
                size_bytes=size_bytes,
            ),
        )


class NextcloudWebDavResolver:
    """Enumerates a Nextcloud public share recursively via a single WebDAV PROPFIND."""

    def can_handle(self, url: str, formato: str | None) -> bool:
        return bool(_NEXTCLOUD_SHARE_PATTERN.search(url))

    def resolve(
        self, url: str, formato: str | None, transport: ApiTransport
    ) -> Sequence[DiscoveredFile]:
        match = _NEXTCLOUD_SHARE_PATTERN.search(url)
        if not match:
            return ()

        token = match.group(1)
        parsed = urlsplit(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        webdav_url = f"{base}/public.php/webdav/"

        auth_value = "Basic " + base64.b64encode(f"{token}:".encode()).decode()
        request = ApiRequest(
            method="PROPFIND",
            url=webdav_url,
            timeout_seconds=30,
            headers=(("Authorization", auth_value), ("Depth", _NEXTCLOUD_RECURSIVE_DEPTH)),
        )
        try:
            response = transport.send(request)
        except _TRANSPORT_EXCEPTIONS as exc:
            _log_resolver_diagnostic(self, url, "transport_error", error=exc)
            return ()

        if response.status_code not in {200, 207}:
            _log_resolver_diagnostic(
                self,
                url,
                "http_status",
                status_code=response.status_code,
            )
            return ()

        files = tuple(_parse_propfind_response(response.body, base, token, formato))
        if not files:
            _log_resolver_diagnostic(self, url, "no_files_found")
        return files


class HtmlLinkResolver:
    """Last-resort resolver: GETs the page and extracts <a href> links by format."""

    def can_handle(self, url: str, formato: str | None) -> bool:
        return True

    def resolve(
        self, url: str, formato: str | None, transport: ApiTransport
    ) -> Sequence[DiscoveredFile]:
        request = ApiRequest(method="GET", url=url, timeout_seconds=60)
        try:
            response = transport.send(request)
        except _TRANSPORT_EXCEPTIONS as exc:
            _log_resolver_diagnostic(self, url, "transport_error", error=exc)
            return ()

        if not (200 <= response.status_code < 400):
            _log_resolver_diagnostic(
                self,
                url,
                "http_status",
                status_code=response.status_code,
            )
            return ()

        try:
            html = response.body.decode("utf-8", errors="replace")
        except AttributeError as exc:
            _log_resolver_diagnostic(self, url, "payload_decode_error", error=exc)
            return ()

        files = []
        for href in _extract_anchor_hrefs(html, base_url=url):
            filename = _filename_from_url(href)
            fmt = _infer_format_name(filename, fallback=formato or "binary")
            if formato and fmt != formato:
                continue
            files.append(
                DiscoveredFile(source_kind="remote", location=href, filename=filename, format=fmt)
            )
        return tuple(files)


DEFAULT_RESOLVER_CHAIN: tuple[LinkResolver, ...] = (
    DirectResolver(),
    RedirectResolver(),
    NextcloudWebDavResolver(),
    HtmlLinkResolver(),
)


def resolve_link(
    url: str,
    formato: str | None,
    transport: ApiTransport,
    resolvers: Sequence[LinkResolver],
) -> Sequence[DiscoveredFile]:
    """Run resolvers in order; return the first non-empty result."""
    for resolver in resolvers:
        if resolver.can_handle(url, formato):
            files = resolver.resolve(url, formato, transport)
            if files:
                return files
    return ()


def build_resolver_chain(link_resolver_mode: str) -> tuple[LinkResolver, ...]:
    """Return the resolver chain for the given link_resolver config value."""
    if link_resolver_mode == "direct":
        return (_DirectPassthroughResolver(),)
    if link_resolver_mode == "nextcloud_webdav":
        return (NextcloudWebDavResolver(),)
    if link_resolver_mode == "html_links":
        return (HtmlLinkResolver(),)
    return DEFAULT_RESOLVER_CHAIN


# --- Internal helpers ---


class _DirectPassthroughResolver:
    """Forces direct passthrough regardless of URL shape — no HTTP call."""

    def can_handle(self, url: str, formato: str | None) -> bool:
        return True

    def resolve(
        self, url: str, formato: str | None, transport: ApiTransport
    ) -> Sequence[DiscoveredFile]:
        filename = _filename_from_url(url)
        fmt = _infer_format_name(filename, fallback=formato or "binary")
        return (DiscoveredFile(source_kind="remote", location=url, filename=filename, format=fmt),)




def _format_from_content_type(content_type: str) -> str | None:
    ct = content_type.split(";")[0].strip().lower()
    return _CONTENT_TYPE_FORMAT_MAP.get(ct)


def _log_resolver_diagnostic(
    resolver: LinkResolver,
    url: str,
    category: str,
    *,
    status_code: int | None = None,
    error: BaseException | None = None,
) -> None:
    _LOGGER.debug(
        "file_link_resolver_diagnostic resolver=%s category=%s url=%s "
        "status_code=%s error_type=%s",
        type(resolver).__name__,
        category,
        redact_url(url),
        status_code,
        type(error).__name__ if error is not None else None,
    )


def _parse_propfind_response(
    xml_body: bytes,
    base_url: str,
    share_token: str,
    formato: str | None,
) -> list[DiscoveredFile]:
    try:
        root = ElementTree.fromstring(xml_body)
    except ElementTree.ParseError:
        return []

    files: list[DiscoveredFile] = []
    for response_el in root.findall(f"{{{_DAV_NS}}}response"):
        href_el = response_el.find(f"{{{_DAV_NS}}}href")
        if href_el is None or not href_el.text:
            continue

        href = href_el.text.strip()
        filename = Path(unquote(href)).name.strip()
        if not filename:
            continue

        if response_el.find(f".//{{{_DAV_NS}}}resourcetype/{{{_DAV_NS}}}collection") is not None:
            continue

        fmt = _infer_format_name(filename, fallback=formato or "binary")
        if formato and fmt != formato:
            continue

        size_bytes: int | None = None
        size_el = response_el.find(f".//{{{_DAV_NS}}}getcontentlength")
        if size_el is not None and size_el.text:
            with contextlib.suppress(ValueError):
                size_bytes = int(size_el.text.strip())

        modified_at = None
        modtime_el = response_el.find(f".//{{{_DAV_NS}}}getlastmodified")
        if modtime_el is not None and modtime_el.text:
            with contextlib.suppress(TypeError, ValueError, IndexError, OverflowError):
                modified_at = parsedate_to_datetime(modtime_el.text.strip())

        files.append(
            DiscoveredFile(
                source_kind="remote",
                location=_nextcloud_public_file_url(base_url, share_token, href),
                filename=filename,
                format=fmt,
                size_bytes=size_bytes,
                modified_at=modified_at,
            )
        )

    return files


def _nextcloud_public_file_url(base_url: str, share_token: str, href: str) -> str:
    href_path = unquote(urlsplit(href).path).strip()
    relative_path = href_path
    if href_path.startswith(_NEXTCLOUD_WEBDAV_PREFIX):
        relative_path = href_path.removeprefix(_NEXTCLOUD_WEBDAV_PREFIX)
    relative_path = relative_path.lstrip("/")
    if not relative_path:
        return f"{base_url}/public.php/dav/files/{share_token}"
    encoded_path = quote(relative_path, safe="/")
    return f"{base_url}/public.php/dav/files/{share_token}/{encoded_path}"


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for attr_name, attr_value in attrs:
            if attr_name.lower() == "href" and attr_value:
                self.links.append(attr_value)


def _extract_anchor_hrefs(html: str, base_url: str) -> list[str]:
    parser = _AnchorParser()
    parser.feed(html)
    result = []
    for href in parser.links:
        if href.startswith(("#", "javascript:", "mailto:")):
            continue
        result.append(urljoin(base_url, href))
    return result
