"""Admission policy for links discovered from remote file listings."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit

from janus.models import AccessConfig
from janus.strategies.files.core import DiscoveredFile
from janus.strategies.http import SUPPORTED_URL_SCHEMES
from janus.utils.logging import redact_url

_LOGGER = logging.getLogger(__name__)
_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True, slots=True)
class RemoteLinkPolicy:
    """Restrict discovered remote links to the configured source boundary."""

    origin_scheme: str
    origin_host: str
    origin_port: int
    allowed_hosts: frozenset[str]
    allowed_suffixes: frozenset[str]

    @classmethod
    def from_access(cls, access: AccessConfig) -> RemoteLinkPolicy | None:
        """Build the policy for a remote source, or ``None`` for a local-only source."""
        if access.url is None:
            return None

        origin = urlsplit(access.url)
        origin_scheme = origin.scheme.lower()
        origin_host = (origin.hostname or "").lower()
        origin_port = _effective_port(origin, origin_scheme)
        if origin_port is None:
            origin_port = -1

        allowed_hosts = frozenset(
            host.lower() for host in access.allowed_hosts if not host.startswith("*.")
        )
        allowed_suffixes = frozenset(
            host[1:].lower() for host in access.allowed_hosts if host.startswith("*.")
        )
        return cls(
            origin_scheme=origin_scheme,
            origin_host=origin_host,
            origin_port=origin_port,
            allowed_hosts=allowed_hosts,
            allowed_suffixes=allowed_suffixes,
        )

    def rejection_reason(self, href: str) -> str | None:
        """Return why ``href`` is rejected, or ``None`` when it is admitted."""
        parsed = urlsplit(href)
        scheme = parsed.scheme.lower()
        if scheme not in SUPPORTED_URL_SCHEMES:
            return "scheme"

        hostname = parsed.hostname
        if hostname is None:
            return "host"
        host = hostname.lower()
        port = _effective_port(parsed, scheme)
        if port is None:
            return "host"

        if (scheme, host, port) == (
            self.origin_scheme,
            self.origin_host,
            self.origin_port,
        ):
            return None
        suffix_allowed = any(
            host.endswith(suffix) and len(host) > len(suffix)
            for suffix in self.allowed_suffixes
        )
        if host in self.allowed_hosts or suffix_allowed:
            return None
        return "host"

    def filter(
        self,
        candidates: Iterable[DiscoveredFile],
        *,
        stage: str,
    ) -> tuple[DiscoveredFile, ...]:
        """Drop rejected remote candidates and log one diagnostic for each drop."""
        admitted: list[DiscoveredFile] = []
        for candidate in candidates:
            if candidate.source_kind != "remote":
                admitted.append(candidate)
                continue

            reason = self.rejection_reason(candidate.location)
            if reason is None:
                admitted.append(candidate)
                continue

            _LOGGER.debug(
                "file_link_dropped href=%s reason=%s stage=%s",
                redact_url(candidate.location),
                reason,
                stage,
            )
        return tuple(admitted)


def _effective_port(parsed: SplitResult, scheme: str) -> int | None:
    try:
        return parsed.port if parsed.port is not None else _DEFAULT_PORTS.get(scheme)
    except ValueError:
        return None
