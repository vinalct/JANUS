"""Remote content does not get to choose the host JANUS connects to."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.registry import load_registry
from janus.strategies.files import DiscoveredFile, FileHook, RemoteLinkPolicy
from janus.strategies.files.discovery import _discover_files
from janus.strategies.files.resolvers import (
    DirectResolver,
    HtmlLinkResolver,
    NextcloudWebDavResolver,
)
from janus.strategies.http import ApiRequest, ApiResponse

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "files"

#: The origin every verdict in ``hostile_listing.html`` is measured against.
LISTING_URL = "https://example.gov.br/dados/"

EXPECTED_KEPT = (
    "https://example.gov.br/dados/dados_2024.csv",
    "https://example.gov.br/dados/dados_2023.csv",
    "https://EXAMPLE.GOV.BR/dados/upper.csv",
    "https://example.gov.br/dados/secret.csv?token=abc",
)


EXPECTED_DROPPED = {
    "http://example.gov.br/dados/plain.csv": "host",
    "https://example.gov.br:8443/dados/port.csv": "host",
    "https://cdn.example.gov.br/dados/cdn.csv": "host",
    "https://evil.example/dados/proto_relative.csv": "host",
    "https://example.gov.br@evil.example/dados/userinfo.csv": "host",
    "http://169.254.169.254/latest/meta-data/": "host",
    "http://minio:9000/janus-bronze/warehouse/x.csv": "host",
    "file:///etc/passwd": "scheme",
    "data:text/csv;base64,YSxiCg==": "scheme",
    "ftp://example.gov.br/dados/x.csv": "scheme",
}

#: The CNPJ share, as the ten checked-in sources configure it.
CNPJ_SHARE_URL = "https://arquivos.receitafederal.gov.br/index.php/s/SHARE_TOKEN"
CNPJ_HOST = "arquivos.receitafederal.gov.br"

DROP_EVENT = "file_link_dropped"


# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponseSpec:
    status_code: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


class FakeTransport:
    def __init__(self, responses: list[ResponseSpec | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[ApiRequest] = []
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request: ApiRequest) -> ApiResponse:
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


def _file_source_config(
    tmp_path: Path,
    *,
    url: str,
    source_id: str = "hostile_listing",
    allowed_hosts: tuple[str, ...] = (),
    link_resolver: str = "auto",
    access_format: str = "binary",
) -> SourceConfig:
    access: dict[str, Any] = {
        "url": url,
        "method": "GET",
        "format": access_format,
        "timeout_seconds": 30,
        "link_resolver": link_resolver,
        "auth": {"type": "none"},
        "pagination": {"type": "none"},
        "rate_limit": {"requests_per_minute": None, "concurrency": 1, "backoff_seconds": 5},
    }
    if allowed_hosts:
        access["allowed_hosts"] = list(allowed_hosts)
    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "file",
            "strategy": "file",
            "strategy_variant": "static_file",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": access,
            "extraction": {
                "mode": "full_refresh",
                "checkpoint_strategy": "none",
                "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
            },
            "schema": {"contract": "conf/contracts/test/minimal_contract.yaml"},
            "spark": {"input_format": "csv", "write_mode": "append"},
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "binary"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _plan(tmp_path: Path, source_config: SourceConfig) -> ExecutionPlan:
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id=f"run-{source_config.source_id}",
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        ),
    )


def _policy_for(access):
    return RemoteLinkPolicy.from_access(access)


def _resolve_listing(tmp_path: Path, html: bytes, *, allowed_hosts: tuple[str, ...] = ()):
    access = _file_source_config(
        tmp_path, url=LISTING_URL, allowed_hosts=allowed_hosts
    ).access
    transport = FakeTransport([ResponseSpec(200, html, {"Content-Type": "text/html"})])
    return HtmlLinkResolver().resolve(LISTING_URL, None, transport, policy=_policy_for(access))


def _drop_records(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if DROP_EVENT in record.getMessage()]


# ---------------------------------------------------------------------------
# AC-1 — the HTML listing


def test_a_hostile_listing_yields_only_same_origin_http_candidates(tmp_path):
    """The whole finding in one assertion: nothing off-origin becomes a candidate."""
    html = (FIXTURES / "hostile_listing.html").read_bytes()

    resolved = _resolve_listing(tmp_path, html)

    assert tuple(item.location for item in resolved) == EXPECTED_KEPT


@pytest.mark.parametrize(
    ("label", "allowed_hosts", "admitted"),
    (
        ("exact", ("cdn.example.gov.br",), "https://cdn.example.gov.br/dados/cdn.csv"),
        ("wildcard", ("*.example.gov.br",), "https://cdn.example.gov.br/dados/cdn.csv"),
        (
            "userinfo_host_not_the_text_before_at",
            ("evil.example",),
            "https://example.gov.br@evil.example/dados/userinfo.csv",
        ),
    ),
    ids=("exact", "wildcard", "userinfo_host_not_the_text_before_at"),
)
def test_allowed_hosts_widens_the_set_by_host(tmp_path, label, allowed_hosts, admitted):
    """``allowed_hosts`` names hosts, and the *host* is what decides."""
    del label
    html = (FIXTURES / "hostile_listing.html").read_bytes()

    locations = {
        item.location for item in _resolve_listing(tmp_path, html, allowed_hosts=allowed_hosts)
    }

    assert admitted in locations
    assert set(EXPECTED_KEPT) <= locations, "widening the host set must not drop a same-origin href"


def test_every_dropped_href_is_logged_once_with_its_reason(tmp_path, caplog):
    """One DEBUG line per dropped href — an invisible boundary is an unauditable one."""
    html = (FIXTURES / "hostile_listing.html").read_bytes()

    with caplog.at_level(logging.DEBUG, logger="janus.strategies.files"):
        _resolve_listing(tmp_path, html)

    messages = [record.getMessage() for record in _drop_records(caplog)]

    assert len(messages) == len(EXPECTED_DROPPED), (
        f"expected one {DROP_EVENT} line per dropped href, got {len(messages)}: {messages}"
    )
    for href, reason in EXPECTED_DROPPED.items():
        matching = [message for message in messages if href in message]
        assert len(matching) == 1, f"{href!r} was not logged exactly once: {matching}"
        assert f"reason={reason}" in matching[0], (
            f"{href!r} was dropped for the wrong stated reason: {matching[0]!r}"
        )


def test_a_dropped_href_is_logged_with_its_query_token_redacted(tmp_path, caplog):
    """A drop line renders a URL, so it goes through ``redact_url`` like every other site."""
    html = (
        b"<html><body>"
        b'<a href="https://evil.example/dados/leak.csv?token=abc">leak</a>'
        b"</body></html>"
    )

    with caplog.at_level(logging.DEBUG, logger="janus.strategies.files"):
        _resolve_listing(tmp_path, html)

    messages = [record.getMessage() for record in _drop_records(caplog)]

    assert len(messages) == 1
    assert "token=abc" not in messages[0]
    assert "***REDACTED***" in messages[0]


# ---------------------------------------------------------------------------
# AC-1 — the PROPFIND listing


def _resolve_propfind(tmp_path: Path, body: bytes, *, url: str = CNPJ_SHARE_URL):
    access = _file_source_config(
        tmp_path, url=url, source_id="hostile_propfind", link_resolver="nextcloud_webdav"
    ).access
    transport = FakeTransport([ResponseSpec(207, body, {"Content-Type": "application/xml"})])
    return NextcloudWebDavResolver().resolve(url, "binary", transport, policy=_policy_for(access))


def test_a_hostile_propfind_yields_only_the_same_origin_file_entry(tmp_path, caplog):
    """A foreign DAV href must be **dropped**, not silently re-homed onto the configured base."""
    body = (FIXTURES / "hostile_propfind.xml").read_bytes()

    with caplog.at_level(logging.DEBUG, logger="janus.strategies.files"):
        resolved = _resolve_propfind(tmp_path, body)

    filenames = [item.filename for item in resolved]
    assert filenames == ["Empresas0.zip", "passwd"], (
        f"the collection, the foreign href and the file: href must all be gone; got {filenames}"
    )

    messages = [record.getMessage() for record in _drop_records(caplog)]
    assert any("evil.example" in message and "reason=host" in message for message in messages)
    assert any("file:///" in message and "reason=scheme" in message for message in messages)


def test_a_traversal_href_stays_under_the_configured_base(tmp_path):
    """Containment, not rejection: the basename is already safe, the *location* is not."""
    body = (FIXTURES / "hostile_propfind.xml").read_bytes()

    traversal = next(
        item for item in _resolve_propfind(tmp_path, body) if item.filename == "passwd"
    )

    assert traversal.filename == "passwd"
    assert "/" not in traversal.filename
    assert traversal.location.startswith(f"https://{CNPJ_HOST}/")
    assert ".." not in traversal.location.split("/"), (
        f"the produced location escapes the configured base: {traversal.location}"
    )


# ---------------------------------------------------------------------------
# NFR-2 — the boundary is the framework's, not each hook's


class _OffOriginHook(FileHook):
    """A hook that bypasses the resolver chain and answers with a foreign candidate."""

    def resolve_links(self, plan, url, formato, transport):
        del plan, url, formato, transport
        return (
            DiscoveredFile(
                source_kind="remote",
                location="https://evil.example/x.csv",
                filename="x.csv",
                format="csv",
            ),
        )


def test_a_hook_cannot_bypass_the_host_policy(tmp_path, caplog):
    """``_discover_files`` filters hook output too — FR-2's "belongs to the framework" clause."""
    source_config = _file_source_config(
        tmp_path, url="https://example.gov.br/dados/", source_id="hook_bypass"
    )
    plan = _plan(tmp_path, source_config)

    with caplog.at_level(logging.DEBUG, logger="janus.strategies.files"):
        discovered = _discover_files(plan, _OffOriginHook(), FakeTransport([]))

    assert all("evil.example" not in item.location for item in discovered)
    messages = [record.getMessage() for record in _drop_records(caplog)]
    assert len(messages) == 1, f"expected exactly one drop line, got {messages}"
    assert "evil.example" in messages[0]


# ---------------------------------------------------------------------------
# risk 1 — the default must break no checked-in source


def _file_sources() -> list[SourceConfig]:
    """Every checked-in file source, disabled ones included — they are opt-in, not absent."""
    registry = load_registry(PROJECT_ROOT)
    return [
        source
        for source in registry.list_sources(enabled_only=False)
        if source.source_type == "file"
    ]


def test_the_checked_in_file_sources_are_actually_found():
    """Non-emptiness: a replay over an empty source list would prove nothing at all."""
    sources = _file_sources()

    assert len(sources) >= 11, f"expected the eleven checked-in file sources, found {len(sources)}"


def test_every_cnpj_source_keeps_every_recorded_candidate():
    """The recorded PROPFIND, replayed per source: the same-origin default drops nothing.

    Every ``<d:href>`` in the recorded body is server-relative, so every candidate is on the
    configured origin by construction — this test is what keeps that true after FR-2 lands.
    """
    body = (FIXTURES / "cnpj_propfind_recorded.xml").read_bytes()
    cnpj_sources = [
        source
        for source in _file_sources()
        if source.access.link_resolver == "nextcloud_webdav"
    ]

    assert len(cnpj_sources) == 10, f"expected ten CNPJ sources, found {len(cnpj_sources)}"

    for source in cnpj_sources:
        access = source.access
        transport = FakeTransport([ResponseSpec(207, body, {"Content-Type": "application/xml"})])
        resolved = NextcloudWebDavResolver().resolve(
            access.url, access.format, transport, policy=_policy_for(access)
        )
        assert resolved, f"{source.source_id} resolved no candidate at all"
        off_origin = [item for item in resolved if f"//{CNPJ_HOST}/" not in item.location]
        assert not off_origin, (
            f"{source.source_id} produced off-origin candidates: "
            f"{[item.location for item in off_origin][:3]}"
        )


def test_the_inep_direct_url_is_admitted_by_the_default_policy():
    """INEP is a direct URL: ``DirectResolver`` answers it with no HTTP call, and it passes."""
    inep = next(
        source for source in _file_sources() if source.source_id.startswith("inep_")
    )
    access = inep.access

    resolved = DirectResolver().resolve(
        access.url, access.format, FakeTransport([]), policy=_policy_for(access)
    )

    assert [item.location for item in resolved] == [access.url]
