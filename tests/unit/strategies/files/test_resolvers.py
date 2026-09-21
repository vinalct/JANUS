from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api import ApiRequest, ApiResponse
from janus.strategies.files import DiscoveredFile, FileHook, FileStrategy
from janus.strategies.files.resolvers import (
    DEFAULT_RESOLVER_CHAIN,
    DirectResolver,
    HtmlLinkResolver,
    NextcloudWebDavResolver,
    RedirectResolver,
    build_resolver_chain,
    resolve_link,
)
from janus.utils.storage import StorageLayout

# ---------------------------------------------------------------------------
# Shared test infrastructure
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


# ---------------------------------------------------------------------------
# AC1 — DirectResolver passthrough for known file extensions (no HTTP call)
# ---------------------------------------------------------------------------


def test_direct_resolver_handles_known_extensions():
    r = DirectResolver()
    assert r.can_handle("https://example.gov.br/data/arquivo.zip", None)
    assert r.can_handle("https://example.gov.br/data/records.csv", None)
    assert r.can_handle("https://example.gov.br/data/dump.json", None)
    assert not r.can_handle("https://example.gov.br/portal/page", None)
    assert not r.can_handle("https://example.gov.br/index.php/s/TOKEN", None)


def test_direct_resolver_returns_one_file_without_http_call():
    transport = FakeTransport([])
    url = "https://example.gov.br/data/arquivo.zip"

    files = DirectResolver().resolve(url, None, transport)

    assert len(files) == 1
    assert files[0].location == url
    assert files[0].filename == "arquivo.zip"
    assert files[0].format == "binary"
    assert files[0].source_kind == "remote"
    assert len(transport.requests) == 0


def test_direct_resolver_infers_csv_format():
    transport = FakeTransport([])
    url = "https://example.gov.br/data/records.csv"

    files = DirectResolver().resolve(url, None, transport)

    assert files[0].format == "csv"
    assert len(transport.requests) == 0


# ---------------------------------------------------------------------------
# AC2 — RedirectResolver follows redirect chain via HEAD
# ---------------------------------------------------------------------------


def test_redirect_resolver_returns_file_on_non_html_content_type():
    transport = FakeTransport([
        ResponseSpec(
            200, b"",
            {"Content-Type": "text/csv", "Content-Length": "1024"},
        )
    ])
    url = "https://dados.gov.br/resource/abc123"

    files = RedirectResolver().resolve(url, "csv", transport)

    assert len(files) == 1
    assert files[0].location == url
    assert files[0].format == "csv"
    assert files[0].size_bytes == 1024
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "HEAD"


def test_redirect_resolver_skips_html_response():
    transport = FakeTransport([
        ResponseSpec(200, b"", {"Content-Type": "text/html; charset=utf-8"})
    ])

    files = RedirectResolver().resolve("https://example.gov.br/page", None, transport)

    assert list(files) == []


def test_redirect_resolver_uses_content_disposition_for_filename():
    transport = FakeTransport([
        ResponseSpec(
            200, b"",
            {
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="data_2026.csv"',
            },
        )
    ])

    files = RedirectResolver().resolve("https://example.gov.br/download/abc", None, transport)

    assert len(files) == 1
    assert files[0].filename == "data_2026.csv"
    assert files[0].format == "csv"


def test_redirect_resolver_returns_empty_on_transport_error():
    from janus.strategies.api import ApiTransportError

    transport = FakeTransport([ApiTransportError("network timeout")])

    files = RedirectResolver().resolve("https://example.gov.br/data", None, transport)

    assert list(files) == []


def test_redirect_resolver_returns_empty_on_error_status():
    transport = FakeTransport([ResponseSpec(404, b"Not Found")])

    files = RedirectResolver().resolve("https://example.gov.br/data", None, transport)

    assert list(files) == []


# ---------------------------------------------------------------------------
# AC3 — NextcloudWebDavResolver enumerates share recursively via PROPFIND
# ---------------------------------------------------------------------------

_PROPFIND_RESPONSE = textwrap.dedent("""\
    <?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response>
        <d:href>/public.php/webdav/</d:href>
        <d:propstat>
          <d:prop>
            <d:resourcetype><d:collection/></d:resourcetype>
          </d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
      <d:response>
        <d:href>/public.php/webdav/CNPJ_2024.zip</d:href>
        <d:propstat>
          <d:prop>
            <d:resourcetype/>
            <d:getcontentlength>10485760</d:getcontentlength>
            <d:getlastmodified>Mon, 01 Jan 2024 00:00:00 GMT</d:getlastmodified>
          </d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
      <d:response>
        <d:href>/public.php/webdav/CNPJ_2024_b.zip</d:href>
        <d:propstat>
          <d:prop>
            <d:resourcetype/>
            <d:getcontentlength>5242880</d:getcontentlength>
          </d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
    </d:multistatus>
""").encode()

_PROPFIND_RECURSIVE_RESPONSE = textwrap.dedent("""\
    <?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response>
        <d:href>/public.php/webdav/</d:href>
        <d:propstat>
          <d:prop>
            <d:resourcetype><d:collection/></d:resourcetype>
          </d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
      <d:response>
        <d:href>/public.php/webdav/2026-04/</d:href>
        <d:propstat>
          <d:prop>
            <d:resourcetype><d:collection/></d:resourcetype>
          </d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
      <d:response>
        <d:href>/public.php/webdav/2026-04/Empresas0.zip</d:href>
        <d:propstat>
          <d:prop>
            <d:resourcetype/>
            <d:getcontentlength>263675156</d:getcontentlength>
          </d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
      <d:response>
        <d:href>/public.php/webdav/2026-04/README.txt</d:href>
        <d:propstat>
          <d:prop>
            <d:resourcetype/>
            <d:getcontentlength>12</d:getcontentlength>
          </d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
    </d:multistatus>
""").encode()


def test_nextcloud_webdav_resolver_detects_share_url():
    r = NextcloudWebDavResolver()
    assert r.can_handle("https://arquivos.receitafederal.gov.br/index.php/s/TOKEN123", None)
    assert not r.can_handle("https://example.gov.br/data/file.zip", None)
    assert not r.can_handle("https://dados.gov.br/resource/abc", None)


def test_nextcloud_webdav_resolver_returns_one_discovered_file_per_zip():
    transport = FakeTransport([ResponseSpec(207, _PROPFIND_RESPONSE)])
    url = "https://arquivos.receitafederal.gov.br/index.php/s/YggdBLfdninEJX9"

    files = NextcloudWebDavResolver().resolve(url, None, transport)

    assert len(files) == 2
    assert len(transport.requests) == 1  # single PROPFIND call
    assert transport.requests[0].method == "PROPFIND"

    req_headers = dict(transport.requests[0].headers)
    assert req_headers.get("Depth") == "infinity"
    assert req_headers.get("Authorization", "").startswith("Basic ")
    assert transport.requests[0].sensitive_headers == ("Authorization",)

    filenames = {f.filename for f in files}
    assert filenames == {"CNPJ_2024.zip", "CNPJ_2024_b.zip"}

    big_file = next(f for f in files if f.filename == "CNPJ_2024.zip")
    assert big_file.size_bytes == 10485760
    assert big_file.modified_at is not None
    assert big_file.modified_at.tzinfo is not None
    assert big_file.location == (
        "https://arquivos.receitafederal.gov.br/public.php/dav/files/"
        "YggdBLfdninEJX9/CNPJ_2024.zip"
    )
    assert big_file.format == "binary"


def test_nextcloud_webdav_resolver_filters_by_formato():
    propfind_mixed = textwrap.dedent("""\
        <?xml version="1.0"?>
        <d:multistatus xmlns:d="DAV:">
          <d:response>
            <d:href>/public.php/webdav/data.zip</d:href>
            <d:propstat>
              <d:prop><d:resourcetype/></d:prop>
              <d:status>HTTP/1.1 200 OK</d:status>
            </d:propstat>
          </d:response>
          <d:response>
            <d:href>/public.php/webdav/readme.txt</d:href>
            <d:propstat>
              <d:prop><d:resourcetype/></d:prop>
              <d:status>HTTP/1.1 200 OK</d:status>
            </d:propstat>
          </d:response>
        </d:multistatus>
    """).encode()

    transport = FakeTransport([ResponseSpec(207, propfind_mixed)])
    files = NextcloudWebDavResolver().resolve(
        "https://host.gov.br/index.php/s/TOKEN", "binary", transport
    )

    assert len(files) == 1
    assert files[0].filename == "data.zip"


def test_nextcloud_webdav_resolver_discovers_nested_files_recursively():
    transport = FakeTransport([ResponseSpec(207, _PROPFIND_RECURSIVE_RESPONSE)])

    files = NextcloudWebDavResolver().resolve(
        "https://host.gov.br/index.php/s/TOKEN", "binary", transport
    )

    assert len(files) == 1
    assert files[0].filename == "Empresas0.zip"
    assert files[0].location == (
        "https://host.gov.br/public.php/dav/files/TOKEN/2026-04/Empresas0.zip"
    )
    assert files[0].size_bytes == 263675156


def test_nextcloud_webdav_resolver_preserves_nested_paths_in_download_url():
    propfind_nested = textwrap.dedent("""\
        <?xml version="1.0"?>
        <d:multistatus xmlns:d="DAV:">
          <d:response>
            <d:href>/public.php/webdav/2026-04/CNPJ%20Dados.zip</d:href>
            <d:propstat>
              <d:prop><d:resourcetype/></d:prop>
              <d:status>HTTP/1.1 200 OK</d:status>
            </d:propstat>
          </d:response>
        </d:multistatus>
    """).encode()

    transport = FakeTransport([ResponseSpec(207, propfind_nested)])
    files = NextcloudWebDavResolver().resolve(
        "https://host.gov.br/index.php/s/TOKEN", None, transport
    )

    assert len(files) == 1
    assert files[0].filename == "CNPJ Dados.zip"
    assert files[0].location == (
        "https://host.gov.br/public.php/dav/files/TOKEN/2026-04/CNPJ%20Dados.zip"
    )


def test_nextcloud_webdav_resolver_uses_token_as_basic_auth():
    transport = FakeTransport([ResponseSpec(207, _PROPFIND_RESPONSE)])
    url = "https://example.gov.br/index.php/s/MYTOKEN99"

    NextcloudWebDavResolver().resolve(url, None, transport)

    import base64

    auth_header = dict(transport.requests[0].headers).get("Authorization", "")
    decoded = base64.b64decode(auth_header.removeprefix("Basic ")).decode()
    assert decoded == "MYTOKEN99:"


def test_nextcloud_webdav_resolver_returns_empty_on_propfind_failure():
    transport = FakeTransport([ResponseSpec(403, b"Forbidden")])

    files = NextcloudWebDavResolver().resolve(
        "https://example.gov.br/index.php/s/TOKEN", None, transport
    )

    assert list(files) == []


# ---------------------------------------------------------------------------
# AC4 — HtmlLinkResolver extracts links matching formato hint from static HTML
# ---------------------------------------------------------------------------

_HTML_FIXTURE = b"""
<html>
<body>
<h1>Dataset downloads</h1>
<a href="/data/cnpj_2024.csv">CNPJ CSV</a>
<a href="/data/cnpj_2024_b.csv">CNPJ CSV part B</a>
<a href="/data/readme.txt">Read me</a>
<a href="https://example.gov.br/other/data.zip">Archive</a>
<a href="#anchor">Skip</a>
</body>
</html>
"""


def test_html_link_resolver_extracts_links_matching_formato():
    transport = FakeTransport([
        ResponseSpec(200, _HTML_FIXTURE, {"Content-Type": "text/html"})
    ])
    url = "https://example.gov.br/portal/datasets"

    files = HtmlLinkResolver().resolve(url, "csv", transport)

    assert len(files) == 2
    assert all(f.format == "csv" for f in files)
    assert all(f.filename.endswith(".csv") for f in files)
    assert transport.requests[0].method == "GET"


def test_html_link_resolver_skips_links_not_matching_formato():
    transport = FakeTransport([
        ResponseSpec(200, _HTML_FIXTURE, {"Content-Type": "text/html"})
    ])

    files = HtmlLinkResolver().resolve("https://example.gov.br/page", "binary", transport)

    assert len(files) == 1
    assert files[0].filename == "data.zip"


def test_html_link_resolver_resolves_relative_links_against_base_url():
    transport = FakeTransport([
        ResponseSpec(200, _HTML_FIXTURE, {"Content-Type": "text/html"})
    ])

    files = HtmlLinkResolver().resolve("https://example.gov.br/portal/datasets", "csv", transport)

    locations = {f.location for f in files}
    assert all(loc.startswith("https://example.gov.br") for loc in locations)


def test_html_link_resolver_returns_empty_on_http_error():
    transport = FakeTransport([ResponseSpec(404, b"Not Found")])

    files = HtmlLinkResolver().resolve("https://example.gov.br/page", "csv", transport)

    assert list(files) == []


# ---------------------------------------------------------------------------
# AC5 — A source hook overriding resolve_links fully replaces the chain
# ---------------------------------------------------------------------------


def test_hook_override_replaces_default_chain(tmp_path):
    class CustomHook(FileHook):
        def __init__(self) -> None:
            self.called_with: list[str] = []

        def resolve_links(self, plan, url, formato, transport):
            self.called_with.append(url)
            return (
                DiscoveredFile(
                    source_kind="remote",
                    location=url,
                    filename="custom.csv",
                    format="csv",
                ),
            )

    payload = b"id\n1\n"
    hook = CustomHook()
    plan = _build_plan(tmp_path, source_id="hook_override", access_url="https://example.gov.br/data")
    strategy, transport = _build_strategy(
        tmp_path, responses=[ResponseSpec(200, payload, {"Content-Type": "text/csv"})]
    )

    strategy.extract(plan, hook=hook)

    assert hook.called_with == ["https://example.gov.br/data"]
    # Only the download request — no HEAD/GET from default resolvers
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "GET"


# ---------------------------------------------------------------------------
# AC6 — link_resolver: direct bypasses all resolvers (no HTTP resolution call)
# ---------------------------------------------------------------------------


def test_build_resolver_chain_direct_always_passes_through():
    transport = FakeTransport([])
    chain = build_resolver_chain("direct")
    url = "https://arquivos.receitafederal.gov.br/index.php/s/TOKEN"

    files = resolve_link(url, None, transport, chain)

    assert len(files) == 1
    assert files[0].location == url
    assert len(transport.requests) == 0


def test_resolve_link_direct_config_preserves_url_for_non_extension_url(tmp_path):
    payload = b"data,value\n1,a\n"
    plan = _build_plan(
        tmp_path,
        source_id="direct_config",
        access_url="https://dados.gov.br/resource/no-extension",
        link_resolver="direct",
        access_format="csv",
    )
    strategy, transport = _build_strategy(
        tmp_path,
        responses=[ResponseSpec(200, payload)],
    )

    result = strategy.extract(plan)

    assert len(result.artifacts) == 1
    # No HEAD/resolution requests — only the download
    assert len(transport.requests) == 1
    assert transport.requests[0].url == "https://dados.gov.br/resource/no-extension"


# ---------------------------------------------------------------------------
# AC7 — A failed URL produces a dead-letter entry and does not crash the run
# ---------------------------------------------------------------------------

_PROPFIND_TWO_FILES = textwrap.dedent("""\
    <?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response>
        <d:href>/public.php/webdav/file_a.zip</d:href>
        <d:propstat>
          <d:prop><d:resourcetype/></d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
      <d:response>
        <d:href>/public.php/webdav/file_b.zip</d:href>
        <d:propstat>
          <d:prop><d:resourcetype/></d:prop>
          <d:status>HTTP/1.1 200 OK</d:status>
        </d:propstat>
      </d:response>
    </d:multistatus>
""").encode()


def test_dead_letter_records_failed_download_and_run_continues(tmp_path):
    # Nextcloud resolver discovers 2 files; first download fails → dead-lettered;
    # second succeeds → extraction continues without crashing.
    plan = _build_plan(
        tmp_path,
        source_id="dead_letter_test",
        access_url="https://host.gov.br/index.php/s/TOKEN",
        access_format="binary",
        link_resolver="nextcloud_webdav",
        dead_letter_max_items=1,
    )
    strategy, transport = _build_strategy(
        tmp_path,
        responses=[
            ResponseSpec(207, _PROPFIND_TWO_FILES),  # PROPFIND discovers 2 files
            ResponseSpec(503, b"Service Unavailable"),  # file_a.zip download fails
            ResponseSpec(200, b"not-a-real-zip"),  # file_b.zip download succeeds
        ],
    )

    result = strategy.extract(plan)

    assert result.metadata_as_dict()["dead_letter_count"] == "1"
    assert len(result.artifacts) == 1


def test_dead_letter_records_html_download_as_failed_file_and_run_continues(tmp_path):
    plan = _build_plan(
        tmp_path,
        source_id="html_dead_letter_test",
        access_url="https://host.gov.br/index.php/s/TOKEN",
        access_format="binary",
        link_resolver="nextcloud_webdav",
        dead_letter_max_items=1,
    )
    strategy, _transport = _build_strategy(
        tmp_path,
        responses=[
            ResponseSpec(207, _PROPFIND_TWO_FILES),
            ResponseSpec(200, b"", {"Content-Type": "text/html; charset=utf-8"}),
            ResponseSpec(200, b"not-a-real-zip", {"Content-Type": "application/octet-stream"}),
        ],
    )

    result = strategy.extract(plan)

    assert result.metadata_as_dict()["dead_letter_count"] == "1"
    assert len(result.artifacts) == 1


# ---------------------------------------------------------------------------
# Chain runner — resolve_link free function
# ---------------------------------------------------------------------------


def test_resolve_link_stops_at_first_successful_resolver():
    transport = FakeTransport([])
    url = "https://example.gov.br/data/file.csv"

    files = resolve_link(url, None, transport, DEFAULT_RESOLVER_CHAIN)

    # DirectResolver handles .csv without HTTP
    assert len(files) == 1
    assert len(transport.requests) == 0


def test_resolve_link_falls_through_when_all_resolvers_return_empty():
    # Non-extension URL, HEAD returns HTML, no Nextcloud pattern, HTML has no matching links
    transport = FakeTransport([
        ResponseSpec(200, b"<html></html>", {"Content-Type": "text/html"}),  # HEAD
        ResponseSpec(200, b"<html></html>", {"Content-Type": "text/html"}),  # GET html resolver
    ])

    files = resolve_link(
        "https://example.gov.br/portal", "csv", transport, DEFAULT_RESOLVER_CHAIN
    )

    assert list(files) == []


def test_build_resolver_chain_nextcloud_webdav():
    chain = build_resolver_chain("nextcloud_webdav")
    assert len(chain) == 1
    assert isinstance(chain[0], NextcloudWebDavResolver)


def test_build_resolver_chain_html_links():
    chain = build_resolver_chain("html_links")
    assert len(chain) == 1
    assert isinstance(chain[0], HtmlLinkResolver)


def test_build_resolver_chain_auto_uses_full_chain():
    chain = build_resolver_chain("auto")
    assert chain is DEFAULT_RESOLVER_CHAIN


# ---------------------------------------------------------------------------
# tar.gz format inference — DirectResolver and HtmlLinkResolver
# ---------------------------------------------------------------------------


def test_direct_resolver_handles_tar_gz_url():
    transport = FakeTransport([])
    url = "https://dados.rfb.gov.br/CNPJ/cnpj.tar.gz"

    files = resolve_link(url, None, transport, DEFAULT_RESOLVER_CHAIN)

    assert len(files) == 1
    assert files[0].filename == "cnpj.tar.gz"
    assert files[0].format == "binary"
    assert len(transport.requests) == 0  # DirectResolver — no HTTP call


def test_direct_resolver_handles_tgz_url():
    transport = FakeTransport([])
    url = "https://dados.rfb.gov.br/CNPJ/cnpj.tgz"

    files = resolve_link(url, None, transport, DEFAULT_RESOLVER_CHAIN)

    assert len(files) == 1
    assert files[0].format == "binary"
    assert len(transport.requests) == 0


def test_html_link_resolver_includes_tar_gz_links_when_format_is_binary():
    html_body = textwrap.dedent("""\
        <html><body>
        <a href="/CNPJ/cnpj.tar.gz">Dados CNPJ</a>
        <a href="/CNPJ/LAYOUT.pdf">Layout</a>
        </body></html>
    """).encode()
    transport = FakeTransport([ResponseSpec(200, html_body)])
    url = "https://dados.rfb.gov.br/CNPJ/"

    files = resolve_link(url, "binary", transport, (HtmlLinkResolver(),))

    filenames = {f.filename for f in files}
    assert "cnpj.tar.gz" in filenames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_plan(
    tmp_path: Path,
    *,
    source_id: str,
    access_url: str = "https://example.gov.br/data/file.csv",
    access_format: str = "csv",
    link_resolver: str = "auto",
    dead_letter_max_items: int = 0,
) -> ExecutionPlan:
    source_config = _build_source_config(
        tmp_path,
        source_id=source_id,
        access_url=access_url,
        access_format=access_format,
        link_resolver=link_resolver,
        dead_letter_max_items=dead_letter_max_items,
    )
    run_context = RunContext.create(
        run_id=f"run-{source_id}",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _build_source_config(
    tmp_path: Path,
    *,
    source_id: str,
    access_url: str,
    access_format: str = "csv",
    link_resolver: str = "auto",
    dead_letter_max_items: int = 0,
) -> SourceConfig:
    access_block: dict[str, Any] = {
        "method": "GET",
        "format": access_format,
        "url": access_url,
        "link_resolver": link_resolver,
        "timeout_seconds": 30,
        "auth": {"type": "none"},
        "pagination": {"type": "none"},
        "rate_limit": {"requests_per_minute": None, "concurrency": 1},
    }
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
            "access": access_block,
            "extraction": {
                "mode": "full_refresh",
                "checkpoint_strategy": "none",
                "dead_letter_max_items": dead_letter_max_items,
                "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": access_format, "write_mode": "overwrite"},
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "binary"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _build_strategy(
    tmp_path: Path,
    responses: list[ResponseSpec | Exception] | None = None,
):
    transport = FakeTransport(responses or [])
    strategy = FileStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: StorageLayout.from_environment_config(
            {
                "storage": {
                    "root_dir": "runtime",
                    "raw_dir": "runtime/raw",
                    "bronze_dir": "runtime/bronze",
                    "metadata_dir": "runtime/metadata",
                }
            },
            tmp_path,
        ),
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    return strategy, transport


# ---------------------------------------------------------------------------
# resolver chain falls through on transport exceptions
# ---------------------------------------------------------------------------


def test_redirect_resolver_returns_empty_on_transport_exception():
    """RedirectResolver must swallow transport errors and return empty — not crash."""
    from urllib.error import URLError

    transport = FakeTransport([URLError("connection refused")])
    url = "https://example.gov.br/data/file"

    files = RedirectResolver().resolve(url, "csv", transport)

    assert list(files) == []


def test_html_link_resolver_returns_empty_on_transport_exception():
    """HtmlLinkResolver must swallow transport errors and return empty — not crash."""
    from urllib.error import URLError

    transport = FakeTransport([URLError("connection refused")])
    url = "https://example.gov.br/portal/page"

    files = HtmlLinkResolver().resolve(url, "csv", transport)

    assert list(files) == []


def test_nextcloud_webdav_resolver_returns_empty_on_transport_exception():
    """NextcloudWebDavResolver must swallow transport errors and return empty."""
    from urllib.error import URLError

    transport = FakeTransport([URLError("connection refused")])
    url = "https://example.gov.br/index.php/s/TOKEN"

    files = NextcloudWebDavResolver().resolve(url, None, transport)

    assert list(files) == []


def test_resolve_link_chain_falls_through_when_redirect_resolver_fails():
    """
    When RedirectResolver raises a transport exception the chain must continue
    and the next matching resolver (HtmlLinkResolver) gets a chance to resolve.
    """
    from urllib.error import URLError

    head_exception = URLError("connection refused")
    html_body = b"<html><body><a href='/data/report.csv'>CSV</a></body></html>"
    transport = FakeTransport([
        head_exception,
        ResponseSpec(200, html_body, {"Content-Type": "text/html"}),
    ])
    url = "https://example.gov.br/portal/page"
    chain = (RedirectResolver(), HtmlLinkResolver())

    files = resolve_link(url, "csv", transport, chain)

    assert len(files) == 1
    assert files[0].filename == "report.csv"


def test_resolver_transport_diagnostic_uses_redacted_url(caplog):
    from urllib.error import URLError

    caplog.set_level(logging.DEBUG, logger="janus.strategies.files.resolvers")
    transport = FakeTransport([URLError("connection refused")])
    url = "https://example.gov.br/index.php/s/TOKEN123?token=secret&page=1"

    files = NextcloudWebDavResolver().resolve(url, None, transport)

    assert list(files) == []
    assert "NextcloudWebDavResolver" in caplog.text
    assert "transport_error" in caplog.text
    assert "TOKEN123" not in caplog.text
    assert "secret" not in caplog.text
    assert "***REDACTED***" in caplog.text
