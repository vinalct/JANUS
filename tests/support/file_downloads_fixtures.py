"""Deterministic CNPJ, flat-file and archive-package download fixtures."""

from __future__ import annotations

import copy
import io
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

import yaml

from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.files import FileStrategy
from janus.utils.storage import StorageLayout
from tests.support.cnpj_csv_generator import DEFAULT_SEED, write_csv

if TYPE_CHECKING:
    from tests.support.file_downloads import ScriptedFileServer, UrlResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_HOST = "https://files.download.test"
SHARE_TOKEN = "fixture-share"
WEBDAV_URL = f"{FIXTURE_HOST}/public.php/webdav/"
ZIP_TIME = (2023, 5, 13, 0, 0, 0)
ZIP_COMPRESSION_LEVEL = 6
DAV = "{DAV:}"


@dataclass(frozen=True, slots=True)
class CnpjFixture:
    """Shared script, editable source mapping and the selected candidate URL order."""

    server: ScriptedFileServer
    source_mapping: dict[str, Any]
    archive_urls: tuple[str, ...]
    storage_layout: StorageLayout

    @property
    def urls(self) -> tuple[str, ...]:
        return self.archive_urls

    def build_plan(
        self, run_id: str, *, concurrency: int | None = None,
        dead_letter_max_items: int | None = None, resume: bool = False,
    ) -> ExecutionPlan:
        mapping = copy.deepcopy(self.source_mapping)
        if concurrency is not None:
            mapping["access"]["rate_limit"]["concurrency"] = concurrency
        if dead_letter_max_items is not None:
            mapping["extraction"]["dead_letter_max_items"] = dead_letter_max_items
        root = self.storage_layout.project_root
        config = SourceConfig.from_mapping(mapping, root / "conf" / "sources" / "fixture.yaml")
        context = RunContext.create(
            run_id=run_id,
            environment="local",
            project_root=root,
            started_at=datetime(2026, 9, 17, 12, tzinfo=UTC),
            attributes={"resume": "true"} if resume else {},
        )
        return FileStrategy().plan(config, context)

    def build_strategy(self, **kwargs: Any) -> FileStrategy:
        defaults = {
            "transport_factory": self.server.transport_factory(),
            "storage_layout_factory": lambda plan: self.storage_layout,
            "sleeper": lambda seconds: None,
            "clock": lambda: 0.0,
        }
        defaults.update(kwargs)
        return FileStrategy(**defaults)


def deterministic_zip(members: Mapping[str, bytes]) -> bytes:
    """Fix metadata and compression settings, including across repeated fixture builds."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in members.items():
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info, body, compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=ZIP_COMPRESSION_LEVEL,
            )
    return buffer.getvalue()


def _source_mapping(
    root: Path, *, source_id: str, concurrency: int, dead_letter_max_items: int,
) -> dict[str, Any]:
    mapping = yaml.safe_load(
        (PROJECT_ROOT / "conf/sources/receita_federal/cnpj/empresas.yaml").read_text("utf-8")
    )
    mapping.update(source_id=source_id, name=source_id, enabled=True)
    mapping["access"].update(url=f"{FIXTURE_HOST}/index.php/s/{SHARE_TOKEN}", timeout_seconds=5)
    mapping["access"]["rate_limit"].update(
        requests_per_minute=None, concurrency=concurrency, backoff_seconds=1,
    )
    mapping["extraction"]["dead_letter_max_items"] = dead_letter_max_items
    mapping["extraction"]["retry"].update(
        max_attempts=1, backoff_strategy="fixed", backoff_seconds=1,
    )
    for zone in ("raw", "bronze", "metadata"):
        mapping["outputs"][zone]["path"] = str(root / "data" / zone / source_id)
    return mapping


def _fixture(root: Path, server: ScriptedFileServer, mapping: dict, urls: tuple) -> CnpjFixture:
    layout = StorageLayout.from_environment_config(
        {"storage": {
            "root_dir": "data", "raw_dir": "data/raw",
            "bronze_dir": "data/bronze", "metadata_dir": "data/metadata",
        }}, root,
    )
    return CnpjFixture(server, mapping, urls, layout)


def _csv_bodies(root: Path, count: int) -> list[bytes]:
    # Reuse the existing headerless, semicolon-separated ISO-8859-1 CNPJ generator.
    bodies = []
    for index in range(count):
        path = root / "inputs" / f"rows-{index}.csv"
        write_csv(path, rows=3 + index, seed=DEFAULT_SEED + index)
        bodies.append(path.read_bytes())
    return bodies


def _propfind_listing(archives: Mapping[str, bytes]) -> bytes:
    root = ElementTree.fromstring(
        (PROJECT_ROOT / "tests/fixtures/files/cnpj_propfind_recorded.xml").read_bytes()
    )
    templates = []
    for response in list(root):
        href = response.find(f"{DAV}href")
        if href is not None and Path(href.text or "").name.startswith("Empresas"):
            templates.append(response)
            root.remove(response)
    assert templates, "Recorded PROPFIND contains no Empresas archive"
    for name, body in archives.items():
        response = copy.deepcopy(templates[0])
        response.find(f"{DAV}href").text = f"/public.php/webdav/2023-05/{name}"
        response.find(f".//{DAV}getcontentlength").text = str(len(body))
        root.append(response)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def build_cnpj_fixture(
    tmp_path: Path, *, archive_count: int = 10,
    failing: Mapping[int, UrlResponse | Exception] | None = None,
    retrying: Mapping[int, int] | None = None,
    concurrency: int = 1, dead_letter_max_items: int = 5,
) -> CnpjFixture:
    """Build real WebDAV-discovered ZIPs; fault/retry indices are 1-based candidates."""
    from tests.support.file_downloads import ScriptedFileServer, UrlResponse

    if archive_count < 1:
        raise ValueError("archive_count must be positive")
    failing, retrying = dict(failing or {}), dict(retrying or {})
    if any(index < 1 or index > archive_count for index in failing.keys() | retrying.keys()):
        raise ValueError("Fault indices must address a 1-based candidate")
    if any(count < 0 for count in retrying.values()):
        raise ValueError("Retry counts must be nonnegative")
    root = tmp_path.resolve()
    archives = {
        f"Empresas{i}.zip": deterministic_zip({f"K3241.K03200Y{i}.D30513.EMPRECSV": body})
        for i, body in enumerate(_csv_bodies(root, archive_count))
    }
    urls = tuple(sorted(
        f"{FIXTURE_HOST}/public.php/dav/files/{SHARE_TOKEN}/2023-05/{name}"
        for name in archives
    ))
    routes = {("PROPFIND", WEBDAV_URL): [UrlResponse(207, _propfind_listing(archives))]}
    for index, url in enumerate(urls, start=1):
        body = archives[Path(url).name]
        answer = failing.get(index, UrlResponse(200, body, (("Content-Type", "application/zip"),)))
        routes[("GET", url)] = [
            *[UrlResponse(503, b"scripted retry") for _ in range(retrying.get(index, 0))], answer,
        ]
    mapping = _source_mapping(
        root, source_id="fixture_cnpj", concurrency=concurrency,
        dead_letter_max_items=dead_letter_max_items,
    )
    mapping["extraction"]["retry"]["max_attempts"] = max(retrying.values(), default=0) + 1
    return _fixture(root, ScriptedFileServer(routes), mapping, urls)


def build_flat_files_fixture(tmp_path: Path, *, file_count: int = 10) -> CnpjFixture:
    from tests.support.file_downloads import ScriptedFileServer, UrlResponse

    root = tmp_path.resolve()
    urls = tuple(f"{FIXTURE_HOST}/flat/part-{i:02d}.csv" for i in range(file_count))
    listing_url = f"{FIXTURE_HOST}/flat/"
    html = "".join(f'<a href="{url}">part</a>' for url in urls).encode("utf-8")
    routes = {("GET", listing_url): [UrlResponse(200, html)]}
    routes.update({
        ("GET", url): [UrlResponse(200, body, (("Content-Type", "text/csv"),))]
        for url, body in zip(urls, _csv_bodies(root, file_count), strict=True)
    })
    mapping = _source_mapping(
        root, source_id="fixture_flat", concurrency=1, dead_letter_max_items=5,
    )
    mapping["access"].update(
        url=listing_url, format="csv", link_resolver="html_links", remote_file_pattern="*.csv",
    )
    return _fixture(root, ScriptedFileServer(routes), mapping, urls)


def build_archive_package_fixture(tmp_path: Path) -> CnpjFixture:
    from tests.support.file_downloads import ScriptedFileServer, UrlResponse

    root = tmp_path.resolve()
    bodies = _csv_bodies(root, 2)
    archive = deterministic_zip({
        "package/dados/escolas.csv": bodies[0],
        "package/dados/turmas.csv": bodies[1],
        "package/documentacao/leia-me.txt": "Documentação do censo\n".encode("iso-8859-1"),
    })
    url = f"{FIXTURE_HOST}/microdados_censo_escolar_2024.zip"
    mapping = _source_mapping(
        root, source_id="fixture_archive_package", concurrency=1, dead_letter_max_items=5,
    )
    mapping["strategy_variant"] = "archive_package"
    mapping["extraction"]["mode"] = "snapshot"
    # The landed model calls its archive-member filter access.file_pattern.
    mapping["access"].update(
        url=url, link_resolver="direct", remote_file_pattern=None, file_pattern="*/dados/*.csv",
    )
    server = ScriptedFileServer({("GET", url): [UrlResponse(200, archive)]})
    return _fixture(root, server, mapping, (url,))
