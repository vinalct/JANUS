from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest

from janus.checkpoints import CheckpointStore
from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.planner import StrategyCatalog
from janus.strategies.api import ApiResponse
from janus.strategies.files import (
    DiscoveredFile,
    FileHook,
    FileIntegrityError,
    FileStrategy,
)
from janus.utils.logging import build_structured_logger
from janus.utils.storage import StorageLayout


@dataclass(frozen=True, slots=True)
class ResponseSpec:
    status_code: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


class FakeTransport:
    def __init__(self, responses: list[ResponseSpec | Exception]) -> None:
        self._responses = list(responses)
        self.requests = []
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def send(self, request):
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


def test_default_strategy_catalog_uses_file_strategy_for_file_variants(tmp_path):
    source_path = tmp_path / "fixtures" / "records.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("id,name\n1,alpha\n", encoding="utf-8")
    source_config = _build_source_config(
        tmp_path,
        source_id="catalog_file",
        access_path=source_path,
    )

    binding = StrategyCatalog.with_defaults().resolve(source_config)

    assert isinstance(binding.strategy, FileStrategy)
    assert binding.dispatch_path == "file.static_file"


def test_file_strategy_logs_archive_progress(tmp_path):
    stream = StringIO()
    logger = build_structured_logger("janus.tests.file.progress", stream=stream)
    archive_path = tmp_path / "fixtures" / "package_2026-04.zip"
    archive_path.parent.mkdir(parents=True)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/records.csv", "id,name\n1,alpha\n")
        archive.writestr("notes/readme.txt", "ignore me\n")
    plan = _build_plan(
        tmp_path,
        source_id="logged_archive_source",
        variant="archive_package",
        access_path=archive_path,
        access_format="binary",
        file_pattern="*.csv",
    )
    strategy, _transport = _build_strategy(tmp_path, logger=logger)

    strategy.extract(plan)

    payloads = [json.loads(line) for line in stream.getvalue().splitlines()]
    events = [payload["event"] for payload in payloads]
    persisted = _event_payload(payloads, "file_payload_persisted")
    archive_extracted = _event_payload(payloads, "file_archive_extracted")
    finished = payloads[-1]

    assert events == [
        "file_extraction_started",
        "file_discovery_finished",
        "file_candidate_started",
        "file_payload_loaded",
        "file_payload_persisted",
        "file_archive_extracted",
        "file_extraction_finished",
    ]
    assert payloads[1]["fields"]["discovered_file_count"] == 1
    assert payloads[1]["fields"]["selected_file_count"] == 1
    assert payloads[2]["fields"]["source_kind"] == "local"
    assert payloads[2]["fields"]["filename"] == "package_2026-04.zip"
    assert payloads[3]["fields"]["attempts_used"] == 1
    assert persisted["fields"]["resolved_version"] == "2026-04"
    assert persisted["fields"]["artifact_format"] == "binary"
    assert persisted["fields"]["checksum_verified"] is False
    assert persisted["fields"]["is_archive"] is True
    assert archive_extracted["fields"]["archive_member_count"] == 1
    assert archive_extracted["fields"]["first_member_artifact_path"].endswith(
        "nested/records.csv"
    )
    assert finished["event"] == "file_extraction_finished"
    assert finished["fields"]["persisted_file_count"] == 1
    assert finished["fields"]["archive_member_count"] == 1
    assert finished["fields"]["normalization_candidate_count"] == 1
    assert finished["fields"]["artifact_count"] == 2
    assert finished["fields"]["checkpoint_value"] == "2026-04"


def test_file_strategy_static_local_file_persists_raw_and_builds_csv_handoff(tmp_path):
    source_path = tmp_path / "fixtures" / "records.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("id,name\n1,alpha\n2,beta\n", encoding="utf-8")
    plan = _build_plan(
        tmp_path,
        source_id="static_local",
        access_path=source_path,
    )
    strategy, transport = _build_strategy(tmp_path)

    result = strategy.extract(plan)
    handoff = strategy.build_normalization_handoff(plan, result)
    metadata = result.metadata_as_dict()

    assert transport.opened is True
    assert transport.closed is True
    assert result.records_extracted == 1
    assert len(result.artifacts) == 1
    assert result.artifacts[0].format == "csv"
    assert metadata["persisted_file_count"] == "1"
    assert metadata["normalization_candidate_count"] == "1"
    assert result.checkpoint_value == "current"

    persisted_path = Path(result.artifacts[0].path)
    assert persisted_path == (
        _raw_run_dir(tmp_path, "static_local") / "downloads" / "current" / "records.csv"
    )
    assert persisted_path.read_text(encoding="utf-8") == "id,name\n1,alpha\n2,beta\n"
    assert handoff.artifact_paths == (str(persisted_path),)

    emitted_metadata = strategy.emit_metadata(plan, result)
    assert emitted_metadata["artifact_count"] == 1
    assert emitted_metadata["input_format"] == "csv"


def test_file_strategy_versioned_full_refresh_selects_latest_local_file(tmp_path):
    source_dir = tmp_path / "fixtures" / "versioned"
    source_dir.mkdir(parents=True)
    (source_dir / "dataset_2026-01.csv").write_text("id\n1\n", encoding="utf-8")
    (source_dir / "dataset_2026-03.csv").write_text("id\n2\n", encoding="utf-8")
    plan = _build_plan(
        tmp_path,
        source_id="versioned_full",
        variant="versioned_file",
        access_path=source_dir,
        file_pattern="*.csv",
    )
    strategy, _ = _build_strategy(tmp_path)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert len(result.artifacts) == 1
    assert Path(result.artifacts[0].path).name == "dataset_2026-03.csv"
    assert result.checkpoint_value == "2026-03"
    assert metadata["selected_file_count"] == "1"
    assert metadata["resolved_versions"] == "2026-03"


def test_file_strategy_incremental_skips_versions_at_or_below_checkpoint(tmp_path):
    source_dir = tmp_path / "fixtures" / "incremental"
    source_dir.mkdir(parents=True)
    (source_dir / "dataset_2026-01.csv").write_text("id\n1\n", encoding="utf-8")
    (source_dir / "dataset_2026-03.csv").write_text("id\n2\n", encoding="utf-8")
    plan = _build_plan(
        tmp_path,
        source_id="versioned_incremental",
        variant="versioned_file",
        access_path=source_dir,
        file_pattern="*.csv",
        extraction_mode="incremental",
        checkpoint_field="publication_version",
        checkpoint_strategy="max_value",
    )
    CheckpointStore().save(plan, "2026-01")
    strategy, _ = _build_strategy(tmp_path)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert len(result.artifacts) == 1
    assert Path(result.artifacts[0].path).name == "dataset_2026-03.csv"
    assert result.checkpoint_value == "2026-03"
    assert metadata["checkpoint_loaded"] == "true"
    assert metadata["selected_file_count"] == "1"


def test_file_strategy_remote_download_retries_and_validates_checksum(tmp_path):
    payload = b"id,name\n1,alpha\n"
    checksum = sha256(payload).hexdigest()
    plan = _build_plan(
        tmp_path,
        source_id="remote_csv",
        access_url="https://example.invalid/files/records.csv",
        requests_per_minute=None,
        retry_max_attempts=3,
        retry_backoff_seconds=2,
        retry_backoff_strategy="exponential",
        rate_limit_backoff_seconds=3,
    )
    sleeps: list[float] = []
    strategy, transport = _build_strategy(
        tmp_path,
        responses=[
            ResponseSpec(503, b"temporarily unavailable"),
            ResponseSpec(503, b"temporarily unavailable"),
            ResponseSpec(200, payload, headers={"X-Checksum-Sha256": checksum}),
        ],
        sleeper=sleeps.append,
    )

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert sleeps == [2.0, 3.0]
    assert transport.opened is True
    assert transport.closed is True
    assert len(transport.requests) == 3
    assert len(result.artifacts) == 1
    assert result.artifacts[0].format == "csv"
    assert metadata["checksum_verified_count"] == "1"
    assert metadata["persisted_file_count"] == "1"


def test_file_strategy_filters_remote_discovery_before_download(tmp_path):
    class MixedRemoteFilesHook(FileHook):
        def resolve_links(self, plan, url, formato, transport):
            del plan
            del url
            del formato
            del transport
            return (
                DiscoveredFile(
                    source_kind="remote",
                    location="https://example.invalid/Empresas0.zip",
                    filename="Empresas0.zip",
                    format="binary",
                ),
                DiscoveredFile(
                    source_kind="remote",
                    location="https://example.invalid/Socios0.zip",
                    filename="Socios0.zip",
                    format="binary",
                ),
            )

    archive_payload = io.BytesIO()
    with ZipFile(archive_payload, "w") as archive:
        archive.writestr("K3241.K03200Y0.D30513.EMPRECSV", "12345678;ACME\n")

    plan = _build_plan(
        tmp_path,
        source_id="cnpj_empresas_remote_filter",
        access_url="https://example.invalid/share",
        access_format="binary",
        spark_input_format="csv",
        remote_file_pattern="Empresas*.zip",
    )
    strategy, transport = _build_strategy(
        tmp_path,
        responses=[ResponseSpec(200, archive_payload.getvalue())],
    )

    result = strategy.extract(plan, hook=MixedRemoteFilesHook())

    assert [request.url for request in transport.requests] == [
        "https://example.invalid/Empresas0.zip"
    ]
    assert len(result.artifacts) == 2
    assert result.artifacts[0].path.endswith("Empresas0.zip")
    assert result.artifacts[1].path.endswith("K3241.K03200Y0.D30513.EMPRECSV")


def test_file_strategy_archive_package_extracts_csv_members_for_handoff(tmp_path):
    archive_path = tmp_path / "fixtures" / "package_2026-04.zip"
    archive_path.parent.mkdir(parents=True)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/records.csv", "id,name\n1,alpha\n")
        archive.writestr("notes/readme.txt", "ignore me\n")

    plan = _build_plan(
        tmp_path,
        source_id="archive_source",
        variant="archive_package",
        access_path=archive_path,
        access_format="binary",
        file_pattern="*.csv",
    )
    strategy, _ = _build_strategy(tmp_path)

    result = strategy.extract(plan)
    handoff = strategy.build_normalization_handoff(plan, result)
    metadata = result.metadata_as_dict()

    assert len(result.artifacts) == 2
    assert sorted(artifact.format for artifact in result.artifacts) == ["binary", "csv"]
    assert metadata["archive_member_count"] == "1"
    assert result.checkpoint_value == "2026-04"
    assert len(handoff.artifacts) == 1
    assert Path(handoff.artifacts[0].path) == (
        _raw_run_dir(tmp_path, "archive_source")
        / "extracted"
        / "2026-04"
        / "package_2026-04"
        / "nested"
        / "records.csv"
    )


def test_file_strategy_rejects_local_checksum_mismatch(tmp_path):
    source_path = tmp_path / "fixtures" / "mismatch.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("id\n1\n", encoding="utf-8")
    source_path.with_name("mismatch.csv.sha256").write_text("deadbeef\n", encoding="utf-8")
    plan = _build_plan(
        tmp_path,
        source_id="checksum_mismatch",
        access_path=source_path,
    )
    strategy, _ = _build_strategy(tmp_path)

    with pytest.raises(FileIntegrityError, match="Checksum mismatch"):
        strategy.extract(plan)


def test_file_strategy_dead_letters_failed_file_and_continues(tmp_path):
    source_dir = tmp_path / "fixtures" / "dead_letter_files"
    source_dir.mkdir(parents=True)
    bad_path = source_dir / "bad.csv"
    good_path = source_dir / "good.csv"
    bad_path.write_text("id\n1\n", encoding="utf-8")
    good_path.write_text("id\n2\n", encoding="utf-8")
    bad_path.with_name("bad.csv.sha256").write_text("deadbeef\n", encoding="utf-8")

    plan = _build_plan(
        tmp_path,
        source_id="file_dead_letter",
        access_path=source_dir,
        file_pattern="*.csv",
        dead_letter_max_items=1,
    )
    strategy, _ = _build_strategy(tmp_path)

    result = strategy.extract(plan)
    metadata = result.metadata_as_dict()

    assert [Path(artifact.path).name for artifact in result.artifacts] == ["good.csv"]
    assert result.records_extracted == 1
    assert metadata["persisted_file_count"] == "1"
    assert metadata["dead_letter_count"] == "1"
    assert metadata["dead_letter_skipped_count"] == "1"

    dead_letter_path = strategy.dead_letter_store.path(plan)
    dead_letter_payload = json.loads(dead_letter_path.read_text(encoding="utf-8"))
    assert dead_letter_payload["entries"][0]["item_type"] == "file_candidate"


def test_file_strategy_extracts_tar_gz_archive_members(tmp_path):
    archive_path = tmp_path / "fixtures" / "cnpj.tar.gz"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(_make_tarball({"estabelecimentos.csv": b"cnpj;nome\n1234;Alpha\n"}))

    plan = _build_plan(
        tmp_path,
        source_id="cnpj_tar_gz",
        variant="static_file",
        access_path=archive_path,
        access_format="binary",
        spark_input_format="csv",
        file_pattern="*.csv",
    )
    strategy, _ = _build_strategy(tmp_path)

    result = strategy.extract(plan)
    handoff = strategy.build_normalization_handoff(plan, result)

    assert len(result.artifacts) == 2
    assert sorted(a.format for a in result.artifacts) == ["binary", "csv"]
    assert len(handoff.artifacts) == 1
    assert Path(handoff.artifacts[0].path).name == "estabelecimentos.csv"


def test_file_strategy_tar_gz_respects_file_pattern(tmp_path):
    archive_path = tmp_path / "fixtures" / "cnpj.tar.gz"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(
        _make_tarball({
            "estabelecimentos.csv": b"cnpj;nome\n1234;Alpha\n",
            "LEIAME.txt": b"ignore me\n",
        })
    )

    plan = _build_plan(
        tmp_path,
        source_id="cnpj_tar_gz_filtered",
        variant="static_file",
        access_path=archive_path,
        access_format="binary",
        spark_input_format="csv",
        file_pattern="*.csv",
    )
    strategy, _ = _build_strategy(tmp_path)

    result = strategy.extract(plan)
    handoff = strategy.build_normalization_handoff(plan, result)

    member_names = [Path(a.path).name for a in handoff.artifacts]
    assert member_names == ["estabelecimentos.csv"]


def _make_tarball(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _event_payload(payloads: list[dict], event: str) -> dict:
    for payload in payloads:
        if payload["event"] == event:
            return payload
    raise AssertionError(f"event {event!r} was not emitted")


def _build_strategy(
    tmp_path: Path,
    responses: list[ResponseSpec | Exception] | None = None,
    *,
    sleeper=None,
    logger=None,
):
    transport = FakeTransport(responses or [])
    strategy = FileStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
        sleeper=sleeper or (lambda seconds: None),
        clock=lambda: 0.0,
        logger=logger,
    )
    return strategy, transport


def _build_plan(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "static_file",
    access_path: Path | None = None,
    access_url: str | None = None,
    file_pattern: str | None = None,
    remote_file_pattern: str | None = None,
    access_format: str = "csv",
    spark_input_format: str = "csv",
    extraction_mode: str = "full_refresh",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
    retry_max_attempts: int = 3,
    retry_backoff_seconds: int = 1,
    retry_backoff_strategy: str = "fixed",
    rate_limit_backoff_seconds: int | None = 5,
    requests_per_minute: int | None = 10,
    dead_letter_max_items: int = 0,
) -> ExecutionPlan:
    source_config = _build_source_config(
        tmp_path,
        source_id=source_id,
        variant=variant,
        access_path=access_path,
        access_url=access_url,
        file_pattern=file_pattern,
        remote_file_pattern=remote_file_pattern,
        access_format=access_format,
        spark_input_format=spark_input_format,
        extraction_mode=extraction_mode,
        checkpoint_field=checkpoint_field,
        checkpoint_strategy=checkpoint_strategy,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        retry_backoff_strategy=retry_backoff_strategy,
        rate_limit_backoff_seconds=rate_limit_backoff_seconds,
        requests_per_minute=requests_per_minute,
        dead_letter_max_items=dead_letter_max_items,
    )
    run_context = RunContext.create(
        run_id=f"run-{source_id}",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _build_source_config(
    tmp_path: Path,
    *,
    source_id: str,
    variant: str = "static_file",
    access_path: Path | None = None,
    access_url: str | None = None,
    file_pattern: str | None = None,
    remote_file_pattern: str | None = None,
    access_format: str = "csv",
    spark_input_format: str = "csv",
    extraction_mode: str = "full_refresh",
    checkpoint_field: str | None = None,
    checkpoint_strategy: str = "none",
    retry_max_attempts: int = 3,
    retry_backoff_seconds: int = 1,
    retry_backoff_strategy: str = "fixed",
    rate_limit_backoff_seconds: int | None = 5,
    requests_per_minute: int | None = 10,
    dead_letter_max_items: int = 0,
) -> SourceConfig:
    access_block: dict[str, Any] = {
        "method": "GET",
        "format": access_format,
        "timeout_seconds": 30,
        "auth": {"type": "none"},
        "pagination": {"type": "none"},
        "rate_limit": {
            "requests_per_minute": requests_per_minute,
            "concurrency": 1,
            "backoff_seconds": rate_limit_backoff_seconds,
        },
    }
    if access_path is not None:
        access_block["path"] = str(access_path)
    if access_url is not None:
        access_block["url"] = access_url
    if file_pattern is not None:
        access_block["file_pattern"] = file_pattern
    if remote_file_pattern is not None:
        access_block["remote_file_pattern"] = remote_file_pattern

    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "file",
            "strategy": "file",
            "strategy_variant": variant,
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": access_block,
            "extraction": {
                "mode": extraction_mode,
                "checkpoint_field": checkpoint_field,
                "checkpoint_strategy": checkpoint_strategy,
                "dead_letter_max_items": dead_letter_max_items,
                "retry": {
                    "max_attempts": retry_max_attempts,
                    "backoff_strategy": retry_backoff_strategy,
                    "backoff_seconds": retry_backoff_seconds,
                },
            },
            "schema": {"mode": "infer"},
            "spark": {
                "input_format": spark_input_format,
                "write_mode": "append",
            },
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "binary"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _raw_run_dir(tmp_path: Path, source_id: str) -> Path:
    return (
        tmp_path
        / "runtime"
        / "raw"
        / "example"
        / source_id
        / "runs"
        / "ingestion_date=2026-04-10"
        / f"run_id=run-{source_id}"
    )


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": "runtime",
                "raw_dir": "runtime/raw",
                "bronze_dir": "runtime/bronze",
                "metadata_dir": "runtime/metadata",
            }
        },
        tmp_path,
    )
