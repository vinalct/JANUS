from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import pytest

import janus.scripts.raw_to_bronze as raw_to_bronze
from janus.models import ExecutionPlan, RunContext, WriteResult
from janus.planner import PlannedRun
from janus.quality import PersistedValidationReport, ValidationCheck, ValidationReport
from janus.registry import load_registry
from janus.scripts.raw_to_bronze import (
    RawToBronzeLoader,
    _artifact_format_for_path,
    _rediscover_raw_artifacts,
    _sha256,
)
from janus.strategies.catalog import CatalogStrategy
from janus.strategies.files import FileStrategy
from janus.utils.storage import StorageLayout
from janus.writers import SIDECAR_SUFFIX, RawArtifactWriter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "dados_abertos_catalog"


@dataclass(slots=True)
class FakeStrategy:
    calls: list[str]

    @property
    def strategy_family(self) -> str:
        return "api"

    def plan(self, source_config, run_context, hook=None):
        raise NotImplementedError

    def extract(self, plan, hook=None, *, spark=None):
        raise NotImplementedError

    def build_normalization_handoff(self, plan, extraction_result, hook=None):
        del plan
        del hook
        self.calls.append("handoff")
        assert len(extraction_result.artifacts) == 1
        assert extraction_result.artifacts[0].format == "json"
        return extraction_result

    def emit_metadata(self, plan, extraction_result, write_results=(), hook=None):
        del plan
        del extraction_result
        del hook
        self.calls.append("metadata")
        return {"write_result_count": len(write_results)}


@dataclass(slots=True)
class FakeReader:
    calls: list[str]
    seen_format_name: str | None = None
    seen_schema: Any | None = None
    seen_options: dict[str, str] | None = None
    read_artifact_counts: list[int] = field(default_factory=list)

    def read_extraction_result(
        self,
        spark,
        extraction_result,
        format_name=None,
        schema=None,
        options=None,
    ):
        del spark
        self.read_artifact_counts.append(len(extraction_result.artifacts))
        self.seen_format_name = format_name
        self.seen_schema = schema
        self.seen_options = None if options is None else dict(options)
        self.calls.append("read")
        return object()


@dataclass(slots=True)
class FakeNormalizer:
    calls: list[str]

    def normalize(self, dataframe, plan):
        del dataframe
        del plan
        self.calls.append("normalize")
        return object()


@dataclass(slots=True)
class FakeWriter:
    calls: list[str]
    expected_namespace: str = "curated"
    expected_table_name: str = "custom_table"
    result_path: str = "curated.custom_table"

    def write(self, dataframe, plan, zone, **kwargs):
        del dataframe
        del kwargs
        self.calls.append("write")
        assert zone == "bronze"
        assert plan.bronze_output.namespace == self.expected_namespace
        assert plan.bronze_output.table_name == self.expected_table_name
        return WriteResult.from_plan(
            plan,
            zone,
            path=self.result_path,
            format_name="iceberg",
            mode="overwrite",
            records_written=1,
        )


@dataclass(slots=True)
class RecordingWriter:
    modes: list[str]
    paths: list[str]

    def write(self, dataframe, plan, zone, **kwargs):
        del dataframe
        assert zone == "bronze"
        mode = kwargs.get("mode") or plan.source_config.spark.write_mode
        path = f"{plan.bronze_output.namespace}.{plan.bronze_output.table_name}"
        self.modes.append(mode)
        self.paths.append(path)
        return WriteResult.from_plan(
            plan,
            zone,
            path=path,
            format_name="iceberg",
            mode=mode,
            records_written=None,
        )


@dataclass(slots=True)
class FakeQualityGate:
    calls: list[str]
    report: ValidationReport
    path: Path

    def validate_and_store(self, plan, **kwargs):
        del plan
        del kwargs
        self.calls.append("validate")
        return PersistedValidationReport(report=self.report, path=self.path)


@dataclass(slots=True)
class FakeObserver:
    calls: list[str]
    metadata_root: Path
    seen_strategy_metadata: Any | None = None

    def start_run(self, plan):
        del plan
        self.calls.append("start")
        return SimpleNamespace()

    def record_success(self, plan, extraction_result, write_results=(), **kwargs):
        del plan
        del extraction_result
        del write_results
        self.seen_strategy_metadata = kwargs.get("strategy_metadata")
        self.calls.append("success")
        return SimpleNamespace(
            run_metadata_path=self.metadata_root / "runs" / "run-001.json",
            lineage_path=self.metadata_root / "lineage" / "run-001.json",
            checkpoint_result=SimpleNamespace(
                current_path=self.metadata_root / "checkpoints" / "current.json",
                history_path=self.metadata_root / "checkpoints" / "history" / "run-001.json",
            ),
        )

    def record_failure(self, *args, **kwargs):
        raise AssertionError("record_failure should not be called in the success test")


@dataclass(slots=True)
class FakeFailingObserver(FakeObserver):
    def record_success(self, *args, **kwargs):
        raise AssertionError("record_success should not be called in the failure test")

    def record_failure(self, plan, error, extraction_result=None, write_results=(), **kwargs):
        del plan
        del error
        del extraction_result
        del write_results
        self.seen_strategy_metadata = kwargs.get("strategy_metadata")
        self.calls.append("failure")
        return SimpleNamespace(
            run_metadata_path=self.metadata_root / "runs" / "run-001.json",
            lineage_path=self.metadata_root / "lineage" / "run-001.json",
            checkpoint_result=None,
        )


def test_raw_to_bronze_loader_reuses_existing_modules_and_overrides_target_table(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    raw_root = tmp_path / "data" / "raw" / "example" / "federal_open_data_example"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "page-0001.json").write_text('{"id": 1}\n', encoding="utf-8")

    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"

    loader = RawToBronzeLoader(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(calls),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    result = loader.ingest(
        planned_run,
        spark=object(),
        environment_config={},
        bronze_table="curated.custom_table",
    )

    assert calls == [
        "start",
        "handoff",
        "read",
        "normalize",
        "write",
        "metadata",
        "validate",
        "success",
    ]
    assert result.is_successful is True
    assert result.write_results[-1].path == "curated.custom_table"
    assert result.to_summary()["target_table"] == "curated.custom_table"
    assert result.to_summary()["raw_artifact_count"] == 1


def test_raw_to_bronze_loader_uses_latest_run_scoped_raw_root(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    raw_root = tmp_path / "data" / "raw" / "example" / "federal_open_data_example"
    old_run = raw_root / "runs" / "ingestion_date=2026-04-08" / "run_id=run-old"
    latest_run = raw_root / "runs" / "ingestion_date=2026-04-09" / "run_id=run-new"
    old_run.mkdir(parents=True, exist_ok=True)
    latest_run.mkdir(parents=True, exist_ok=True)
    (old_run / "page-0001.json").write_text('{"id": "old"}\n', encoding="utf-8")
    (latest_run / "page-0001.json").write_text('{"id": "new"}\n', encoding="utf-8")

    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"

    result = RawToBronzeLoader(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(calls),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    ).ingest(
        planned_run,
        spark=object(),
        environment_config={},
        bronze_table="curated.custom_table",
    )

    assert result.is_successful is True
    assert result.extraction_result.artifact_paths == (str(latest_run / "page-0001.json"),)
    assert result.extraction_result.metadata_as_dict()["raw_artifact_root"] == str(latest_run)


def test_artifact_format_inference_uses_file_suffix_before_fallback():
    assert _artifact_format_for_path(Path("page-0001.json"), fallback="csv") == "json"
    assert _artifact_format_for_path(Path("dataset.jsonl"), fallback="json") == "jsonl"
    assert _artifact_format_for_path(Path("download.zip"), fallback="csv") == "binary"
    assert _artifact_format_for_path(Path("data.unknown"), fallback="parquet") == "parquet"


def test_rediscover_raw_artifacts_excludes_checksum_sidecars(tmp_path):
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        source_config,
        outputs=replace(
            source_config.outputs,
            raw=replace(source_config.outputs.raw, path=str(tmp_path / "raw")),
        ),
    )
    run_context = RunContext.create(
        run_id="run-rediscover-sidecar-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
    )
    plan = ExecutionPlan.from_source_config(source_config, run_context)

    raw_root = Path(plan.raw_output.path)
    (raw_root / "pages").mkdir(parents=True, exist_ok=True)
    (raw_root / "pages" / "page-0001.json").write_text('{"page": 1}\n', encoding="utf-8")
    (raw_root / "pages" / "page-0002.json").write_text('{"page": 2}\n', encoding="utf-8")

    before = _rediscover_raw_artifacts(plan)

    # Sidecars added by the writer must not change the rediscovered artifact set.
    (raw_root / "pages" / "page-0001.json.sha256").write_text("deadbeef\n", encoding="utf-8")
    (raw_root / "pages" / "page-0002.json.sha256").write_text("cafef00d\n", encoding="utf-8")

    after = _rediscover_raw_artifacts(plan)

    assert [artifact.path for artifact in before] == [artifact.path for artifact in after]
    assert all(not artifact.path.endswith(".sha256") for artifact in after)


def _rediscovery_plan(tmp_path: Path) -> ExecutionPlan:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        source_config,
        outputs=replace(
            source_config.outputs,
            raw=replace(source_config.outputs.raw, path=str(tmp_path / "raw")),
        ),
    )
    run_context = RunContext.create(
        run_id="run-rediscover-checksum-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source_config, run_context)


def _write_raw_page(pages_dir: Path, name: str, body: str) -> Path:
    pages_dir.mkdir(parents=True, exist_ok=True)
    page = pages_dir / name
    page.write_text(body, encoding="utf-8")
    return page


def _spy_on_sha256(monkeypatch) -> list[Path]:
    hashed: list[Path] = []

    def spy(path: Path) -> str:
        hashed.append(Path(path))
        return _sha256(path)

    monkeypatch.setattr(raw_to_bronze, "_sha256", spy)
    return hashed


def test_rediscover_raw_artifacts_reads_sidecar_without_re_hashing(tmp_path, monkeypatch):
    plan = _rediscovery_plan(tmp_path)
    pages_dir = Path(plan.raw_output.path) / "pages"
    page_one = _write_raw_page(pages_dir, "page-0001.json", '{"page": 1}\n')
    page_two = _write_raw_page(pages_dir, "page-0002.json", '{"page": 2}\n')

    # Sidecars hold arbitrary-but-non-empty digests so a cache hit is observable
    # independently of what re-hashing would produce.
    digest_one = "a" * 64
    digest_two = "b" * 64
    page_one.with_name(page_one.name + SIDECAR_SUFFIX).write_text(
        f"{digest_one}\n", encoding="utf-8"
    )
    page_two.with_name(page_two.name + SIDECAR_SUFFIX).write_text(
        f"{digest_two}\n", encoding="utf-8"
    )

    hashed = _spy_on_sha256(monkeypatch)
    artifacts = _rediscover_raw_artifacts(plan)

    checksums = {artifact.path: artifact.checksum for artifact in artifacts}
    assert checksums[str(page_one)] == digest_one
    assert checksums[str(page_two)] == digest_two
    # sidecar-backed files are never re-hashed (no file-body read).
    assert hashed == []


def test_rediscover_raw_artifacts_checksums_match_direct_sha256(tmp_path):
    plan = _rediscovery_plan(tmp_path)
    pages_dir = Path(plan.raw_output.path) / "pages"
    page_one = _write_raw_page(pages_dir, "page-0001.json", '{"page": 1}\n')
    page_two = _write_raw_page(pages_dir, "page-0002.json", '{"page": 2}\n')

    # Sidecars written exactly as extraction would: the real digest.
    expected = {page_one: _sha256(page_one), page_two: _sha256(page_two)}
    for page, digest in expected.items():
        page.with_name(page.name + SIDECAR_SUFFIX).write_text(f"{digest}\n", encoding="utf-8")

    artifacts = _rediscover_raw_artifacts(plan)

    checksums = {artifact.path: artifact.checksum for artifact in artifacts}
    # the cached digest is byte-identical to the pre-change _sha256 value.
    assert checksums[str(page_one)] == expected[page_one]
    assert checksums[str(page_two)] == expected[page_two]


def test_rediscover_raw_artifacts_falls_back_to_hashing_without_sidecar(tmp_path, monkeypatch):
    plan = _rediscovery_plan(tmp_path)
    pages_dir = Path(plan.raw_output.path) / "pages"
    page_one = _write_raw_page(pages_dir, "page-0001.json", '{"page": 1}\n')
    page_two = _write_raw_page(pages_dir, "page-0002.json", '{"page": 2}\n')

    hashed = _spy_on_sha256(monkeypatch)
    artifacts = _rediscover_raw_artifacts(plan)

    checksums = {artifact.path: artifact.checksum for artifact in artifacts}
    # Legacy raw zone (no sidecars): identical behaviour to pre-change hashing.
    assert checksums[str(page_one)] == _sha256(page_one)
    assert checksums[str(page_two)] == _sha256(page_two)
    assert set(hashed) == {page_one, page_two}


def test_rediscover_raw_artifacts_hashes_only_sidecarless_files(tmp_path, monkeypatch):
    plan = _rediscovery_plan(tmp_path)
    pages_dir = Path(plan.raw_output.path) / "pages"
    cached_page = _write_raw_page(pages_dir, "page-0001.json", '{"page": 1}\n')
    legacy_page = _write_raw_page(pages_dir, "page-0002.json", '{"page": 2}\n')

    cached_digest = _sha256(cached_page)
    cached_page.with_name(cached_page.name + SIDECAR_SUFFIX).write_text(
        f"{cached_digest}\n", encoding="utf-8"
    )

    hashed = _spy_on_sha256(monkeypatch)
    artifacts = _rediscover_raw_artifacts(plan)

    checksums = {artifact.path: artifact.checksum for artifact in artifacts}
    assert checksums[str(cached_page)] == cached_digest
    assert checksums[str(legacy_page)] == _sha256(legacy_page)
    # Only the sidecarless file is re-hashed during rediscovery.
    assert hashed == [legacy_page]


def test_rediscover_raw_artifacts_round_trips_writer_sidecars(tmp_path, monkeypatch):
    plan = _rediscovery_plan(tmp_path)
    storage_layout = _storage_layout(tmp_path)
    writer = RawArtifactWriter(storage_layout)

    persisted_one = writer.write_json(plan, "pages/page-0001.json", {"page": 1})
    persisted_two = writer.write_json(plan, "pages/page-0002.json", {"page": 2})

    raw_root = Path(persisted_one.artifact.path).parents[1]
    rediscovery_plan = replace(
        plan, raw_output=replace(plan.raw_output, path=str(raw_root))
    )

    hashed = _spy_on_sha256(monkeypatch)
    artifacts = _rediscover_raw_artifacts(rediscovery_plan)

    checksums = {artifact.path: artifact.checksum for artifact in artifacts}
    # The rediscovered digest equals the write-time ExtractedArtifact.checksum.
    assert checksums[persisted_one.artifact.path] == persisted_one.artifact.checksum
    assert checksums[persisted_two.artifact.path] == persisted_two.artifact.checksum
    # Writer emitted sidecars, so replay never re-reads the bodies.
    assert hashed == []


def test_rediscover_raw_artifacts_verify_mode_detects_sidecar_mismatch(tmp_path):
    plan = _rediscovery_plan(tmp_path)
    pages_dir = Path(plan.raw_output.path) / "pages"
    page = _write_raw_page(pages_dir, "page-0001.json", '{"page": 1}\n')
    page.with_name(page.name + SIDECAR_SUFFIX).write_text("f" * 64 + "\n", encoding="utf-8")

    # Default path trusts the sidecar even when it disagrees with the body.
    trusted = _rediscover_raw_artifacts(plan)
    assert trusted[0].checksum == "f" * 64

    # Opt-in verification re-hashes and rejects the divergent sidecar.
    with pytest.raises(ValueError, match="Checksum sidecar mismatch"):
        _rediscover_raw_artifacts(plan, verify_checksums=True)


def test_raw_to_bronze_loader_regenerates_catalog_jsonl_handoff_from_raw_pages(tmp_path):
    source_config = load_registry(PROJECT_ROOT).get_source(
        "dados_abertos_catalog__conjunto_dados__full_refresh",
        include_disabled=True,
    )
    source_config = replace(
        source_config,
        outputs=replace(
            source_config.outputs,
            raw=replace(
                source_config.outputs.raw,
                path="data/raw/dados_abertos/catalog",
            ),
            bronze=replace(
                source_config.outputs.bronze,
                path="data/bronze/dados_abertos/catalog",
            ),
            metadata=replace(
                source_config.outputs.metadata,
                path="data/metadata/dados_abertos/catalog",
            ),
        ),
    )
    run_context = RunContext.create(
        run_id="run-catalog-raw-to-bronze-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=CatalogStrategy(),
        hook=None,
    )

    raw_pages_dir = tmp_path / "data" / "raw" / "dados_abertos" / "catalog" / "pages"
    raw_pages_dir.mkdir(parents=True, exist_ok=True)
    (raw_pages_dir / "page-0001.json").write_text(
        (FIXTURES_ROOT / "package_search_page_1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "dados_abertos" / "catalog"
    calls: list[str] = []

    loader = RawToBronzeLoader(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-catalog-raw-to-bronze-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(calls),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    result = loader.ingest(
        planned_run,
        spark=object(),
        environment_config={},
        bronze_table="curated.custom_table",
    )

    assert result.is_successful is True
    assert result.extraction_result.records_extracted == 9
    assert result.handoff.artifacts
    assert all(artifact.format == "jsonl" for artifact in result.handoff.artifacts)
    assert (
        tmp_path / "data" / "raw" / "dados_abertos" / "catalog" / "normalized" / "datasets.jsonl"
    ).exists()


def test_raw_to_bronze_loader_reads_using_handoff_format_instead_of_source_config_format(tmp_path):
    calls: list[str] = []
    source_config = load_registry(PROJECT_ROOT).get_source(
        "receita_federal__cnpj__empresas_full_refresh",
        include_disabled=True,
    )
    run_context = RunContext.create(
        run_id="run-raw-to-bronze-jsonl-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
    )

    class JsonlHandoffStrategy(FakeStrategy):
        def build_normalization_handoff(self, plan, extraction_result, hook=None):
            del plan
            del hook
            self.calls.append("handoff")
            assert extraction_result.artifacts[0].format == "jsonl"
            return extraction_result

    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=JsonlHandoffStrategy(calls),
        hook=None,
    )

    raw_root = tmp_path / "data" / "raw" / "receita_federal" / "cnpj" / "empresas"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "rows.jsonl").write_text('{"id": 1}\n', encoding="utf-8")

    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "receita_federal" / "cnpj" / "empresas"
    reader = FakeReader(calls)

    result = RawToBronzeLoader(
        reader=reader,
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-raw-to-bronze-jsonl-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(
            calls,
            expected_namespace="bronze__receita_federal",
            expected_table_name="cnpj_jsonl",
            result_path="bronze__receita_federal.cnpj_jsonl",
        ),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    ).ingest(
        planned_run,
        spark=object(),
        environment_config={},
        bronze_table="bronze__receita_federal.cnpj_jsonl",
    )

    assert result.is_successful is True
    assert reader.seen_format_name == "jsonl"
    assert reader.seen_schema is None
    assert reader.seen_options is None


def test_raw_to_bronze_loader_rehydrates_file_archive_members_from_raw_downloads(tmp_path):
    source_config = _source_config_with_absolute_schema(
        load_registry(PROJECT_ROOT).get_source(
            "receita_federal__cnpj__empresas_full_refresh",
            include_disabled=True,
        )
    )
    run_context = RunContext.create(
        run_id="run-file-raw-to-bronze-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=FileStrategy(),
        hook=None,
    )

    raw_download_dir = (
        tmp_path
        / "data"
        / "raw"
        / "receita_federal"
        / "cnpj"
        / "empresas"
        / "downloads"
        / "current"
    )
    raw_download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_download_dir / "Empresas0.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "K3241.K03200Y0.D30513.EMPRECSV",
            "12345678;ACME LTDA;2062;49;1000,00;01;\n",
        )

    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "receita_federal" / "cnpj" / "empresas"
    calls: list[str] = []
    reader = FakeReader(calls)

    loader = RawToBronzeLoader(
        reader=reader,
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-file-raw-to-bronze-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: FakeWriter(
            calls,
            expected_namespace="bronze__receita_federal",
            expected_table_name="cnpj_empresas",
            result_path="bronze__receita_federal.cnpj_empresas",
        ),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )

    result = loader.ingest(
        planned_run,
        spark=object(),
        environment_config={},
        bronze_table="bronze__receita_federal.cnpj_empresas",
    )

    extracted_path = (
        tmp_path
        / "data"
        / "raw"
        / "receita_federal"
        / "cnpj"
        / "empresas"
        / "extracted"
        / "current"
        / "Empresas0"
        / "K3241.K03200Y0.D30513.EMPRECSV"
    )

    assert result.is_successful is True
    assert any(artifact.format == "csv" for artifact in result.handoff.artifacts)
    assert reader.seen_schema is not None
    assert reader.seen_schema.fieldNames() == [
        "cnpj_basico",
        "razao_social",
        "natureza_juridica",
        "qualificacao_do_responsavel",
        "capital_social",
        "porte_empresa",
        "ente_federativo_responsavel",
    ]
    assert reader.seen_options == source_config.spark.read_options
    assert extracted_path.exists()
    assert (
        extracted_path.read_text(encoding="utf-8")
        == "12345678;ACME LTDA;2062;49;1000,00;01;\n"
    )
    assert calls == [
        "start",
        "read",
        "normalize",
        "write",
        "validate",
        "success",
    ]


def test_file_raw_to_bronze_writes_handoff_artifacts_in_append_batches(tmp_path):
    source_config = _source_config_with_absolute_schema(
        load_registry(PROJECT_ROOT).get_source(
            "receita_federal__cnpj__empresas_full_refresh",
            include_disabled=True,
        )
    )
    run_context = RunContext.create(
        run_id="run-file-raw-to-bronze-batches-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
    )
    planned_run = PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=FileStrategy(),
        hook=None,
    )

    raw_root = tmp_path / "data" / "raw" / "receita_federal" / "cnpj" / "empresas"
    raw_root.mkdir(parents=True, exist_ok=True)
    for part_index in range(6):
        (raw_root / f"part-{part_index:03d}.csv").write_text(
            f"1234567{part_index};EMPRESA {part_index:02d} LTDA;2062;49;1000,00;01;\n",
            encoding="utf-8",
        )

    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "receita_federal" / "cnpj" / "empresas"
    calls: list[str] = []
    reader = FakeReader(calls)
    writer = RecordingWriter(modes=[], paths=[])

    result = RawToBronzeLoader(
        reader=reader,
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-file-raw-to-bronze-batches-001.json",
        ),
        observer=FakeObserver(calls, metadata_root),
        writer_factory=lambda storage_layout: writer,
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    ).ingest(
        planned_run,
        spark=object(),
        environment_config={},
        bronze_table="bronze__receita_federal.cnpj_empresas",
    )

    assert result.is_successful is True
    assert reader.read_artifact_counts == [5, 1]
    assert writer.modes == ["overwrite", "append"]
    assert [write.mode for write in result.write_results if write.zone == "bronze"] == [
        "overwrite",
        "append",
    ]


def test_raw_to_bronze_loader_forwards_strategy_metadata_to_observer_on_success(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    raw_root = tmp_path / "data" / "raw" / "example" / "federal_open_data_example"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "page-0001.json").write_text('{"id": 1}\n', encoding="utf-8")

    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.passed("output", "materialized_outputs", "ok")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    observer = FakeObserver(calls, metadata_root)

    result = RawToBronzeLoader(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=observer,
        writer_factory=lambda storage_layout: FakeWriter(calls),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    ).ingest(
        planned_run,
        spark=object(),
        environment_config={},
        bronze_table="curated.custom_table",
    )

    assert result.is_successful is True
    assert observer.seen_strategy_metadata == {"write_result_count": 2}


def test_raw_to_bronze_loader_forwards_strategy_metadata_on_quality_failure(tmp_path):
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    raw_root = tmp_path / "data" / "raw" / "example" / "federal_open_data_example"
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "page-0001.json").write_text('{"id": 1}\n', encoding="utf-8")

    validation_report = ValidationReport.from_plan(
        planned_run.plan,
        [ValidationCheck.failed("data", "required_fields", "missing fields")],
    )
    metadata_root = tmp_path / "data" / "metadata" / "example" / "federal_open_data_example"
    observer = FakeFailingObserver(calls, metadata_root)

    result = RawToBronzeLoader(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(
            calls,
            validation_report,
            metadata_root / "validations" / "run-001.json",
        ),
        observer=observer,
        writer_factory=lambda storage_layout: FakeWriter(calls),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    ).ingest(
        planned_run,
        spark=object(),
        environment_config={},
        bronze_table="curated.custom_table",
    )

    assert result.is_successful is False
    assert calls[-1] == "failure"
    assert observer.seen_strategy_metadata == {"write_result_count": 2}


def _planned_run(tmp_path: Path, calls: list[str]) -> PlannedRun:
    source_config = load_registry(PROJECT_ROOT).get_source("federal_open_data_example")
    source_config = replace(
        source_config,
        outputs=replace(
            source_config.outputs,
            raw=replace(
                source_config.outputs.raw,
                path="data/raw/example/federal_open_data_example",
            ),
            bronze=replace(
                source_config.outputs.bronze,
                path="data/bronze/example/federal_open_data_example",
            ),
            metadata=replace(
                source_config.outputs.metadata,
                path="data/metadata/example/federal_open_data_example",
            ),
        ),
    )
    run_context = RunContext.create(
        run_id="run-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )
    return PlannedRun(
        plan=ExecutionPlan.from_source_config(source_config, run_context),
        strategy=FakeStrategy(calls),
        hook=None,
    )


def _source_config_with_absolute_schema(source_config):
    if source_config.schema.path is None:
        return source_config
    return replace(
        source_config,
        schema=replace(
            source_config.schema,
            path=str(PROJECT_ROOT / source_config.schema.path),
        ),
    )


def _storage_layout(tmp_path: Path) -> StorageLayout:
    return StorageLayout.from_environment_config(
        {
            "storage": {
                "root_dir": str(tmp_path / "data"),
                "raw_dir": str(tmp_path / "data" / "raw"),
                "bronze_dir": str(tmp_path / "data" / "bronze"),
                "metadata_dir": str(tmp_path / "data" / "metadata"),
            }
        },
        tmp_path,
    )
