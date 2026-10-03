"""`--verify-checksums`: replay refuses a changed raw zone."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

# The scripts suite is not a package (no ``__init__.py``), so pytest puts this directory
# on ``sys.path`` and the sibling module is imported by plain name.
from test_raw_to_bronze import (
    FIXTURES_ROOT,
    PROJECT_ROOT,
    FakeFailingObserver,
    FakeNormalizer,
    FakeObserver,
    FakeQualityGate,
    FakeReader,
    FakeWriter,
    _planned_run,
    _rediscovery_plan,
    _storage_layout,
    _with_contract,
    _write_raw_page,
)

import janus.writers.sidecar as sidecar
from janus.models import ExecutionPlan, RunContext
from janus.planner import PlannedRun
from janus.quality import ValidationCheck, ValidationReport
from janus.registry import load_registry
from janus.runtime import SparkSessionProvider
from janus.scripts.raw_to_bronze import RawToBronzeLoader, _rediscover_raw_artifacts
from janus.strategies.catalog import CatalogStrategy
from janus.writers import SIDECAR_SUFFIX
from tests.support.operator_cli import run_janus

BRONZE_TABLE = "curated.custom_table"
EXAMPLE_RAW = Path("data") / "raw" / "example" / "federal_open_data_example"
EXAMPLE_METADATA = Path("data") / "metadata" / "example" / "federal_open_data_example"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_sidecar(artifact: Path, digest: str) -> None:
    artifact.with_name(artifact.name + SIDECAR_SUFFIX).write_text(f"{digest}\n", encoding="utf-8")


@dataclass(slots=True)
class _CapturingObserver(FakeObserver):
    seen_extraction_metadata: dict[str, str] | None = None
    seen_run_attributes: dict[str, str] | None = None

    def record_success(self, plan, extraction_result, write_results=(), **kwargs):
        self.seen_extraction_metadata = extraction_result.metadata_as_dict()
        self.seen_run_attributes = plan.run_context.attributes_as_dict()
        return FakeObserver.record_success(self, plan, extraction_result, write_results, **kwargs)


def _session_refusing_provider(calls: list[str]) -> SparkSessionProvider:
    """The spy: a replay that reaches for Spark records it, and the test asserts it did not."""

    def factory() -> Any:
        calls.append("spark_started")
        pytest.fail("a refused raw zone must never reach the materialization boundary")

    return SparkSessionProvider({}, {}, session_factory=factory)


def _loader(tmp_path: Path, calls: list[str], observer: FakeObserver, plan: ExecutionPlan):
    report = ValidationReport.from_plan(
        plan, [ValidationCheck.passed("output", "materialized_outputs", "ok")]
    )
    metadata_root = tmp_path / EXAMPLE_METADATA
    return RawToBronzeLoader(
        reader=FakeReader(calls),
        normalizer=FakeNormalizer(calls),
        quality_gate=FakeQualityGate(calls, report, metadata_root / "validations" / "run.json"),
        observer=observer,
        writer_factory=lambda storage_layout: FakeWriter(calls),
        storage_layout_resolver=lambda plan, config: _storage_layout(tmp_path),
    )


def _replayable_page(tmp_path: Path, *, sidecar: str | None) -> Path:
    raw_root = tmp_path / EXAMPLE_RAW
    raw_root.mkdir(parents=True, exist_ok=True)
    page = raw_root / "page-0001.json"
    page.write_text('{"id": 1}\n', encoding="utf-8")
    if sidecar is not None:
        _write_sidecar(page, sidecar)
    return page


class _SessionStub:
    def table(self, identifier: str) -> Any:
        del identifier
        return object()

    def stop(self) -> None:
        return None


# ---------------------------------------------------------------------------------------
# The error


def test_the_integrity_error_names_the_path_the_sidecar_digest_and_the_computed_digest(
    tmp_path: Path,
) -> None:
    from janus.scripts.checksums import RawArtifactIntegrityError, _resolve_raw_checksum

    page = _write_raw_page(tmp_path / "pages", "page-0001.json", '{"page": 1}\n')
    _write_sidecar(page, "f" * 64)

    with pytest.raises(RawArtifactIntegrityError) as refused:
        _resolve_raw_checksum(page, verify=True)

    assert isinstance(refused.value, ValueError), "D-17: no new failure shape"
    message = str(refused.value)
    assert str(page) in message
    assert "f" * 64 in message
    assert _digest(page) in message


def test_rediscovery_names_the_first_mismatching_sidecar_in_sorted_order(tmp_path: Path) -> None:
    from janus.scripts.checksums import RawArtifactIntegrityError

    plan = _rediscovery_plan(tmp_path)
    pages = Path(plan.raw_output.path) / "pages"
    first = _write_raw_page(pages, "page-0001.json", '{"page": 1}\n')
    second = _write_raw_page(pages, "page-0002.json", '{"page": 2}\n')
    _write_sidecar(second, "e" * 64)
    _write_sidecar(first, "f" * 64)

    with pytest.raises(RawArtifactIntegrityError) as refused:
        _rediscover_raw_artifacts(plan, verify_checksums=True)

    assert str(first) in str(refused.value)
    assert str(second) not in str(refused.value)


# ---------------------------------------------------------------------------------------
# Through the replay loader


def test_a_tampered_artifact_fails_the_replay_before_any_spark_work(tmp_path: Path) -> None:
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    _replayable_page(tmp_path, sidecar="f" * 64)
    observer = FakeFailingObserver(calls, tmp_path / EXAMPLE_METADATA)

    result = _loader(tmp_path, calls, observer, planned_run.plan).ingest(
        planned_run,
        spark=_session_refusing_provider(calls),
        environment_config={},
        bronze_table=BRONZE_TABLE,
        verify_checksums=True,
    )

    assert result.is_successful is False
    assert result.error_type == "RawArtifactIntegrityError"
    assert "spark_started" not in calls
    assert "write" not in calls
    assert calls[-1] == "failure"


def test_a_verified_replay_records_checksums_verified(tmp_path: Path) -> None:
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    page = _replayable_page(tmp_path, sidecar=None)
    _write_sidecar(page, _digest(page))
    observer = _CapturingObserver(calls, tmp_path / EXAMPLE_METADATA)

    result = _loader(tmp_path, calls, observer, planned_run.plan).ingest(
        planned_run,
        spark=_SessionStub(),
        environment_config={},
        bronze_table=BRONZE_TABLE,
        verify_checksums=True,
    )

    assert result.is_successful is True
    assert result.extraction_result.metadata_as_dict()["checksums_verified"] == "true"
    assert observer.seen_extraction_metadata["checksums_verified"] == "true"
    # Run-metadata JSON is written from the plan's run attributes, not extraction metadata.
    assert observer.seen_run_attributes["checksums_verified"] == "true"
    assert '"checksums_verified": "true"' in json.dumps(result.to_summary(), sort_keys=True)


def test_an_artifact_without_a_sidecar_still_replays_under_verification(tmp_path: Path) -> None:
    """A legacy zone has no recorded digest to verify against; it keeps replaying."""
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    _replayable_page(tmp_path, sidecar=None)
    observer = FakeObserver(calls, tmp_path / EXAMPLE_METADATA)

    result = _loader(tmp_path, calls, observer, planned_run.plan).ingest(
        planned_run,
        spark=_SessionStub(),
        environment_config={},
        bronze_table=BRONZE_TABLE,
        verify_checksums=True,
    )

    assert result.is_successful is True


def test_without_the_flag_replay_trusts_the_sidecar_and_records_nothing(tmp_path: Path) -> None:
    calls: list[str] = []
    planned_run = _planned_run(tmp_path, calls)
    _replayable_page(tmp_path, sidecar="f" * 64)
    observer = _CapturingObserver(calls, tmp_path / EXAMPLE_METADATA)

    result = _loader(tmp_path, calls, observer, planned_run.plan).ingest(
        planned_run, spark=_SessionStub(), environment_config={}, bronze_table=BRONZE_TABLE
    )

    assert result.is_successful is True
    assert result.extraction_result.artifacts[0].checksum == "f" * 64
    assert "checksums_verified" not in result.extraction_result.metadata_as_dict()
    assert "checksums_verified" not in observer.seen_extraction_metadata
    assert "checksums_verified" not in observer.seen_run_attributes
    assert "checksums_verified" not in result.to_summary()


def _catalog_planned_run(tmp_path: Path) -> PlannedRun:
    source_config = load_registry(PROJECT_ROOT).get_source(
        "dados_abertos_catalog__conjunto_dados__full_refresh", include_disabled=True
    )
    outputs = source_config.outputs
    source_config = replace(
        source_config,
        outputs=replace(
            outputs,
            raw=replace(outputs.raw, path="data/raw/dados_abertos/catalog"),
            bronze=replace(outputs.bronze, path="data/bronze/dados_abertos/catalog"),
            metadata=replace(outputs.metadata, path="data/metadata/dados_abertos/catalog"),
        ),
    )
    run_context = RunContext.create(
        run_id="run-catalog-verify-001",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
    )
    return PlannedRun(
        plan=_with_contract(ExecutionPlan.from_source_config(source_config, run_context)),
        strategy=CatalogStrategy(),
        hook=None,
    )


def _catalog_page(tmp_path: Path) -> Path:
    return _write_raw_page(
        tmp_path / "data" / "raw" / "dados_abertos" / "catalog" / "pages",
        "page-0001.json",
        (FIXTURES_ROOT / "package_search_page_1.json").read_text(encoding="utf-8"),
    )


def _spy_on_hashing(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    hashed: list[Path] = []
    real_sha256 = sidecar._sha256

    def spy(path: Path) -> str:
        hashed.append(Path(path))
        return real_sha256(path)

    monkeypatch.setattr(sidecar, "_sha256", spy)
    return hashed


def test_the_catalog_family_honours_the_flag(tmp_path: Path) -> None:
    planned_run = _catalog_planned_run(tmp_path)
    _write_sidecar(_catalog_page(tmp_path), "f" * 64)
    calls: list[str] = []
    observer = FakeFailingObserver(calls, tmp_path / "data" / "metadata" / "dados_abertos")

    result = _loader(tmp_path, calls, observer, planned_run.plan).ingest(
        planned_run,
        spark=_session_refusing_provider(calls),
        environment_config={},
        bronze_table=BRONZE_TABLE,
        verify_checksums=True,
    )

    assert result.is_successful is False
    assert result.error_type == "RawArtifactIntegrityError"
    assert "spark_started" not in calls


@pytest.mark.parametrize(
    "verify_checksums", [False, True], ids=["unverified-reads-the-sidecar", "verified-once"]
)
def test_a_catalog_replay_takes_its_digests_from_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verify_checksums: bool
) -> None:

    planned_run = _catalog_planned_run(tmp_path)
    page = _catalog_page(tmp_path)
    recorded = _digest(page) if verify_checksums else "a" * 64
    _write_sidecar(page, recorded)
    calls: list[str] = []
    observer = _CapturingObserver(calls, tmp_path / "data" / "metadata" / "dados_abertos")
    hashed = _spy_on_hashing(monkeypatch)

    result = _loader(tmp_path, calls, observer, planned_run.plan).ingest(
        planned_run,
        spark=_SessionStub(),
        environment_config={},
        bronze_table=BRONZE_TABLE,
        verify_checksums=verify_checksums,
    )

    assert result.is_successful is True, result.failure_reason
    assert hashed == ([page] if verify_checksums else [])
    raw_pages = [a for a in result.extraction_result.artifacts if a.path == str(page)]
    assert [artifact.checksum for artifact in raw_pages] == [recorded]
    metadata = result.extraction_result.metadata_as_dict()
    assert (metadata.get("checksums_verified") == "true") is verify_checksums


# ---------------------------------------------------------------------------------------
# The CLI flag


def test_verify_checksums_without_ingest_raw_to_bronze_is_an_argument_error() -> None:
    result = run_janus(("--environment", "local", "--verify-checksums"))

    assert result.exit_code == 2
    assert "--verify-checksums requires --ingest-raw-to-bronze" in result.stderr


@pytest.mark.parametrize("flag", [(), ("--verify-checksums",)], ids=["without", "with"])
def test_the_flag_reaches_the_replay_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: tuple[str, ...]
) -> None:
    shutil.copytree(
        PROJECT_ROOT / "conf", tmp_path / "conf", ignore=shutil.ignore_patterns("*.env")
    )
    seen: dict[str, Any] = {}

    class _ReplayedRun:
        is_successful = True

        def to_summary(self) -> dict[str, Any]:
            return {"status": "succeeded"}

    def loader_spy(planned_run, spark_provider, environment_config, **kwargs):
        del planned_run, spark_provider, environment_config
        seen.update(kwargs)
        return _ReplayedRun()

    monkeypatch.setattr("janus.cli.run.ingest_raw_to_bronze", loader_spy)

    result = run_janus(
        (
            "--environment",
            "local",
            "--project-root",
            str(tmp_path),
            "--source-id",
            "ibge_pib_brasil",
            "--include-disabled",
            "--ingest-raw-to-bronze",
            "--bronze-table",
            BRONZE_TABLE,
            *flag,
        )
    )

    assert result.exit_code == 0, result.stderr
    assert seen["verify_checksums"] is bool(flag)
