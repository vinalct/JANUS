"""`--verify-checksums`: replay refuses a changed raw zone."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

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

from janus.models import ExecutionPlan, RunContext
from janus.planner import PlannedRun
from janus.quality import ValidationCheck, ValidationReport
from janus.registry import load_registry
from janus.runtime import SparkSessionProvider
from janus.scripts.raw_to_bronze import RawToBronzeLoader, _rediscover_raw_artifacts
from janus.strategies.catalog import CatalogStrategy
from janus.writers import SIDECAR_SUFFIX
from tests.support.operator_cli import run_janus

RED = pytest.mark.xfail(
    strict=True,
    reason="RawArtifactIntegrityError and the verify_checksums keyword on "
    "RawToBronzeLoader.ingest / --verify-checksums do not exist yet",
)

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

    def record_success(self, plan, extraction_result, write_results=(), **kwargs):
        self.seen_extraction_metadata = extraction_result.metadata_as_dict()
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


@RED
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


@RED
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


@RED
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


@RED
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
    assert '"checksums_verified": "true"' in json.dumps(result.to_summary(), sort_keys=True)


@RED
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


@RED
def test_the_catalog_family_honours_the_flag(tmp_path: Path) -> None:
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
    planned_run = PlannedRun(
        plan=_with_contract(ExecutionPlan.from_source_config(source_config, run_context)),
        strategy=CatalogStrategy(),
        hook=None,
    )
    page = _write_raw_page(
        tmp_path / "data" / "raw" / "dados_abertos" / "catalog" / "pages",
        "page-0001.json",
        (FIXTURES_ROOT / "package_search_page_1.json").read_text(encoding="utf-8"),
    )
    _write_sidecar(page, "f" * 64)
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


# ---------------------------------------------------------------------------------------
# The CLI flag


@RED
def test_verify_checksums_without_ingest_raw_to_bronze_is_an_argument_error() -> None:
    result = run_janus(("--environment", "local", "--verify-checksums"))

    assert result.exit_code == 2
    assert "--verify-checksums requires --ingest-raw-to-bronze" in result.stderr
