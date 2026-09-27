"""A corrupt resume record is a reason to stop, not a path to follow."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from janus.checkpoints import ExtractionProgressStore
from janus.checkpoints.progress import ProgressRecordError
from janus.models import ExecutionPlan, RunContext, SourceConfig
from janus.strategies.api import ApiResponse, ApiStrategy
from janus.strategies.api.artifacts import _pages_dir, _rediscover_raw_artifacts
from janus.strategies.catalog import CatalogStrategy
from janus.strategies.common import _raw_run_path_prefix
from janus.utils.storage import (
    StorageLayout,
    _normalize_relative_path,
    normalize_relative_path,
)
from janus.writers import RawArtifactWriter

TAMPERED_PREFIXES = ("/etc", "../../x", "runs/../..", " ")

LEGITIMATE_PREFIX = "runs/ingestion_date=2026-09-20/run_id=run-interrupted"


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


def _api_source_config(tmp_path: Path, *, source_id: str) -> SourceConfig:
    access: dict[str, Any] = {
        "base_url": "https://api.example.gov.br",
        "path": "/v1/records",
        "method": "GET",
        "format": "json",
        "timeout_seconds": 30,
        "auth": {"type": "none"},
        "pagination": {
            "type": "page_number",
            "page_param": "pagina",
            "size_param": "tamanhoPagina",
            "page_size": 100,
        },
        "rate_limit": {"requests_per_minute": None, "concurrency": 1, "backoff_seconds": 1},
    }
    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "api",
            "strategy": "api",
            "strategy_variant": "page_number_api",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": access,
            "extraction": {
                "mode": "full_refresh",
                "checkpoint_strategy": "none",
                "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "json", "write_mode": "append"},
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _catalog_source_config(tmp_path: Path, *, source_id: str) -> SourceConfig:
    return SourceConfig.from_mapping(
        {
            "source_id": source_id,
            "name": source_id,
            "owner": "janus",
            "enabled": True,
            "source_type": "catalog",
            "strategy": "catalog",
            "strategy_variant": "metadata_catalog",
            "federation_level": "federal",
            "domain": "example",
            "public_access": True,
            "access": {
                "base_url": "https://dados.gov.br",
                "path": "/api/3/conjuntos",
                "method": "GET",
                "format": "json",
                "timeout_seconds": 30,
                "auth": {"type": "none"},
                "pagination": {
                    "type": "page_number",
                    "page_param": "pagina",
                    "size_param": "tamanhoPagina",
                    "page_size": 100,
                },
                "rate_limit": {
                    "requests_per_minute": None,
                    "concurrency": 1,
                    "backoff_seconds": 1,
                },
            },
            "extraction": {
                "mode": "full_refresh",
                "checkpoint_strategy": "none",
                "retry": {"max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1},
            },
            "schema": {"mode": "infer"},
            "spark": {"input_format": "jsonl", "write_mode": "append"},
            "outputs": {
                "raw": {"path": f"data/raw/example/{source_id}", "format": "json"},
                "bronze": {"path": f"data/bronze/example/{source_id}", "format": "iceberg"},
                "metadata": {"path": f"data/metadata/example/{source_id}", "format": "json"},
            },
            "quality": {"allow_schema_evolution": True},
        },
        tmp_path / "conf" / "sources" / f"{source_id}.yaml",
    )


def _plan(
    tmp_path: Path, source_config: SourceConfig, *, run_id: str, resume: bool
) -> ExecutionPlan:
    return ExecutionPlan.from_source_config(
        source_config,
        RunContext.create(
            run_id=run_id,
            environment="local",
            project_root=tmp_path,
            started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
            attributes={"resume": "true"} if resume else {},
        ),
    )


class FakeTransport:
    """Refuses every request: a run that reaches the transport has already failed the test."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def send(self, request):
        self.requests.append(request)
        return ApiResponse(request=request, status_code=200, body=b'{"records": []}')


def _seed_progress(plan: ExecutionPlan, raw_path_prefix: str | None) -> Path:
    """Write a progress record carrying ``raw_path_prefix`` verbatim, blanks included.

    ``ExtractionProgressStore.save`` strips and omits a blank value, so the tampered record
    is patched onto disk afterwards — which is the only way such a record could appear.
    """
    path = ExtractionProgressStore().save(
        plan,
        page_number=9686,
        artifact_count=9686,
        current_input_key="__none__",
        current_input_index=1,
        request_input_count=1,
        raw_path_prefix=LEGITIMATE_PREFIX,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if raw_path_prefix is None:
        payload.pop("raw_path_prefix", None)
    else:
        payload["raw_path_prefix"] = raw_path_prefix
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _progress_mapping(raw_path_prefix: str) -> dict[str, Any]:
    return {"source_id": "any", "raw_path_prefix": raw_path_prefix}


# ---------------------------------------------------------------------------
# FR-7 — the resolver itself


@pytest.mark.parametrize("prefix", TAMPERED_PREFIXES, ids=[repr(p) for p in TAMPERED_PREFIXES])
def test_a_tampered_prefix_is_refused_by_the_resolver(tmp_path, prefix):
    """The one function every consumer goes through, so the rule cannot be applied in only some."""
    plan = _plan(
        tmp_path,
        _api_source_config(tmp_path, source_id="prefix_resolver"),
        run_id="run-resume",
        resume=True,
    )

    with pytest.raises(ProgressRecordError) as excinfo:
        _raw_run_path_prefix(plan, _progress_mapping(prefix))

    message = str(excinfo.value)
    assert repr(prefix) in message or prefix.strip() in message, (
        f"the error must name the offending value; got {message!r}"
    )
    assert "extraction_progress.json" in message, (
        f"the error must name the file to inspect; got {message!r}"
    )


def test_a_legitimate_prefix_resolves_verbatim(tmp_path):
    """Green on arrival: a pin, not a red test."""
    plan = _plan(
        tmp_path,
        _api_source_config(tmp_path, source_id="prefix_legit"),
        run_id="run-resume",
        resume=True,
    )

    assert _raw_run_path_prefix(plan, _progress_mapping(LEGITIMATE_PREFIX)) == Path(
        LEGITIMATE_PREFIX
    )


def test_a_record_without_the_field_still_resolves_the_legacy_flat_layout(tmp_path):
    """Green on arrival: a pin, not a red test.

    ``None`` means the pre-prefix flat layout and stays supported — the contract
    ``test_resume_state_survival.py`` pins, restated here so the new rule cannot swallow it.
    """
    plan = _plan(
        tmp_path,
        _api_source_config(tmp_path, source_id="prefix_legacy"),
        run_id="run-resume",
        resume=True,
    )

    assert _raw_run_path_prefix(plan, {"source_id": "any"}) is None


# ---------------------------------------------------------------------------
# FR-7 — refuse to resume, and leave the record alone


@pytest.mark.parametrize("prefix", TAMPERED_PREFIXES, ids=[repr(p) for p in TAMPERED_PREFIXES])
def test_an_api_run_refuses_to_resume_and_leaves_the_record_in_place(tmp_path, prefix):
    """Refusing beats starting over: the position a 9,686-page run reached must survive."""
    source_config = _api_source_config(tmp_path, source_id="api_refuses_resume")
    seed_plan = _plan(tmp_path, source_config, run_id="run-interrupted", resume=False)
    progress_path = _seed_progress(seed_plan, prefix)

    resume_plan = _plan(tmp_path, source_config, run_id="run-resume", resume=True)
    transport = FakeTransport()
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )

    with pytest.raises(ProgressRecordError):
        strategy.extract(resume_plan)

    assert progress_path.exists(), "the tampered record was deleted instead of kept for inspection"
    assert json.loads(progress_path.read_text(encoding="utf-8"))["raw_path_prefix"] == prefix
    raw_root = Path(_storage_layout(tmp_path).resolve_output(resume_plan, "raw").resolved_path)
    assert not raw_root.exists() or not any(path.is_file() for path in raw_root.rglob("*"))


def test_a_catalog_run_refuses_to_resume_too(tmp_path):
    """The rule lives in ``strategies/common.py``, so both families inherit it.

    Proven rather than assumed: the api case above and this one are the two call sites.
    """
    source_config = _catalog_source_config(tmp_path, source_id="catalog_refuses_resume")
    seed_plan = _plan(tmp_path, source_config, run_id="run-interrupted", resume=False)
    _seed_progress(seed_plan, "/etc")

    resume_plan = _plan(tmp_path, source_config, run_id="run-resume", resume=True)
    strategy = CatalogStrategy(
        transport_factory=FakeTransport,
        storage_layout_factory=lambda plan: _storage_layout(tmp_path),
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )

    with pytest.raises(ProgressRecordError):
        strategy.extract(resume_plan)


def test_rediscovery_never_globs_outside_the_raw_zone(tmp_path):
    """The escape with teeth: a flat-layout raw zone rehydrated into an unrelated run."""
    source_config = _api_source_config(tmp_path, source_id="rediscovery_contained")
    plan = _plan(tmp_path, source_config, run_id="run-resume", resume=True)

    with pytest.raises(ProgressRecordError):
        _rediscover_raw_artifacts(
            plan,
            _storage_layout(tmp_path),
            {
                "source_id": "rediscovery_contained",
                "raw_path_prefix": "../../..",
                "last_page_number": 1,
            },
        )


# ---------------------------------------------------------------------------
# FR-7 — defence in depth: no caller can bypass the rule


@pytest.mark.parametrize("prefix", ("/abs", "../x"))
def test_the_raw_writer_refuses_an_uncontained_prefix(tmp_path, prefix):
    """A future caller constructing the writer directly must not reopen the hole."""
    with pytest.raises(ValueError):
        RawArtifactWriter(_storage_layout(tmp_path), raw_path_prefix=prefix)


@pytest.mark.parametrize("prefix", ("/abs", "../x"))
def test_the_pages_directory_refuses_an_uncontained_prefix(tmp_path, prefix):
    """Same rule at the read side: ``_pages_dir`` is where rediscovery resolves its directory."""
    plan = _plan(
        tmp_path,
        _api_source_config(tmp_path, source_id="pages_dir_contained"),
        run_id="run-resume",
        resume=True,
    )

    with pytest.raises(ValueError):
        _pages_dir(plan, _storage_layout(tmp_path), 1, 1, Path(prefix))


def test_normalize_relative_path_is_the_one_public_spelling():
    """FR-7 promotes the private helper; the alias stays for one order and is documented as such."""
    assert normalize_relative_path is _normalize_relative_path
    assert normalize_relative_path("runs/ingestion_date=2026-09-20") == Path(
        "runs/ingestion_date=2026-09-20"
    )
    for bad in ("/etc", "../x"):
        with pytest.raises(ValueError):
            normalize_relative_path(bad)
