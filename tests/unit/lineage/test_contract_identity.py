"""Contract identity is copied from the planner snapshot into run records."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from janus.lineage import (
    LineageRecord,
    RunMetadata,
    compute_schema_version,
)
from janus.models import ExecutionPlan, RunContext
from janus.models.data_contracts import load_data_contract
from janus.registry import load_registry
from tests.support.contracts import MINIMAL_CONTRACT_FIXTURE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SHARED_CONTRACT_SOURCES = (
    "transparencia__gastos_cartoes__cartoes__full_refresh",
    "transparencia__gastos_cartoes__cartoes__incremental",
)


def _plan(source_id: str = "federal_open_data_example") -> ExecutionPlan:
    source = load_registry(PROJECT_ROOT).get_source(source_id, include_disabled=True)
    run_context = RunContext.create(
        run_id=f"contract-identity-{source_id}",
        environment="local",
        project_root=PROJECT_ROOT,
        started_at=datetime(2026, 9, 23, tzinfo=UTC),
    )
    return ExecutionPlan.from_source_config(source, run_context)


def _records(plan: ExecutionPlan) -> tuple[RunMetadata, LineageRecord]:
    return (
        RunMetadata.started(plan),
        LineageRecord.from_runtime(plan, status="succeeded"),
    )


def test_real_contract_identity_is_copied_without_rereading_contract_bytes(tmp_path, monkeypatch):
    contract_path = tmp_path / "minimal_contract.yaml"
    contract_path.write_bytes(MINIMAL_CONTRACT_FIXTURE.read_bytes())
    contract = load_data_contract(contract_path)
    plan = _plan().with_data_contract(contract)
    expected_hash = sha256(contract_path.read_bytes()).hexdigest()
    assert compute_schema_version(contract_path) == expected_hash

    original_read_bytes = Path.read_bytes
    contract_reads: list[Path] = []

    def observe_reads(path: Path) -> bytes:
        if path.resolve() == contract_path.resolve():
            contract_reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observe_reads)
    run_metadata, lineage = _records(plan)

    for record in (run_metadata, lineage):
        payload = record.to_dict()
        assert payload["schema_version"] == expected_hash
        assert payload["contract_id"] == contract.id
        assert payload["contract_version"] == contract.version
    assert (
        run_metadata.schema_version == lineage.schema_version == plan.data_contract.schema_version
    )
    assert run_metadata.contract_id == lineage.contract_id == plan.data_contract.id
    assert run_metadata.contract_version == lineage.contract_version == plan.data_contract.version
    assert contract_reads == []


def test_every_checked_in_source_records_a_contract_version():
    """With the legacy conversion gone, no new record can carry an unversioned contract."""
    registry = load_registry(PROJECT_ROOT)
    sources = registry.list_sources(enabled_only=False)
    assert sources

    for source in sources:
        context = RunContext.create(
            run_id=f"contract-identity-{source.source_id}",
            environment="local",
            project_root=PROJECT_ROOT,
            started_at=datetime(2026, 9, 23, tzinfo=UTC),
        )
        plan = ExecutionPlan.from_source_config(
            source, context, data_contract=registry.contract_for(source.source_id)
        )
        for record in _records(plan):
            assert record.contract_version is not None
            assert record.to_dict()["contract_version"] == plan.data_contract.version


def test_no_contract_identity_is_omitted_from_both_serialized_records():
    run_metadata, lineage = _records(_plan())
    identity_fields = {"schema_version", "contract_id", "contract_version"}

    for record in (run_metadata, lineage):
        assert all(getattr(record, field) is None for field in identity_fields)
        assert identity_fields.isdisjoint(record.to_dict())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "A" * 64),
        ("schema_version", "not-a-digest"),
        ("contract_id", "  "),
        ("contract_version", "1.0"),
        ("contract_version", "v1.0.0"),
    ],
)
@pytest.mark.parametrize("record_type", (RunMetadata, LineageRecord))
def test_malformed_identity_values_are_rejected(field, value, record_type):
    plan = _plan().with_data_contract(load_data_contract(MINIMAL_CONTRACT_FIXTURE))
    run_metadata, lineage = _records(plan)
    record = run_metadata if record_type is RunMetadata else lineage

    with pytest.raises(ValueError):
        replace(record, **{field: value})


def test_sources_sharing_gastos_contract_keep_the_same_identity():
    registry = load_registry(PROJECT_ROOT)
    records = []
    for source_id in _SHARED_CONTRACT_SOURCES:
        source = registry.get_source(source_id, include_disabled=True)
        context = RunContext.create(
            run_id=f"contract-identity-{source_id}",
            environment="local",
            project_root=PROJECT_ROOT,
            started_at=datetime(2026, 9, 23, tzinfo=UTC),
        )
        plan = ExecutionPlan.from_source_config(
            source, context, data_contract=registry.contract_for(source_id)
        )
        run_metadata, lineage = _records(plan)
        records.extend((run_metadata, lineage))

    assert len({record.schema_version for record in records}) == 1
    assert len({record.contract_id for record in records}) == 1
