
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

# The registry suite is not a package (no ``__init__.py``), so pytest puts this directory
# on ``sys.path`` and the sibling module is imported by plain name.
from test_source_registry import (
    _create_project,
    _grouped_sources_yaml,
    _valid_source_yaml,
)

from janus.models.data_contracts import DataContract
from janus.models.source_config import SourceConfigValidationError
from janus.planner import Planner, PlanningRequest
from janus.registry import load_registry
from tests.support.contracts import DECLARED_CONTRACT_PATH, minimal_contract_yaml

LEGACY_SCHEMA_PATH = "conf/schemas/example/legacy_schema.json"
LEGACY_SCHEMA = {
    "type": "struct",
    "fields": [
        {"name": "id", "type": "string", "nullable": False, "metadata": {}},
        {"name": "amount", "type": "long", "nullable": True, "metadata": {}},
    ],
}


def _declaring(source_id: str, schema_block: str, **kwargs: object) -> str:
    """Render a valid source whose ``schema`` block is replaced wholesale."""
    return _valid_source_yaml(source_id, enabled=True, **kwargs).replace(  # type: ignore[arg-type]
        f"schema:\n  contract: {DECLARED_CONTRACT_PATH}\n",
        schema_block,
    )


def _contract_yaml(contract_id: str, *, status: str = "active") -> str:
    return (
        minimal_contract_yaml()
        .replace("id: example.minimal", f"id: {contract_id}")
        .replace("status: active", f"status: {status}")
    )


def _write_legacy_schema(project_root: Path) -> Path:
    path = project_root / LEGACY_SCHEMA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(LEGACY_SCHEMA), encoding="utf-8")
    return path


# ── the snapshot ──────────────────────────────────────────────────────────────


def test_two_entries_sharing_one_contract_file_share_one_object(tmp_path: Path) -> None:
    """Read once, keyed by resolved path: the digest is of the file, not of the entry."""
    project_root = _create_project(
        tmp_path,
        {
            "sources.yaml": _grouped_sources_yaml(
                _valid_source_yaml("first", enabled=True),
                _valid_source_yaml("second", enabled=True),
            )
        },
    )

    registry = load_registry(project_root)

    assert registry.contract_for("first") is registry.contract_for("second")
    assert registry.contract_for("first").schema_version == (
        registry.contract_for("second").schema_version
    )


def test_a_legacy_entry_yields_a_synthetic_draft_contract(tmp_path: Path) -> None:
    project_root = _create_project(
        tmp_path,
        {
            "legacy.yaml": _declaring(
                "legacy_source",
                f"schema:\n  mode: explicit\n  path: {LEGACY_SCHEMA_PATH}\n",
            )
        },
    )
    _write_legacy_schema(project_root)

    with pytest.warns(DeprecationWarning):
        registry = load_registry(project_root)

    contract = registry.contract_for("legacy_source")
    assert contract.id == f"legacy:{LEGACY_SCHEMA_PATH}"
    assert contract.status == "draft"
    assert contract.version == "0.0.0"
    assert contract.schema.name == "bronze.example__legacy_source"
    assert [
        (prop.name, prop.physical_type, prop.required)
        for prop in contract.schema.properties
    ] == [("id", "string", True), ("amount", "long", False)]


def test_two_legacy_entries_sharing_one_file_get_their_own_bronze_table(
    tmp_path: Path,
) -> None:
    """The synthetic schema is named after the entry; the digest still names the file."""
    project_root = _create_project(
        tmp_path,
        {
            "legacy.yaml": _grouped_sources_yaml(
                _declaring(
                    "legacy_first",
                    f"schema:\n  mode: explicit\n  path: {LEGACY_SCHEMA_PATH}\n",
                ),
                _declaring(
                    "legacy_second",
                    f"schema:\n  mode: explicit\n  path: {LEGACY_SCHEMA_PATH}\n",
                ),
            )
        },
    )
    _write_legacy_schema(project_root)

    with pytest.warns(DeprecationWarning):
        registry = load_registry(project_root)

    first = registry.contract_for("legacy_first")
    second = registry.contract_for("legacy_second")
    assert first is not second
    assert first.schema_version == second.schema_version
    assert first.schema.name == "bronze.example__legacy_first"
    assert second.schema.name == "bronze.example__legacy_second"


def test_an_inferred_entry_contributes_no_contract(tmp_path: Path) -> None:
    """``infer`` is still a declaration this order; a stand-in would be indistinguishable."""
    project_root = _create_project(
        tmp_path,
        {
            "sources.yaml": _grouped_sources_yaml(
                _declaring("inferred", "schema:\n  mode: infer\n"),
                _valid_source_yaml("declared", enabled=True),
            )
        },
    )

    with pytest.warns(DeprecationWarning):
        registry = load_registry(project_root)

    assert registry.contract_for("inferred") is None
    assert "inferred" not in registry.contracts
    assert isinstance(registry.contract_for("declared"), DataContract)


def test_the_snapshot_is_immutable_in_practice(tmp_path: Path) -> None:
    project_root = _create_project(
        tmp_path, {"sources.yaml": _valid_source_yaml("only", enabled=True)}
    )

    registry = load_registry(project_root)

    with pytest.raises(TypeError):
        registry.contracts["only"] = None  # type: ignore[index]


# ── read-once ─────────────────────────────────────────────────────────────────


@pytest.fixture
def counted_reads(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record every ``Path.read_bytes`` a load makes, so re-reads are countable."""
    reads: list[str] = []
    original = Path.read_bytes

    def recording(self: Path) -> bytes:
        reads.append(self.name)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", recording)
    yield reads


def test_a_shared_contract_file_is_read_exactly_once_per_load(
    tmp_path: Path, counted_reads: list[str]
) -> None:
    project_root = _create_project(
        tmp_path,
        {
            "sources.yaml": _grouped_sources_yaml(
                *(_valid_source_yaml(f"entry_{index}", enabled=True) for index in range(4))
            )
        },
    )
    counted_reads.clear()

    load_registry(project_root)

    assert counted_reads.count("minimal_contract.yaml") == 1


def test_a_shared_legacy_file_is_read_exactly_once_per_load(
    tmp_path: Path, counted_reads: list[str]
) -> None:
    """Both entries need their own synthetic contract; neither needs a second read."""
    project_root = _create_project(
        tmp_path,
        {
            "legacy.yaml": _grouped_sources_yaml(
                _declaring(
                    "legacy_first",
                    f"schema:\n  mode: explicit\n  path: {LEGACY_SCHEMA_PATH}\n",
                ),
                _declaring(
                    "legacy_second",
                    f"schema:\n  mode: explicit\n  path: {LEGACY_SCHEMA_PATH}\n",
                ),
            )
        },
    )
    _write_legacy_schema(project_root)
    counted_reads.clear()

    with pytest.warns(DeprecationWarning):
        load_registry(project_root)

    assert counted_reads.count("legacy_schema.json") == 1


def test_deprecations_are_warned_once_per_config_path(tmp_path: Path) -> None:
    project_root = _create_project(
        tmp_path,
        {
            "legacy.yaml": _grouped_sources_yaml(
                _declaring("legacy_first", "schema:\n  mode: infer\n"),
                _declaring("legacy_second", "schema:\n  mode: infer\n"),
            ),
            "other.yaml": _declaring("legacy_other", "schema:\n  mode: infer\n"),
        },
    )

    with pytest.warns(DeprecationWarning) as records:
        load_registry(project_root)

    assert len(records) == 2


# ── the four-step search ──────────────────────────────────────────────────────


def test_an_absolute_declaration_is_used_as_written(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere" / "contract.yaml"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(_contract_yaml("example.absolute"), encoding="utf-8")
    project_root = _create_project(
        tmp_path / "project",
        {"sources.yaml": _declaring("absolute", f"schema:\n  contract: {elsewhere}\n")},
    )

    registry = load_registry(project_root)

    assert registry.contract_for("absolute").id == "example.absolute"


def test_a_project_relative_declaration_wins_when_it_exists(tmp_path: Path) -> None:
    project_root = _create_project(
        tmp_path, {"example/source.yaml": _valid_source_yaml("relative", enabled=True)}
    )

    registry = load_registry(project_root)

    assert registry.contract_for("relative").contract_path == (
        project_root / DECLARED_CONTRACT_PATH
    )


def test_a_parent_of_the_config_is_searched_next(tmp_path: Path) -> None:
    """Neither absolute nor project-relative: found by walking up from the config file."""
    project_root = _create_project(
        tmp_path,
        {
            "example/source.yaml": _declaring(
                "beside", "schema:\n  contract: contracts/beside.yaml\n"
            )
        },
    )
    beside = project_root / "conf" / "contracts" / "beside.yaml"
    beside.parent.mkdir(parents=True, exist_ok=True)
    beside.write_text(_contract_yaml("example.beside"), encoding="utf-8")
    assert not (project_root / "contracts" / "beside.yaml").exists()

    registry = load_registry(project_root)

    assert registry.contract_for("beside").contract_path == beside


def test_nothing_found_is_reported_against_the_runtime_path(tmp_path: Path) -> None:
    """A missing file must surface as "does not exist", never as a silently absent entry."""
    project_root = _create_project(
        tmp_path,
        {
            "example/source.yaml": _declaring(
                "missing", "schema:\n  contract: conf/contracts/example/absent.yaml\n"
            )
        },
    )

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    issue = exc_info.value.issues[0]
    assert issue.path == "missing.schema.contract"
    assert "conf/contracts/example/absent.yaml" in issue.message
    assert "the file does not exist" in issue.message
    assert str(project_root / "conf" / "contracts" / "example" / "absent.yaml") in issue.message


# ── failures ──────────────────────────────────────────────────────────────────


def test_a_malformed_contract_carries_the_loaders_issues(tmp_path: Path) -> None:
    project_root = _create_project(
        tmp_path, {"example/source.yaml": _valid_source_yaml("broken", enabled=True)}
    )
    (project_root / DECLARED_CONTRACT_PATH).write_text(
        "apiVersion: v3.2.0\nkind: DataContract\n", encoding="utf-8"
    )

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    sources_dir = project_root / "conf" / "sources"
    assert exc_info.value.config_path == sources_dir / "example" / "source.yaml"
    message = exc_info.value.issues[0].message
    assert "could not load" in message
    assert "id: is required" in message
    assert "schema: is required" in message


def test_two_sources_with_two_broken_contracts_raise_once_listing_both(
    tmp_path: Path,
) -> None:
    """Nothing is planned from a half-read snapshot, so every failure is reported now."""
    project_root = _create_project(
        tmp_path,
        {
            "first.yaml": _declaring(
                "first", "schema:\n  contract: conf/contracts/example/absent_a.yaml\n"
            ),
            "second.yaml": _declaring(
                "second", "schema:\n  contract: conf/contracts/example/absent_b.yaml\n"
            ),
        },
    )

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    assert exc_info.value.config_path == project_root / "conf" / "sources"
    assert [issue.path for issue in exc_info.value.issues] == [
        "first.schema.contract",
        "second.schema.contract",
    ]
    assert "absent_a.yaml" in str(exc_info.value)
    assert "absent_b.yaml" in str(exc_info.value)


def test_a_broken_legacy_file_is_reported_under_the_field_it_declared(
    tmp_path: Path,
) -> None:
    project_root = _create_project(
        tmp_path,
        {
            "legacy.yaml": _declaring(
                "legacy_source",
                f"schema:\n  mode: explicit\n  path: {LEGACY_SCHEMA_PATH}\n",
            )
        },
    )
    path = project_root / LEGACY_SCHEMA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(SourceConfigValidationError) as exc_info:
        load_registry(project_root)

    issue = exc_info.value.issues[0]
    assert issue.path == "legacy_source.schema.path"
    assert "must be valid JSON" in issue.message


# ── plan carriage ─────────────────────────────────────────────────────────────


def test_the_plan_carries_the_object_the_snapshot_holds(tmp_path: Path) -> None:
    """Identity, not equality: a second read would produce an equal, different object."""
    project_root = _create_project(
        tmp_path, {"example/source.yaml": _valid_source_yaml("planned", enabled=True)}
    )
    registry = load_registry(project_root)

    planned = Planner().plan(
        PlanningRequest.create(
            source_id="planned", environment="local", project_root=project_root
        ),
        registry=registry,
    )

    assert planned.plan.data_contract is registry.contract_for("planned")


def test_the_planning_summary_describes_the_contract(tmp_path: Path) -> None:
    project_root = _create_project(
        tmp_path, {"example/source.yaml": _valid_source_yaml("summarized", enabled=True)}
    )
    registry = load_registry(project_root)

    planned = Planner().plan(
        PlanningRequest.create(
            source_id="summarized", environment="local", project_root=project_root
        ),
        registry=registry,
    )

    assert planned.to_summary()["contract"] == {
        "id": "example.minimal",
        "version": "1.0.0",
        "status": "active",
        "schema_version": registry.contract_for("summarized").schema_version,
        "path": DECLARED_CONTRACT_PATH,
    }


def test_the_planning_summary_carries_a_null_contract_for_an_inferred_source(
    tmp_path: Path,
) -> None:
    """The key is always present: a key whose presence depends on config is harder to query."""
    project_root = _create_project(
        tmp_path, {"example/source.yaml": _declaring("inferred", "schema:\n  mode: infer\n")}
    )
    with pytest.warns(DeprecationWarning):
        registry = load_registry(project_root)

    planned = Planner().plan(
        PlanningRequest.create(
            source_id="inferred", environment="local", project_root=project_root
        ),
        registry=registry,
    )

    summary = planned.to_summary()
    assert "contract" in summary
    assert summary["contract"] is None
