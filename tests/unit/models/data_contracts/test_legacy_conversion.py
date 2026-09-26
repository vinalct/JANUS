from __future__ import annotations

import json
from pathlib import Path

import pytest

from janus.models.data_contracts import (
    ContractValidationError,
    contract_from_legacy_schema_file,
    legacy_contract_id,
    load_data_contract,
    spark_struct_json,
)
from janus.models.data_contracts.legacy import (
    LEGACY_CONTRACT_OWNER,
    LEGACY_CONTRACT_PURPOSE,
    LEGACY_CONTRACT_STATUS,
    LEGACY_CONTRACT_VERSION,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
LEGACY_SCHEMAS = PROJECT_ROOT / "conf" / "schemas"
SPARK_SCHEMA_GOLDENS = (
    PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "baseline" / "spark_schema"
)
LEGACY_SCHEMA_FILE_COUNT = 16


def _legacy_files() -> tuple[Path, ...]:
    return tuple(sorted(LEGACY_SCHEMAS.rglob("*.json")))


def _goldens() -> tuple[Path, ...]:
    return tuple(sorted(SPARK_SCHEMA_GOLDENS.glob("*.json")))


def _convert(path: Path, *, bronze_table: str = "bronze_example.table") -> object:
    return contract_from_legacy_schema_file(
        path,
        source_id="legacy_source",
        bronze_table=bronze_table,
        domain="example",
        project_root=PROJECT_ROOT,
    )


def _write(tmp_path: Path, payload: object, *, name: str = "legacy.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ── the sixteen checked-in files ──────────────────────────────────────────────


def test_every_checked_in_legacy_schema_file_converts() -> None:
    files = _legacy_files()

    assert len(files) == LEGACY_SCHEMA_FILE_COUNT, files
    for path in files:
        contract = _convert(path)
        assert contract.schema.properties


@pytest.mark.parametrize("golden_path", _goldens(), ids=lambda path: path.stem)
def test_converted_contract_reproduces_the_m0_spark_schema(golden_path: Path) -> None:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    schema_path = PROJECT_ROOT / golden["schema_path"]

    contract = contract_from_legacy_schema_file(
        schema_path,
        source_id=golden["source_id"],
        bronze_table="bronze_example.table",
        domain="example",
        project_root=PROJECT_ROOT,
    )

    assert spark_struct_json(contract.schema.properties) == golden["struct_type"]
    assert contract.schema_version == golden["schema_sha256"]


def test_the_golden_sweep_covers_every_explicit_entry() -> None:
    """A glob that matched nothing would make the comparison above vacuous."""
    goldens = _goldens()

    assert len(goldens) == 17
    referenced = {
        json.loads(path.read_text(encoding="utf-8"))["schema_path"] for path in goldens
    }
    assert len(referenced) == 15


def test_nested_structs_keep_their_shape_and_nullability() -> None:
    contract = _convert(LEGACY_SCHEMAS / "transparencia" / "gastos_cartoes_cartoes_schema.json")

    unidade = next(
        prop for prop in contract.schema.properties if prop.name == "unidadeGestora"
    )
    assert unidade.physical_type == "struct"
    assert {child.name for child in unidade.properties} >= {"orgaoMaximo", "orgaoVinculado"}
    assert next(
        prop for prop in contract.schema.properties if prop.name == "id"
    ).required is True


# ── identity ──────────────────────────────────────────────────────────────────


def test_a_converted_file_declares_itself_unreviewed() -> None:
    path = LEGACY_SCHEMAS / "inep" / "censo_escolar_microdados_schema.json"

    contract = contract_from_legacy_schema_file(
        path,
        source_id="inep_censo_escolar_microdados",
        bronze_table="bronze_inep.censo_escolar_microdados",
        domain="education",
        project_root=PROJECT_ROOT,
    )

    assert contract.id == "legacy:conf/schemas/inep/censo_escolar_microdados_schema.json"
    assert contract.status == LEGACY_CONTRACT_STATUS == "draft"
    assert contract.version == LEGACY_CONTRACT_VERSION == "0.0.0"
    assert contract.name == "Legacy schema for inep_censo_escolar_microdados"
    assert contract.purpose == LEGACY_CONTRACT_PURPOSE
    assert contract.owners == (LEGACY_CONTRACT_OWNER,)
    assert contract.domain == "education"
    assert contract.schema.name == "bronze_inep.censo_escolar_microdados"
    assert contract.janus.compatibility == "additive"
    assert contract.janus.enforcement == "lenient"
    assert contract.contract_path == path


def test_two_entries_sharing_one_file_differ_only_in_the_table_they_describe() -> None:
    path = LEGACY_SCHEMAS / "dados_abertos_catalog" / "catalog_metadata_schema.json"

    first = _convert(path, bronze_table="bronze__dados_abertos_catalog.conjuntos_dados")
    second = _convert(
        path, bronze_table="bronze__dados_abertos_catalog.conjuntos_dados_details"
    )

    assert first.schema.name != second.schema.name
    assert first.schema_version == second.schema_version
    assert first.id == second.id


def test_a_file_outside_the_project_keeps_its_own_path_as_identity(tmp_path: Path) -> None:
    path = _write(tmp_path, ["id"])

    assert legacy_contract_id(path, PROJECT_ROOT) == f"legacy:{path.as_posix()}"
    assert legacy_contract_id(path) == f"legacy:{path.as_posix()}"


# ── the columns-only shapes ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        ["id", "updated_at"],
        {"fields": ["id", "updated_at"]},
        {"columns": ["id", "updated_at"]},
        {"schema": {"fields": ["id", "updated_at"]}},
        {"schema": {"columns": ["id", "updated_at"]}},
        {"columns": [{"name": "id"}, {"name": "updated_at"}]},
        {"columns": ["id", {"name": "updated_at"}]},
    ],
)
def test_every_columns_only_shape_becomes_nullable_strings(
    tmp_path: Path, payload: object
) -> None:
    contract = _convert(_write(tmp_path, payload))

    assert [prop.name for prop in contract.schema.properties] == ["id", "updated_at"]
    assert {prop.physical_type for prop in contract.schema.properties} == {"string"}
    assert {prop.logical_type for prop in contract.schema.properties} == {"string"}
    assert not any(prop.required for prop in contract.schema.properties)


def test_columns_only_names_are_stripped(tmp_path: Path) -> None:
    contract = _convert(_write(tmp_path, {"columns": [" id ", "updated_at"]}))

    assert [prop.name for prop in contract.schema.properties] == ["id", "updated_at"]


# ── the refusals ──────────────────────────────────────────────────────────────


def _rendered_issues(path: Path) -> str:
    with pytest.raises(ContractValidationError) as exc_info:
        _convert(path)
    return str(exc_info.value)


def test_an_unreadable_shape_is_refused_by_name(tmp_path: Path) -> None:
    message = _rendered_issues(_write(tmp_path, {"unexpected": True}))

    assert "must be a JSON array of field names or a mapping with fields/columns" in message


def test_an_unusable_entry_is_refused_by_name(tmp_path: Path) -> None:
    message = _rendered_issues(_write(tmp_path, {"columns": ["id", 7]}))

    assert "entries must be strings or objects containing a 'name' field" in message
    assert "columns[1]" in message


def test_an_empty_name_is_refused(tmp_path: Path) -> None:
    message = _rendered_issues(_write(tmp_path, {"columns": ["id", "  "]}))

    assert "columns[1].name: must not be empty" in message


def test_a_duplicate_name_is_refused(tmp_path: Path) -> None:
    message = _rendered_issues(_write(tmp_path, {"columns": ["id", "id"]}))

    assert "columns[1].name: must be unique; duplicate 'id'" in message


def test_a_file_declaring_no_field_is_refused(tmp_path: Path) -> None:
    message = _rendered_issues(_write(tmp_path, {"columns": []}))

    assert "must declare at least one field" in message


def test_malformed_json_is_refused_rather_than_raised_raw(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    message = _rendered_issues(path)

    assert "must be valid JSON" in message


def test_every_entry_problem_is_reported_at_once(tmp_path: Path) -> None:
    path = _write(tmp_path, {"columns": ["id", 7, "id", ""]})

    with pytest.raises(ContractValidationError) as exc_info:
        _convert(path)

    assert len(exc_info.value.issues) == 3


def test_a_spark_type_outside_the_vocabulary_is_refused(tmp_path: Path) -> None:
    payload = {
        "type": "struct",
        "fields": [
            {"name": "amount", "type": "interval", "nullable": True, "metadata": {}}
        ],
    }

    message = _rendered_issues(_write(tmp_path, payload))

    assert "no JANUS vocabulary spelling" in message


# ── the checked-in contracts ──────────────────────────────────────────────────


def test_migrated_contract_column_order_matches_every_m0_entry() -> None:
    """Contracts preserve bronze field order for all seventeen explicit entries."""
    from janus.registry.loader import load_registry

    registry = load_registry(PROJECT_ROOT)
    goldens = _goldens()

    assert len(goldens) == 17
    for golden_path in goldens:
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        source = registry.get_source(golden["source_id"], include_disabled=True)
        assert source.schema.contract is not None
        contract = load_data_contract(PROJECT_ROOT / source.schema.contract)

        assert contract.column_names == tuple(
            field["name"] for field in golden["struct_type"]["fields"]
        )
        assert spark_struct_json(contract.schema.properties) == golden["struct_type"]


def test_migrated_contract_spark_schemas_match_every_m0_entry() -> None:
    """The contract-generated Spark schemas retain names, types, nullability, and order."""
    pytest.importorskip("pyspark.sql")
    from janus.registry.loader import load_registry
    from janus.schema_contracts import spark_schema_from_contract

    registry = load_registry(PROJECT_ROOT)
    goldens = _goldens()

    assert len(goldens) == 17
    for golden_path in goldens:
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        source = registry.get_source(golden["source_id"], include_disabled=True)
        assert source.schema.contract is not None
        contract = load_data_contract(PROJECT_ROOT / source.schema.contract)

        assert spark_schema_from_contract(contract).jsonValue() == golden["struct_type"]
