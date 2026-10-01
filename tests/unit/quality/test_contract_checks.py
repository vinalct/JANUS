"""The pure structural check: is this frame what the contract declares?"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from janus.models.data_contracts import ContractProperty, ContractSchema, DataContract
from janus.models.data_contracts import load_data_contract as _load_data_contract
from janus.quality.contract_checks import (
    CORRUPT_RECORD_COLUMN,
    ContractCheck,
    ContractEnforcementError,
    ContractViolationError,
    FrameColumn,
    check_frame_against_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOSTILE = PROJECT_ROOT / "tests" / "fixtures" / "contracts" / "hostile"
MODULE_PATH = PROJECT_ROOT / "src" / "janus" / "quality" / "contract_checks.py"
DETECTOR = (
    PROJECT_ROOT / "tests" / "unit" / "models" / "data_contracts"
    / "test_contract_package_imports.py"
)
CORRUPT = "_janus_corrupt_record"

BASE_FRAME_TYPES = (
    ("id", "string"),
    ("label", "string"),
    ("amount", "long"),
    ("when", "timestamp"),
)


# ── builders ────────────────────────────────────────────────────────────────


def _contract(name: str) -> DataContract:
    return _load_data_contract(HOSTILE / f"{name}.yaml")


def _with_properties(contract: DataContract, *properties: ContractProperty) -> DataContract:
    """The same contract identity over a hand-built property list (nested shapes)."""
    return replace(contract, schema=replace(contract.schema, properties=properties))


def _without_properties(contract: DataContract) -> DataContract:
    """A contract the loader could never produce: its table declares nothing.

    The check must still answer; ``ContractSchema`` refuses an empty property list at
    construction, so the instance is assembled field by field.
    """
    schema = object.__new__(ContractSchema)
    object.__setattr__(schema, "name", contract.schema.name)
    object.__setattr__(schema, "physical_type", contract.schema.physical_type)
    object.__setattr__(schema, "properties", ())
    empty = object.__new__(DataContract)
    for field_name in DataContract.__dataclass_fields__:
        object.__setattr__(empty, field_name, getattr(contract, field_name))
    object.__setattr__(empty, "schema", schema)
    return empty


def _frame(*columns: tuple[str, Any] | tuple[str, Any, bool]) -> tuple[FrameColumn, ...]:
    return tuple(
        FrameColumn(column[0], column[1], column[2] if len(column) > 2 else True)
        for column in columns
    )


def _base_frame(**overrides: Any) -> tuple[FrameColumn, ...]:
    """The base frame with some types replaced; a ``None`` override drops the column."""
    columns = []
    for name, spark_type in BASE_FRAME_TYPES:
        replacement = overrides.get(name, spark_type)
        if replacement is not None:
            columns.append((name, replacement))
    return _frame(*columns)


def _struct(*fields: tuple[str, Any]) -> dict[str, Any]:
    return {
        "type": "struct",
        "fields": [
            {"name": name, "type": spark_type, "nullable": True, "metadata": {}}
            for name, spark_type in fields
        ],
    }


def _array(element: Any) -> dict[str, Any]:
    return {"type": "array", "elementType": element, "containsNull": True}


def _map(key: Any, value: Any) -> dict[str, Any]:
    return {"type": "map", "keyType": key, "valueType": value, "valueContainsNull": True}


def _scalar(name: str, physical_type: str, **options: Any) -> ContractProperty:
    logical = {
        "string": "string",
        "long": "integer",
        "integer": "integer",
        "timestamptz": "date",
        "timestamp": "date",
    }.get(physical_type, "number")
    return ContractProperty(name, physical_type, logical, **options)


def _kinds(check: ContractCheck) -> list[tuple[str, str]]:
    return [(mismatch.kind, mismatch.column) for mismatch in check.mismatches]


# ── names ───────────────────────────────────────────────────────────────────


def test_an_identical_frame_is_ok_with_no_mismatches():
    check = check_frame_against_contract(_base_frame(), _contract("base"))

    assert check.ok is True
    assert check.mismatches == ()
    assert check.checked_columns == 4


def test_a_missing_column_is_one_mismatch_named_after_it():
    check = check_frame_against_contract(_base_frame(amount=None), _contract("base"))

    assert check.ok is False
    assert _kinds(check) == [("missing_column", "amount")]
    assert check.mismatches[0].expected == "long"


@pytest.mark.parametrize("fixture", ["base", "base_backward", "base_frozen"])
def test_an_undeclared_frame_column_is_a_mismatch_in_every_compatibility_mode(fixture):
    """D-4: a new upstream field is admitted by declaring it, never by a batch carrying it."""
    frame = (*_base_frame(), *_frame(("note", "string")))
    check = check_frame_against_contract(frame, _contract(fixture))

    assert _kinds(check) == [("unexpected_column", "note")]
    assert check.mismatches[0].observed == "string"


def test_names_compare_exactly_and_case_sensitively():
    frame = _frame(("ID", "string"), ("label", "string"), ("amount", "long"), ("when", "timestamp"))
    check = check_frame_against_contract(frame, _contract("base"))

    assert sorted(_kinds(check)) == [("missing_column", "id"), ("unexpected_column", "ID")]


# ── types, in vocabulary spellings ──────────────────────────────────────────


def test_a_scalar_type_mismatch_is_reported_in_vocabulary_spellings():
    check = check_frame_against_contract(_base_frame(amount="string"), _contract("base"))

    assert _kinds(check) == [("type_mismatch", "amount")]
    mismatch = check.mismatches[0]
    assert (mismatch.expected, mismatch.observed) == ("long", "string")
    assert "bigint" not in f"{mismatch.expected} {mismatch.observed} {mismatch.render()}"
    assert mismatch.render() == "amount: type mismatch (contract long, frame string)"


def test_timestamp_without_zone_is_not_timestamptz():
    check = check_frame_against_contract(_base_frame(when="timestamp_ntz"), _contract("base"))

    assert _kinds(check) == [("type_mismatch", "when")]
    assert (check.mismatches[0].expected, check.mismatches[0].observed) == (
        "timestamptz",
        "timestamp",
    )


def test_a_decimal_scale_difference_is_a_type_mismatch():
    contract = _contract("base_decimal_18_2")
    check = check_frame_against_contract(_base_frame(amount="decimal(18,4)"), contract)

    assert _kinds(check) == [("type_mismatch", "amount")]
    assert (check.mismatches[0].expected, check.mismatches[0].observed) == (
        "decimal(18,2)",
        "decimal(18,4)",
    )


def test_an_unmappable_spark_type_is_a_mismatch_carrying_the_raw_spelling():
    check = check_frame_against_contract(_base_frame(amount="short"), _contract("base"))

    assert _kinds(check) == [("type_mismatch", "amount")]
    assert check.mismatches[0].observed == "short"


def test_a_nested_struct_child_mismatch_is_named_by_its_dotted_path():
    code = _scalar("code", "string")
    inner = ContractProperty("inner", "struct", "object", properties=(code,))
    payload = ContractProperty("payload", "struct", "object", properties=(inner,))
    contract = _with_properties(_contract("base"), _scalar("id", "string"), payload)
    frame = _frame(("id", "string"), ("payload", _struct(("inner", _struct(("code", "long"))))))

    check = check_frame_against_contract(frame, contract)

    assert _kinds(check) == [("type_mismatch", "payload.inner.code")]
    assert (check.mismatches[0].expected, check.mismatches[0].observed) == ("string", "long")


def test_array_elements_and_map_values_have_their_own_path_suffixes():
    items = ContractProperty(
        "items", "array", "array", items=_scalar("element", "string")
    )
    attrs = ContractProperty(
        "attrs",
        "map",
        "object",
        keys=_scalar("key", "string", required=True),
        values=_scalar("value", "long"),
    )
    contract = _with_properties(_contract("base"), _scalar("id", "string"), items, attrs)
    frame = _frame(
        ("id", "string"),
        ("items", _array("long")),
        ("attrs", _map("string", "string")),
    )

    check = check_frame_against_contract(frame, contract)

    assert _kinds(check) == [("type_mismatch", "attrs{}"), ("type_mismatch", "items[]")]


def test_a_container_against_a_scalar_is_one_mismatch_at_the_container():
    inner = ContractProperty("inner", "struct", "object", properties=(_scalar("code", "string"),))
    payload = ContractProperty("payload", "struct", "object", properties=(inner,))
    contract = _with_properties(_contract("base"), _scalar("id", "string"), payload)

    check = check_frame_against_contract(_frame(("id", "string"), ("payload", "string")), contract)

    assert _kinds(check) == [("type_mismatch", "payload")]


def test_a_map_key_type_difference_is_one_mismatch_at_the_map():
    attrs = ContractProperty(
        "attrs",
        "map",
        "object",
        keys=_scalar("key", "string", required=True),
        values=_scalar("value", "long"),
    )
    contract = _with_properties(_contract("base"), _scalar("id", "string"), attrs)
    frame = _frame(("id", "string"), ("attrs", _map("integer", "long")))

    check = check_frame_against_contract(frame, contract)

    assert _kinds(check) == [("type_mismatch", "attrs")]


# ── nullability is recorded, never judged (D-3) ─────────────────────────────


def test_a_nullable_frame_column_the_contract_requires_is_recorded_not_a_mismatch():
    frame = _frame(("id", "string", True), *((n, t) for n, t in BASE_FRAME_TYPES[1:]))
    check = check_frame_against_contract(frame, _contract("base"))

    assert check.ok is True
    assert check.nullability_relaxed == ("id",)


def test_a_non_nullable_frame_column_the_contract_leaves_optional_is_nothing():
    frame = _frame(
        ("id", "string", False),
        ("label", "string", False),
        ("amount", "long", False),
        ("when", "timestamp", False),
    )
    check = check_frame_against_contract(frame, _contract("base"))

    assert check.ok is True
    assert check.mismatches == ()
    assert check.nullability_relaxed == ()


# ── the corrupt-record column ───────────────────────────────────────────────


def test_the_corrupt_record_column_is_ignored_only_while_the_reader_still_carries_it():
    assert CORRUPT_RECORD_COLUMN == CORRUPT
    frame = (*_base_frame(), *_frame((CORRUPT, "string")))

    tolerated = check_frame_against_contract(frame, _contract("base"), allow_corrupt_column=True)
    leaked = check_frame_against_contract(frame, _contract("base"))

    assert tolerated.ok is True
    assert _kinds(leaked) == [("corrupt_column_leaked", CORRUPT)]


# ── totality and ordering ───────────────────────────────────────────────────


def test_an_empty_frame_reports_every_declared_column_missing():
    check = check_frame_against_contract((), _contract("base"))

    assert sorted(_kinds(check)) == [
        ("missing_column", "amount"),
        ("missing_column", "id"),
        ("missing_column", "label"),
        ("missing_column", "when"),
    ]


def test_an_empty_contract_still_produces_a_result():
    check = check_frame_against_contract(_base_frame(), _without_properties(_contract("base")))

    assert check.checked_columns == 0
    assert {kind for kind, _ in _kinds(check)} == {"unexpected_column"}


def test_mismatches_are_sorted_by_column_then_kind():
    frame = (*_base_frame(amount="string", label=None), *_frame(("extra", "string")))
    check = check_frame_against_contract(frame, _contract("base"))

    assert _kinds(check) == [
        ("type_mismatch", "amount"),
        ("unexpected_column", "extra"),
        ("missing_column", "label"),
    ]


def test_the_check_is_frozen_and_carries_the_contract_identity():
    contract = _contract("base")
    check = check_frame_against_contract(_base_frame(), contract)

    assert (check.contract_id, check.contract_version, check.schema_version) == (
        contract.id,
        contract.version,
        contract.schema_version,
    )
    with pytest.raises(AttributeError):
        check.mismatches = ()


# ── what the check hands on ─────────────────────────────────────────────────


def test_the_check_renders_as_the_schema_expectations_validation_check():
    passed = check_frame_against_contract(_base_frame(), _contract("base")).to_validation_check()
    failed = check_frame_against_contract(
        _base_frame(amount="string"), _contract("base")
    ).to_validation_check()

    assert (passed.phase, passed.name, passed.outcome) == (
        "data",
        "schema_expectations",
        "passed",
    )
    assert (failed.phase, failed.name, failed.outcome) == (
        "data",
        "schema_expectations",
        "failed",
    )
    assert "amount" in failed.details_as_dict()["mismatches"]


def test_the_violation_names_the_contract_and_lists_one_mismatch_per_line():
    contract = _contract("base")
    check = check_frame_against_contract(_base_frame(amount="string", label=None), contract)
    error = ContractViolationError(check, batch_index=2, batch_count=3)

    assert isinstance(error, ContractEnforcementError)
    assert error.failure_stage == "contract_check"
    assert error.batch_index == 2
    lines = str(error).splitlines()
    assert lines[0].startswith(f"{contract.id} v{contract.version}")
    assert "batch 2/3" in lines[0]
    mismatch_lines = [line.strip() for line in lines[1:]]
    assert mismatch_lines == [mismatch.render() for mismatch in check.mismatches]
    assert [line.split(":", 1)[0] for line in mismatch_lines] == ["amount", "label"]


# ── the module is engine-free (NFR-3) ───────────────────────────────────────


def _order_18_import_detector() -> Callable[[str], tuple[str, ...]]:
    """engine-import detector, loaded by path: one detector, two packages."""
    spec = importlib.util.spec_from_file_location("import_detector", DETECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._forbidden_imports


def test_module_imports_no_engine():
    detector = _order_18_import_detector()

    assert detector("def f():\n    import pyspark\n") == ("pyspark",), "detector is live"
    assert detector(MODULE_PATH.read_text(encoding="utf-8")) == ()


def test_importing_the_module_loads_no_engine_at_any_depth():
    import_paths = (str(PROJECT_ROOT / "src"), *(entry for entry in sys.path if entry))
    command = (
        "import sys; "
        f"sys.path[:0] = {import_paths!r}; "
        "import janus.quality.contract_checks; "
        "assert not {'pyspark', 'pyiceberg', 'pyarrow'} & sys.modules.keys()"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
