from __future__ import annotations

import json

import pytest

from tests.support.contract_baseline import _CASE_BY_NAME, BRONZE_CASES
from tests.support.handoff_matrix import (
    GOLDENS_DIR,
    SYNTHETIC_CASE,
    identifier_for,
    plan_cell,
    prepare_fixture,
    run_mode,
)
from tests.support.spark_sessions import sqlite_catalog_target, start_session

MATRIX_CASES = [
    pytest.param("transparencia_servidores_por_orgao", False, id="transparencia-explicit"),
    pytest.param("transparencia_servidores_por_orgao", True, id="transparencia-inferred"),
    pytest.param("ibge_agro_abacaxi_pronaf", False, id="ibge-agro-hook"),
    pytest.param("ibge_pib_brasil", False, id="ibge-pib-hook"),
    pytest.param("dados_abertos_catalog_conjunto_dados", False, id="catalog-unaffected"),
    pytest.param("bronze_materializer_inep", False, id="inep-batched"),
    pytest.param("inep_censo_escolar_microdados", False, id="inep-sample"),
    pytest.param(SYNTHETIC_CASE, False, id="synthetic-500-split-envelopes"),
]
assert set(BRONZE_CASES) <= {case.values[0] for case in MATRIX_CASES}


@pytest.fixture(scope="module")
def catalog_target(tmp_path_factory):
    target = sqlite_catalog_target(tmp_path_factory.mktemp("ac1-handoff-catalog"))
    target.prepare()
    return target


@pytest.fixture(scope="module")
def spark(catalog_target):
    session = start_session("janus-ac1-handoff-matrix", catalog_target.session_options())
    yield session
    session.stop()


@pytest.mark.parametrize(("case_name", "inferred"), MATRIX_CASES)
def test_bronze_identity_across_compaction_and_replay(
    spark, catalog_target, tmp_path, monkeypatch, case_name, inferred
):
    mapping, config_path, strategy_factory = prepare_fixture(case_name, tmp_path)
    if case_name != SYNTHETIC_CASE and _CASE_BY_NAME[case_name].token is not None:
        monkeypatch.setenv(*_CASE_BY_NAME[case_name].token)
    identifier = identifier_for(mapping)
    matrix = {}
    for mode in ("pages", "compact"):
        planned = plan_cell(
            mapping,
            config_path,
            strategy_factory(),
            mode=mode,
            case_name=case_name,
            inferred=inferred,
        )
        cells = run_mode(
            spark,
            catalog_target,
            planned,
            case_name=case_name,
            canonical_identifier=identifier,
            inferred=inferred,
        )
        matrix.update({f"{mode}/{path}": cell for path, cell in cells.items()})

    baseline_name = f"{case_name}_inferred" if inferred else case_name
    filename = (
        "additional_pages_goldens.json"
        if inferred or case_name == SYNTHETIC_CASE
        else ("baseline_goldens.json")
    )
    baseline = json.loads((GOLDENS_DIR / filename).read_text())["baseline_goldens"][baseline_name]
    evidence = {"fixture": baseline_name, "baseline": baseline, "matrix": matrix}
    diagnostic = json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True)

    print("AC1_MATRIX=" + json.dumps(evidence, ensure_ascii=False, sort_keys=True), flush=True)
    assert_matrix(matrix, baseline, diagnostic)
    for label, cell in matrix.items():
        if "ibge" in case_name:
            assert cell["handoff_compaction"] == "hook", diagnostic
        elif _family(case_name) in {"catalog", "file"}:
            assert cell["handoff_compaction"] is None, diagnostic
            assert all(
                "handoff_compaction" not in read["handoff_metadata"] for read in cell["reads"]
            ), diagnostic
        else:
            assert cell["handoff_compaction"] == (
                "pages" if label.startswith("pages/") else "jsonl"
            ), diagnostic
        if case_name == "bronze_materializer_inep":
            assert cell["batch_artifact_counts"] == [5, 1], diagnostic
            assert cell["bronze_commit_count"] == 2, diagnostic
        if case_name == SYNTHETIC_CASE and label.startswith("compact/"):
            assert cell["handoff_artifact_count"] > 1, diagnostic
        if _family(case_name) == "api" and "ibge" not in case_name:
            assert all(read["schema_applied"] is not inferred for read in cell["reads"]), diagnostic


def assert_matrix(matrix, baseline, diagnostic):
    """A failed cell, digest, identifier or replay must report the entire matrix."""
    expected_cells = {"pages/live", "compact/live", "pages/replay", "compact/replay"}
    assert set(matrix) == expected_cells, diagnostic
    for label, cell in matrix.items():
        assert cell["status"] == "succeeded", diagnostic
        assert cell["actual_identifiers"] == [cell["expected_identifier"]], diagnostic
        if label.endswith("/replay"):
            metadata = cell["extraction_metadata"]
            assert metadata.get("raw_to_bronze") == "true", diagnostic
            assert int(metadata.get("rediscovered_raw_artifact_count", "0")) > 0, diagnostic
            assert metadata.get("checksums_verified") == "true", diagnostic
    assert len({cell["expected_identifier"] for cell in matrix.values()}) == 4, diagnostic
    goldens = {tuple(sorted(cell["golden"].items())) for cell in matrix.values()}
    assert len(goldens) == 1, diagnostic
    assert matrix["pages/live"]["golden"] == baseline, diagnostic


def _family(case_name):
    return "api" if case_name == SYNTHETIC_CASE else _CASE_BY_NAME[case_name].family
