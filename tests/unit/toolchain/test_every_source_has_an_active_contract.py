"""Read the checked-in YAML directly so the inventory can report invalid declarations."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from janus.models.data_contracts import load_data_contract
from janus.utils.storage import bronze_table_identifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCES_ROOT = PROJECT_ROOT / "conf" / "sources"
CONTRACTS_ROOT = PROJECT_ROOT / "conf" / "contracts"
EXPECTED_SOURCE_FILES = 21
EXPECTED_SOURCE_ENTRIES = 31
EXPECTED_CONTRACT_FILES = 30

# Live Transparency sources require a token and sampled response before review.
PENDING_ACTIVATION: dict[str, dict[str, str]] = {
    source_id: {
        "since": "2026-09-23",
        "review_by": "2026-10-23",
        "reason": "No token or saved run; draft reflects the published DTO only.",
    }
    for source_id in (
        "transparencia__contratos__contratos__full_refresh",
        "transparencia__emendas_parlamentares__emendas__full_refresh",
        "transparencia__emendas_parlamentares__documentos__full_refresh",
        "transparencia__licitacoes__licitacoes__full_refresh",
        "transparencia__licitacoes__unidades_gestoras__full_refresh",
        "transparencia__licitacoes__modalidades__full_refresh",
        "transparencia__orgaos__siape__full_refresh",
        "transparencia__orgaos__siafi__full_refresh",
        "transparencia__renuncias_fiscais__renuncias_valores__full_refresh",
        "transparencia__renuncias_fiscais__empresas_imunes_isentas__full_refresh",
        "transparencia__renuncias_fiscais__empresas_habilitadas__full_refresh",
        "transparencia__poder_executivo_federal__servidores__full_refresh",
    )
}


def _source_entries() -> tuple[tuple[Path, dict[str, Any]], ...]:
    files = tuple(sorted(SOURCES_ROOT.rglob("*.yaml")))
    entries: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(document, dict), f"{path}: expected a YAML mapping"
        sources = document.get("sources", [document])
        assert isinstance(sources, list) and sources, f"{path}: expected a nonempty sources list"
        for index, source in enumerate(sources):
            assert isinstance(source, dict), f"{path}: sources[{index}] must be a mapping"
            entries.append((path, source))
    assert len(files) == EXPECTED_SOURCE_FILES and len(entries) == EXPECTED_SOURCE_ENTRIES, (
        f"source inventory changed: {len(entries)} entries in {len(files)} files; "
        "update these counts and the README inventory when sources change"
    )
    return tuple(entries)


def _contract_path(config_path: Path, declared: str) -> Path:
    relative = Path(declared)
    if relative.is_absolute():
        return relative.resolve()
    candidates = (
        PROJECT_ROOT / relative,
        *(parent / relative for parent in config_path.resolve().parents),
    )
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())


def _declared_contracts() -> tuple[tuple[dict[str, Any], Path], ...]:
    declared: list[tuple[dict[str, Any], Path]] = []
    for config_path, entry in _source_entries():
        schema = entry.get("schema", {})
        assert isinstance(schema, dict), f"{config_path}: schema must be a mapping"
        if "contract" in schema:
            assert isinstance(schema["contract"], str), f"{config_path}: contract must be a path"
            declared.append((entry, _contract_path(config_path, schema["contract"])))
    return tuple(declared)


def test_no_spark_json_under_conf() -> None:
    assert not (PROJECT_ROOT / "conf" / "schemas").exists()
    offenders = []
    for path in (PROJECT_ROOT / "conf").rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and (
            payload.get("type") == "struct"
            or isinstance(payload.get("columns"), list)
            or isinstance(payload.get("fields"), list)
        ):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert not offenders, f"Spark-shaped JSON remains under conf: {offenders}"


def test_no_entry_uses_schema_path_or_mode_explicit() -> None:
    offenders = [
        entry["source_id"]
        for _, entry in _source_entries()
        if "path" in entry.get("schema", {}) or entry.get("schema", {}).get("mode") == "explicit"
    ]
    assert not offenders, f"legacy explicit declarations: {offenders}"


def test_no_entry_uses_schema_mode_infer() -> None:
    offenders = [
        entry["source_id"]
        for _, entry in _source_entries()
        if entry.get("schema", {}).get("mode") == "infer"
    ]
    assert not offenders, f"inferred declarations: {offenders}"


def test_every_entry_declares_a_contract_that_exists() -> None:
    missing = []
    for config_path, entry in _source_entries():
        declared = entry.get("schema", {}).get("contract")
        if not isinstance(declared, str):
            missing.append(f"{entry['source_id']}: no schema.contract")
            continue
        path = _contract_path(config_path, declared)
        if not path.is_file() or not path.is_relative_to(CONTRACTS_ROOT.resolve()):
            missing.append(f"{entry['source_id']}: {path} is not a contract under conf/contracts")
    assert not missing, "\n".join(missing)


def test_every_entry_references_an_active_or_documented_pending_contract() -> None:
    unaccounted = []
    for config_path, entry in _source_entries():
        source_id = entry["source_id"]
        declared = entry.get("schema", {}).get("contract")
        if not isinstance(declared, str):
            unaccounted.append(f"{source_id}: no contract (enabled={entry.get('enabled')})")
            continue
        path = _contract_path(config_path, declared)
        if not path.is_file():
            unaccounted.append(f"{source_id}: missing contract {path}")
            continue
        contract = load_data_contract(path)
        if contract.status == "active":
            if source_id in PENDING_ACTIVATION:
                unaccounted.append(f"{source_id}: pending entry is already active")
            continue
        if source_id not in PENDING_ACTIVATION:
            unaccounted.append(f"{source_id}: unallowlisted {contract.status} contract")
        elif entry.get("enabled") is not False:
            unaccounted.append(f"{source_id}: pending source must remain disabled")
    assert not unaccounted, "\n".join(unaccounted)


def test_pending_activation_allowlist_has_current_drafts_and_expiry() -> None:
    entries = {entry["source_id"]: entry for _, entry in _source_entries()}
    drafts = set()
    for config_path, entry in _source_entries():
        declared = entry.get("schema", {}).get("contract")
        if isinstance(declared, str):
            contract = load_data_contract(_contract_path(config_path, declared))
            if contract.status == "draft":
                drafts.add(entry["source_id"])

    assert set(PENDING_ACTIVATION) == drafts, (
        f"pending activation entries do not match drafts: "
        f"missing={sorted(drafts - set(PENDING_ACTIVATION))}, "
        f"stale={sorted(set(PENDING_ACTIVATION) - drafts)}"
    )
    for source_id, allowance in PENDING_ACTIVATION.items():
        assert source_id in entries, f"stale pending source id: {source_id}"
        assert allowance["reason"].strip(), f"missing reason for {source_id}"
        since = date.fromisoformat(allowance["since"])
        review_by = date.fromisoformat(allowance["review_by"])
        assert since <= date.today(), f"future pending date for {source_id}: {since}"
        assert review_by > date.today(), f"pending activation review is overdue for {source_id}"


def test_contract_domain_matches_entry_domain() -> None:
    for entry, path in _declared_contracts():
        assert load_data_contract(path).domain == entry["domain"], entry["source_id"]


def test_contract_schema_name_matches_bronze_table() -> None:
    for entry, path in _declared_contracts():
        bronze = entry["outputs"]["bronze"]
        identifier = bronze_table_identifier(
            bronze["path"],
            fallback_name=entry["source_id"],
            namespace=bronze.get("namespace"),
            table_name=bronze.get("table_name"),
        )
        assert load_data_contract(path).schema.name == identifier.rsplit(".", 1)[1], entry[
            "source_id"
        ]


def test_every_contract_is_referenced() -> None:
    files = {path.resolve() for path in CONTRACTS_ROOT.rglob("*.yaml")}
    assert len(files) == EXPECTED_CONTRACT_FILES, (
        f"expected {EXPECTED_CONTRACT_FILES} contracts, found {len(files)}"
    )
    referenced = {path for _, path in _declared_contracts()}
    assert files == referenced, f"orphan contracts: {sorted(files - referenced)}"
