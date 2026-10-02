from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from janus.hooks import built_in_hooks
from janus.models.data_contracts import DataContract
from janus.models.source_config import SourceConfig
from janus.registry.contracts import load_contract_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_ROOT = REPO_ROOT / "tests" / "fixtures" / "semantics"
ENVIRONMENT_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "environments"

CLEAN = "clean"
MULTI_ISSUE = "multi_issue"

CLEAN_PRODUCER = "semantics_clean_producer"
CLEAN_CONSUMER = "semantics_clean_consumer"

REFUSED_AT_LOAD: Mapping[str, str] = {
    "rule_a_required_not_in_schema": "quality.required_fields",
    "rule_d_schema_path_missing": "semantics_rule_d_source.schema.contract",
    "graph_cycle": "semantics_cycle_a → semantics_cycle_b → semantics_cycle_a",
    "graph_missing_producer": "semantics_orphan_consumer",
    "unregistered_variant": "strategy_variant",
}

BUILT_IN_HOOK_IDS = ", ".join(sorted(hook_id for hook_id, _ in built_in_hooks()))


@dataclass(frozen=True, slots=True)
class RuleCase:
    """One engine-owned rule, the tree that violates it, and the issue it must report."""

    rule: str
    fixture: str
    source_id: str
    path: str
    message: str

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.source_id, self.path, self.message)

    @property
    def rendered(self) -> str:
        return f"- {self.source_id}: {self.path}: {self.message}"



RULE_CASES: Mapping[str, RuleCase] = {
    "b": RuleCase(
        "b",
        "rule_b_unique_not_in_required",
        "semantics_rule_b_source",
        "schema.contract",
        "primaryKey columns must also be required: code",
    ),
    "c": RuleCase(
        "c",
        "rule_c_iceberg_column_absent",
        "semantics_rule_c_consumer",
        "access.request_inputs.columns",
        "reads column(s) the producer semantics_rule_c_producer does not declare: "
        "column_that_is_not_there",
    ),
    "e": RuleCase(
        "e",
        "rule_e_unknown_hook",
        "semantics_rule_e_source",
        "source_hook",
        f"is not a registered hook; known hooks: {BUILT_IN_HOOK_IDS}",
    ),
    "f": RuleCase(
        "f",
        "rule_f_partition_column_unknown",
        "semantics_rule_f_source",
        "spark.partition_by",
        "names column(s) that are neither normalization metadata nor contract columns: "
        "not_a_column",
    ),
}

MULTI_ISSUE_EXPECTED: tuple[tuple[str, str, str], ...] = (
    (
        "semantics_multi_alpha",
        "schema.contract",
        "primaryKey columns must also be required: code",
    ),
    (
        "semantics_multi_alpha",
        "spark.partition_by",
        "names column(s) that are neither normalization metadata nor contract columns: "
        "not_a_column",
    ),
    (
        "semantics_multi_zulu",
        "source_hook",
        f"is not a registered hook; known hooks: {BUILT_IN_HOOK_IDS}",
    ),
)

LOADS_CLEAN_TODAY: tuple[str, ...] = (
    CLEAN,
    MULTI_ISSUE,
    *(case.fixture for case in RULE_CASES.values()),
)


def fixture_root(name: str) -> Path:
    root = SEMANTICS_ROOT / name
    assert (root / "conf" / "app.yaml").is_file(), f"not a semantics fixture registry: {root}"
    return root


def materialize(name: str, workspace: Path) -> Path:
    """Copy one fixture registry into ``workspace`` so a test may write state under it."""
    shutil.copytree(fixture_root(name), workspace, dirs_exist_ok=True)
    return workspace


def install_profile(project_root: Path, name: str, *, source: Path | None = None) -> Path:
    """Copy one environment profile into ``<project_root>/conf/environments/``."""
    origin = source if source is not None else REPO_ROOT / "conf" / "environments"
    target = project_root / "conf" / "environments" / f"{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin / f"{name}.yaml", target)
    return target


def load_semantic_inputs(
    project_root: Path,
) -> tuple[tuple[SourceConfig, ...], dict[str, DataContract]]:
    root = project_root.resolve()
    sources_dir = root / "conf" / "sources"
    sources: list[SourceConfig] = []
    for config_path in sorted(sources_dir.rglob("*.yaml")):
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        entries = document.get("sources", [document])
        sources.extend(SourceConfig.from_mapping(entry, config_path) for entry in entries)
    contracts = load_contract_snapshot(sources, project_root=root, sources_dir=sources_dir)
    return tuple(sources), contracts


def tree_snapshot(root: Path) -> frozenset[tuple[str, int, int]]:
    """Every path under ``root`` with its size and mtime: equal snapshots, nothing written."""
    return frozenset(
        (str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
    )
