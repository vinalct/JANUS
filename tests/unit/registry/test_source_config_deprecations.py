"""Legacy declarations remain usable and warn once per file, including grouped entries."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from janus.models import SourceConfigValidationError, SparkWriteDefaults, ValidationIssue
from janus.registry import load_registry
from tests.fixtures.handoff.cases import source_mapping
from tests.support.contracts import write_minimal_contract

MESSAGE = (
    "is deprecated and will be removed in the next order; declare spark.write.partitions instead"
)


def _project(root: Path, entries: int, *, legacy: bool = True) -> Path:
    sources = root / "conf/sources"
    sources.mkdir(parents=True)
    (root / "conf/app.yaml").write_text("registry:\n  sources_dir: conf/sources\n")
    write_minimal_contract(root)
    mappings = []
    for index in range(entries):
        mapping = source_mapping(spark={"repartition": 4} if legacy else {})
        mapping["source_id"] = f"legacy_{index}"
        mapping["outputs"]["bronze"]["path"] = f"data/bronze/legacy_{index}"
        mappings.append(mapping)
    document = {"sources": mappings} if entries > 1 else mappings[0]
    config_path = sources / "legacy.yaml"
    config_path.write_text(yaml.safe_dump(document, sort_keys=False))
    return config_path


@pytest.mark.parametrize("entries", [1, 4])
def test_loader_warns_once_per_config_path_and_preserves_each_deprecation(tmp_path, entries):
    config_path = _project(tmp_path, entries)
    with pytest.warns(DeprecationWarning) as emitted:
        registry = load_registry(tmp_path, write_defaults=SparkWriteDefaults(256, "range"))
    assert len(emitted) == 1
    assert str(emitted[0].message) == f"{config_path}: spark.repartition: {MESSAGE}"
    assert len(registry.sources) == entries
    for source in registry.sources:
        assert source.deprecations == (ValidationIssue("spark.repartition", MESSAGE),)
        assert source.spark.repartition == source.spark.write.partitions == 4
        assert source.spark.write.target_file_size_mb == 256
        assert source.spark.write.distribution_mode == "range"


def test_loader_warns_once_for_each_deprecated_file_in_each_load(tmp_path):
    first = _project(tmp_path, 1)
    document = yaml.safe_load(first.read_text())
    document["source_id"] = "other"
    document["outputs"]["bronze"]["path"] = "data/bronze/other"
    second = first.with_name("other.yaml")
    second.write_text(yaml.safe_dump(document))
    for _ in range(2):
        with pytest.warns(DeprecationWarning) as emitted:
            registry = load_registry(tmp_path)
        assert len(registry.sources) == 2
        assert [str(warning.message) for warning in emitted] == [
            f"{path}: spark.repartition: {MESSAGE}" for path in (first, second)
        ]


def test_current_config_emits_no_warning(tmp_path):
    _project(tmp_path, 4, legacy=False)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", DeprecationWarning)
        registry = load_registry(tmp_path)
    assert all(source.deprecations == () for source in registry.sources)
    assert emitted == []


def test_grouped_error_preserves_deprecations_and_entry_issue_paths(tmp_path):
    path = _project(tmp_path, 4)
    document = yaml.safe_load(path.read_text())
    document["sources"][1]["owner"] = None
    document["sources"][3]["owner"] = None
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(SourceConfigValidationError) as raised:
        load_registry(tmp_path)
    assert raised.value.issues == (
        ValidationIssue("sources[1].owner", "is required"),
        ValidationIssue("sources[3].owner", "is required"),
    )
    assert [issue.path for issue in raised.value.deprecations] == [
        f"sources[{index}].spark.repartition" for index in range(4)
    ]
