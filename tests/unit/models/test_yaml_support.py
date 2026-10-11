"""The accelerated YAML reader preserves safe parsing and configuration values."""

from __future__ import annotations

import io
import runpy
from pathlib import Path

import pytest
import yaml

from janus.models.config import yaml_support

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOADERS = [yaml.SafeLoader]
if hasattr(yaml, "CSafeLoader"):
    LOADERS.append(yaml.CSafeLoader)


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "conf").rglob("*.yaml")))
def test_checked_in_configuration_matches_the_original_safe_parser(path):
    document = path.read_bytes()
    assert yaml_support.safe_load_yaml(document) == yaml.safe_load(document)


@pytest.mark.parametrize("loader", LOADERS, ids=lambda loader: loader.__name__)
@pytest.mark.parametrize(
    "document",
    [
        "!!python/object:types.SimpleNamespace {value: 1}",
        "!!python/name:os.getcwd ''",
        "!!python/tuple [1, 2]",
        "{broken: [yaml",
        "first: document\n---\nsecond: document\n",
    ],
)
def test_both_backends_reject_unsafe_tags_and_invalid_documents(monkeypatch, loader, document):
    monkeypatch.setattr(yaml_support, "_SAFE_LOADER", loader)
    with pytest.raises(yaml.YAMLError):
        yaml_support.safe_load_yaml(document)


@pytest.mark.parametrize("loader", LOADERS, ids=lambda loader: loader.__name__)
def test_streams_aliases_merges_and_typed_scalars_keep_their_values(monkeypatch, loader):
    document = (
        "defaults: &defaults {enabled: true, missing: null, version: '1.0.0'}\n"
        "source: {<<: *defaults, started_at: 2026-10-11, size: 42}\n"
    )
    monkeypatch.setattr(yaml_support, "_SAFE_LOADER", loader)
    assert yaml_support.safe_load_yaml(io.StringIO(document)) == yaml.safe_load(document)


def test_import_without_libyaml_uses_the_python_safe_parser(monkeypatch):
    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    namespace = runpy.run_path(yaml_support.__file__)
    load = namespace["safe_load_yaml"]
    assert load("enabled: true\n") == {"enabled": True}
    with pytest.raises(yaml.YAMLError):
        load("!!python/tuple [1, 2]")
