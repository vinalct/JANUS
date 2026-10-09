"""Profile values enter source parsing explicitly, without a session or process state."""

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from janus.models import SparkWriteDefaults
from janus.planner import Planner, PlanningRequest
from janus.registry import SourceRegistry, load_registry
from janus.registry.loader import _load_source_configs
from janus.utils.environment import load_environment_config, read_spark_write_defaults
from tests.fixtures.handoff.cases import source_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def clear_write_environment(monkeypatch):
    monkeypatch.delenv("JANUS_SPARK_TARGET_FILE_SIZE_MB", raising=False)
    monkeypatch.delenv("JANUS_SPARK_DISTRIBUTION_MODE", raising=False)


@pytest.mark.parametrize(("profile", "size"), [("local", 128), ("cluster", 256)])
def test_profile_defaults_load_and_plan_all_31_sources(profile, size):
    defaults = read_spark_write_defaults(load_environment_config(profile, PROJECT_ROOT))
    assert defaults == SparkWriteDefaults(size, None)
    registry = load_registry(PROJECT_ROOT, write_defaults=defaults)
    assert len(registry.sources) == 31
    planner = Planner()
    for source in registry.sources:
        planned = planner.plan(
            PlanningRequest.create(
                source_id=source.source_id, environment=profile,
                project_root=PROJECT_ROOT, include_disabled=True,
            ),
            registry=registry,
        )
        write = planned.plan.source_config.spark.write
        assert write.target_file_size_mb == size
        assert write.distribution_mode == ("hash" if source.spark.partition_by else "none")
        assert write.partitions is None


def test_environment_overrides_profile_values(monkeypatch):
    monkeypatch.setenv("JANUS_SPARK_TARGET_FILE_SIZE_MB", "64")
    monkeypatch.setenv("JANUS_SPARK_DISTRIBUTION_MODE", "range")
    for profile in ("local", "cluster"):
        defaults = read_spark_write_defaults(load_environment_config(profile, PROJECT_ROOT))
        assert defaults == SparkWriteDefaults(64, "range")


@pytest.mark.parametrize("mode", [None, "", "   "])
def test_empty_profile_distribution_is_undeclared(mode):
    config = {"spark": {"write_defaults": {"distribution_mode": mode}}}
    assert read_spark_write_defaults(config) == SparkWriteDefaults()


def test_absent_profile_block_is_valid():
    assert read_spark_write_defaults({}) == SparkWriteDefaults()
    assert read_spark_write_defaults({"spark": {}}) == SparkWriteDefaults()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("target_file_size_mb", 0, "must be >= 1"),
        ("target_file_size_mb", True, "must be an integer"),
        ("target_file_size_mb", 1.5, "must be an integer"),
        ("target_file_size_mb", "128000", "must be <= 8192"),
        ("target_file_size_mb", "invalid", "must be an integer"),
        ("distribution_mode", "hashh", "must be one of: hash, none, range"),
        ("distribution_mode", False, "must be one of: hash, none, range"),
    ],
)
def test_invalid_profile_values_have_precise_issues(key, value, message):
    with pytest.raises(ValueError) as raised:
        read_spark_write_defaults({"spark": {"write_defaults": {key: value}}})
    assert str(raised.value) == (
        f"Invalid Spark write defaults:\nspark.write_defaults.{key}: {message}"
    )


@pytest.mark.parametrize("value", [False, "hash", []])
def test_malformed_profile_block_is_rejected(value):
    with pytest.raises(ValueError, match="spark.write_defaults: must be a mapping"):
        read_spark_write_defaults({"spark": {"write_defaults": value}})


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("JANUS_SPARK_TARGET_FILE_SIZE_MB", "128000", "must be <= 8192"),
        ("JANUS_SPARK_DISTRIBUTION_MODE", "hashh", "must be one of: hash, none, range"),
    ],
)
def test_invalid_environment_override_is_rejected_at_profile_load(
    monkeypatch, variable, value, message
):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=message):
        load_environment_config("local", PROJECT_ROOT)


@pytest.mark.parametrize("grouped", [False, True])
def test_profile_defaults_reach_single_and_grouped_source_parsers(tmp_path, grouped):
    config_path = tmp_path / "sources.yaml"
    plain = source_mapping()
    override = source_mapping(spark={"write": {"target_file_size_mb": 32}})
    document = {"sources": [plain, override]} if grouped else override
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    defaults = SparkWriteDefaults(256, "range")
    sources = _load_source_configs(config_path, write_defaults=defaults)
    assert sources[-1][0].spark.write.target_file_size_mb == 32
    assert all(source.spark.write.distribution_mode == "range" for source, _entry in sources)
    if grouped:
        assert sources[0][0].spark.write.target_file_size_mb == 256


def test_registry_loads_do_not_retain_profile_defaults():
    cluster = SourceRegistry.load(PROJECT_ROOT, write_defaults=SparkWriteDefaults(256))
    builtin = SourceRegistry.load(PROJECT_ROOT)
    assert all(source.spark.write.target_file_size_mb == 256 for source in cluster.sources)
    assert all(source.spark.write.target_file_size_mb == 128 for source in builtin.sources)
    assert "write_defaults" not in {field.name for field in fields(SourceRegistry)}
