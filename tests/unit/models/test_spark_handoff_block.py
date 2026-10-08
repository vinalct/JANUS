from pathlib import Path

import pytest

from janus.models.config.outputs import _build_spark_config
from janus.models.source_config import (
    SourceConfig,
    SourceConfigValidationError,
    SparkConfig,
    SparkHandoffConfig,
)
from tests.fixtures.handoff.cases import source_mapping


@pytest.mark.parametrize(
    ("declaration", "mode", "size"),
    [
        (None, "compact", 256),
        ({}, "compact", 256),
        ("compact", "compact", 256),
        ({"mode": "pages"}, "pages", 256),
        ("pages", "pages", 256),
        ({"mode": "pages", "max_artifact_mb": 512}, "pages", 512),
        ({"mode": "compact", "max_artifact_mb": 2}, "compact", 2),
        ({"max_artifact_mb": 1}, "compact", 1),
        ({"future_option": "ignored"}, "compact", 256),
    ],
)
def test_handoff_defaults_and_supported_shapes(declaration, mode, size):
    spark = {} if declaration is None else {"handoff": declaration}
    config = SourceConfig.from_mapping(source_mapping(spark=spark), Path("source.yaml"))
    assert config.spark.handoff.mode == mode
    assert config.spark.handoff.max_artifact_mb == size
    assert config.spark.handoff.max_artifact_bytes == size * 1024 * 1024
    assert config.spark.handoff.is_compacting == (mode == "compact")


@pytest.mark.parametrize(
    ("declaration", "path", "message"),
    [
        ("nonsense", "spark.handoff.mode", "must be one of: compact, pages"),
        ({"mode": "parquet"}, "spark.handoff.mode", "must be one of: compact, pages"),
        ({"mode": 7}, "spark.handoff.mode", "must be a string"),
        ("", "spark.handoff.mode", "must not be empty"),
        ({"max_artifact_mb": 0}, "spark.handoff.max_artifact_mb", "must be >= 1"),
        ({"max_artifact_mb": -1}, "spark.handoff.max_artifact_mb", "must be >= 1"),
        ({"max_artifact_mb": 1.5}, "spark.handoff.max_artifact_mb", "must be >= 1"),
        ({"max_artifact_mb": True}, "spark.handoff.max_artifact_mb", "must be >= 1"),
        ({"max_artifact_mb": "256"}, "spark.handoff.max_artifact_mb", "must be >= 1"),
        ([], "spark.handoff", "must be a mapping"),
        (7, "spark.handoff", "must be a mapping"),
        (True, "spark.handoff", "must be a mapping"),
    ],
)
def test_invalid_handoff_collects_one_issue_without_raising_in_builder(declaration, path, message):
    issues = []
    _build_spark_config(source_mapping(spark={"handoff": declaration})["spark"], issues)
    assert [issue.path for issue in issues] == [path]
    assert issues[0].render() == f"{path}: {message}"
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(
            source_mapping(spark={"handoff": declaration}), Path("source.yaml")
        )
    assert raised.value.issues == tuple(issues)


def test_handoff_error_does_not_mask_unrelated_error():
    mapping = source_mapping(spark={"handoff": "nonsense"})
    mapping["owner"] = None
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(mapping, Path("source.yaml"))
    assert [issue.path for issue in raised.value.issues] == ["owner", "spark.handoff.mode"]


def test_absent_handoff_has_no_validation_issue():
    issues = []
    config = _build_spark_config(source_mapping()["spark"], issues)
    assert issues == []
    assert config.handoff == SparkHandoffConfig()


def test_direct_spark_config_construction_keeps_handoff_defaults():
    config = SparkConfig(input_format="json", write_mode="append")
    assert config.handoff == SparkHandoffConfig()
    assert config.handoff.is_compacting
    assert config.handoff.max_artifact_bytes == 268435456


def test_invalid_mode_and_size_are_collected_with_unrelated_error_in_order():
    mapping = source_mapping(spark={"handoff": {"mode": "parquet", "max_artifact_mb": 0}})
    mapping["owner"] = None
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(mapping, Path("source.yaml"))
    assert [issue.render() for issue in raised.value.issues] == [
        "owner: is required",
        "spark.handoff.mode: must be one of: compact, pages",
        "spark.handoff.max_artifact_mb: must be >= 1",
    ]
