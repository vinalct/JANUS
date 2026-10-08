from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from janus.models.config.outputs import _build_spark_config
from janus.models.source_config import SourceConfig, SourceConfigValidationError, ValidationIssue
from tests.fixtures.handoff.cases import source_mapping


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize(
    ("partition_by", "mode"),
    [((), "none"), (("ingestion_date",), "hash")],
)
def test_absent_write_uses_builtin_defaults_and_partition_aware_distribution(partition_by, mode):
    config = SourceConfig.from_mapping(
        source_mapping(spark={"partition_by": list(partition_by)}), Path("source.yaml")
    )
    assert config.spark.write.target_file_size_mb == 128
    assert config.spark.write.target_file_size_bytes == 134217728
    assert config.spark.write.distribution_mode == mode
    assert config.spark.write.partitions is None


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize(
    ("declaration", "profile_size", "profile_mode", "expected_size", "expected_mode"),
    [
        (None, None, None, 128, "none"),
        (None, 256, "range", 256, "range"),
        ({}, 256, None, 256, "none"),
        ({"target_file_size_mb": 64}, 256, "range", 64, "range"),
        ({"distribution_mode": "hash"}, 256, "range", 256, "hash"),
        ({"target_file_size_mb": 32, "distribution_mode": "none"}, 256, "range", 32, "none"),
    ],
)
def test_source_overrides_profile_overrides_builtin(
    declaration, profile_size, profile_mode, expected_size, expected_mode
):
    from janus.models.config.outputs import _build_spark_write_config

    # Profile parsing owns the concrete defaults type; this spy pins its value interface.
    defaults = (
        SimpleNamespace(target_file_size_mb=profile_size, distribution_mode=profile_mode)
        if profile_size is not None
        else None
    )
    issues = []
    write = _build_spark_write_config(
        declaration,
        issues,
        partition_by=(),
        defaults=defaults,
    )
    assert issues == []
    assert write.target_file_size_mb == expected_size
    assert write.distribution_mode == expected_mode


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize(
    ("declaration", "path"),
    [
        ({"target_file_size_mb": 0}, "spark.write.target_file_size_mb"),
        ({"target_file_size_mb": -1}, "spark.write.target_file_size_mb"),
        ({"target_file_size_mb": 1.5}, "spark.write.target_file_size_mb"),
        ({"distribution_mode": "random"}, "spark.write.distribution_mode"),
        ({"partitions": 0}, "spark.write.partitions"),
    ],
)
def test_invalid_write_value_collects_exactly_one_issue(declaration, path):
    issues = []
    _build_spark_config(source_mapping(spark={"write": declaration})["spark"], issues)
    assert [issue.path for issue in issues] == [path]
    if path.endswith("distribution_mode"):
        assert all(value in issues[0].message for value in ("none", "hash", "range"))


@pytest.mark.xfail(strict=True, reason="green")
def test_null_partitions_means_no_exact_partition_count():
    config = SourceConfig.from_mapping(
        source_mapping(spark={"write": {"partitions": None}}), Path("source.yaml")
    )
    assert config.spark.write.partitions is None


@pytest.mark.xfail(strict=True, reason="green")
def test_repartition_loads_as_write_partitions_with_one_deprecation_and_no_error():
    config = SourceConfig.from_mapping(
        source_mapping(spark={"repartition": 4}), Path("source.yaml")
    )
    assert config.spark.write.partitions == 4
    assert len(config.deprecations) == 1
    assert config.deprecations[0].path == "spark.repartition"
    assert "spark.write.partitions" in config.deprecations[0].message


@pytest.mark.xfail(strict=True, reason="green")
def test_legacy_and_new_partitions_together_are_rejected():
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(
            source_mapping(spark={"repartition": 4, "write": {"partitions": 2}}),
            Path("source.yaml"),
        )
    assert len(raised.value.issues) == 1
    assert raised.value.issues[0].path in {"spark.repartition", "spark.write.partitions"}
    assert "repartition" in raised.value.issues[0].message
    assert "write.partitions" in raised.value.issues[0].message


@pytest.mark.xfail(strict=True, reason="green")
def test_deprecation_never_masks_an_unrelated_error_and_survives_the_raise():
    mapping = source_mapping(spark={"repartition": 4})
    mapping["owner"] = None
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(mapping, Path("source.yaml"))
    assert [issue.path for issue in raised.value.issues] == ["owner"]
    assert len(raised.value.deprecations) == 1
    assert "spark.write.partitions" in raised.value.deprecations[0].message


def test_validation_issue_keeps_its_two_field_collect_all_contract():
    assert [field.name for field in fields(ValidationIssue)] == ["path", "message"]


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize("mode", ["none", "hash", "range"])
def test_valid_source_write_override_preserves_all_declared_fields(mode):
    config = SourceConfig.from_mapping(
        source_mapping(
            spark={
                "write": {
                    "target_file_size_mb": 32,
                    "distribution_mode": mode,
                    "partitions": 4,
                }
            }
        ),
        Path("source.yaml"),
    )
    assert config.spark.write.target_file_size_mb == 32
    assert config.spark.write.target_file_size_bytes == 33554432
    assert config.spark.write.distribution_mode == mode
    assert config.spark.write.partitions == 4
