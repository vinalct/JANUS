from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from janus.models.config.outputs import _build_spark_config
from janus.models.source_config import (
    SourceConfig,
    SourceConfigValidationError,
    SparkWriteConfig,
    SparkWriteDefaults,
    ValidationIssue,
)
from tests.fixtures.handoff.cases import source_mapping

DEPRECATION = ValidationIssue(
    "spark.repartition",
    "is deprecated and will be removed in the next order; declare spark.write.partitions instead",
)
CONFLICT = ValidationIssue(
    "spark.write.partitions",
    "must not be declared together with the deprecated spark.repartition",
)


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


def test_null_partitions_means_no_exact_partition_count():
    config = SourceConfig.from_mapping(
        source_mapping(spark={"write": {"partitions": None}}), Path("source.yaml")
    )
    assert config.spark.write.partitions is None


def test_repartition_loads_as_write_partitions_with_one_deprecation_and_no_error():
    config = SourceConfig.from_mapping(
        source_mapping(spark={"repartition": 4}), Path("source.yaml")
    )
    assert config.spark.write.partitions == 4
    assert config.spark.repartition == 4
    assert config.deprecations == (DEPRECATION,)


@pytest.mark.parametrize("partitions", [2, None])
def test_legacy_and_new_partitions_together_are_rejected(partitions):
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(
            source_mapping(spark={"repartition": 4, "write": {"partitions": partitions}}),
            Path("source.yaml"),
        )
    assert raised.value.issues == (CONFLICT,)
    assert raised.value.deprecations == ()


def test_deprecation_never_masks_an_unrelated_error_and_survives_the_raise():
    mapping = source_mapping(spark={"repartition": 4})
    mapping["owner"] = None
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(mapping, Path("source.yaml"))
    assert [issue.path for issue in raised.value.issues] == ["owner"]
    assert raised.value.deprecations == (DEPRECATION,)
    assert str(raised.value) == "Invalid source config: source.yaml\n- owner: is required"


def test_validation_issue_keeps_its_two_field_collect_all_contract():
    assert [field.name for field in fields(ValidationIssue)] == ["path", "message"]


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


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("target_file_size_mb", 0, "must be >= 1"),
        ("target_file_size_mb", -1, "must be >= 1"),
        ("target_file_size_mb", 1.5, "must be an integer"),
        ("target_file_size_mb", True, "must be an integer"),
        ("target_file_size_mb", "128", "must be an integer"),
        ("target_file_size_mb", 8193, "must be <= 8192"),
        ("target_file_size_mb", 128000, "must be <= 8192"),
        ("distribution_mode", "hashh", "must be one of: hash, none, range"),
        ("distribution_mode", "", "must be one of: hash, none, range"),
        ("distribution_mode", 1, "must be one of: hash, none, range"),
        ("distribution_mode", [], "must be one of: hash, none, range"),
        ("partitions", 0, "must be >= 1"),
        ("partitions", -1, "must be >= 1"),
        ("partitions", 1.5, "must be an integer"),
        ("partitions", False, "must be an integer"),
        ("partitions", "4", "must be an integer"),
    ],
)
def test_invalid_write_values_have_one_precise_issue(key, value, message):
    issues = []
    _build_spark_config(source_mapping(spark={"write": {key: value}})["spark"], issues)
    assert issues == [ValidationIssue(f"spark.write.{key}", message)]


@pytest.mark.parametrize("value", [False, 128, "hash", []])
def test_write_block_requires_a_mapping(value):
    issues = []
    _build_spark_config(source_mapping(spark={"write": value})["spark"], issues)
    assert issues == [ValidationIssue("spark.write", "must be a mapping")]


def test_write_builder_collects_all_errors_without_hiding_other_source_errors():
    mapping = source_mapping(
        spark={"write": {"target_file_size_mb": 0, "distribution_mode": "bad", "partitions": -1}}
    )
    mapping["owner"] = None
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(mapping, Path("source.yaml"))
    assert [issue.path for issue in raised.value.issues] == [
        "owner", "spark.write.target_file_size_mb", "spark.write.distribution_mode",
        "spark.write.partitions",
    ]


@pytest.mark.parametrize("size", [1, 8192])
def test_file_size_limits_are_inclusive(size):
    config = SourceConfig.from_mapping(
        source_mapping(spark={"write": {"target_file_size_mb": size}}), Path("source.yaml")
    )
    assert config.spark.write.target_file_size_bytes == size * 1024**2


@pytest.mark.parametrize("partitions", [None, 1, 4])
def test_exact_partition_count_is_preserved(partitions):
    config = SourceConfig.from_mapping(
        source_mapping(spark={"write": {"partitions": partitions}}), Path("source.yaml")
    )
    assert config.spark.write.partitions == partitions


def test_profile_defaults_are_injected_per_load_without_mutation_or_global_state():
    mapping = source_mapping(spark={"write": {"target_file_size_mb": 32}})
    profile = SparkWriteDefaults(target_file_size_mb=256, distribution_mode="range")
    injected = SourceConfig.from_mapping(mapping, Path("source.yaml"), write_defaults=profile)
    builtin = SourceConfig.from_mapping(mapping, Path("source.yaml"))
    assert injected.spark.write == SparkWriteConfig(32, "range")
    assert builtin.spark.write == SparkWriteConfig(32, "none")
    assert profile == SparkWriteDefaults(256, "range")
    assert mapping["spark"]["write"] == {"target_file_size_mb": 32}


def test_write_config_is_frozen_and_has_slots():
    write = SparkWriteConfig()
    with pytest.raises(FrozenInstanceError):
        write.partitions = 4
    assert not hasattr(write, "__dict__")


def test_legacy_repartition_preserves_the_model_field_for_one_order():
    config = SourceConfig.from_mapping(
        source_mapping(spark={"repartition": 4}), Path("source.yaml")
    )
    assert config.spark.repartition == 4
    assert config.spark.write.partitions == 4


@pytest.mark.parametrize("write", [None, {}, {"target_file_size_mb": 32}])
def test_legacy_shim_preserves_other_write_overrides_and_profile_defaults(write):
    mapping = source_mapping(spark={"repartition": 4, "write": write})
    config = SourceConfig.from_mapping(
        mapping, Path("source.yaml"), write_defaults=SparkWriteDefaults(256, "range")
    )
    assert config.spark.write == SparkWriteConfig(
        32 if write else 256, "range", partitions=4
    )
    assert mapping["spark"]["write"] == write
    assert config.deprecations == (DEPRECATION,)
    with pytest.raises(FrozenInstanceError):
        config.deprecations = ()


@pytest.mark.parametrize("value", [0, -1, True, "4", 1.5])
def test_invalid_legacy_value_still_raises_and_carries_deprecation(value):
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(
            source_mapping(spark={"repartition": value}), Path("source.yaml")
        )
    assert [issue.path for issue in raised.value.issues] == ["spark.repartition"]
    assert raised.value.deprecations == (DEPRECATION,)


def test_legacy_deprecation_preserves_multi_issue_order_and_rendering():
    mapping = source_mapping(spark={"repartition": 0, "partition_by": [1]})
    mapping["owner"] = None
    mapping["outputs"]["bronze"]["path"] = None
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(mapping, Path("source.yaml"))
    assert [issue.path for issue in raised.value.issues] == [
        "owner", "spark.repartition", "spark.partition_by[0]", "outputs.bronze.path"
    ]
    assert raised.value.deprecations == (DEPRECATION,)


@pytest.mark.parametrize("spark", [{}, {"write": {"partitions": 4}}])
def test_current_config_has_no_deprecations(spark):
    config = SourceConfig.from_mapping(source_mapping(spark=spark), Path("source.yaml"))
    assert config.deprecations == ()


def test_partitioned_source_can_explicitly_disable_distribution():
    config = SourceConfig.from_mapping(
        source_mapping(
            spark={"partition_by": ["ingestion_date"], "write": {"distribution_mode": "none"}}
        ),
        Path("source.yaml"),
        write_defaults=SparkWriteDefaults(256, "hash"),
    )
    assert config.spark.write.distribution_mode == "none"
    assert config.spark.write.target_file_size_mb == 256
