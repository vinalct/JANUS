from pathlib import Path

import pytest

from janus.models.config.outputs import _build_spark_config
from janus.models.source_config import SourceConfig, SourceConfigValidationError
from tests.fixtures.handoff.cases import source_mapping


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize(
    ("declaration", "mode", "size"),
    [
        (None, "compact", 256),
        ({"mode": "pages"}, "pages", 256),
        ("pages", "pages", 256),
        ({"mode": "compact", "max_artifact_mb": 2}, "compact", 2),
    ],
)
def test_handoff_defaults_and_supported_shapes(declaration, mode, size):
    spark = {} if declaration is None else {"handoff": declaration}
    config = SourceConfig.from_mapping(source_mapping(spark=spark), Path("source.yaml"))
    assert config.spark.handoff.mode == mode
    assert config.spark.handoff.max_artifact_mb == size
    assert config.spark.handoff.max_artifact_bytes == size * 1024 * 1024
    assert config.spark.handoff.is_compacting == (mode == "compact")


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize(
    ("declaration", "path"),
    [
        ("nonsense", "spark.handoff.mode"),
        ({"max_artifact_mb": 0}, "spark.handoff.max_artifact_mb"),
        ({"max_artifact_mb": -1}, "spark.handoff.max_artifact_mb"),
        ({"max_artifact_mb": 1.5}, "spark.handoff.max_artifact_mb"),
        ([], "spark.handoff"),
        (7, "spark.handoff"),
    ],
)
def test_invalid_handoff_collects_one_issue_without_raising_in_builder(declaration, path):
    issues = []
    _build_spark_config(source_mapping(spark={"handoff": declaration})["spark"], issues)
    assert [issue.path for issue in issues] == [path]


@pytest.mark.xfail(strict=True, reason="green")
def test_handoff_error_does_not_mask_unrelated_error():
    mapping = source_mapping(spark={"handoff": "nonsense"})
    mapping["owner"] = None
    with pytest.raises(SourceConfigValidationError) as raised:
        SourceConfig.from_mapping(mapping, Path("source.yaml"))
    assert [issue.path for issue in raised.value.issues] == ["owner", "spark.handoff.mode"]
