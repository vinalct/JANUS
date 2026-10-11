"""The declared Transparência contract survives JSON page to JSONL compaction."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from janus.models import RunContext
from janus.normalizers import BaseNormalizer
from janus.planner import PlannedRun
from janus.readers import SparkDatasetReader
from janus.registry import load_registry
from janus.runtime.materialize import BronzeMaterializer
from janus.strategies.api import ApiStrategy
from janus.utils.storage import StorageLayout
from janus.writers import SparkDatasetWriter
from tests.fixtures.handoff.cases import ENVIRONMENT_CONFIG
from tests.integration.transparencia.test_transparencia_integration import FixtureTransport
from tests.support.bronze_digest import bronze_golden
from tests.support.contracts import with_registry_contract

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transparencia"
SOURCE_ID = "transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh"


def test_transparencia_contract_schema_and_rows_match_with_compaction_on_and_off(
    spark, tmp_path, monkeypatch
):
    config = load_registry(PROJECT_ROOT).get_source(SOURCE_ID, include_disabled=True)
    config = replace(
        config,
        access=replace(
            config.access,
            pagination=replace(config.access.pagination, page_size=2),
            rate_limit=replace(config.access.rate_limit, concurrency=1),
        ),
    )
    context = RunContext.create(
        run_id="format-seam",
        environment="local",
        project_root=tmp_path,
        started_at=datetime(2026, 9, 16, 12, tzinfo=UTC),
    )
    layout = StorageLayout.from_environment_config(ENVIRONMENT_CONFIG, tmp_path)
    transport = FixtureTransport(
        [FIXTURES_DIR / f"servidores_por_orgao_page_{index}.json" for index in (1, 2)]
    )
    strategy = ApiStrategy(
        transport_factory=lambda: transport,
        storage_layout_factory=lambda plan: layout,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )
    monkeypatch.setenv("TRANSPARENCIA_API_TOKEN", "fixture-token")
    plan = with_registry_contract(strategy.plan(config, context))
    extraction = strategy.extract(plan)
    materializer = BronzeMaterializer(SparkDatasetReader(), BaseNormalizer(), SparkDatasetWriter)
    goldens = {}
    for mode in ("pages", "compact"):
        mode_plan = replace(
            plan,
            source_config=replace(
                config,
                spark=replace(config.spark, handoff=replace(config.spark.handoff, mode=mode)),
            ),
        )
        handoff = strategy.build_normalization_handoff(mode_plan, extraction)
        assert handoff.single_artifact_format() == ("json" if mode == "pages" else "jsonl")
        assert handoff.metadata_as_dict()["handoff_compaction"] == (
            "pages" if mode == "pages" else "jsonl"
        )
        results, _, _, evidence = materializer.materialize(
            PlannedRun(mode_plan, strategy), mode_plan, spark, handoff, layout, None
        )
        assert evidence[0].malformed_count == 0
        table = results[0].path
        goldens[mode] = bronze_golden(spark.table(table), identifier=table)

    print(json.dumps({"source_id": SOURCE_ID, "goldens": goldens}, sort_keys=True))
    assert goldens["pages"]["row_count"] == "3"
    assert goldens["compact"] == goldens["pages"]
