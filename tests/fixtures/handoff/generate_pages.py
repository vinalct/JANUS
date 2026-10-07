"""Generate deterministic API pages and real RawArtifactWriter checksum sidecars."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from janus.models import ExecutionPlan, ExtractedArtifact, RunContext, SourceConfig
from janus.strategies.api import ApiStrategy
from janus.utils.storage import StorageLayout
from janus.writers import RawArtifactWriter

FIXTURE_DIR = Path(__file__).resolve().parent
SOURCE_ID = "synthetic_pages"
LATE_FIELD_NULL_PAGES = 100


def synthetic_plan(target_dir: Path) -> ExecutionPlan:
    """Plan an offline source with the envelope schema today's reader preserves."""
    target_dir = target_dir.resolve()
    source = SourceConfig.from_mapping(
        {
            "source_id": SOURCE_ID,
            "name": "synthetic API pages",
            "description": "Deterministic Transparência-shaped raw baseline",
            "owner": "janus-tests",
            "enabled": True,
            "source_type": "api",
            "strategy": "api",
            "strategy_variant": "page_number_api",
            "federation_level": "federal",
            "domain": "tests",
            "public_access": True,
            "access": {
                "base_url": "https://example.invalid",
                "path": "/synthetic",
                "method": "GET",
                "format": "json",
                "auth": {"type": "none"},
                "pagination": {
                    "type": "page_number", "page_size": 15,
                    "page_param": "pagina", "size_param": "tamanhoPagina",
                },
                "rate_limit": {"requests_per_minute": 90, "concurrency": 1},
            },
            "extraction": {
                "mode": "full_refresh", "checkpoint_strategy": "none",
                "retry": {
                    "max_attempts": 1, "backoff_strategy": "fixed", "backoff_seconds": 1,
                },
            },
            "schema": {"contract": str(FIXTURE_DIR / "envelope_contract.yaml")},
            "spark": {
                "input_format": "json",
                "write_mode": "overwrite",
                "partition_by": ["ingestion_date"],
            },
            "outputs": {
                "raw": {"path": str(target_dir), "format": "json"},
                "bronze": {
                    "path": "data/bronze/order22/synthetic",
                    "format": "iceberg",
                    "namespace": "bronze_order22",
                    "table_name": "synthetic_pages",
                },
                "metadata": {"path": "data/metadata/order22", "format": "json"},
            },
            "quality": {},
        },
        config_path=target_dir / "synthetic_source.yaml",
    )
    context = RunContext.create(
        run_id="baseline",
        environment="local",
        project_root=target_dir,
        started_at=datetime(2026, 9, 16, 12, tzinfo=UTC),
    )
    return ApiStrategy().plan(source, context)


def generate_pages(
    target_dir: Path, *, page_count: int, records_per_page: int = 15, seed: int = 22
) -> tuple[ExtractedArtifact, ...]:
    """Write pages/page-NNNN.json plus .sha256 sidecars; return the page artifacts.

    Page numbering is one-based and widens naturally past 9,999, like production.
    Only one page of records is kept in memory. No network or Spark session is used.
    """
    if page_count < 0 or records_per_page < 1:
        raise ValueError("page_count must be nonnegative and records_per_page must be positive")
    plan = synthetic_plan(target_dir)
    layout = StorageLayout.from_environment_config(
        {"storage": {key: "data" for key in ("root_dir", "raw_dir", "bronze_dir", "metadata_dir")}},
        target_dir,
    )
    writer = RawArtifactWriter(layout)
    template = json.loads((FIXTURE_DIR / "page_template.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    artifacts = []
    for page_number in range(1, page_count + 1):
        records = []
        for offset in range(records_per_page):
            record = {
                **template,
                "id": (page_number - 1) * records_per_page + offset + 1,
                "valor": rng.randrange(100, 10_000_000) / 100,
                "ativo": bool(rng.randrange(2)),
                "observacaoTardia": None if page_number <= LATE_FIELD_NULL_PAGES else "Disponível",
            }
            records.append(record)
        persisted = writer.write_json(
            plan,
            f"pages/page-{page_number:04d}.json",
            {"data": records, "pagina": page_number, "totalPaginas": page_count},
        )
        artifacts.append(persisted.artifact)
    return tuple(artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--page-count", type=int, default=10_000)
    parser.add_argument("--records-per-page", type=int, default=15)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    artifacts = generate_pages(
        args.target_dir,
        page_count=args.page_count,
        records_per_page=args.records_per_page,
        seed=args.seed,
    )
    print(json.dumps({"page_count": len(artifacts), "seed": args.seed}))


if __name__ == "__main__":
    main()
