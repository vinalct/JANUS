from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import yaml

from janus.lineage import ConfiguredOutput, RunMetadata
from janus.lineage.persistence import write_json_atomic
from tests.support.contracts import write_keyed_contract
from tests.support.observability_baseline import source_payload

# (name, declared max_age, expected outcome, (age in hours, status) history)
SOURCE_CASES = (
    ("fresh", "24h", "fresh", ((30, "succeeded"), (12, "failed"), (2, "succeeded"))),
    ("stale", "24h", "stale", ((50, "succeeded"), (1, "failed"))),
    ("never", "6h", "never_ran", ((8, "failed"), (1, "failed"))),
    ("undeclared", None, "undeclared", ((200, "succeeded"),)),
    ("unreadable", "24h", "fresh", ((3, "succeeded"),)),
    ("empty", "24h", "never_ran", ()),
)


@dataclass(frozen=True, slots=True)
class MetadataSourceFixture:
    source_id: str
    config_path: Path
    metadata_dir: Path
    max_age: str | None
    expected_outcome: str
    runs: tuple[RunMetadata, ...]
    unreadable_run_files: int = 0


@dataclass(frozen=True, slots=True)
class MetadataZoneFixture:
    root: Path
    now: datetime
    seed: int
    sources: tuple[MetadataSourceFixture, ...]

    @property
    def expected_outcomes(self) -> dict[str, str]:
        return {source.source_id: source.expected_outcome for source in self.sources}


def build_metadata_zone(root: Path, *, now: datetime, seed: int = 0) -> MetadataZoneFixture:
    """Write a per-source metadata zone covering every ``status`` outcome.

    ``root`` is a new project root, containing ``conf/`` and ``data/metadata/``.
    Record config paths are project-relative so the same seed and instant yield the
    same bytes under different temporary roots. No production parser is mocked.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(UTC)
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("metadata fixture root must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    sources_dir = root / "conf" / "sources" / "fixture"
    sources_dir.mkdir(parents=True)
    (root / "conf" / "app.yaml").write_text(
        "registry:\n  sources_dir: conf/sources\n  file_pattern: '*.yaml'\n", encoding="utf-8"
    )
    write_keyed_contract(root)
    generator = random.Random(seed)
    sources: list[MetadataSourceFixture] = []
    for name, max_age, outcome, history in SOURCE_CASES:
        source_id = f"fixture.{name}"
        config_path = sources_dir / f"{name}.yaml"
        document = source_payload("api_success")
        document.update(source_id=source_id, name=f"Freshness fixture: {name}")
        document["tags"] = ["freshness-fixture"]
        document["access"]["path"] = f"/{name}"
        for zone, output in document["outputs"].items():
            output["path"] = f"data/{zone}/fixture/{name}"
        if max_age is not None:
            document["operations"] = {"freshness": {"max_age": max_age}}
        config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        metadata_dir = root / document["outputs"]["metadata"]["path"]
        metadata_dir.mkdir(parents=True)
        outputs = tuple(
            ConfiguredOutput(zone=zone, path=output["path"], format=output["format"])
            for zone, output in document["outputs"].items()
        )
        runs: list[RunMetadata] = []
        for age_hours, status in history:
            ended_at = now - timedelta(hours=age_hours)
            run = RunMetadata(
                run_id=str(UUID(int=generator.getrandbits(128), version=4)),
                source_id=source_id,
                source_name=document["name"],
                environment="local",
                strategy_family=document["strategy"],
                strategy_variant=document["strategy_variant"],
                extraction_mode=document["extraction"]["mode"],
                checkpoint_strategy=document["extraction"]["checkpoint_strategy"],
                status=status,
                started_at=ended_at - timedelta(minutes=5),
                ended_at=ended_at,
                duration_seconds=300.0,
                source_config_path=str(config_path.relative_to(root)),
                configured_outputs=outputs,
                failure_reason="fixture failure" if status == "failed" else None,
                error_type="RuntimeError" if status == "failed" else None,
            )
            write_json_atomic(metadata_dir / "runs" / f"{run.run_id}.json", run.to_dict())
            runs.append(run)
        unreadable = int(name == "unreadable")
        if unreadable:
            (metadata_dir / "runs" / "truncated.json").write_bytes(b'{"run_id":')
        sources.append(
            MetadataSourceFixture(
                source_id, config_path, metadata_dir, max_age, outcome, tuple(runs), unreadable
            )
        )
    return MetadataZoneFixture(root, now, seed, tuple(sources))
