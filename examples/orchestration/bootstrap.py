"""Prepare only the isolated runtime state required by the orchestration example."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]

PINNED_JARS = (
    "org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar",
    "org.xerial_sqlite-jdbc-3.53.2.1.jar",
)


def prepare_example_runtime(
    project_root: Path = PROJECT_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Create local directories and seed the pinned jars without contacting Maven."""
    runtime_root = project_root / "runtime"
    ivy_jars = runtime_root / "ivy" / "jars"
    dagster_home = runtime_root / "dagster"
    fixture_state = runtime_root / "fixture"
    for path in (ivy_jars, dagster_home, fixture_state):
        path.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for filename in PINNED_JARS:
        source = repository_root / "deps" / filename
        if not source.is_file():
            raise FileNotFoundError(
                f"Pinned runtime jar is missing: {source}. Restore the repository deps/ file."
            )
        target = ivy_jars / filename
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        copied.append(str(target))

    dagster_config = project_root / "conf" / "dagster" / "dagster.yaml"
    shutil.copy2(dagster_config, dagster_home / "dagster.yaml")
    return {
        "runtime_root": str(runtime_root),
        "dagster_home": str(dagster_home),
        "ivy_jars": copied,
    }


def main() -> int:
    print(json.dumps(prepare_example_runtime(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
