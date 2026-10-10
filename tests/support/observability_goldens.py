"""Capture exact authoritative bytes in the reference container, without metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.support.observability_baseline import BASELINE_CASES, capture_case

AUTHORITATIVE_DIRECTORIES = frozenset({"runs", "lineage", "checkpoints", "validations"})


def capture_authoritative_goldens(
    destination: Path, *, project_root: Path
) -> dict[str, list[str]]:
    """Capture every terminal shape, preserving the writer's bytes and absolute paths.

    Both directories must be new. Use the same container project root when comparing
    later captures: these goldens intentionally do not normalize or reserialize JSON.
    ``capture_case`` supplies a profile with no observability block and a no-op observer.
    """
    destination = destination.resolve()
    project_root = project_root.resolve()
    if destination == project_root or destination in project_root.parents:
        raise ValueError("the capture project must be outside the golden destination")
    if project_root in destination.parents:
        raise ValueError("the golden destination must be outside the capture project")
    if destination.exists() or project_root.exists():
        raise FileExistsError("golden destination and capture project must both be new")
    destination.mkdir(parents=True)
    project_root.mkdir(parents=True)

    captured: dict[str, list[str]] = {}
    for case in BASELINE_CASES:
        root = project_root / case
        capture_case(root, case)
        metadata_root = root / "data" / "metadata"
        written: list[str] = []
        for path in sorted(metadata_root.rglob("*.json")):
            relative = path.relative_to(metadata_root)
            if not AUTHORITATIVE_DIRECTORIES.intersection(relative.parts):
                continue
            target = destination / case / "metadata" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            written.append(str(Path("metadata") / relative))
        captured[case] = written
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    for case, files in capture_authoritative_goldens(
        args.destination, project_root=args.project_root
    ).items():
        print(f"{case}: {len(files)} file(s)")
        for name in files:
            print(f"    {name}")


if __name__ == "__main__":
    main()
