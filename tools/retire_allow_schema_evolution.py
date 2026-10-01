"""Remove the retired ``quality.allow_schema_evolution`` key from source YAML.

Schema evolution is governed by the contract's ``janus.compatibility`` now, and the model
rejects the key by name. The edit is line-local so comments and ordering stay intact.
Rerunning after a successful retirement makes no changes.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RETIRED_KEY = "allow_schema_evolution"
ROOTS = (
    REPO_ROOT / "conf" / "sources",
    REPO_ROOT / "examples" / "orchestration" / "conf" / "sources",
    REPO_ROOT / "tests" / "fixtures" / "full_refresh_history" / "conf" / "sources",
    REPO_ROOT / "tests" / "fixtures" / "catalog_commits" / "conf" / "sources",
)


def retire_file(path: Path) -> bool:
    """Drop every standalone declaration of the retired key; report whether any existed."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if line.lstrip().split(":", 1)[0] != RETIRED_KEY]
    if len(kept) == len(lines):
        return False
    path.write_text("".join(kept), encoding="utf-8")
    return True


def main() -> None:
    changed = [path for root in ROOTS for path in sorted(root.rglob("*.yaml")) if retire_file(path)]
    for path in changed:
        print(path.relative_to(REPO_ROOT))
    print(f"Retired {RETIRED_KEY} from {len(changed)} YAML files")


if __name__ == "__main__":
    main()
