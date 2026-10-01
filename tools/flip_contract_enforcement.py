"""Patch-bump and enable strict enforcement in the 30 checked-in contracts.

The edit is deliberately line-local so ODCS ordering and formatting stay intact.
Rerunning after a successful flip makes no changes.
"""

from __future__ import annotations

import argparse
import os
import stat
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "conf" / "contracts"
EXPECTED_COUNTS = {"active": 18, "draft": 12}
VERSIONS = {"active": ("1.0.0", "1.0.1"), "draft": ("0.1.0", "0.1.1")}


def flipped_text(path: Path) -> tuple[str, bool, str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fields: dict[str, int] = {}
    for name in ("version", "status"):
        matches = [index for index, line in enumerate(lines) if line.startswith(f"{name}: ")]
        if len(matches) != 1:
            raise ValueError(f"{path}: expected exactly one top-level {name}")
        fields[name] = matches[0]
    status = lines[fields["status"]].split(":", 1)[1].strip()
    if status not in VERSIONS:
        raise ValueError(f"{path}: unexpected status {status!r}")
    version = lines[fields["version"]].split(":", 1)[1].strip()
    matches = [
        index for index, line in enumerate(lines)
        if line.strip() == "- property: janus.enforcement"
    ]
    if len(matches) != 1 or matches[0] + 1 >= len(lines):
        raise ValueError(f"{path}: expected exactly one janus.enforcement property")
    mode_index = matches[0] + 1
    mode_line = lines[mode_index]
    if mode_line.strip() not in {"value: lenient", "value: strict"}:
        raise ValueError(f"{path}: unexpected janus.enforcement value")
    old, new = VERSIONS[status]
    if version == new and mode_line.strip() == "value: strict":
        return "".join(lines), False, status
    if version != old or mode_line.strip() != "value: lenient":
        raise ValueError(f"{path}: version/enforcement are not an expected old or new pair")
    lines[fields["version"]] = lines[fields["version"]].replace(old, new, 1)
    lines[mode_index] = mode_line.replace("value: lenient", "value: strict", 1)
    return "".join(lines), True, status


def flip(root: Path = ROOT, *, check: bool = False) -> int:
    paths = sorted(root.rglob("*.yaml"))
    if len(paths) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"{root}: expected 30 contracts, found {len(paths)}")
    prepared = [(path, *flipped_text(path)) for path in paths]
    counts = {status: sum(item[3] == status for item in prepared) for status in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"{root}: expected status counts {EXPECTED_COUNTS}, found {counts}")
    pending = sum(changed for _, _, changed, _ in prepared)
    if check:
        print(
            f"contracts={len(paths)} active={counts['active']} "
            f"draft={counts['draft']} pending={pending}"
        )
        return 1 if pending else 0

    for path, content, changed, status in prepared:
        if changed:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                handle.write(content)
                staged = Path(handle.name)
            os.chmod(staged, stat.S_IMODE(path.stat().st_mode))
            os.replace(staged, path)
        print(f"{'flipped' if changed else 'already strict'} {status} {path.relative_to(root)}")
    print(f"contracts={len(paths)} flipped={pending} already_strict={len(paths) - pending}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify that all contracts were flipped"
    )
    args = parser.parse_args()
    return flip(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
