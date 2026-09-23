"""Render pip-audit ignore flags from time-bounded advisory exceptions."""

from __future__ import annotations

import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path

IGNORE_ENTRY = re.compile(
    r"^(?P<id>[A-Za-z0-9_.:-]+)\s+(?P<expiry>\d{4}-\d{2}-\d{2})\s+(?P<reason>\S.*)$"
)


def render_ignore_flags(path: Path, *, today: date | None = None) -> list[str]:
    """Return pip-audit flags, refusing malformed or expired exceptions."""
    current_date = today or datetime.now(tz=UTC).date()
    flags: list[str] = []

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        match = IGNORE_ENTRY.fullmatch(line)
        if match is None:
            print(
                f"{path}:{number}: expected `<ID> <YYYY-MM-DD> <reason>`",
                file=sys.stderr,
            )
            raise SystemExit(1)

        advisory_id = match.group("id")
        try:
            expiry = date.fromisoformat(match.group("expiry"))
        except ValueError as error:
            print(
                f"{path}:{number}: invalid expiry for {advisory_id}: {match.group('expiry')}",
                file=sys.stderr,
            )
            raise SystemExit(1) from error

        if expiry < current_date:
            print(
                f"{path}:{number}: {advisory_id} expired on {expiry.isoformat()}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        flags.extend(("--ignore-vuln", advisory_id))

    return flags


def main(argv: list[str] | None = None) -> int:
    """Print flags for command substitution in the CI audit step."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: python -m tests.support.pip_audit_ignores <allowlist>")
    print(" ".join(render_ignore_flags(Path(arguments[0]))))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CI command
    raise SystemExit(main())
