"""Inspect only an integration subprocess's isolated session and drain its events."""

import os
import select
import time
from pathlib import Path

from tests.support.parallel import parse_log_events


def session_processes(session_id):
    """Read ps's session fields from procfs, available in the pinned slim image."""
    rows = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            stat = (directory / "stat").read_text()
            end = stat.rindex(")")
            fields = stat[end + 2 :].split()
            if int(fields[3]) == session_id:
                rows.append(
                    {
                        "pid": int(directory.name),
                        "ppid": int(fields[1]),
                        "pgid": int(fields[2]),
                        "session_id": int(fields[3]),
                        "command": stat[stat.index("(") + 1 : end],
                        "state": fields[0],
                    }
                )
        except (OSError, ValueError, IndexError):
            continue
    return sorted(rows, key=lambda row: row["pid"])


def wait_for_events(process, event, *, count=1, timeout=30):
    """Drain stderr as it arrives so the CLI cannot fill its pipe while starting Spark."""
    deadline = time.monotonic() + timeout
    captured = b""
    while time.monotonic() < deadline:
        ready, _, _ = select.select([process.stderr], [], [], max(0, deadline - time.monotonic()))
        if not ready:
            break
        data = os.read(process.stderr.fileno(), 65536)
        if not data:
            break
        captured += data
        # Ignore a trailing partial JSON line until the next read completes it.
        complete = captured[: captured.rfind(b"\n") + 1].decode("utf-8", errors="replace")
        if sum(item["event"] == event for item in parse_log_events(complete)) >= count:
            return captured.decode("utf-8", errors="replace")
    raise AssertionError(
        f"CLI did not emit {count} {event} events: {captured.decode(errors='replace')}"
    )


def wait_for_session_exit(session_id, *, timeout=5):
    deadline = time.monotonic() + timeout
    while rows := session_processes(session_id):
        if time.monotonic() >= deadline:
            raise AssertionError(f"Interrupted CLI left processes in its session: {rows}")
        time.sleep(0.05)
    return rows
