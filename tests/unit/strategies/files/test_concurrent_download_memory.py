"""AC-5/D-18: RSS measurements must start in a fresh process, never pytest's high-water mark."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from janus.writers import SPOOL_THRESHOLD_BYTES

pytestmark = pytest.mark.xfail(
    strict=True, reason="multi-file memory probe flags missing"
)
RSS_SLACK_BYTES = 32 * 2**20
THREAD_SLACK_BYTES = 8 * 2**20
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("concurrency", [1, 4])
def test_four_large_downloads_cost_spool_buffers_instead_of_payloads(tmp_path, concurrency):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(Path(path or ROOT).resolve()) for path in sys.path)
    probe = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.support.memory_probe",
            "256",
            "--files",
            "4",
            "--concurrency",
            str(concurrency),
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert probe.returncode == 0, f"probe exited {probe.returncode}: {probe.stderr}"
    match = re.search(r"^peak - baseline\s*:\s*([\d,]+) bytes", probe.stdout, re.MULTILINE)
    assert match, f"probe omitted the RSS measurement:\n{probe.stdout}"
    delta = int(match.group(1).replace(",", ""))
    assert delta > 0, "fresh-process probe reported no extraction RSS growth"
    bound = (
        SPOOL_THRESHOLD_BYTES + RSS_SLACK_BYTES
        if concurrency == 1
        else 4 * SPOOL_THRESHOLD_BYTES + RSS_SLACK_BYTES + 4 * THREAD_SLACK_BYTES
    )
    assert delta < bound, f"RSS grew {delta:,} bytes; bound is {bound:,} bytes"
    assert "artifacts written : 4 (1,073,741,824 bytes)" in probe.stdout
