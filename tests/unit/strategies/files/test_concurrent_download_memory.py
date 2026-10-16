from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from janus.writers import SPOOL_THRESHOLD_BYTES

RSS_SLACK_BYTES = 32 * 2**20
THREAD_SLACK_BYTES = 8 * 2**20
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("concurrency", [1, 4])
def test_four_large_downloads_cost_spool_buffers_instead_of_payloads(tmp_path, concurrency):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(Path(path or ROOT).resolve()) for path in sys.path)
    command = [
        sys.executable,
        "-m",
        "tests.support.memory_probe",
        "256",
        str(tmp_path),
        "--files",
        "4",
        "--concurrency",
        str(concurrency),
    ]
    # The launcher inherits pytest's ru_maxrss, but its small live heap is all the
    # next child inherits. Keep the real CLI's server inside that measured child.
    launcher = "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))"
    probe = subprocess.run(
        [sys.executable, "-c", launcher, *command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    rendered = f"{probe.stdout}\n{probe.stderr}"
    assert probe.returncode == 0, f"probe exited {probe.returncode}:\n{rendered}"
    match = re.search(r"^peak - baseline\s*:\s*([\d,]+) bytes", probe.stdout, re.MULTILINE)
    assert match, f"probe omitted the RSS measurement:\n{rendered}"
    delta = int(match.group(1).replace(",", ""))
    assert delta > 0, f"fresh-process probe reported no extraction RSS growth:\n{rendered}"
    bound = (
        SPOOL_THRESHOLD_BYTES + RSS_SLACK_BYTES
        if concurrency == 1
        else 4 * SPOOL_THRESHOLD_BYTES + RSS_SLACK_BYTES + 4 * THREAD_SLACK_BYTES
    )
    assert delta < bound, f"RSS grew {delta:,} bytes; bound is {bound:,} bytes:\n{rendered}"
    assert "artifacts written : 4 (1,073,741,824 bytes)" in probe.stdout, rendered
