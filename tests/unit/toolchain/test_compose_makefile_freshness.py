"""Local setup refreshes a Makefile mount after an atomic editor save."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("compose_name", ["docker-compose", "podman-compose"])
def test_setup_refreshes_replaced_makefile_and_reuses_unchanged_container(tmp_path, compose_name):
    if shutil.which("make") is None:
        pytest.skip("GNU make is required for the local container lifecycle check")
    checkout = tmp_path / "checkout"
    (checkout / "docker").mkdir(parents=True)
    makefile = checkout / "Makefile"
    original = (PROJECT_ROOT / "Makefile").read_bytes()
    makefile.write_bytes(original)
    shutil.copyfile(
        PROJECT_ROOT / "docker/docker-compose.yml", checkout / "docker/docker-compose.yml",
    )
    compose = tmp_path / compose_name
    # A hard link reproduces a single-file mount retaining the old inode. This
    # Compose stand-in remounts only when the service's interpolated label changes.
    compose.write_text(
        f"#!{sys.executable}\n"
        "import json, os, re, sys\n"
        "from pathlib import Path\n"
        "import yaml\n"
        "assert sys.argv[1:3] == ['-f', 'docker/docker-compose.yml']\n"
        "if 'up' not in sys.argv:\n"
        "    sys.exit(0)\n"
        "root = Path(os.environ['JANUS_PROJECT_ROOT'])\n"
        "config = yaml.safe_load((root / 'docker/docker-compose.yml').read_text())\n"
        "template = config['services']['janus'].get('labels', {}).get(\n"
        "    'io.janus.makefile-checksum', 'unknown')\n"
        "label = re.sub(r'\\$\\{(\\w+):-([^}]*)\\}',\n"
        "    lambda m: os.environ.get(m[1]) or m[2], template)\n"
        "previous = root / 'previous-label'\n"
        "mounted = root / 'mounted-Makefile'\n"
        "recreated = not previous.exists() or previous.read_text() != label\n"
        "if recreated:\n"
        "    mounted.unlink(missing_ok=True)\n"
        "    os.link(root / 'Makefile', mounted)\n"
        "    previous.write_text(label)\n"
        "with (root / 'compose-calls.jsonl').open('a') as stream:\n"
        "    stream.write(json.dumps({'recreated': recreated}) + '\\n')\n",
        encoding="utf-8",
    )
    compose.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key not in {
        "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "JANUS_MAKEFILE_CHECKSUM",
    }}

    def setup():
        result = subprocess.run(
            [
                "make", "--no-print-directory", "ensure-up",
                f"DETECT_COMPOSE=printf '%s' '{compose}'", "IVY_JAR_NAMES=",
            ],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr

    setup()
    mounted = checkout / "mounted-Makefile"
    assert mounted.read_bytes() == original
    replacement = checkout / "Makefile.replacement"
    updated = original + b"\n# An atomic editor save changes the local CI recipe.\n"
    replacement.write_bytes(updated)
    replacement.replace(makefile)
    assert mounted.read_bytes() == original

    setup()
    assert mounted.read_bytes() == updated
    setup()
    calls = [
        json.loads(line) for line in (checkout / "compose-calls.jsonl").read_text().splitlines()
    ]
    assert [call["recreated"] for call in calls] == [True, True, False]
