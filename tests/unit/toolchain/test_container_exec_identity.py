"""A fresh exec process resolves its own uid without inheriting PID 1's exports."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NSS_LIBRARY = Path("/usr/lib/x86_64-linux-gnu/libnss_wrapper.so")


def test_fresh_exec_resolves_unknown_uid_and_parallel_commands_use_private_files(tmp_path):
    shell = shutil.which("sh")
    if shell is None or not NSS_LIBRARY.is_file():
        pytest.skip("the image's native NSS wrapper is required for the identity regression")
    # Deliberately omit the current uid so a fresh exec sees the same unresolved
    # account as the GitHub runner's uid 1001 in the uid-1000 image.
    absent_uid = os.getuid() + 100000
    passwd = tmp_path / "unresolved-passwd"
    group = tmp_path / "unresolved-group"
    passwd.write_text(f"other:x:{absent_uid}:{absent_uid}:Other:/tmp:/bin/sh\n")
    group.write_text(f"other:x:{absent_uid}:\n")
    env = {**os.environ, "LD_PRELOAD": str(NSS_LIBRARY),
           "NSS_WRAPPER_PASSWD": str(passwd), "NSS_WRAPPER_GROUP": str(group)}
    code = (
        "import json, os, pwd; account = pwd.getpwuid(os.getuid()); "
        "print(json.dumps({'uid': account.pw_uid, 'name': account.pw_name, "
        "'home': account.pw_dir, 'passwd': os.environ['NSS_WRAPPER_PASSWD'], "
        "'group': os.environ['NSS_WRAPPER_GROUP']}))"
    )
    unresolved = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10,
        check=False,
    )
    assert unresolved.returncode != 0 and "getpwuid" in unresolved.stderr

    def execute(_):
        result = subprocess.run(
            [shell, str(PROJECT_ROOT / "docker/entrypoint.sh"), sys.executable, "-c", code],
            env=env, capture_output=True, text=True, timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    with ThreadPoolExecutor(max_workers=2) as pool:
        accounts = list(pool.map(execute, range(2)))
    try:
        assert all(account["uid"] == os.getuid() for account in accounts)
        assert all(account["name"] == "janus" and account["home"] == "/tmp" for account in accounts)
        assert len({account["passwd"] for account in accounts}) == 2
        assert len({account["group"] for account in accounts}) == 2
        for account in accounts:
            assert f"janus:x:{os.getuid()}:" in Path(account["passwd"]).read_text()
    finally:
        for account in accounts:
            Path(account["passwd"]).unlink()
            Path(account["group"]).unlink()
