"""The default JANUS import surface must not require the Dagster extra."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import janus


def _pythonpath() -> str:
    package_root = str(Path(janus.__file__).parents[1])
    return os.pathsep.join(
        part for part in (package_root, os.environ.get("PYTHONPATH")) if part
    )


def test_importing_janus_does_not_import_dagster():
    env = dict(os.environ)
    env["PYTHONPATH"] = _pythonpath()
    program = (
        "import sys, janus\n"
        "print(any(name == 'dagster' or name.startswith('dagster.') "
        "for name in sys.modules))"
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_requesting_the_missing_extra_has_an_actionable_error():
    env = dict(os.environ)
    env["PYTHONPATH"] = _pythonpath()
    program = (
        "import importlib.util\n"
        "original = importlib.util.find_spec\n"
        "importlib.util.find_spec = lambda name, *args: "
        "None if name == 'dagster' else original(name, *args)\n"
        "try:\n"
        " import janus.adapters.dagster\n"
        "except ModuleNotFoundError as exc:\n"
        " print(exc)\n"
        "else:\n"
        " raise AssertionError('adapter imported without its optional dependency')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "pip install 'janus[dagster]'" in result.stdout
