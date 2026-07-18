"""Regression guard that the mypy type gate is wired to *fail* on a type error."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mypy")

# A deliberately mistyped snippet: assign a str to an int-annotated name. mypy
# must reject this; if it ever passes, the gate is silently broken.
_MISTYPED_SNIPPET = "x: int = 'not an int'\n"


def _run_mypy(target: Path) -> subprocess.CompletedProcess[str]:
    """Invoke mypy on a single file, isolated from the repo's config and cache.

    ``--no-incremental`` and a throwaway ``--cache-dir`` keep the run from
    touching or being influenced by the project cache; ``--config-file=`` (empty)
    detaches it from ``pyproject.toml`` so only the snippet under test matters.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-incremental",
            "--cache-dir",
            str(target.parent / ".mypy_cache"),
            "--config-file=",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _require_mypy_on_path() -> None:
    if shutil.which("mypy") is None:
        try:
            import mypy 
        except ImportError:  
            pytest.skip("mypy is not available in this environment")


def test_gate_fails_on_type_error(tmp_path: Path) -> None:
    bad = tmp_path / "mistyped_snippet.py"
    bad.write_text(_MISTYPED_SNIPPET)

    result = _run_mypy(bad)

    assert result.returncode != 0, (
        "mypy accepted a deliberately mistyped snippet — the type gate is not "
        f"wired to fail.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "error:" in combined, (
        "mypy exited non-zero but emitted no diagnostic — expected a type error.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_gate_passes_on_well_typed_snippet(tmp_path: Path) -> None:
    """Sanity check that the harness is not just always-failing: a clean snippet
    must pass, so the failure above is attributable to the type error, not to a
    broken invocation."""
    good = tmp_path / "well_typed_snippet.py"
    good.write_text("x: int = 1\n")

    result = _run_mypy(good)

    assert result.returncode == 0, (
        "mypy rejected a well-typed snippet — the meta-test invocation is broken, "
        f"not the code under test.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
