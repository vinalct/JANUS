"""Engine-neutral import boundary for janus.models.data_contracts."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import janus.models.data_contracts as data_contracts

PACKAGE_ROOT = Path(data_contracts.__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parents[2]
ANCHORS = {"errors.py", "loader.py", "model.py"}
FORBIDDEN_IMPORTS = (
    "janus.observability",
    "janus.readers",
    "janus.runtime",
    "janus.strategies",
    "janus.writers",
    "pyarrow",
    "pyiceberg",
    "pyspark",
)


def _modules() -> tuple[Path, ...]:
    return tuple(sorted(PACKAGE_ROOT.glob("*.py")))


def _import_targets(source: str) -> tuple[str, ...]:
    targets: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.append(node.module)
    return tuple(targets)


def _matches_root(module: str, root: str) -> bool:
    return module == root or module.startswith(f"{root}.")


def _forbidden_imports(source: str) -> tuple[str, ...]:
    return tuple(
        target
        for target in _import_targets(source)
        if any(_matches_root(target, root) for root in FORBIDDEN_IMPORTS)
    )


def test_data_contracts_package_imports_no_engine():
    violations = {
        module.name: forbidden
        for module in _modules()
        if (forbidden := _forbidden_imports(module.read_text(encoding="utf-8")))
    }

    assert violations == {}


def test_sweep_covers_the_package():
    matched = {module.name for module in _modules()}

    assert matched >= ANCHORS
    assert matched


def test_detector_flags_a_deliberate_violation():
    source = "def f():\n    import pyspark\n"

    assert _forbidden_imports(source) == ("pyspark",)


def test_detector_does_not_flag_clean_code():
    source = "from pathlib import Path\nfrom janus.models.config import ValidationIssue\n"

    assert _forbidden_imports(source) == ()


def test_package_imports_cleanly_without_engines():
    import_paths = (str(SRC_ROOT), *(entry for entry in sys.path if entry))
    command = (
        "import sys; "
        f"sys.path[:0] = {import_paths!r}; "
        "import janus.models.data_contracts; "
        "assert not {'pyspark', 'pyiceberg', 'pyarrow'} & sys.modules.keys()"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
