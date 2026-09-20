"""A jar cache is code, and a shared scratch directory is not a safe place to put one."""

from __future__ import annotations

import ast
import inspect
import os
import stat
from pathlib import Path

import pytest

import janus.utils.environment as environment
from janus.utils.environment import FALLBACK_RUNTIME_PATH_KEYS, prepare_runtime

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="red until: user-private runtime fallback, no ivy_dir relocation",
)

WAREHOUSE_DIR_VALUE = "data/metadata/spark-warehouse"
IVY_DIR_VALUE = "data/metadata/ivy"


def _config() -> dict:
    return {
        "name": "local",
        "runtime": {"log_level": "WARNING"},
        "storage": {
            "root_dir": "data",
            "raw_dir": "data/raw",
            "bronze_dir": "data/bronze",
            "metadata_dir": "data/metadata",
        },
        "spark": {
            "app_name": "janus-runtime-fallback-test",
            "master": "local[1]",
            "warehouse_dir": WAREHOUSE_DIR_VALUE,
            "ivy_dir": IVY_DIR_VALUE,
        },
    }


def _deny(monkeypatch, *denied_suffixes: str) -> None:
    """Make ``_ensure_writable_directory`` refuse exactly the named paths, nothing else."""
    real = environment._ensure_writable_directory

    def _guarded(path: Path) -> None:
        if any(str(path).endswith(suffix) for suffix in denied_suffixes):
            raise PermissionError(13, "Permission denied", str(path))
        real(path)

    monkeypatch.setattr(environment, "_ensure_writable_directory", _guarded)


def _reset_process_root(monkeypatch) -> None:
    """Clear the per-process fallback root so each test observes a fresh decision."""
    if hasattr(environment, "_PROCESS_FALLBACK_ROOT"):
        monkeypatch.setattr(environment, "_PROCESS_FALLBACK_ROOT", None)


# ---------------------------------------------------------------------------
# FR-10 — the fallback root is user-private


def test_the_fallback_lands_under_xdg_runtime_dir_when_it_is_set(tmp_path, monkeypatch):
    """``$XDG_RUNTIME_DIR`` is a per-user, mode-0700 directory by definition — use it."""
    xdg = tmp_path / "xdg"
    xdg.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    monkeypatch.delenv(environment.RUNTIME_SCRATCH_DIR_ENV, raising=False)
    _reset_process_root(monkeypatch)
    _deny(monkeypatch, WAREHOUSE_DIR_VALUE)

    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = prepare_runtime(_config(), project_root)

    warehouse = Path(str(paths["warehouse_dir"]))
    assert warehouse.is_relative_to(xdg / "janus"), warehouse
    assert stat.S_IMODE((xdg / "janus").stat().st_mode) == 0o700


def test_without_xdg_the_fallback_is_a_per_process_private_temp_root(tmp_path, monkeypatch):
    """One root per process: two relocations in one run must not scatter across directories."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv(environment.RUNTIME_SCRATCH_DIR_ENV, raising=False)
    _reset_process_root(monkeypatch)
    _deny(monkeypatch, WAREHOUSE_DIR_VALUE)

    project_root = tmp_path / "project"
    project_root.mkdir()
    first = Path(str(prepare_runtime(_config(), project_root)["warehouse_dir"]))
    second = Path(str(prepare_runtime(_config(), project_root)["warehouse_dir"]))

    assert first == second
    assert "janus-runtime-" in str(first), first
    assert not str(first).startswith("/tmp/janus/runtime"), (
        "the fixed, shared scratch path is exactly what SEC-06 is about"
    )


def test_a_fallback_root_owned_by_another_user_is_refused(tmp_path, monkeypatch):
    """Pre-creating the directory is the attack; refusing to adopt it is the control."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    monkeypatch.setenv(environment.RUNTIME_SCRATCH_DIR_ENV, str(scratch))
    _reset_process_root(monkeypatch)
    _deny(monkeypatch, WAREHOUSE_DIR_VALUE)
    monkeypatch.setattr(os, "getuid", lambda: os.getuid() + 1)

    project_root = tmp_path / "project"
    project_root.mkdir()

    with pytest.raises(PermissionError):
        prepare_runtime(_config(), project_root)


def test_a_group_or_other_writable_fallback_root_is_refused(tmp_path, monkeypatch):
    """Ownership is not enough: a 0777 directory the current user owns is still shared."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o777)
    scratch.chmod(0o777)
    monkeypatch.setenv(environment.RUNTIME_SCRATCH_DIR_ENV, str(scratch))
    _reset_process_root(monkeypatch)
    _deny(monkeypatch, WAREHOUSE_DIR_VALUE)

    project_root = tmp_path / "project"
    project_root.mkdir()

    with pytest.raises(PermissionError):
        prepare_runtime(_config(), project_root)


# ---------------------------------------------------------------------------
# FR-10 — the jar cache is never relocated


def test_ivy_dir_is_not_a_relocatable_key():
    """risk 5: the existing permission error already names ``JANUS_SPARK_IVY_DIR``."""
    assert "ivy_dir" not in FALLBACK_RUNTIME_PATH_KEYS
    assert "warehouse_dir" in FALLBACK_RUNTIME_PATH_KEYS


def test_an_unwritable_ivy_dir_fails_the_run_instead_of_moving_the_cache(tmp_path, monkeypatch):
    """Failing loudly beats loading a jar from somewhere the operator never chose."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv(environment.RUNTIME_SCRATCH_DIR_ENV, raising=False)
    _reset_process_root(monkeypatch)
    _deny(monkeypatch, IVY_DIR_VALUE)

    project_root = tmp_path / "project"
    project_root.mkdir()

    with pytest.raises(PermissionError):
        prepare_runtime(_config(), project_root)


def test_the_cli_message_names_the_variable_that_moves_the_jar_cache():
    """An error that states the failure without stating the remedy just relocates the problem."""
    from janus.cli.common import format_runtime_permission_error

    message = format_runtime_permission_error(
        PermissionError(13, "Permission denied", "/read-only/data/metadata/ivy")
    )

    assert "JANUS_SPARK_IVY_DIR" in message


# ---------------------------------------------------------------------------
# FR-10 — no fixed shared path survives in the module


def test_the_environment_module_holds_no_tmp_path_literal():
    """An AST sweep, so a constant reintroduced under another name is still caught."""
    tree = ast.parse(Path(inspect.getfile(environment)).read_text(encoding="utf-8"))

    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("/tmp")
    ]

    assert literals == [], (
        f"utils/environment.py still names a fixed shared scratch path: {literals}"
    )
