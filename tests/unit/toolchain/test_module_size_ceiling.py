"""AC-1 as an executable measurement: the module size ceiling for ``src/janus``."""

from __future__ import annotations

import inspect
from pathlib import Path

import janus

#: Hard ceiling. No module may exceed this, ever, without an explicit decision.
HARD_CEILING = 800

#: Soft target. Modules above this are listed below with a recorded reason.
SOFT_TARGET = 600

#: Modules deliberately allowed between SOFT_TARGET and HARD_CEILING.
#: Each entry is a decision, not a backlog: state why the module is not being split.
SOFT_CEILING_ALLOWANCES: dict[str, str] = {
    "quality/validators.py": (
        "662 LOC; neither a strategy core nor a config module, so outside"
        "scope. Cohesive: one validator class per quality rule."
    ),
}

#: Root of the package under measurement, resolved through the import system rather than a
#: relative path so the sweep is independent of the working directory pytest runs from.
PACKAGE_ROOT = Path(inspect.getfile(janus)).parent


def _physical_line_count(path: Path) -> int:
    """Physical lines in ``path``, matching ``wc -l`` for newline-terminated files."""
    text = path.read_text(encoding="utf-8")
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _module_line_counts() -> dict[str, int]:
    """Every module in ``src/janus`` keyed by its path relative to the package root."""
    return {
        str(path.relative_to(PACKAGE_ROOT)): _physical_line_count(path)
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }


def _format_counts(counts: dict[str, int]) -> str:
    """Render ``path: loc`` sorted by size descending — largest offender first."""
    return "\n".join(
        f"  {path}: {loc}" for path, loc in sorted(counts.items(), key=lambda kv: -kv[1])
    )


def test_the_sweep_actually_measures_the_package():
    """A glob that silently matches nothing would pass every ceiling test below."""
    counts = _module_line_counts()

    assert "strategies/api/core.py" in counts, (
        "the size sweep did not find strategies/api/core.py — PACKAGE_ROOT is wrong and "
        f"every ceiling assertion below is vacuous (swept {len(counts)} modules)"
    )
    assert len(counts) >= 40, f"only {len(counts)} modules swept; the sweep looks truncated"


def test_no_module_exceeds_the_hard_ceiling():
    """No module in ``src/janus`` may exceed HARD_CEILING physical lines."""
    over = {path: loc for path, loc in _module_line_counts().items() if loc > HARD_CEILING}

    assert not over, (
        f"{len(over)} module(s) exceed the hard ceiling of {HARD_CEILING} physical lines "
        ":\n"
        + _format_counts(over)
        + "\nSplit the module, or record an explicit decision to raise the ceiling."
    )


def test_modules_over_the_soft_target_are_explicitly_allowed():
    """Every module in (SOFT_TARGET, HARD_CEILING] must carry a written reason.

    Modules *over* the hard ceiling are deliberately out of scope here — they are
    ``test_no_module_exceeds_the_hard_ceiling``'s business, and listing them in the
    allowance table would grant them a pass they have not earned.
    """
    in_band = {
        path: loc
        for path, loc in _module_line_counts().items()
        if SOFT_TARGET < loc <= HARD_CEILING
    }
    unrecorded = {
        path: loc for path, loc in in_band.items() if path not in SOFT_CEILING_ALLOWANCES
    }

    assert not unrecorded, (
        f"module(s) between the soft target ({SOFT_TARGET}) and the hard ceiling "
        f"({HARD_CEILING}) with no recorded reason:\n"
        + _format_counts(unrecorded)
        + "\nEither split the module or add it to SOFT_CEILING_ALLOWANCES with a reason "
        "explaining why it is not being split. The allowance is a decision, not a backlog."
    )


def test_allowances_are_not_stale():
    """An allowance must describe a module that exists and is still over the soft target.

    Without this, a module that gets split later keeps its allowance as fiction — the table
    would document a decision about code that no longer looks like that.
    """
    counts = _module_line_counts()

    missing = sorted(path for path in SOFT_CEILING_ALLOWANCES if path not in counts)
    assert not missing, (
        "SOFT_CEILING_ALLOWANCES names module(s) that no longer exist:\n"
        + "\n".join(f"  {path}" for path in missing)
        + "\nRemove the allowance — it was granted to a module that has since moved or gone."
    )

    shrunk = {
        path: counts[path]
        for path in SOFT_CEILING_ALLOWANCES
        if counts[path] <= SOFT_TARGET
    }
    assert not shrunk, (
        f"SOFT_CEILING_ALLOWANCES names module(s) now at or under the soft target "
        f"({SOFT_TARGET}):\n"
        + _format_counts(shrunk)
        + "\nRemove the allowance — the exception it records no longer applies."
    )
