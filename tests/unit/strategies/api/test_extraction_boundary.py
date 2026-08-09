"""Architectural guardrails for the API family's orchestration/mechanics seam.

Two rules, both structural rather than behavioural, so they are asserted over source text
and ASTs instead of over a run:

* the family composes the shared HTTP layer, it never
  copies it. A retry loop, a backoff calculation or a status-code branch appearing inside
  ``strategies/api`` means the composition was unwound.
* **AC-2** — orchestration decides *what to request next*; mechanics perform *one request*.
  The seam is real only if the orchestration module cannot reach the transport at all.

Both sweeps are scoped to the *package*, not to a filename: a rule pinned to one module is
silently disarmed by the next split rather than by anyone deciding to weaken it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    PageScript,
    ScriptedPageTransport,
    build_concurrent_plan,
    build_concurrent_strategy,
    build_storage_layout,
    build_transport_factory,
)

import janus.strategies.api as api_package
import janus.strategies.api.core as api_core
import janus.strategies.api.extraction as api_extraction
import janus.strategies.api.requests as api_requests
from janus.checkpoints import DeadLetterStore
from janus.runtime import SparkSessionProvider
from janus.strategies.api import ApiStrategy
from janus.strategies.api.request_inputs import ApiParameterBindingError

#: Tokens that only appear when the retry loop has been re-implemented locally.
BANNED_RETRY_TOKENS = ("while attempt", "RETRYABLE_STATUS_CODES", "time.sleep(")

#: Names that would let the orchestration layer speak HTTP directly.
BANNED_TRANSPORT_TOKENS = ("ApiTransport", "ApiClient", "urllib")


def _api_package_sources() -> dict[str, str]:
    """Source text of every module in the API strategy package."""
    package_dir = Path(inspect.getfile(api_package)).parent
    return {
        path.name: path.read_text(encoding="utf-8") for path in sorted(package_dir.glob("*.py"))
    }


def test_the_sweep_actually_covers_the_api_package():
    """A glob that silently matched nothing would pass every assertion below."""
    names = set(_api_package_sources())

    expected = {"core.py", "requests.py", "extraction.py", "pagination_loop.py", "run_state.py"}
    assert expected <= names, (
        f"the api package sweep is missing modules it must cover (found {sorted(names)})"
    )


def test_request_executor_does_not_reimplement_the_retry_loop():
    """single-source rule: mechanics compose the shared layer, never copy it."""
    source = inspect.getsource(api_requests)

    for banned in BANNED_RETRY_TOKENS:
        assert banned not in source, (
            f"{banned!r} appears in api/requests.py. The retry loop, its backoff arithmetic "
            "and its status-code classification live once, in strategies/http/retry.py — "
            "ApiRequestExecutor composes send_with_retries, it does not re-derive it."
        )


def test_no_module_in_the_api_package_reimplements_the_retry_loop():
    """The same rule, swept over the package, so a new module inherits it automatically."""
    offenders = {
        name: banned
        for name, source in _api_package_sources().items()
        for banned in BANNED_RETRY_TOKENS
        if banned in source
    }

    assert not offenders, (
        f"retry mechanics reappeared inside strategies/api: {offenders}. "
        f"reject-list for order-03 forbids a private retry loop in a family core."
    )


def test_orchestration_does_not_import_the_transport():
    """AC-2 as a test rather than a structure review: the seam is an import boundary."""
    source = inspect.getsource(api_extraction)

    for banned in BANNED_TRANSPORT_TOKENS:
        assert banned not in source, (
            f"{banned!r} appears in api/extraction.py. Orchestration decides what to request "
            "next and what to do when something fails; performing a request belongs to "
            "ApiRequestExecutor in api/requests.py."
        )


def _top_level_definitions(module) -> set[str]:
    tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }


def test_core_is_a_facade():
    """``core.py`` defines the strategy and the hook ABC — everything else is re-exported."""
    defined = _top_level_definitions(api_core)

    assert defined == {"ApiStrategy", "ApiHook"}, (
        f"api/core.py defines {sorted(defined)}. After the split it is a façade: the "
        "ApiStrategy contract methods, the ApiHook ABC, the contract constant and the "
        "compatibility re-exports. Mechanics go to requests.py, loops to pagination_loop.py, "
        "orchestration to extraction.py."
    )


def test_core_still_exports_the_concurrency_contract():
    """order-08's declared output contract stays where the tests and the docs point."""
    assert frozenset(
        {
            "speculative_request_count",
            "speculative_discarded_count",
            "past_end_terminated_count",
            "past_end_status",
            "lookahead_ceiling_source",
            "total_records_reported",
        }
    ) == api_core.CONCURRENCY_ONLY_METADATA_KEYS


# ── behavioural guardrails for the decomposed orchestration ──────────────────


def _binding_failure_strategy(tmp_path: Path) -> tuple[ApiStrategy, Any, DeadLetterStore]:
    """A run whose only request input cannot satisfy its parameter binding."""
    transport = ScriptedPageTransport(scripts={}, key_param="page")
    dead_letter_store = DeadLetterStore()
    strategy = ApiStrategy(
        transport_factory=build_transport_factory(transport),
        storage_layout_factory=lambda plan: build_storage_layout(tmp_path),
        sleeper=lambda seconds: None,
        dead_letter_store=dead_letter_store,
    )
    plan = build_concurrent_plan(
        tmp_path,
        source_id="binding_failure",
        concurrency=1,
        request_inputs={
            "type": "date_window",
            "start": "2026-01-01",
            "end": "2026-01-03",
            "step": "day",
        },
        # Valid at config-load time, unresolvable at run time: this full-refresh run has no
        # checkpoint to bind. Three windows and a dead-letter budget of ten mean a run that
        # *did* dead-letter the failure would finish quietly instead of raising.
        parameter_bindings={"since": {"from": "checkpoint_value"}},
        dead_letter_max_items=10,
    )
    return strategy, plan, dead_letter_store


def test_parameter_binding_failure_is_not_dead_lettered(tmp_path):
    """A binding failure is a config bug affecting every input — it must fail the run.

    Dead-lettering it would turn one fail-fast error into N silently skipped inputs, which
    is why ``bind_request_input`` is called outside the driver's dead-letter ``try``.
    """
    strategy, plan, dead_letter_store = _binding_failure_strategy(tmp_path)

    with pytest.raises(ApiParameterBindingError):
        strategy.extract(plan)

    assert dead_letter_store.load(plan) is None, (
        "a parameter-binding failure was recorded as a dead letter. It must propagate: "
        "the binding is resolved outside the try that dead-letters, precisely so a config "
        "error fails the run instead of quietly skipping every request input."
    )


class _SessionSpyProvider(SparkSessionProvider):
    """Refuses to hand out a session — extraction must never ask for one."""

    def __init__(self) -> None:
        super().__init__({}, {}, session_factory=lambda: object())

    def get(self):
        raise AssertionError("a Spark session was requested during extraction")


def test_extraction_never_acquires_a_session(tmp_path):
    """Belt-and-braces beside the spy provider in tests/unit/runtime."""
    transport = ScriptedPageTransport(scripts={1: PageScript(records=({"id": "1"},))})
    strategy = build_concurrent_strategy(tmp_path, transport)
    plan = build_concurrent_plan(tmp_path, source_id="session_free", concurrency=1)

    result = strategy.extract(plan, spark=_SessionSpyProvider())

    assert result.records_extracted == 1
