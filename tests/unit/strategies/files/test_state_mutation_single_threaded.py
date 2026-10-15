from __future__ import annotations

import ast
import inspect
import threading
from collections import Counter
from functools import wraps
from pathlib import Path

import pytest

import janus.strategies.files as files_package
from janus.checkpoints import CheckpointStore, DeadLetterStore
from janus.strategies.files.core import FileHook
from janus.writers.raw import StagedWrite
from tests.support.file_downloads import UrlResponse, build_cnpj_fixture
from tests.unit.strategies.files.test_concurrent_dead_letter_order import (
    assert_no_staging,
    assert_worker_downloads,
    run_leg,
)
from tests.unit.strategies.files.test_download_boundary import _aliases, _qualified_name, _resolve

BANNED_NAMES = frozenset(
    {
        "DeadLetterStore",
        "ExtractionProgressStore",
        "CheckpointStore",
        "can_continue_after_dead_letter",
        "_extract_archive",
        "_complete_candidate",
        "file_hook",
        "FileHook",
        "context",
        "execute_source_attempt",
        "SourceExecutor",
        "SourceExecutionService",
        "SparkSessionProvider",
    }
)
BANNED_METHODS = frozenset({"record", "save", "clear", "release", "reset", "persist", "commit"})
LOG_METHODS = frozenset({"info", "warning", "error", "exception", "debug"})


def _worker_references(function):
    violations = []
    for node in ast.walk(function):
        if isinstance(node, ast.Name | ast.arg):
            name = node.id if isinstance(node, ast.Name) else node.arg
            if name in BANNED_NAMES:
                violations.append(name)
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            violations.append(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            receiver = _qualified_name(node.func.value).rsplit(".", 1)[-1]
            if method in BANNED_METHODS or (receiver.endswith("logger") and method in LOG_METHODS):
                violations.append(_qualified_name(node.func))
            if method == "get" and receiver.endswith("provider"):
                violations.append(_qualified_name(node.func))
    return violations


def _submission_targets(tree):
    aliases = _aliases(tree)
    window_names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _resolve(node.value.func, aliases).endswith("OrderedDownloadWindow")
        ):
            window_names.update(_qualified_name(target) for target in node.targets)
        if (
            isinstance(node, ast.withitem)
            and isinstance(node.context_expr, ast.Call)
            and _resolve(node.context_expr.func, aliases).endswith("OrderedDownloadWindow")
        ):
            window_names.add(_qualified_name(node.optional_vars))
    targets, violations = [], []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"submit", "map"}
        ):
            continue
        receiver = _qualified_name(node.func.value)
        kind = "window" if receiver in window_names or receiver.endswith("window") else "pool"
        offset = 1 if kind == "window" else 0
        if len(node.args) <= offset:
            violations.append("submission target missing")
            continue
        target = node.args[offset]
        if isinstance(target, ast.Call) and _resolve(target.func, aliases) in {
            "partial",
            "functools.partial",
        }:
            target = target.args[0]
        if isinstance(target, ast.Name):
            targets.append((kind, target.id))
        else:
            violations.append("anonymous or unresolved worker target")
    return targets, violations


def _detect(sources):
    trees = {filename: ast.parse(source) for filename, source in sources.items()}
    definitions = {}
    nested = set()
    for filename, tree in trees.items():
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                definitions.setdefault(node.name, []).append((filename, node))
                parent = parents.get(node)
                while parent is not None:
                    if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                        nested.add((filename, node.name))
                    parent = parents.get(parent)
    pool_targets, window_targets, violations = [], [], []
    for filename, tree in trees.items():
        targets, problems = _submission_targets(tree)
        violations.extend((filename, problem) for problem in problems)
        for kind, name in targets:
            (pool_targets if kind == "pool" else window_targets).append(name)
            resolved = definitions.get(name, [])
            if len(resolved) != 1:
                violations.append((filename, f"unresolved worker {name}"))
                continue
            defining_file, function = resolved[0]
            if (defining_file, name) in nested:
                violations.append((filename, f"nested worker {name}"))
            bodies = [function]
            direct_names = {
                _qualified_name(call.func).rsplit(".", 1)[-1]
                for call in ast.walk(function)
                if isinstance(call, ast.Call)
            }
            for helper in direct_names:
                bodies.extend(
                    body
                    for module, body in definitions.get(helper, [])
                    if module == defining_file and body is not function
                )
            for body in bodies:
                violations.extend(
                    (defining_file, reference) for reference in _worker_references(body)
                )
    return pool_targets, window_targets, violations


def test_package_worker_targets_have_no_state_hook_commit_log_or_source_attempt_access():
    root = Path(inspect.getfile(files_package)).parent
    sources = {path.name: path.read_text("utf-8") for path in sorted(root.glob("*.py"))}
    assert len(sources) >= 10
    pool, window, violations = _detect(sources)
    assert pool == ["_download_in_worker"], f"pool-submitted callable anchors: {pool}"
    assert window == ["_download_candidate"], f"window-submitted download anchors: {window}"
    assert "download_window.py" in sources
    assert not violations, violations


@pytest.mark.parametrize(
    "source",
    [
        "pool.submit(lambda: 1)",
        "def worker():\n dead_letter_store.record(plan)\npool.submit(worker)",
        "def worker():\n payload.persist(writer)\npool.submit(worker)",
        "def worker():\n self.file_hook.resolve_version(plan, file)\npool.submit(worker)",
        "def worker():\n logger.warning('event')\npool.map(worker, indices)",
        "def worker():\n spark_provider.get()\npool.submit(worker)",
        "def worker(context):\n return context.plan\npool.submit(worker)",
        "def helper():\n store.save(plan)\ndef worker():\n helper()\npool.submit(worker)",
        "def loop():\n def worker(): return 1\n pool.submit(worker)",
        "def target():\n payload.commit()\nwindow.submit(index, partial(target, plan))",
    ],
)
def test_detector_rejects_deliberate_worker_violations(source):
    assert _detect({"sample.py": source})[2]


@pytest.mark.parametrize(
    "source",
    [
        """def worker():
 return downloader.load_payload_with_dedicated_client(plan, logger=logger)
pool.submit(worker)
""",
        """def worker():
 return DownloadOutcome(index, downloader.load_payload_with_dedicated_client(plan), None, 0)
pool.submit(worker)
store.record(plan)
""",
        """from functools import partial as bind
def target():
 return downloader.load_payload_with_dedicated_client(plan)
window.submit(index, bind(target))
""",
    ],
)
def test_detector_accepts_value_only_workers_and_loop_thread_records(source):
    assert not _detect({"sample.py": source})[2]


@pytest.mark.parametrize(
    "failures,budget", [((3, 6), 1), ((1,), 5)], ids=["budget-exceeded", "early-failure-continues"]
)
def test_state_hooks_commits_and_archive_expansion_stay_on_the_loop_thread(
    tmp_path,
    monkeypatch,
    failures,
    budget,
):
    import janus.strategies.files.download as download
    import janus.strategies.files.download_loop as loop

    identities, counts = [], Counter()

    def spy(original, label):
        @wraps(original)
        def wrapped(*args, **kwargs):
            identities.append(threading.get_ident())
            counts[label] += 1
            return original(*args, **kwargs)

        return wrapped

    for store, methods in (
        (DeadLetterStore, ("record", "load", "clear")),
        (CheckpointStore, ("load",)),
    ):
        for method in methods:
            monkeypatch.setattr(
                store, method, spy(getattr(store, method), f"{store.__name__}.{method}")
            )

    class RecordingHook(FileHook):
        pass

    hook_methods = [
        name
        for name, member in vars(FileHook).items()
        if callable(member) and not name.startswith("_")
    ]
    for name in hook_methods:
        monkeypatch.setattr(RecordingHook, name, spy(getattr(FileHook, name), f"hook.{name}"))
    monkeypatch.setattr(StagedWrite, "commit", spy(StagedWrite.commit, "commit"))
    monkeypatch.setattr(loop, "_extract_archive", spy(loop._extract_archive, "archive"))
    monkeypatch.setattr(download, "SPOOL_THRESHOLD_BYTES", 32)
    monkeypatch.setattr(loop, "SPOOL_THRESHOLD_BYTES", 32)
    fixture = build_cnpj_fixture(
        tmp_path,
        failing=dict.fromkeys(failures, UrlResponse(500, b"failure")),
        dead_letter_max_items=budget,
    )
    leg = run_leg(fixture, 4, hook=RecordingHook())
    assert (leg.error is not None) == (budget == 1)
    required = {
        "DeadLetterStore.record",
        "DeadLetterStore.load",
        "DeadLetterStore.clear",
        "CheckpointStore.load",
        "commit",
        "archive",
    }
    assert required <= set(counts)
    assert {f"hook.{name}" for name in hook_methods} <= set(counts)
    assert identities and set(identities) == {threading.get_ident()}
    assert_no_staging(fixture)
    assert_worker_downloads(leg)
