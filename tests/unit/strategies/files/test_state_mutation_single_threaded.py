from __future__ import annotations

import ast
import inspect
import json
import threading
from collections import Counter
from functools import wraps
from pathlib import Path

import pytest

import janus.strategies.files as files_package
from janus.checkpoints import CheckpointStore, DeadLetterStore
from janus.strategies.files.core import FileHook
from janus.writers.raw import SPOOL_THRESHOLD_BYTES, RawArtifactWriter, StagedWrite
from tests.support.file_downloads import UrlResponse, build_cnpj_fixture
from tests.unit.strategies.files.test_concurrent_dead_letter_order import (
    assert_no_staging,
    assert_worker_downloads,
    download_requests,
    run_leg,
)
from tests.unit.strategies.files.test_download_boundary import (
    _aliases,
    _enclosing_scopes,
    _qualified_name,
    _resolve,
)

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
LOG_METHODS = frozenset({"info", "warning", "error", "exception", "debug", "critical", "log"})


def _package_sources():
    root = Path(inspect.getfile(files_package)).parent
    return {path.name: path.read_text("utf-8") for path in sorted(root.glob("*.py"))}


def _worker_references(function, aliases):
    violations = [function.name] if function.name in BANNED_NAMES else []
    for node in ast.walk(function):
        if isinstance(node, ast.Name | ast.Attribute | ast.arg):
            name = (
                node.arg
                if isinstance(node, ast.arg)
                else _resolve(node, aliases).rsplit(".", 1)[-1]
            )
            if name in BANNED_NAMES:
                violations.append(name)
        if not isinstance(node, ast.Call):
            continue
        called = _resolve(node.func, aliases)
        method = called.rsplit(".", 1)[-1]
        if method in BANNED_METHODS:
            violations.append(called)
        if isinstance(node.func, ast.Attribute):
            receiver = _qualified_name(node.func.value).rsplit(".", 1)[-1]
            if receiver.endswith("logger") and method in LOG_METHODS:
                violations.append(called)
            if method == "get" and receiver.endswith("provider"):
                violations.append(called)
    return violations


def _receiver_bindings(tree, aliases, *, track_pool):
    pools, windows = set(), set()
    for node in ast.walk(tree):
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.annotation
        elif isinstance(node, ast.arg):
            targets, value = [ast.Name(id=node.arg)], node.annotation
        elif isinstance(node, ast.withitem):
            targets, value = [node.optional_vars], node.context_expr
        values = [value]
        if isinstance(node, ast.AnnAssign):
            values.append(node.value)
        constructors = {_resolve(item, aliases).rsplit(".", 1)[-1] for item in values}
        if "OrderedDownloadWindow" in constructors:
            windows.update(_qualified_name(target) for target in targets)
        elif track_pool and "ThreadPoolExecutor" in constructors:
            pools.update(_qualified_name(target) for target in targets)
    return pools, windows


def _submission_targets(tree, filename):
    aliases = _aliases(tree)
    pools, windows = _receiver_bindings(tree, aliases, track_pool=filename == "download_window.py")
    targets, violations = [], []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"submit", "map"}
        ):
            continue
        receiver = _qualified_name(node.func.value)
        resolved = _resolve(node.func.value, aliases).rsplit(".", 1)[-1]
        if (
            receiver in windows
            or receiver.endswith("window")
            or resolved == "OrderedDownloadWindow"
        ):
            kind, offset = "window", 1
        elif "pool" in receiver.lower() or receiver in pools or resolved == "ThreadPoolExecutor":
            kind, offset = "pool", 0
        else:
            continue  # An unrelated object's submit/map is not a worker dispatch.
        if len(node.args) <= offset:
            violations.append("submission target missing")
            continue
        target = node.args[offset]
        if isinstance(target, ast.Call) and _resolve(target.func, aliases) in {
            "partial",
            "functools.partial",
        }:
            if not target.args:
                violations.append("partial target missing")
                continue
            target = target.args[0]
        if isinstance(target, ast.Name | ast.Attribute):
            targets.append((kind, target))
        else:
            violations.append("anonymous or unresolved worker target")
    return targets, violations


def _definitions(trees):
    definitions, nested = {}, set()
    for filename, tree in trees.items():
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            definitions.setdefault(node.name, []).append((filename, node))
            scopes = _enclosing_scopes(node, parents)
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                    nested.add(node)
                parent = parents.get(parent)
            if scopes:
                definitions.setdefault(".".join((*scopes, node.name)), []).append((filename, node))
    return definitions, nested


def _resolve_function(reference, filename, trees, definitions):
    raw = _qualified_name(reference)
    aliases = _aliases(trees[filename])
    name = _resolve(reference, aliases)
    candidates = definitions.get(name, [])
    local = [item for item in candidates if item[0] == filename]
    if local:
        return local
    # Relative imports and absolute janus.strategies.files imports both point to
    # sibling module definitions, including imports renamed with `as`.
    parts = name.strip(".").split(".")
    if len(parts) > 1 and f"{parts[-2]}.py" in trees:
        return [item for item in definitions.get(parts[-1], []) if item[0] == f"{parts[-2]}.py"]
    if raw.split(".")[0] in aliases and name != raw:
        return []  # Imported external functions do not belong to this sweep.
    # Instance methods (e.g. downloader.load_payload) can be resolved when their
    # definition is unique within the package. Ambiguous entry points fail closed.
    return definitions.get(parts[-1], [])


def _worker_inventory(sources):
    trees = {filename: ast.parse(source) for filename, source in sources.items()}
    definitions, nested = _definitions(trees)
    targets, violations = [], []
    for filename, tree in trees.items():
        submitted, problems = _submission_targets(tree, filename)
        violations.extend((filename, problem) for problem in problems)
        targets.extend((filename, kind, reference) for kind, reference in submitted)
    return trees, definitions, nested, targets, violations


def _worker_violations(sources):
    trees, definitions, nested, targets, violations = _worker_inventory(sources)
    for filename, _kind, reference in targets:
        resolved = _resolve_function(reference, filename, trees, definitions)
        if len(resolved) != 1:
            violations.append((filename, f"unresolved worker {_qualified_name(reference)}"))
            continue
        defining_file, function = resolved[0]
        if function in nested:
            violations.append((filename, f"nested worker {function.name}"))
        bodies = {(defining_file, function)}
        for call in ast.walk(function):
            if isinstance(call, ast.Call):
                bodies.update(_resolve_function(call.func, defining_file, trees, definitions))
        for module, body in bodies:
            violations.extend(
                (module, body.name, name)
                for name in _worker_references(body, _aliases(trees[module]))
            )
    return violations


def _dead_letter_receivers(tree, aliases):
    receivers = set()
    for node in ast.walk(tree):
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.annotation
        elif isinstance(node, ast.arg):
            targets, value = [ast.Name(id=node.arg)], node.annotation
        if value is not None and any(
            _resolve(part, aliases).rsplit(".", 1)[-1] == "DeadLetterStore"
            for part in ast.walk(value)
        ):
            receivers.update(_qualified_name(target) for target in targets)
    return receivers


def _policy_violations(sources):
    callers = {"record": set(), "can_continue_after_dead_letter": set()}
    for filename, source in sources.items():
        tree = ast.parse(source)
        aliases = _aliases(tree)
        receivers = _dead_letter_receivers(tree, aliases)
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _resolve(node.func, aliases).rsplit(".", 1)[-1]
            if called == "record" and isinstance(node.func, ast.Attribute):
                receiver = _qualified_name(node.func.value)
                if (
                    receiver not in receivers
                    and "deadletterstore" not in receiver.replace("_", "").lower()
                ):
                    continue
            elif called != "can_continue_after_dead_letter":
                continue
            callers[called].add((filename, _enclosing_scopes(node, parents)))
    return [
        (policy, sorted(sites))
        for policy, sites in callers.items()
        if len(sites) != 1 or next(iter(sites))[1] != ("_consume_candidate",)
    ] + (
        ["record and budget have different callers"]
        if callers["record"] != callers["can_continue_after_dead_letter"]
        else []
    )


def test_package_worker_targets_have_no_state_hook_commit_log_or_source_attempt_access(
    record_property,
):
    sources = _package_sources()
    assert len(sources) >= 10
    assert "download_window.py" in sources
    _trees, _definitions_map, _nested, targets, problems = _worker_inventory(sources)
    pool = [_qualified_name(target) for _filename, kind, target in targets if kind == "pool"]
    window = [_qualified_name(target) for _filename, kind, target in targets if kind == "window"]
    assert pool == ["_download_in_worker"], f"pool-submitted callable anchors: {pool}"
    assert window == ["_download_candidate"], f"window-submitted download anchors: {window}"
    assert not problems
    assert not _worker_violations(sources)
    record_property("swept_modules", len(sources))
    record_property("pool_targets", pool)
    record_property("window_targets", window)


def test_dead_letter_record_and_budget_policy_have_one_shared_caller():
    assert not _policy_violations(_package_sources())


@pytest.mark.parametrize(
    "source",
    [
        "pool.submit(lambda: download())",
        "def worker():\n dead_letter_store.record(plan)\npool.submit(worker)",
        "def worker():\n payload.persist(writer, plan, path)\npool.submit(worker)",
        "def worker():\n context.file_hook.resolve_version(plan, file)\npool.submit(worker)",
        "def worker():\n logger.info('x')\npool.map(worker, indices)",
        "def worker():\n spark_provider.get()\npool.submit(worker)",
        "def worker(context):\n return context.plan\npool.submit(worker)",
        "def helper():\n store.save(plan)\ndef worker():\n helper()\npool.submit(worker)",
        "def loop():\n def worker(): return 1\n pool.submit(worker)",
        "def _complete_candidate(): pass\nwindow.submit(i, partial(_complete_candidate, plan))",
        "callback = lambda: download()\npool.submit(callback)",
        "pool.submit()",
        "window.submit(index, partial())",
        "from janus.checkpoints import can_continue_after_dead_letter as budget\n"
        "def worker(): budget()\npool.submit(worker)",
        "from janus.runtime import SourceExecutor as Executor\n"
        "def worker(): Executor().execute(plan)\npool.submit(worker)",
        *[f"def worker(): {name}()\npool.submit(worker)" for name in sorted(BANNED_NAMES)],
        *[
            f"def worker(): payload.{name}()\npool.submit(worker)"
            for name in sorted(BANNED_METHODS)
        ],
    ],
)
def test_detector_rejects_deliberate_worker_violations(source):
    assert _worker_violations({"sample.py": source})


@pytest.mark.parametrize(
    "source",
    [
        "def _download_in_worker(i, download): return download()\n"
        "pool.submit(_download_in_worker, i, download)",
        "def worker():\n"
        " return downloader.load_payload_with_dedicated_client(plan, logger=logger)\n"
        "pool.submit(worker)",
        "def worker():\n return DownloadOutcome(i, d, None, t)\npool.submit(worker)\n"
        "def loop(): record(plan)",
        "from functools import partial as bind\ndef _download_candidate():\n"
        " return downloader.load_payload_with_dedicated_client(plan)\n"
        "window.submit(i, bind(_download_candidate))",
        "def worker(): return 1\nOrderedDownloadWindow.submit(i, partial(worker))",
        "def worker(): return 1\nservice.map(lambda: context)\npool.submit(worker)",
    ],
)
def test_detector_accepts_value_only_workers_and_loop_thread_records(source):
    assert not _worker_violations({"sample.py": source})


@pytest.mark.parametrize(
    "setup,submission",
    [
        (
            "from concurrent.futures import ThreadPoolExecutor as E\nself._executor = E()",
            "self._executor.submit(worker)",
        ),
        ("from concurrent.futures import ThreadPoolExecutor as E", "E().map(worker, indices)"),
        (
            "from concurrent.futures import ThreadPoolExecutor as E\nwith E() as executor:",
            " executor.submit(worker)",
        ),
        (
            "from .download_window import OrderedDownloadWindow as W\nwith W(4) as pending:",
            " pending.submit(i, partial(worker))",
        ),
        ("pending: OrderedDownloadWindow = factory()", "pending.submit(i, partial(worker))"),
        ("self._executor: object = ThreadPoolExecutor()", "self._executor.submit(worker)"),
    ],
)
def test_detector_resolves_constructed_receivers_without_pool_or_window_names(setup, submission):
    source = "def worker(): payload.commit()\n" + setup + "\n" + submission
    assert _worker_violations({"download_window.py": source})


@pytest.mark.parametrize(
    "imported",
    [
        "from .helpers import helper as call",
        "from janus.strategies.files.helpers import helper as call",
        "from . import helpers as h",
    ],
)
def test_detector_follows_a_direct_callee_in_another_package_module(imported):
    call = "h.helper" if " as h" in imported else "call"
    sources = {
        "dispatch.py": f"{imported}\ndef worker(): {call}()\npool.submit(worker)",
        "helpers.py": "def helper(): payload.commit()",
    }
    assert _worker_violations(sources)
    sources["helpers.py"] = "def helper(): return DownloadOutcome(i, d, None, t)"
    assert not _worker_violations(sources)


def test_detector_resolves_imported_worker_alias_and_distinguishes_same_named_functions():
    sources = {
        "dispatch.py": "from .helpers import worker as fetch\npool.submit(fetch)",
        "helpers.py": "def worker(): return 1",
        "loop.py": "def worker(): dead_letter_store.record(plan)",
    }
    assert not _worker_violations(sources)
    sources["helpers.py"] = "def worker(): payload.commit()"
    assert _worker_violations(sources)


def test_static_depth_is_one_direct_callee():
    source = (
        "def inner(): payload.commit()\ndef helper(): inner()\n"
        "def worker(): helper()\npool.submit(worker)"
    )
    assert not _worker_violations({"sample.py": source})


_POLICY_CLEAN = """
from janus.checkpoints import DeadLetterStore, can_continue_after_dead_letter as budget
def _consume_candidate(store: DeadLetterStore):
    store.record(plan)
    budget()
"""


@pytest.mark.parametrize(
    "extra",
    [
        "def other(): dead_letter_store.record(plan)",
        "def other(): budget()",
        "class Worker:\n def _consume_candidate(self): dead_letter_store.record(plan)",
        "def other(store: DeadLetterStore): store.record(plan)",
        "store = DeadLetterStore()\ndef other(): store.record(plan)",
    ],
)
def test_policy_detector_rejects_a_second_caller(extra):
    assert _policy_violations({"sample.py": _POLICY_CLEAN + extra})


def test_policy_detector_accepts_one_caller_and_unrelated_record_methods():
    assert not _policy_violations(
        {"sample.py": _POLICY_CLEAN + "def other(): metrics.record(event)"}
    )


@pytest.mark.parametrize(
    "source",
    [
        "",
        _POLICY_CLEAN.replace("    store.record(plan)\n", ""),
        _POLICY_CLEAN.replace("    budget()\n", ""),
        _POLICY_CLEAN.replace("_consume_candidate", "other"),
    ],
)
def test_policy_detector_rejects_missing_or_misplaced_calls(source):
    assert _policy_violations({"sample.py": source})


@pytest.mark.parametrize("spool_threshold", [32, SPOOL_THRESHOLD_BYTES], ids=["staged", "inline"])
@pytest.mark.parametrize(
    "failures,budget",
    [((3, 6), 1), ((1,), 5), ((), 5)],
    ids=["budget-exceeded", "early-failure-continues", "ten-archive-success"],
)
def test_state_hooks_commits_and_archive_expansion_stay_on_the_loop_thread(
    tmp_path,
    monkeypatch,
    record_property,
    failures,
    budget,
    spool_threshold,
):
    import janus.strategies.files.concurrent_extraction as concurrent_loop
    import janus.strategies.files.download as download
    import janus.strategies.files.download_loop as loop

    identities, counts = [], Counter()
    loop_ident = threading.get_ident()

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
        (StagedWrite, ("commit",)),
        (RawArtifactWriter, ("write_bytes",)),
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
        # Setting prepare_download on the subclass makes it an override, ensuring
        # that the compatibility hook is really invoked in both payload paths.
        monkeypatch.setattr(RecordingHook, name, spy(getattr(FileHook, name), f"hook.{name}"))
    monkeypatch.setattr(loop, "_extract_archive", spy(loop._extract_archive, "_extract_archive"))
    monkeypatch.setattr(
        concurrent_loop,
        "can_continue_after_dead_letter",
        spy(concurrent_loop.can_continue_after_dead_letter, "can_continue_after_dead_letter"),
    )
    monkeypatch.setattr(download, "SPOOL_THRESHOLD_BYTES", spool_threshold)
    monkeypatch.setattr(loop, "SPOOL_THRESHOLD_BYTES", spool_threshold)
    fixture = build_cnpj_fixture(
        tmp_path,
        failing=dict.fromkeys(failures, UrlResponse(500, b"failure")),
        dead_letter_max_items=budget,
    )
    fixture.server.reverse_completion(fixture.urls[:4])
    leg = run_leg(fixture, 4, hook=RecordingHook())
    assert (leg.error is not None) == (budget == 1)
    required = {
        "DeadLetterStore.clear",
        "CheckpointStore.load",
        "_extract_archive",
    }
    if failures:
        required.update(
            {"DeadLetterStore.load", "DeadLetterStore.record", "can_continue_after_dead_letter"}
        )
        assert (
            counts["DeadLetterStore.record"]
            == counts["can_continue_after_dead_letter"]
            == len(failures)
        )
    else:
        assert not counts["DeadLetterStore.record"]
        assert not counts["can_continue_after_dead_letter"]
        assert leg.result.metadata_as_dict()["persisted_file_count"] == "10"
    if spool_threshold == SPOOL_THRESHOLD_BYTES:
        required.add("RawArtifactWriter.write_bytes")
    else:
        required.add("StagedWrite.commit")
    assert required <= set(counts), counts
    assert {f"hook.{name}" for name in hook_methods} <= set(counts)
    assert identities and set(identities) == {loop_ident}
    assert_no_staging(fixture)
    assert_worker_downloads(leg)
    worker_names = {request.thread_name for request in download_requests(leg)}
    assert len(worker_names) >= 2
    record_property("spy_counts", json.dumps(dict(sorted(counts.items()))))
    record_property("state_touch_count", len(identities))
    record_property("distinct_state_idents", len(set(identities)))
    record_property("worker_names", sorted(worker_names))
