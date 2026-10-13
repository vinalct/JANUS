"""Keep attempts in spawned processes and parallel bootstrap session-free."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

import janus

PACKAGE = Path(janus.__file__).resolve().parent
SCOPED_PACKAGES = ("runtime", "orchestration", "cli", "adapters")
ANCHORS = {
    "runtime/parallel.py",
    "runtime/batch.py",
    "runtime/executor.py",
    "orchestration/scheduling.py",
    "cli/run_all.py",
    "adapters/dagster/runtime.py",
}
ATTEMPTS = {
    "execute_source_attempt",
    "SourceExecutor.execute",
    "SourceExecutionService.execute",
    "run_worker",
    "_execute_source",
}
ALLOWED_THREADING = {
    "runtime/contract_preflight.py": (
        "Bounds a catalog metadata lookup with a daemon thread and timeout; its worker "
        "calls the catalog loader and preflight decision, never a source attempt."
    ),
}
SPARK_REFERENCES = {
    "build_spark_session",
    "SparkSession",
    "SparkContext",
    "getOrCreate",
    "scoped_request_input_session",
    "hold_source_state_lock",
}
SPAWN_CONTEXT = "<spawn-context>"


def qualified(node: ast.AST | None, aliases: Mapping[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        receiver = qualified(node.value, aliases)
        raw = f"{qualified(node.value, {})}.{node.attr}"
        return aliases.get(raw, f"{receiver}.{node.attr}" if receiver else node.attr)
    if isinstance(node, ast.Call):
        callee = qualified(node.func, aliases)
        if callee == "multiprocessing.get_context" and _is_spawn(node):
            return SPAWN_CONTEXT
        return callee
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        # Forward annotations matter even with postponed annotation evaluation.
        return node.value
    return ""


def _is_spawn(call: ast.Call) -> bool:
    argument = (
        call.args[0]
        if call.args
        else next((keyword.value for keyword in call.keywords if keyword.arg == "method"), None)
    )
    return isinstance(argument, ast.Constant) and argument.value == "spawn"


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            aliases.update(
                {item.asname or item.name: f"{node.module}.{item.name}" for item in node.names}
            )
    return aliases


def _bindings(tree: ast.AST) -> dict[str, str]:
    aliases = _import_bindings(tree)
    # Iterate so assignments and with-bindings also resolve chains of local aliases.
    for _ in range(3):
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            value = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.annotation
            elif isinstance(node, ast.arg) and node.annotation is not None:
                targets, value = [ast.Name(id=node.arg)], node.annotation
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                targets, value = [node.optional_vars], node.context_expr
            name = qualified(value, aliases)
            if (
                isinstance(node, ast.AnnAssign | ast.arg)
                and value is not None
                and any(
                    "SparkSessionProvider" in qualified(part, aliases) for part in ast.walk(value)
                )
            ):
                name = "SparkSessionProvider"
            for target in targets:
                if isinstance(target, ast.Name | ast.Attribute) and name:
                    aliases[qualified(target, {})] = name
    return aliases


def _wrappers(tree: ast.AST) -> dict[str, ast.AST]:
    wrappers: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            wrappers[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    wrappers[target.id] = node.value
    return wrappers


def _calls_attempt(
    target: ast.AST,
    aliases: Mapping[str, str],
    wrappers: Mapping[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> bool:
    for node in ast.walk(target):
        name = qualified(node, aliases)
        if any(name == attempt or name.endswith(f".{attempt}") for attempt in ATTEMPTS):
            return True
        short = name.rsplit(".", 1)[-1]
        if (
            short in wrappers
            and short not in seen
            and _calls_attempt(wrappers[short], aliases, wrappers, seen | {short})
        ):
            return True
    return False


def _threading_imports(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(item.name for item in node.names if item.name.startswith("threading"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("threading"):
            roots.add(node.module or "")
    return roots


def threaded_attempt(source: str, module: str = "runtime/parallel.py") -> list[str]:
    tree = ast.parse(source)
    aliases, wrappers = _bindings(tree), _wrappers(tree)
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = qualified(node.func, aliases)
        targets = []
        if callee in {"threading.Thread", "Thread"}:
            targets = [keyword.value for keyword in node.keywords if keyword.arg == "target"]
            targets.extend(node.args[1:2])
        elif callee.endswith((".submit", ".map")) and "ThreadPoolExecutor" in callee:
            targets = node.args[:1]
            targets.extend(keyword.value for keyword in node.keywords if keyword.arg == "fn")
        if any(_calls_attempt(target, aliases, wrappers) for target in targets):
            findings.append(f"line {node.lineno}: thread executes a source attempt")
    return findings


def unsafe_spawn(source: str, module: str = "runtime/parallel.py") -> list[str]:
    tree = ast.parse(source)
    aliases = _bindings(tree)
    forbidden = {
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.posix_spawnp",
        "multiprocessing.Process",
        "multiprocessing.Pool",
        "multiprocessing.set_start_method",
    }
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = qualified(node.func, aliases)
        bad_context = callee == "multiprocessing.get_context" and not _is_spawn(node)
        bad_process = callee.endswith((".Process", ".Pool")) and (
            callee != f"{SPAWN_CONTEXT}.Process" or module != "runtime/parallel.py"
        )
        bad_pool = False
        if callee.rsplit(".", 1)[-1] == "ProcessPoolExecutor":
            context = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "mp_context"), None
            )
            bad_pool = qualified(context, aliases) != SPAWN_CONTEXT
        if callee in forbidden or bad_context or bad_process or bad_pool:
            findings.append(f"line {node.lineno}: unsafe process creation: {callee}")
    return findings


def _provider_receiver(node: ast.AST, aliases: Mapping[str, str]) -> bool:
    name = qualified(node, {})
    resolved = aliases.get(name, qualified(node, aliases))
    return "provider" in name.lower() or "SparkSessionProvider" in resolved


def eager_spark(source: str, module: str = "runtime/parallel.py") -> list[str]:
    tree = ast.parse(source)
    aliases = _bindings(tree)
    findings = []
    for node in ast.walk(tree):
        name = qualified(node, aliases)
        if isinstance(node, ast.Import):
            if any(item.name.split(".")[0] == "pyspark" for item in node.names):
                findings.append(f"line {node.lineno}: pyspark import")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "pyspark" or any(
                item.name in SPARK_REFERENCES for item in node.names
            ):
                findings.append(f"line {node.lineno}: Spark/lock import")
        elif isinstance(node, ast.Name | ast.Attribute) and (
            name.rsplit(".", 1)[-1] in SPARK_REFERENCES
        ):
            findings.append(f"line {node.lineno}: forbidden bootstrap reference: {name}")
        elif isinstance(node, ast.Call):
            if qualified(node.func, aliases).rsplit(".", 1)[-1] == "SourceStateLock":
                findings.append(f"line {node.lineno}: parent acquired source lock")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and (_provider_receiver(node.func.value, aliases))
            ):
                findings.append(f"line {node.lineno}: provider acquired in bootstrap")
    return findings


def production_test_hooks(source: str, module: str = "runtime/parallel.py") -> list[str]:
    return [
        f"line {node.lineno}: production test hook: {node.value}"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("JANUS_TEST_")
    ]


DETECTORS: dict[str, Callable[[str, str], list[str]]] = {
    "threads": threaded_attempt,
    "spawn": unsafe_spawn,
    "spark": eager_spark,
    "hooks": production_test_hooks,
}


def _assert_surface(sources: Mapping[str, str]) -> None:
    assert sources, "the attempt sweep matched no Python modules"
    assert sources.keys() >= ANCHORS, f"missing anchors: {sorted(ANCHORS - sources.keys())}"
    assert len(sources) >= 40, f"only {len(sources)} modules swept"


def modules(*, whole_package: bool = False) -> dict[str, str]:
    paths = (
        PACKAGE.rglob("*.py")
        if whole_package
        else (path for package in SCOPED_PACKAGES for path in (PACKAGE / package).rglob("*.py"))
    )
    sources = {
        path.relative_to(PACKAGE).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(paths)
    }
    _assert_surface(sources)
    return sources


def _threading_allowance_findings(
    sources: Mapping[str, str], allowances: Mapping[str, str]
) -> list[str]:
    observed = {
        module for module, source in sources.items() if _threading_imports(ast.parse(source))
    }
    return (
        [
            f"{module}: unregistered threading import"
            for module in sorted(observed - allowances.keys())
        ]
        + [
            f"{module}: stale threading allowance"
            for module in sorted(allowances.keys() - observed)
        ]
        + [
            f"{module}: threading allowance needs a reason"
            for module, reason in allowances.items()
            if not reason.strip()
        ]
    )


@pytest.mark.parametrize("rule", DETECTORS)
def test_package_sweep(rule: str, record_property) -> None:
    sources = modules(whole_package=rule in {"spawn", "hooks"})
    if rule == "spark":
        sources = {"runtime/parallel.py": sources["runtime/parallel.py"]}
    record_property("swept_module_count", len(sources))
    findings = {
        module: found
        for module, source in sources.items()
        if (found := DETECTORS[rule](source, module))
    }
    assert findings == {}


def test_threading_imports_are_registered_with_current_reasons() -> None:
    assert _threading_allowance_findings(modules(), ALLOWED_THREADING) == []


def test_parallel_has_exactly_one_explicit_spawn_context() -> None:
    tree = ast.parse(modules()["runtime/parallel.py"])
    aliases = _bindings(tree)
    sites = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and qualified(node.func, aliases) == "multiprocessing.get_context"
        and _is_spawn(node)
    ]
    assert len(sites) == 1, f"expected exactly one spawn context site, found {sites}"


@pytest.mark.parametrize("missing", sorted(ANCHORS))
def test_surface_rejects_missing_anchor(missing: str) -> None:
    sources = modules()
    sources.pop(missing)
    with pytest.raises(AssertionError, match=missing):
        _assert_surface(sources)


def test_surface_rejects_empty_and_truncated_sweeps() -> None:
    for sources in ({}, dict.fromkeys(ANCHORS, "")):
        with pytest.raises(AssertionError):
            _assert_surface(sources)


@pytest.mark.parametrize(
    ("sources", "allowances", "expected"),
    [
        ({"runtime/new.py": "import threading"}, {}, "unregistered"),
        ({}, {"runtime/old.py": "Old worker"}, "stale"),
        ({"runtime/old.py": "# import threading"}, {"runtime/old.py": "Old worker"}, "stale"),
        ({"runtime/a.py": "from threading import Thread"}, {"runtime/a.py": " "}, "reason"),
    ],
)
def test_threading_registration_detector_rejects_bad_inventory(sources, allowances, expected):
    assert any(
        expected in finding for finding in _threading_allowance_findings(sources, allowances)
    )


def test_threading_registration_detector_accepts_registered_imports() -> None:
    assert (
        _threading_allowance_findings(
            {"runtime/a.py": "if TYPE_CHECKING:\n    from threading import Thread as T"},
            {"runtime/a.py": "Bounded metadata lookup"},
        )
        == []
    )


BAD_EXAMPLES = {
    "threads": [
        "import threading as t\nt.Thread(target=execute_source_attempt)",
        "from threading import Thread\nThread(target=lambda: SourceExecutor().execute(plan))",
        "from concurrent.futures import ThreadPoolExecutor\n"
        "ThreadPoolExecutor().submit(execute_source_attempt, plan)",
        "from concurrent.futures import ThreadPoolExecutor as E\n"
        "with E() as pool: pool.map(SourceExecutionService.execute, plans)",
        "import concurrent.futures.thread as t\nt.ThreadPoolExecutor().submit(run_worker, task)",
        "from concurrent.futures import thread\n"
        "thread.ThreadPoolExecutor().map(_execute_source, plans)",
        "from threading import Thread\n"
        "def wrapper():\n    execute_source_attempt(plan)\nThread(target=wrapper)",
        "from threading import Thread\n"
        "def inner():\n    run_worker(task)\ndef outer():\n    inner()\nThread(target=outer)",
        "from threading import Thread\n"
        "class Worker:\n    def run(self):\n        _execute_source(plan)\n"
        "Thread(target=Worker().run)",
        "from concurrent.futures import ThreadPoolExecutor as E\n"
        "x = SourceExecutor()\npool = E()\npool.submit(x.execute, plan)",
        "from threading import Thread\ncallback = lambda: run_worker(task)\nThread(None, callback)",
        "from concurrent.futures import ThreadPoolExecutor\n"
        "self.pool = ThreadPoolExecutor()\nself.pool.submit(run_worker, task)",
        "from threading import Thread\n"
        "def recursive():\n    recursive()\n    run_worker(task)\nThread(target=recursive)",
        "from threading import Thread\n"
        "from janus.runtime.batch import execute_source_attempt as go\n"
        "Thread(target=go)",
    ],
    "spawn": [
        "import multiprocessing as mp\nmp.get_context('fork')",
        "from multiprocessing import get_context\nget_context('forkserver')",
        "import multiprocessing\nmultiprocessing.get_context()",
        "import multiprocessing\nmultiprocessing.get_context(method=method)",
        "from multiprocessing import Process as P\nP(target=run_worker)",
        "import multiprocessing as mp\nmp.Pool(3)",
        "import multiprocessing as mp\nmp.set_start_method('spawn')",
        "import os as o\no.fork()",
        "from os import forkpty as fork\nfork()",
        "import os\nos.posix_spawn('python', [], {})",
        "from concurrent.futures import ProcessPoolExecutor as E\nE()",
        "from concurrent.futures import ProcessPoolExecutor\n"
        "import multiprocessing as mp\nProcessPoolExecutor(mp_context=mp.get_context('fork'))",
        "context.Process(target=run_worker)",
        "import multiprocessing as mp\nctx = mp.get_context('spawn')\nctx.Pool(3)",
    ],
    "spark": [
        "def f():\n    provider.get()",
        "spark_provider.get()",
        "self.compute_provider.get()",
        "def f(compute: SparkSessionProvider):\n    compute.get()",
        "def f(compute: 'SparkSessionProvider'):\n    compute.get()",
        "compute: SparkSessionProvider = factory()\ncompute.get()",
        "def f(compute: SparkSessionProvider | None):\n    compute.get()",
        "from typing import Optional\n"
        "def f(compute: Optional[SparkSessionProvider]):\n    compute.get()",
        "self.compute: SparkSessionProvider = factory()\nself.compute.get()",
        "compute = SparkSessionProvider(config, paths)\ncompute.get()",
        "from pyspark.sql import SparkSession",
        "if TYPE_CHECKING:\n    import pyspark as spark",
        "def f():\n    from pyspark import SparkContext",
        "build_spark_session(config, paths)",
        "session_factory = build_spark_session",
        "SparkContext(config)",
        "builder.getOrCreate()",
        "from janus.utils.spark import build_spark_session as build\nbuild(config, paths)",
        "scoped_request_input_session(provider)",
        "from janus.runtime.locking import hold_source_state_lock as hold\nhold(plan)",
        "SourceStateLock(plan)",
    ],
    "hooks": [
        "hook = 'JANUS_TEST_EAGER_SPARK'",
        "def f():\n    return os.getenv('JANUS_TEST_BOOTSTRAP')",
        "if TYPE_CHECKING:\n    key = 'JANUS_TEST_'",
    ],
}
CLEAN_EXAMPLES = {
    "threads": [
        "import threading\nthreading.Thread(target=self._run_terminal_worker)",
        "from concurrent.futures import ThreadPoolExecutor as E\n"
        "with E() as pool: pool.submit(download_page, request)",
        "from threading import Thread\nThread(target=lambda: download_page(request))",
        "from threading import Thread\n"
        "def worker():\n    loader(config, paths, identifier)\nThread(target=worker)",
        "from multiprocessing import get_context\nget_context('spawn').Process(target=run_worker)",
        "execute_source_attempt(plan)",
        "from concurrent.futures import ProcessPoolExecutor as E\n"
        "with E(mp_context=ctx) as pool: pool.submit(run_worker, task)",
    ],
    "spawn": [
        "from multiprocessing import get_context\nget_context('spawn').Process(target=run_worker)",
        "import multiprocessing as mp\nctx = mp.get_context(method='spawn')\n"
        "ctx.Process(target=run_worker)",
        "from concurrent.futures import ProcessPoolExecutor as E\n"
        "import multiprocessing as mp\nE(mp_context=mp.get_context('spawn'))",
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from multiprocessing import get_context\nctx = get_context('spawn')\n"
        "ProcessPoolExecutor(mp_context=ctx)",
        "import os\nos.getpid()",
        "# multiprocessing.get_context('fork')\ntext = 'os.fork()'",
    ],
    "spark": [
        "SparkSessionProvider(config, paths, logger)",
        "from janus.runtime.spark_lifecycle import SparkSessionProvider",
        "settings.get('key')",
        "def f(compute: dict):\n    return compute.get('key')",
        "provider.was_started",
        "provider.stop()",
        "source_state_lock_path(plan)",
        "# import pyspark\ntext = 'SparkSession and build_spark_session'",
    ],
    "hooks": [
        "key = 'JANUS_MAX_PARALLEL'",
        "# JANUS_TEST_EAGER_SPARK\npass",
        "text = 'mentions JANUS_TEST_ inside a description'",
    ],
}


@pytest.mark.parametrize(
    ("rule", "source"),
    [(rule, source) for rule, examples in BAD_EXAMPLES.items() for source in examples],
)
def test_detector_flags_planted_violation(rule: str, source: str) -> None:
    assert DETECTORS[rule](source, "runtime/parallel.py"), (rule, source)


@pytest.mark.parametrize(
    ("rule", "source"),
    [(rule, source) for rule, examples in CLEAN_EXAMPLES.items() for source in examples],
)
def test_detector_accepts_clean_case(rule: str, source: str) -> None:
    assert DETECTORS[rule](source, "runtime/parallel.py") == []


@pytest.mark.parametrize("module", ["runtime/other.py", "strategies/worker.py"])
def test_spawned_process_is_only_allowed_in_parallel_module(module: str) -> None:
    source = (
        "from multiprocessing import get_context\nget_context('spawn').Process(target=run_worker)"
    )
    assert unsafe_spawn(source, module)
