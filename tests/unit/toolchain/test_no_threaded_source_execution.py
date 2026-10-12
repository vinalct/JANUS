"""Package sweeps with alias-aware detectors and planted/clean counterexamples."""

import ast

import pytest

from tests.support.parallel_execution import REPO_ROOT

PACKAGE = REPO_ROOT / "src/janus"
ATTEMPTS = {
    "execute_source_attempt",
    "SourceExecutor.execute",
    "SourceExecutionService.execute",
    "run_worker",
}


def names(tree):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({item.asname or item.name: item.name for item in node.names})
        elif isinstance(node, ast.ImportFrom):
            aliases.update(
                {item.asname or item.name: f"{node.module}.{item.name}" for item in node.names}
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value.func if isinstance(node.value, ast.Call) else node.value
            for target in node.targets:
                if isinstance(target, ast.Name) and qualified(value, aliases):
                    aliases[target.id] = qualified(value, aliases)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            aliases[node.target.id] = qualified(node.annotation, aliases)
        elif isinstance(node, ast.arg) and node.annotation is not None:
            aliases[node.arg] = qualified(node.annotation, aliases)
    return aliases


def qualified(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{qualified(node.value, aliases)}.{node.attr}"
    return ""


def threaded_attempt(calls, aliases):
    targets = []
    for call in calls:
        callee = qualified(call.func, aliases)
        if callee == "threading.Thread":
            targets.extend(keyword.value for keyword in call.keywords if keyword.arg == "target")
            targets.extend(call.args[1:2])
        if callee.endswith((".submit", ".map")) and call.args:
            targets.append(call.args[0])
    return any(
        qualified(node, aliases).endswith(name)
        for target in targets
        for node in ast.walk(target)
        for name in ATTEMPTS
    )


def unsafe_spawn(calls, aliases):
    for call in calls:
        callee = qualified(call.func, aliases)
        if callee in {
            "os.fork",
            "multiprocessing.Process",
            "multiprocessing.Pool",
            "multiprocessing.set_start_method",
        }:
            return True
        if callee == "multiprocessing.get_context":
            argument = (
                call.args[0]
                if call.args
                else next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "method"), None
                )
            )
            if not isinstance(argument, ast.Constant) or argument.value != "spawn":
                return True
    return False


def eager_spark(calls, aliases):
    return any(value.startswith("pyspark") for value in aliases.values()) or any(
        qualified(call.func, aliases).endswith(
            ("build_spark_session", "SparkSession", "getOrCreate")
        )
        or (
            qualified(call.func, aliases).endswith(".get")
            and "provider" in qualified(call.func, aliases).lower()
        )
        for call in calls
    )


def detects(source, rule):
    tree = ast.parse(source)
    aliases = names(tree)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return {"threads": threaded_attempt, "spawn": unsafe_spawn, "spark": eager_spark}[rule](
        calls, aliases
    )


def modules():
    paths = sorted(
        path
        for package in ("runtime", "orchestration", "cli", "adapters")
        for path in (PACKAGE / package).rglob("*.py")
    )
    assert paths
    relative = {path.relative_to(PACKAGE).as_posix() for path in paths}
    assert {"runtime/parallel.py", "runtime/batch.py", "adapters/dagster/runtime.py"} <= relative
    return paths


@pytest.mark.parametrize("rule", ["threads", "spawn", "spark"])
def test_package_sweep(rule):
    paths = modules()
    if rule == "spawn":
        paths = sorted(PACKAGE.rglob("*.py"))
    if rule == "spark":
        paths = [PACKAGE / "runtime/parallel.py"]
    assert paths
    assert [
        path.relative_to(PACKAGE).as_posix() for path in paths if detects(path.read_text(), rule)
    ] == []


@pytest.mark.parametrize(
    "rule,bad,clean",
    [
        (
            "threads",
            "import threading as t\nt.Thread(target=execute_source_attempt)",
            "import threading as t\nt.Thread(target=download_page)",
        ),
        (
            "threads",
            "from concurrent.futures import ThreadPoolExecutor as E\n"
            "with E() as e: e.submit(SourceExecutor.execute, plan)",
            "from concurrent.futures import ThreadPoolExecutor as E\n"
            "with E() as e: e.submit(download_page, request)",
        ),
        (
            "threads",
            "executor.map(SourceExecutionService.execute, plans)",
            "executor.map(parse_page, pages)",
        ),
        ("threads", "executor.submit(run_worker, request)", "run_worker(request, pipe)"),
        (
            "spawn",
            "import multiprocessing as mp\nmp.get_context('fork')",
            "import multiprocessing as mp\nmp.get_context('spawn')",
        ),
        (
            "spawn",
            "from multiprocessing import Process as P\nP(target=run_worker)",
            "context.Process(target=run_worker)",
        ),
        ("spawn", "import os as o\no.fork()", "import os as o\no.getpid()"),
        ("spawn", "import multiprocessing as mp\nmp.Pool(3)", "ctx.Pool(3)"),
        (
            "spawn",
            "import multiprocessing as mp\nmp.set_start_method('fork')",
            "mp.get_context('spawn')",
        ),
        (
            "threads",
            "import threading as t\nt.Thread(target=lambda: execute_source_attempt(plan))",
            "import threading as t\nt.Thread(target=lambda: download_page(request))",
        ),
        (
            "threads",
            "x = SourceExecutor()\nexecutor.submit(x.execute, plan)",
            "x = Parser()\nexecutor.submit(x.execute, page)",
        ),
        (
            "spark",
            "from pyspark.sql import SparkSession",
            "from janus.runtime import SparkSessionProvider",
        ),
        ("spark", "spark_provider.get()", "spark_provider.was_started"),
        (
            "spark",
            "def f(compute: SparkSessionProvider): compute.get()",
            "def f(compute: SparkSessionProvider): return compute.was_started",
        ),
        ("spark", "build_spark_session(config, paths)", "SparkSessionProvider(config, paths)"),
    ],
)
def test_detector_flags_planted_violation_and_accepts_clean_case(rule, bad, clean):
    # Run both meta-tests before the missing production anchor can xfail this case.
    assert detects(bad, rule) is True
    assert detects(clean, rule) is False
    modules()


def test_parallel_module_contains_no_test_only_bootstrap_hooks():
    paths = modules()
    assert paths
    assert "JANUS_TEST_" not in (PACKAGE / "runtime/parallel.py").read_text()
