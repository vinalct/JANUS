import ast
from pathlib import Path

import pytest

from janus.readers.spark import _compact_complete_parent_paths
from janus.scripts.rehydrate import _rediscover_raw_artifacts as replay_artifacts
from janus.strategies.api.artifacts import (
    _pages_dir,
    _rediscover_all_artifacts_for_input,
    _rediscover_raw_artifacts,
)
from tests.fixtures.handoff.cases import build_raw_zone

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "src/janus"
WALK_METHODS = {"rglob", "glob", "iterdir", "walk"}
RAW_NAMES = {"raw_root", "raw_dir", "runs_root"}
PREFIX = Path("runs/ingestion_date=2026-09-16/run_id=fixture")


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _path_kind(node, bindings):
    if isinstance(node, ast.Name):
        return bindings.get(node.id, "raw" if node.id in RAW_NAMES else "")
    if isinstance(node, ast.Call) and _call_name(node) == "_pages_dir":
        return "pages"
    if isinstance(node, ast.BinOp):
        return _path_kind(node.left, bindings)
    children = list(ast.walk(node))
    if any(isinstance(child, ast.Attribute) and child.attr == "raw_output" for child in children):
        return "raw"
    if any(
        isinstance(child, ast.Call)
        and _call_name(child) == "resolve_output"
        and any(isinstance(arg, ast.Constant) and arg.value == "raw" for arg in child.args)
        for child in children
    ):
        return "raw"
    return ""


def _bindings(function, inherited=None):
    bindings = dict(inherited or {})
    for _ in range(4):
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                kind = _path_kind(node.value, bindings)
                if kind:
                    bindings.update(
                        (target.id, kind) for target in node.targets if isinstance(target, ast.Name)
                    )
    return bindings


def _excludes_handoff(function, call):
    variables = {
        name.id
        for node in ast.walk(function)
        if isinstance(node, ast.comprehension | ast.For)
        and any(child is call for child in ast.walk(node.iter))
        for name in ast.walk(node.target)
        if isinstance(name, ast.Name)
    }
    # Tie the exclusion to this walker, not an unrelated path elsewhere in the function.
    return any(
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotIn)
        and isinstance(node.left, ast.Name)
        and node.left.id == "HANDOFF_DIRECTORY_NAME"
        and isinstance(node.comparators[0], ast.Attribute)
        and node.comparators[0].attr == "parts"
        and isinstance(node.comparators[0].value, ast.Name)
        and node.comparators[0].value.id in variables
        for node in ast.walk(function)
    )


def _selects_run_directories(call, function):
    # replay_plan chooses run roots, not payloads: exact non-recursive layout + is_dir filter.
    return (
        _call_name(call) == "glob"
        and bool(call.args)
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "ingestion_date=*/run_id=*"
        and any(
            isinstance(node, ast.Call) and _call_name(node) == "is_dir"
            for node in ast.walk(function)
        )
    )


def _scan(modules):
    functions = [
        (path, node)
        for path, tree in modules.items()
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    function_names = {function.name for _, function in functions}
    callers = {}
    # An anchor is safe only when every caller passes a pages directory.
    for _, function in functions:
        bindings = _bindings(function)
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            if call.args and _call_name(call) in function_names:
                callers.setdefault(_call_name(call), []).append(_path_kind(call.args[0], bindings))
    inherited = {
        name: "pages" if all(kind == "pages" for kind in kinds) else "raw"
        for name, kinds in callers.items()
        if any(kind in {"pages", "raw"} for kind in kinds)
    }
    inventory, violations = [], []
    for path, function in functions:
        parameter_bindings = {}
        if function.name in inherited and function.args.args:
            parameter_bindings[function.args.args[0].arg] = inherited[function.name]
        bindings = _bindings(function, parameter_bindings)
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            method = _call_name(call)
            if method in inherited:
                inventory.append((path, function.name, "delegated pages walk"))
            if method not in WALK_METHODS or not isinstance(call.func, ast.Attribute):
                continue
            kind = _path_kind(call.func.value, bindings)
            if not kind and method == "walk" and call.args:
                kind = _path_kind(call.args[0], bindings)
            if not kind:
                continue
            inventory.append((path, function.name, method))
            if (
                kind != "pages"
                and not _excludes_handoff(function, call)
                and not _selects_run_directories(call, function)
            ):
                violations.append((path, function.name, call.lineno, method))
    return inventory, violations


def _modules():
    return {
        path.relative_to(SOURCE_ROOT).as_posix(): ast.parse(path.read_text())
        for package in ("strategies/api", "scripts")
        for path in sorted((SOURCE_ROOT / package).rglob("*.py"))
    }


def test_walker_sweep_covers_both_packages_and_all_four_known_walkers():
    modules = _modules()
    assert (
        modules and "strategies/api/artifacts.py" in modules and "scripts/rehydrate.py" in modules
    )
    inventory, _ = _scan(modules)
    matched = {(path, name) for path, name, _ in inventory}
    assert {
        ("strategies/api/artifacts.py", "_rediscover_all_artifacts_for_input"),
        ("strategies/api/artifacts.py", "_rediscover_raw_artifacts"),
        ("strategies/api/artifacts.py", "_numbered_pages_up_to"),
        ("scripts/rehydrate.py", "_rediscover_raw_artifacts"),
    } <= matched


@pytest.mark.parametrize("method", ["rglob", "glob", "iterdir", "walk"])
def test_walker_detector_flags_planted_raw_root_walks(method):
    argument = "" if method == "iterdir" else '"*"'
    source = f"""
def future_raw_walk(plan):
    root = Path(plan.raw_output.path)
    return root.{method}({argument})
"""
    _, violations = _scan({"scripts/future_replay.py": ast.parse(source)})
    assert violations
    print(f"walker detector fired: {violations}")


def test_walker_detector_accepts_real_api_artifacts_and_explicit_exclusion():
    guarded = """
def future_raw_walk(plan):
    root = Path(plan.raw_output.path)
    return [p for p in root.rglob("*") if HANDOFF_DIRECTORY_NAME not in p.parts]
"""
    _, violations = _scan(
        {
            "strategies/api/artifacts.py": ast.parse(
                (SOURCE_ROOT / "strategies/api/artifacts.py").read_text()
            ),
            "scripts/future_replay.py": ast.parse(guarded),
        }
    )
    assert violations == []


@pytest.mark.xfail(strict=True, reason="green")
def test_raw_walkers_are_page_anchored_or_exclude_derived_handoff():
    _, violations = _scan(_modules())
    assert violations == [], violations


@pytest.mark.parametrize(
    ("index", "count", "prefix", "directory"),
    [
        (1, 1, None, "pages"),
        (1, 1, PREFIX, "pages"),
        (1, 2, None, "request-input-000001"),
        (2, 2, None, "request-input-000002"),
        (1, 2, PREFIX, "request-input-000001"),
        (2, 2, PREFIX, "request-input-000002"),
        (1, 3, PREFIX, "request-input-000001"),
        (2, 3, PREFIX, "request-input-000002"),
        (3, 3, PREFIX, "request-input-000003"),
    ],
)
def test_pages_dir_keeps_prefixed_and_legacy_flat_layouts(
    tmp_path, index, count, prefix, directory
):
    zone = build_raw_zone(tmp_path)
    raw_root = zone.layout.resolve_output(zone.plan, "raw").resolved_path
    expected = raw_root / directory if prefix is None else raw_root / prefix / directory
    assert _pages_dir(zone.plan, zone.layout, index, count, prefix) == expected


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize("case", ["single_input", "two_inputs"])
def test_real_compacted_handoff_is_a_sibling_inside_the_same_run_prefix(tmp_path, case):
    zone = build_raw_zone(tmp_path, case)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction)
    assert handoff.artifacts and all(artifact.format == "jsonl" for artifact in handoff.artifacts)
    handoff_dir = zone.run_root / "handoff"
    assert handoff_dir.exists()
    for page in zone.extraction.artifacts:
        pages_dir = Path(page.path).parent
        assert handoff_dir.parent == pages_dir.parent
        assert pages_dir not in handoff_dir.parents and handoff_dir not in pages_dir.parents
        assert "handoff" not in pages_dir.parts
        assert not str(handoff_dir).startswith(str(pages_dir))
    assert all(handoff_dir in Path(artifact.path).parents for artifact in handoff.artifacts)


@pytest.mark.xfail(strict=True, reason="green")
@pytest.mark.parametrize("case", ["single_input", "two_inputs"])
def test_all_rediscovery_paths_return_only_pages_after_real_compaction(tmp_path, case):
    zone = build_raw_zone(tmp_path, case)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction)
    assert handoff.single_artifact_format() == "jsonl"
    count = 2 if case == "two_inputs" else 1
    recovered, resumed = [], []
    for index in range(1, count + 1):
        recovered.extend(
            _rediscover_all_artifacts_for_input(zone.plan, zone.layout, index, count, zone.prefix)
        )
        resumed.extend(
            _rediscover_raw_artifacts(
                zone.plan,
                zone.layout,
                {"last_page_number": 3, "raw_path_prefix": str(zone.prefix)},
                index,
                count,
            )
        )
    replayed = replay_artifacts(zone.replay_plan(), verify_checksums=True)
    for artifacts in (tuple(recovered), tuple(resumed), replayed):
        assert artifacts == zone.extraction.artifacts
        assert all("handoff" not in Path(artifact.path).parts for artifact in artifacts)
        assert all(not artifact.path.endswith(".jsonl") for artifact in artifacts)


def test_sidecars_keep_handoff_files_individually_listed_for_spark(tmp_path):
    zone = build_raw_zone(tmp_path)
    artifacts = tuple(
        zone.writer.write_json_lines(
            zone.plan, f"handoff/pages/records-{index:04d}.jsonl", [{"id": index}]
        ).artifact
        for index in (1, 2)
    )
    paths = tuple(artifact.path for artifact in artifacts)
    assert all(Path(path + ".sha256").is_file() for path in paths)
    assert _compact_complete_parent_paths(paths) == paths


def test_walker_detector_flags_os_walk_and_raw_aliases():
    source = """
def future_raw_walk(plan):
    raw_root = Path(plan.raw_output.path)
    alias = raw_root
    return list(os.walk(alias))
"""
    _, violations = _scan({"scripts/future_replay.py": ast.parse(source)})
    assert violations


def test_walker_detector_rejects_an_exclusion_for_an_unrelated_path():
    source = """
def future_raw_walk(raw_root, other):
    unrelated = HANDOFF_DIRECTORY_NAME not in other.parts
    return [p for p in raw_root.rglob("*")]
"""
    _, violations = _scan({"scripts/future_replay.py": ast.parse(source)})
    assert violations


def test_raw_caller_cannot_borrow_another_callers_pages_anchor():
    source = """
def page_discovery(plan):
    directory = _pages_dir(plan)
    return helper(directory)
def raw_discovery(raw_root):
    return helper(raw_root)
def helper(directory):
    return list(directory.rglob("*"))
"""
    _, violations = _scan({"scripts/future_replay.py": ast.parse(source)})
    assert any(item[1] == "helper" for item in violations)
