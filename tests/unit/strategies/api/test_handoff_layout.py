"""NFR-2 / AC-2: derived handoffs never become raw resume or replay input."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from janus.readers.spark import _compact_complete_parent_paths
from janus.scripts.rehydrate import _rediscover_raw_artifacts as replay_artifacts
from janus.strategies.api.artifacts import (
    _numbered_pages_up_to,
    _pages_dir,
    _rediscover_all_artifacts_for_input,
    _rediscover_raw_artifacts,
)
from janus.strategies.api.handoff import HANDOFF_DIRECTORY_NAME
from tests.fixtures.handoff.cases import build_raw_zone, page_snapshot

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
    kind = _derived_path_kind(node, bindings)
    if kind:
        return kind
    if (
        isinstance(node, ast.Call)
        and _call_name(node) == "resolve_output"
        and any(isinstance(arg, ast.Constant) and arg.value == "raw" for arg in node.args)
    ):
        return "raw"
    return ""


def _derived_path_kind(node, bindings):
    if isinstance(node, ast.Attribute):
        kind = "raw" if node.attr == "raw_output" else _path_kind(node.value, bindings)
        return "raw" if kind == "pages" and node.attr in {"parent", "parents"} else kind
    if isinstance(node, ast.Subscript):
        return _path_kind(node.value, bindings)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return _path_kind(node.func.value, bindings)
        if _call_name(node) == "Path" and node.args:
            return _path_kind(node.args[0], bindings)
    if isinstance(node, ast.BinOp):
        return _path_kind(node.left, bindings)
    return ""


def _bindings(function, inherited=None):
    bindings = dict(inherited or {})
    assignments = [
        node for node in ast.walk(function) if isinstance(node, ast.Assign | ast.AnnAssign)
    ]
    for _ in range(len(assignments) + 1):
        previous = dict(bindings)
        for node in assignments:
            kind = _path_kind(node.value, bindings) if node.value is not None else ""
            if kind:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                bindings.update(
                    (target.id, kind) for target in targets if isinstance(target, ast.Name)
                )
        if bindings == previous:
            break
    return bindings


def _walker_filters(function, call):
    """Only predicates that filter this walk can establish an exclusion."""
    return [
        (node.target.id, predicate)
        for node in ast.walk(function)
        if isinstance(node, ast.comprehension)
        and any(child is call for child in ast.walk(node.iter))
        and isinstance(node.target, ast.Name)
        for predicate in node.ifs
    ]


def _conjuncts(predicate):
    if isinstance(predicate, ast.BoolOp) and isinstance(predicate.op, ast.And):
        return [term for value in predicate.values for term in _conjuncts(value)]
    return [predicate]


def _excludes_handoff(function, call):
    return any(
        (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
            and isinstance(node.left, ast.Name)
            and node.left.id == "HANDOFF_DIRECTORY_NAME"
            and isinstance(node.comparators[0], ast.Attribute)
            and node.comparators[0].attr == "parts"
            and isinstance(node.comparators[0].value, ast.Name)
            and node.comparators[0].value.id == variable
        )
        or (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Call)
            and _call_name(node.operand) == "_is_derived_handoff_artifact"
            and len(node.operand.args) == 2
            and ast.dump(node.operand.args[0]) == ast.dump(call.func.value)
            and isinstance(node.operand.args[1], ast.Name)
            and node.operand.args[1].id == variable
        )
        for variable, predicate in _walker_filters(function, call)
        for node in _conjuncts(predicate)
    )


def _selects_run_directories(call, function):
    # replay_plan chooses run roots, not payloads: exact non-recursive layout + is_dir filter.
    return (
        _call_name(call) == "glob"
        and bool(call.args)
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "ingestion_date=*/run_id=*"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _call_name(node) == "is_dir"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == variable
            for variable, predicate in _walker_filters(function, call)
            for node in _conjuncts(predicate)
        )
    )


def _walk_stays_in_pages(call, kind):
    if kind != "pages":
        return False
    if _call_name(call) == "iterdir":
        return True
    if _call_name(call) != "glob" or not call.args:
        return False
    pattern = call.args[0]
    # A recursive glob or a parent traversal escapes the immediate page-file set.
    literals = [node.value for node in ast.walk(pattern) if isinstance(node, ast.Constant)]
    return (
        isinstance(pattern, ast.Constant | ast.JoinedStr)
        and bool(literals)
        and all(
            isinstance(value, str) and "/" not in value and "**" not in value for value in literals
        )
    )


def _scan(modules):
    functions = [
        (path, node)
        for path, tree in modules.items()
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    for path, tree in modules.items():
        scope = ast.parse("def _module_scope(): pass").body[0]
        scope.name = "<module>"
        scope.body = [
            node
            for node in tree.body
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]
        functions.append((path, scope))
    function_names = {function.name for _, function in functions}
    walker_functions = {
        function.name
        for _, function in functions
        if any(
            isinstance(node, ast.Call) and _call_name(node) in WALK_METHODS
            for node in ast.walk(function)
        )
    }
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
            if method in inherited and method in walker_functions:
                inventory.append((path, function.name, f"delegated {inherited[method]} walk"))
            if method not in WALK_METHODS or not isinstance(call.func, ast.Attribute):
                continue
            kind = _path_kind(call.func.value, bindings)
            if not kind and method == "walk" and call.args:
                kind = _path_kind(call.args[0], bindings)
            if not kind:
                continue
            inventory.append((path, function.name, method))
            if (
                not _walk_stays_in_pages(call, kind)
                and not _excludes_handoff(function, call)
                and not _selects_run_directories(call, function)
            ):
                violations.append((path, function.name, call.lineno, method))
    return sorted(set(inventory)), violations


def _modules():
    return {
        path.relative_to(SOURCE_ROOT).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for package in ("strategies/api", "scripts")
        for path in sorted((SOURCE_ROOT / package).rglob("*.py"))
    }


def test_walker_sweep_covers_both_packages_and_all_four_known_walkers():
    modules = _modules()
    assert (
        modules and "strategies/api/artifacts.py" in modules and "scripts/rehydrate.py" in modules
    )
    inventory, _ = _scan(modules)
    assert inventory, "the package sweep found no raw walkers"
    matched = {(path, name) for path, name, _ in inventory}
    assert {
        ("strategies/api/artifacts.py", "_rediscover_all_artifacts_for_input"),
        ("strategies/api/artifacts.py", "_rediscover_raw_artifacts"),
        ("strategies/api/artifacts.py", "_numbered_pages_up_to"),
        ("scripts/rehydrate.py", "_rediscover_raw_artifacts"),
    } <= matched
    for path, name, method in inventory:
        print(f"walker: {path}::{name} ({method})")


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


def test_raw_walkers_are_page_anchored_or_exclude_derived_handoff():
    _, violations = _scan(_modules())
    assert violations == [], violations


@pytest.mark.parametrize(
    ("index", "count", "prefix", "directory"),
    [
        (1, 1, None, "pages"),
        (1, 1, PREFIX, "pages"),
        (1, 3, None, "request-input-000001"),
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
    print(f"pages_dir: count={count} index={index} prefix={prefix} -> {expected}")


@pytest.fixture(params=["single_input", "two_inputs"])
def compacted_zone(tmp_path, request):
    zone = build_raw_zone(tmp_path, request.param)
    # Non-padded names make lexical order disagree with numeric page order.
    directory = "pages" if request.param == "single_input" else "request-input-000001"
    extra = tuple(
        zone.writer.write_json(
            zone.plan, f"{directory}/page-{index}.json", [{"id": index}]
        ).artifact
        for index in (10, 9)
    )
    artifacts = tuple(
        sorted(
            (*zone.extraction.artifacts, *extra),
            key=lambda artifact: (
                Path(artifact.path).parent,
                int(Path(artifact.path).stem.removeprefix("page-")),
                Path(artifact.path),
            ),
        )
    )
    zone = replace(
        zone,
        extraction=replace(
            zone.extraction,
            artifacts=artifacts,
            records_extracted=zone.extraction.records_extracted + len(extra),
        ),
    )
    before = page_snapshot(zone)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, zone.extraction)
    assert page_snapshot(zone) == before
    assert handoff.artifacts and handoff.single_artifact_format() == "jsonl"
    return zone, handoff


def test_real_compacted_handoff_is_a_sibling_inside_the_same_run_prefix(compacted_zone):
    zone, handoff = compacted_zone
    handoff_dir = zone.run_root / HANDOFF_DIRECTORY_NAME
    assert handoff_dir.exists()
    for page in zone.extraction.artifacts:
        pages_dir = Path(page.path).parent
        assert handoff_dir.parent == pages_dir.parent
        assert pages_dir not in handoff_dir.parents and handoff_dir not in pages_dir.parents
        assert HANDOFF_DIRECTORY_NAME not in pages_dir.parts
        assert HANDOFF_DIRECTORY_NAME not in Path(page.path).parts
        print(f"siblings: pages={pages_dir} handoff={handoff_dir}")
    assert all(handoff_dir in Path(artifact.path).parents for artifact in handoff.artifacts)
    assert all(
        HANDOFF_DIRECTORY_NAME in Path(artifact.path).parts for artifact in handoff.artifacts
    )


@pytest.mark.parametrize("last_page", [0, 1, 2, 9, 10])
def test_all_rediscovery_paths_return_only_pages_after_real_compaction(compacted_zone, last_page):
    zone, _ = compacted_zone
    count = 2 if zone.extraction.metadata_as_dict()["fixture"] == "two_inputs" else 1
    recovered, resumed, numbered = [], [], []
    for index in range(1, count + 1):
        recovered.extend(
            _rediscover_all_artifacts_for_input(zone.plan, zone.layout, index, count, zone.prefix)
        )
        resumed.extend(
            _rediscover_raw_artifacts(
                zone.plan,
                zone.layout,
                {"last_page_number": last_page, "raw_path_prefix": str(zone.prefix)},
                index,
                count,
            )
        )
        numbered.extend(
            _numbered_pages_up_to(
                _pages_dir(zone.plan, zone.layout, index, count, zone.prefix),
                prefix="page-",
                suffix=".json",
                last_index=last_page,
            )
        )
    replayed = replay_artifacts(zone.replay_plan(), verify_checksums=True)
    expected_resume = tuple(
        artifact
        for artifact in zone.extraction.artifacts
        if int(Path(artifact.path).stem.removeprefix("page-")) <= last_page
    )
    assert tuple(recovered) == zone.extraction.artifacts
    assert tuple(resumed) == expected_resume
    assert numbered == [Path(artifact.path) for artifact in expected_resume]
    # Replay has its own path ordering; its page/checksum set must be identical.
    assert replayed == tuple(sorted(zone.extraction.artifacts, key=lambda artifact: artifact.path))
    for name, artifacts in (("completed", recovered), ("resume", resumed), ("replay", replayed)):
        assert all(
            HANDOFF_DIRECTORY_NAME not in Path(artifact.path).parts for artifact in artifacts
        )
        assert all(Path(artifact.path).suffix != ".jsonl" for artifact in artifacts)
        paths = [
            Path(artifact.path).relative_to(zone.run_root).as_posix() for artifact in artifacts
        ]
        print(f"rediscovery {name} last_page={last_page}: {paths}")


def test_sidecars_keep_handoff_files_individually_listed_for_spark(tmp_path):
    zone = build_raw_zone(tmp_path, spark={"handoff": {"max_artifact_mb": 1}})
    records = [{"id": index, "payload": "x" * 600_000} for index in range(5)]
    page = zone.writer.write_json(zone.plan, "pages/page-0010.json", records).artifact
    extraction = replace(
        zone.extraction,
        artifacts=(*zone.extraction.artifacts, page),
        records_extracted=zone.extraction.records_extracted + len(records),
    )
    zone = replace(zone, extraction=extraction)
    before = page_snapshot(zone)
    handoff = zone.strategy().build_normalization_handoff(zone.plan, extraction)
    artifacts = handoff.artifacts
    assert len(artifacts) > 1
    paths = tuple(artifact.path for artifact in artifacts)
    parent = Path(paths[0]).parent
    assert all(Path(path).parent == parent for path in paths)
    assert HANDOFF_DIRECTORY_NAME in parent.parts
    assert all(Path(path + ".sha256").is_file() for path in paths)
    assert _compact_complete_parent_paths(paths) == paths
    assert page_snapshot(zone) == before
    print(f"reader keeps {len(paths)} handoff files: {paths}")
    # Prove these files would trigger the collapse without their actual sidecars.
    for path in paths:
        Path(path + ".sha256").unlink()
    assert _compact_complete_parent_paths(paths) == (str(parent),)


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


@pytest.mark.parametrize(
    ("root", "candidate", "excluded"),
    [("raw_root", "p", True), ("other_root", "p", False), ("raw_root", "other", False)],
)
def test_walker_detector_ties_the_handoff_helper_to_the_walked_root_and_candidate(
    root, candidate, excluded
):
    source = f"""
def future_raw_walk(raw_root, other_root, other):
    return [p for p in raw_root.rglob("*")
            if not _is_derived_handoff_artifact({root}, {candidate})]
"""
    _, violations = _scan({"scripts/future_replay.py": ast.parse(source)})
    assert (violations == []) is excluded


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


@pytest.mark.parametrize("package", ["strategies/api", "scripts"])
@pytest.mark.parametrize(
    "source",
    [
        'def discover(raw_root):\n    return raw_root.rglob("*")\n',
        '\nraw_root.rglob("*")\n',
    ],
)
def test_package_sweep_detects_a_new_module(tmp_path, monkeypatch, package, source):
    for directory in ("strategies/api", "scripts"):
        (tmp_path / directory).mkdir(parents=True)
    path = tmp_path / package / "future_discovery.py"
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(f"{__name__}.SOURCE_ROOT", tmp_path)
    modules = _modules()
    assert f"{package}/future_discovery.py" in modules
    _, violations = _scan(modules)
    name = "discover" if source.startswith("def ") else "<module>"
    assert violations == [(f"{package}/future_discovery.py", name, 2, "rglob")]


@pytest.mark.parametrize("expression", ["alias.resolve()", "Path(alias)", "alias / 'nested'"])
def test_walker_detector_tracks_derived_raw_paths(expression):
    source = f"""
def future_raw_walk(plan):
    root = storage_layout.resolve_output(plan, "raw").resolved_path
    alias = root
    return ({expression}).rglob("*")
"""
    inventory, violations = _scan({"strategies/api/future.py": ast.parse(source)})
    assert inventory and violations


def test_walker_detector_tracks_annotated_raw_aliases():
    source = """
def future_raw_walk(plan):
    root: Path = Path(plan.raw_output.path)
    directory: Path = root
    return directory.rglob("*")
"""
    _, violations = _scan({"scripts/future.py": ast.parse(source)})
    assert violations


@pytest.mark.parametrize(
    "body",
    [
        "excluded = HANDOFF_DIRECTORY_NAME not in p.parts\n        result.append(p)",
        "if HANDOFF_DIRECTORY_NAME not in p.parts:\n            pass\n        result.append(p)",
    ],
)
def test_walker_detector_rejects_checks_that_do_not_filter_the_walk(body):
    source = f"""
def future_raw_walk(raw_root):
    result = []
    for p in raw_root.rglob("*"):
        {body}
    return result
"""
    _, violations = _scan({"scripts/future.py": ast.parse(source)})
    assert violations


def test_walker_detector_rejects_a_bypass_in_an_exclusion():
    source = """
def future_raw_walk(raw_root):
    return [p for p in raw_root.rglob("*")
            if HANDOFF_DIRECTORY_NAME not in p.parts or p.is_file()]
"""
    _, violations = _scan({"scripts/future.py": ast.parse(source)})
    assert violations


@pytest.mark.parametrize("subject", ["directory.parent", "directory.parents[0]"])
def test_walker_detector_rejects_escaping_the_pages_anchor(subject):
    source = f"""
def future_raw_walk(plan):
    directory = _pages_dir(plan)
    return {subject}.glob("*.jsonl")
"""
    _, violations = _scan({"strategies/api/future.py": ast.parse(source)})
    assert violations


@pytest.mark.parametrize("walk", ['rglob("*")', 'glob("**/*")', 'glob("../handoff/*")'])
def test_walker_detector_rejects_recursive_or_parent_globs_in_pages(walk):
    source = f"""
def future_raw_walk(plan):
    directory = _pages_dir(plan)
    return directory.{walk}
"""
    _, violations = _scan({"strategies/api/future.py": ast.parse(source)})
    assert violations


@pytest.mark.parametrize("guard", ["other.is_dir()", "p.is_dir() or p.is_file()"])
def test_run_root_selection_requires_a_filter_on_the_walked_candidate(guard):
    source = f"""
def future_raw_walk(runs_root, other):
    return [p for p in runs_root.glob("ingestion_date=*/run_id=*") if {guard}]
"""
    _, violations = _scan({"scripts/future.py": ast.parse(source)})
    assert violations
