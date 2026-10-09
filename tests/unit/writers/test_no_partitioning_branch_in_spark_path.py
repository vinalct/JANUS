import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[3] / "src/janus/writers"
WRITE_PATHS = {"_write_bronze_iceberg", "_merge_bronze_iceberg"}

PLANNER_BOUNDARIES = {
    "plan_write_partitioning", "apply_write_partitioning", "plan_full_refresh_overwrite",
}
PARTITION_NAMES = {
    "current_partitions",
    "target_partitions",
    "estimated_input_bytes",
    "estimated_bytes",
    "input_bytes",
}
SIZE_NAMES = {"target_file_size_bytes"}
HELPER_BOUNDARIES = {
    ("table_state.py", "read_target_partitions"),
}


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _tracked_name(name, names):
    return (
        name in names
        or "partitions" in name.lower()
        or ("bytes" in name.lower() and "estimat" in name.lower())
    )


def _partition_value(node, names):
    return any(
        (isinstance(child, ast.Name) and _tracked_name(child.id, names))
        or (isinstance(child, ast.Attribute) and _tracked_name(child.attr, names))
        or (isinstance(child, ast.Call) and _call_name(child) == "getNumPartitions")
        for child in ast.walk(node)
    )


def _decision_violations(function):
    names = set(PARTITION_NAMES)
    while True:
        before = set(names)
        for node in ast.walk(function):
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr):
                value = node.value
                if isinstance(value, ast.Call) and _call_name(value) in PLANNER_BOUNDARIES:
                    continue  # The return value is a plan or frame, not a count/estimate.
                if value is not None and _partition_value(value, names | SIZE_NAMES):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names.update(
                        child.id for target in targets for child in ast.walk(target)
                        if isinstance(child, ast.Name)
                    )
        if names == before:
            break
    violations = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"coalesce", "repartition"}:
                violations.append((function.name, node.lineno, "direct " + node.func.attr))
        elif isinstance(node, ast.Compare) and _partition_value(node, names):
            violations.append((function.name, node.lineno, "partition-count comparison"))
        elif isinstance(node, ast.BinOp) and _partition_value(node, names | SIZE_NAMES):
            violations.append((function.name, node.lineno, "partition/byte arithmetic"))
    return violations


def _scan(modules):
    functions = {}
    for path, tree in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.setdefault(node.name, []).append((path, node))
    roots = {name for name in functions if name.startswith(("_write_bronze", "_merge_bronze"))}
    visited = set()
    pending = [item for name in roots for item in functions[name]]
    violations = []
    while pending:
        path, function = pending.pop()
        key = (path, function.lineno)
        if (
            key in visited
            or function.name in PLANNER_BOUNDARIES
            or (path, function.name) in HELPER_BOUNDARIES
        ):
            continue
        visited.add(key)
        violations.extend((path, *item) for item in _decision_violations(function))
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                pending.extend(functions.get(_call_name(node), []))
    return roots, violations


def _modules():
    return {
        path.relative_to(PACKAGE).as_posix(): ast.parse(path.read_text())
        for path in sorted(PACKAGE.rglob("*.py"))
    }


def test_partition_sweep_matches_package_and_both_write_paths():
    modules = _modules()
    assert modules and "spark.py" in modules and "overwrite.py" in modules
    roots, _ = _scan(modules)
    assert roots >= WRITE_PATHS
    spark_functions = {
        node.name for node in ast.walk(modules["spark.py"])
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert spark_functions >= WRITE_PATHS


@pytest.mark.parametrize(
    "statement",
    [
        "if df.rdd.getNumPartitions() > n: df = df.coalesce(n)",
        "if df.rdd.getNumPartitions() > n: pass",
        "df.coalesce(1)",
        "df.repartition(4)",
        "n = df.rdd.getNumPartitions();\n    if n > 1: pass",
        "ideal = estimated_input_bytes / target_file_size_bytes",
        "ideal = estimated_bytes // 128",
        "if output_partitions != n: pass",
        "if declaration.partitions < n: pass",
        "if payload_estimated_bytes > 0: pass",
        "ideal = n / declaration.target_file_size_bytes",
        "size: int = df.rdd.getNumPartitions();\n    if size > 1: pass",
        "if (size := df.rdd.getNumPartitions()) > 1: pass",
        "size = declaration.target_file_size_bytes;\n    ideal = n // size",
    ],
)
def test_partition_detector_flags_planted_violations(statement):
    source = "def _write_bronze_iceberg(df, n, estimated_input_bytes):\n    " + statement
    _, violations = _scan({"future_spark_merge.py": ast.parse(source)})
    assert violations, source
    print(f"partition detector fired: {violations}")


def test_partition_detector_follows_hidden_decisions_into_helpers():
    source = """
def _merge_bronze_iceberg(df):
    return hidden_write(df)
def hidden_write(df):
    count = df.rdd.getNumPartitions()
    if count > 1:
        return df.coalesce(1)
    return df
"""
    _, violations = _scan({"future_spark_merge.py": ast.parse(source)})
    assert any(item[1] == "hidden_write" for item in violations)


def test_partition_detector_accepts_clean_overwrite_and_planner_dispatch():
    clean = """
def _write_bronze_iceberg(df, declaration):
    plan = plan_write_partitioning(
        current_partitions=df.rdd.getNumPartitions(),
        estimated_input_bytes=None, declaration=declaration)
    return apply_write_partitioning(df, plan)
def apply_write_partitioning(df, plan):
    if plan.action == "coalesce":
        return df.coalesce(plan.target_partitions)
    return df
"""
    _, violations = _scan(
        {
            "overwrite.py": ast.parse((PACKAGE / "overwrite.py").read_text()),
            "future_spark_merge.py": ast.parse(clean),
        }
    )
    assert violations == []


def test_real_write_paths_contain_no_partition_decisions_even_in_reachable_helpers():
    _, violations = _scan(_modules())
    assert violations == [], violations


def test_real_partitioning_module_is_clean_outside_write_paths():
    path = PACKAGE / "partitioning.py"
    assert path.is_file(), "introduces the pure planner/executor module"
    _, violations = _scan({
        "partitioning.py": ast.parse(path.read_text()),
        "overwrite.py": ast.parse((PACKAGE / "overwrite.py").read_text()),
    })
    assert violations == []


def test_same_named_write_functions_in_different_modules_cannot_hide_a_violation():
    dirty = "def _write_bronze_iceberg(df): return df.coalesce(1)"
    clean = "def _write_bronze_iceberg(df): return df"
    _, violations = _scan(
        {
            "spark.py": ast.parse(dirty),
            "future/spark.py": ast.parse(clean),
        }
    )
    assert any(item[0] == "spark.py" for item in violations)


def test_existing_spec_reader_boundary_cannot_hide_a_new_helper_with_the_same_name():
    source = """
def _write_bronze_iceberg(df):
    return read_target_partitions(df)
def read_target_partitions(df):
    return df.coalesce(1)
"""
    _, violations = _scan({"future/table_state.py": ast.parse(source)})
    assert any(item[1] == "read_target_partitions" for item in violations)
