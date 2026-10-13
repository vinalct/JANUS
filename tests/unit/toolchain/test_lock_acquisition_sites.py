"""Forward/backward coverage and one-inode locking over every JANUS module."""

import ast

import pytest

from janus.runtime.locking import LOCKED_ENTRY_POINTS, REACHED_THROUGH, STATE_MUTATING_PRIMITIVES
from tests.support.lock_acquisition import (
    HELPERS,
    acquisition_violation,
    acquisitions,
    annotation_name,
    parse_scopes,
    primitive_sites,
    qualified,
    reached_roots,
    replaces_lock,
    scope_nodes,
    second_lock,
    unregistered_mutations,
)
from tests.support.parallel_execution import REPO_ROOT

PACKAGE = REPO_ROOT / "src/janus"
BASELINE_MODULE_COUNT = 205  # TASK-01/TASK-03's package inventory; splits may only increase it.
MINIMUM_ENTRY_POINTS = 7
MINIMUM_PRIMITIVE_SITES = 15
MINIMUM_SOURCES = 31

# Only ambiguous, unrelated receivers belong here; resolved state stores cannot be exempted.
NOT_A_PRIMITIVE = {
    (
        "janus.maintenance.locking:SourceStateMaintenanceLock.release",
        "held",
    ): "Releases the shared SourceStateLock, not dead-letter state.",
    (
        "janus.maintenance.locking:source_lock",
        "lock",
    ): "Releases the injected MaintenanceLock protocol's acquisition.",
    (
        "janus.observability.openlineage.transport:_append_line",
        "os",
    ): "os.write appends the shared OpenLineage stream; it is not a source-state store.",
    (
        "janus.orchestration.persistence:_write_json_document",
        "stream",
    ): "A text stream writes a pipeline-scoped summary, not source state.",
    (
        "janus.runtime.locking:SourceStateLock.__exit__",
        "self",
    ): "The lock context manager releases its kernel acquisition.",
    (
        "janus.runtime.materialize:BronzeMaterializer.materialize",
        "writer",
    ): "The injected SparkDatasetWriter writes bronze tables, not metadata state.",
    (
        "janus.runtime.spark_lifecycle:SparkSessionProvider.take_cleanup_failures",
        "self._cleanup_failures",
    ): "Clears an in-memory list of Spark cleanup failures.",
    (
        "janus.strategies.api.speculation:_cancel_pending",
        "pending",
    ): "Clears the in-memory map of speculative futures.",
    (
        "janus.strategies.files.download:_load_streamed_payload",
        "buffered",
    ): "Writes downloaded bytes to an in-memory BytesIO buffer.",
    (
        "janus.utils.logging:build_structured_logger",
        "logger.handlers",
    ): "Clears in-memory logging handlers while composing a logger.",
    (
        "janus.writers.raw:_write_bytes",
        "stream",
    ): "Writes raw artifact bytes through a binary stream, not source metadata state.",
    (
        "janus.writers.raw:_copy_hashing",
        "destination",
    ): "Copies raw artifact bytes through a BinaryIO stream, not metadata state.",
    (
        "janus.writers.spark:SparkDatasetWriter.write",
        "writer",
    ): "Spark's DataFrameWriter.save materializes a dataset, not a state store.",
    (
        "janus.writers.spark:SparkDatasetWriter.write",
        "plan.source_config.spark",
    ): "spark.write is a declarative sizing field passed as data, not a callable.",
    (
        "janus.writers.spark:SparkDatasetWriter._write_bronze_iceberg",
        "plan.source_config.spark",
    ): "spark.write is declarative table sizing configuration passed as data.",
    (
        "janus.writers.spark:SparkDatasetWriter._merge_bronze_iceberg",
        "plan.source_config.spark",
    ): "spark.write is declarative table sizing configuration passed as data.",
}


@pytest.fixture(scope="module")
def package():
    paths = sorted(PACKAGE.rglob("*.py"))
    assert len(paths) >= BASELINE_MODULE_COUNT
    modules = {}
    scopes = []
    for path in paths:
        module = "janus." + ".".join(path.relative_to(PACKAGE).with_suffix("").parts)
        source = path.read_text()
        modules[path.relative_to(PACKAGE).as_posix()] = source
        scopes.extend(parse_scopes(source, module, include_toplevel=True))
    index = {scope.reference: scope for scope in scopes}
    assert scopes and index
    # Constructor assignments may instead call a factory with an annotated return type.
    returns = {
        reference.replace(":", "."): annotation_name(scope.node.returns, scope.aliases)
        for reference, scope in index.items()
        if isinstance(scope.node, ast.FunctionDef | ast.AsyncFunctionDef)
        and scope.node.returns is not None
    }
    free_returns = {
        reference.split(":")[1]: annotation_name(scope.node.returns, scope.aliases)
        for reference, scope in index.items()
        if "." not in reference.split(":")[1]
        and isinstance(scope.node, ast.FunctionDef | ast.AsyncFunctionDef)
        and scope.node.returns is not None
    }
    for scope in scopes:
        module = scope.reference.split(":")[0]
        for key, value in scope.bindings.items():
            scope.bindings[key] = returns.get(
                value,
                returns.get(f"{module}.{value}", free_returns.get(value.rsplit(".", 1)[-1], value)),
            )
    assert len(modules) == len(paths)
    return modules, scopes, index


def test_registry_and_primitives_resolve_without_importing_named_modules(package):
    _, _, index = package
    assert len(LOCKED_ENTRY_POINTS) >= MINIMUM_ENTRY_POINTS
    entries = {entry.qualified_name for entry in LOCKED_ENTRY_POINTS}
    assert len(entries) == len(LOCKED_ENTRY_POINTS)
    assert STATE_MUTATING_PRIMITIVES and REACHED_THROUGH
    assert len(set(STATE_MUTATING_PRIMITIVES)) == len(STATE_MUTATING_PRIMITIVES)
    for entry in LOCKED_ENTRY_POINTS:
        assert entry.qualified_name in index
        assert entry.command.strip() and entry.reason.strip()
    for reference in STATE_MUTATING_PRIMITIVES:
        assert reference in index, f"stale primitive: {reference}"
    for reference, (targets, reason) in REACHED_THROUGH.items():
        assert reference in index, reference
        assert reason.strip() and targets and set(targets) <= entries
        assert all(target in index for target in targets)
        scope = index[reference]
        assert any(
            not ambiguous for _, ambiguous in primitive_sites(scope, STATE_MUTATING_PRIMITIVES)
        ), f"stale REACHED_THROUGH entry: {reference}"


def test_registered_entry_points_acquire_before_and_through_mutation(package):
    _, scopes, index = package
    matched = {}
    for entry in LOCKED_ENTRY_POINTS:
        scope = index[entry.qualified_name]
        mutations = [
            node
            for nested in scopes
            if nested.reference == scope.reference or scope.reference in nested.ancestors
            for node, ambiguous in primitive_sites(nested, STATE_MUTATING_PRIMITIVES)
            if not ambiguous
        ]
        assert not acquisition_violation(scope, mutations, REACHED_THROUGH), entry.qualified_name
        matched[entry.qualified_name] = {form for _, _, form in acquisitions(scope)}
        # Pin the holder label along with the site, resolving imported command constants by AST.
        constants = {
            node.targets[0].id: node.value.value
            for node in ast.parse((PACKAGE / "runtime/locking.py").read_text()).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        labels = [
            constants.get(keyword.value.id)
            if isinstance(keyword.value, ast.Name)
            else keyword.value.value
            if isinstance(keyword.value, ast.Constant)
            else None
            for call in scope_nodes(scope.node)
            if isinstance(call, ast.Call)
            for keyword in call.keywords
            if keyword.arg == "command"
        ]
        if entry.command == "maintain":
            assert matched[entry.qualified_name] == {"ExitStack.enter_context"}
        else:
            assert entry.command in labels
    assert len(matched) >= MINIMUM_ENTRY_POINTS
    assert {form for forms in matched.values() for form in forms} == {
        "with",
        "ExitStack.enter_context",
    }


def test_package_primitive_callers_are_registered_or_reached_through(package):
    _, scopes, _ = package
    registered = {entry.qualified_name for entry in LOCKED_ENTRY_POINTS} | REACHED_THROUGH.keys()
    violations, sites, used = unregistered_mutations(
        scopes, STATE_MUTATING_PRIMITIVES, registered, NOT_A_PRIMITIVE
    )
    assert len(sites) >= MINIMUM_PRIMITIVE_SITES
    assert not violations, "\n".join(violations)
    assert used == NOT_A_PRIMITIVE.keys(), (
        f"stale NOT_A_PRIMITIVE entries: {NOT_A_PRIMITIVE.keys() - used}"
    )
    assert all(reason.strip() for reason in NOT_A_PRIMITIVE.values())


def test_reached_helpers_have_only_declared_locked_roots(package):
    _, scopes, index = package
    entries = {entry.qualified_name for entry in LOCKED_ENTRY_POINTS}
    roots = reached_roots(scopes, index, entries)
    assert REACHED_THROUGH
    for reference, (targets, _) in REACHED_THROUGH.items():
        assert roots(reference) == set(targets), (reference, roots(reference), targets)


def test_exactly_one_lock_implementation(package):
    modules, _, _ = package
    implementations = [name for name, source in modules.items() if second_lock(source)]
    assert implementations == ["runtime/locking.py"]


def test_lock_inode_is_never_replaced_or_deleted(package):
    modules, _, _ = package
    assert "runtime/locking.py" in modules
    assert not replaces_lock(modules["runtime/locking.py"])


def test_read_only_operator_entry_points_do_not_acquire_locks(package):
    _, _, index = package
    references = (
        "janus.cli.validate:validate_command",
        "janus.cli.list_sources:list_command",
        "janus.cli.checkpoint:_show",
        "janus.cli.dead_letters:_list",
        "janus.cli.dead_letters:_replay",
    )
    assert len(references) == 5
    for reference in references:
        scope = index[reference]
        assert not acquisitions(scope), f"read-only verb takes a lock: {reference}"
        assert not any(
            qualified(call.func, scope.aliases).rsplit(".", 1)[-1] in HELPERS
            for call in scope_nodes(scope.node)
            if isinstance(call, ast.Call)
        ), reference


@pytest.mark.parametrize("profile", ["local", "cluster"])
def test_default_lock_path_agrees_for_every_checked_in_source(profile, monkeypatch):
    import os

    from janus.lineage import MetadataZonePaths
    from janus.models import ExecutionPlan, RunContext
    from janus.registry import load_registry
    from janus.runtime.locking import source_state_lock_path
    from janus.utils.environment import load_environment_config
    from janus.utils.storage import StorageLayout
    from tests.support.parallel_baseline import STARTED_AT

    for key in tuple(os.environ):
        if key.startswith("JANUS_"):
            monkeypatch.delenv(key)
    sources = load_registry(REPO_ROOT).list_sources(enabled_only=False)
    assert len(sources) >= MINIMUM_SOURCES
    layout = StorageLayout.from_environment_config(
        load_environment_config(profile, REPO_ROOT), REPO_ROOT
    )
    for source in sources:
        context = RunContext.create("paths", profile, REPO_ROOT, started_at=STARTED_AT)
        plan = ExecutionPlan.from_source_config(source, context)
        assert (
            source_state_lock_path(plan, layout).parent
            == MetadataZonePaths.from_plan(plan).base_dir
        )


def forward(source):
    scopes = parse_scopes(source)
    # Include writes in nested lexical scopes in the entry point's coverage check.
    writes = [
        node for scope in scopes for node, _ in primitive_sites(scope, STATE_MUTATING_PRIMITIVES)
    ]
    return acquisition_violation(scopes[0], writes)


def backward(source, registered=()):
    return unregistered_mutations(parse_scopes(source), STATE_MUTATING_PRIMITIVES, set(registered))[
        0
    ]


@pytest.mark.parametrize(
    "bad,clean",
    [
        (
            "def f():\n CheckpointStore().reset(plan)\n with hold_source_state_lock(): pass",
            "def f():\n with hold_source_state_lock(): CheckpointStore().reset(plan)",
        ),
        (
            "def f():\n hold_source_state_lock()\n CheckpointStore().reset(plan)",
            "def f():\n"
            " stack.enter_context(source_lock(lock, source_id))\n"
            " CheckpointStore().reset(plan)",
        ),
        (
            "def f():\n with hold_source_state_lock(): pass\n CheckpointStore().reset(plan)",
            "def f():\n"
            " with hold_source_state_lock():\n"
            "  def nested(): CheckpointStore().reset(plan)\n"
            "  nested()",
        ),
        (
            "def f():\n"
            " def nested():\n"
            "  with hold_source_state_lock(): pass\n"
            " CheckpointStore().reset(plan)",
            "def f():\n with hold_source_state_lock(): CheckpointStore().reset(plan)",
        ),
        (
            "def f():\n"
            " stack.enter_context(source_lock(lock, source_id))\n"
            " stack.close()\n"
            " CheckpointStore().reset(plan)",
            "def f():\n"
            " stack.enter_context(source_lock(lock, source_id))\n"
            " CheckpointStore().reset(plan)\n"
            " stack.close()",
        ),
    ],
)
def test_forward_detector_flags_violation_and_accepts_clean_form(bad, clean):
    assert forward(bad)
    assert not forward(clean)


@pytest.mark.parametrize(
    "source",
    [
        "def f(): DeadLetterStore().release(plan)",
        "def f(store: CheckpointStore): store.save(plan)",
        "class F:\n checkpoint: CheckpointStore\n def f(self): self.checkpoint.reset(plan)",
        "def f():\n"
        " with hold_source_state_lock():\n"
        "  def nested(): DeadLetterStore().clear(plan)\n"
        "  nested()",
        "def f(): operation(CheckpointStore().clear_state, plan)",
        "def f():\n reset = CheckpointStore().reset\n reset(plan)",
        "from janus.lineage import RunObserver as O\n"
        "def f(observer): O.record_success(observer, plan)",
        "from janus.checkpoints import DeadLetterStore as Store\ndef f(): Store().record(plan)",
    ],
)
def test_backward_detector_flags_new_caller_and_accepts_registered_caller(source):
    assert backward(source)
    registered = {scope.reference for scope in parse_scopes(source) if not scope.ancestors}
    assert not backward(source, registered)


def test_ambiguous_receiver_fails_until_explicitly_explained():
    scopes = parse_scopes("def f(): unknown.save(plan)")
    registered = {"snippet:f"}
    assert unregistered_mutations(scopes, STATE_MUTATING_PRIMITIVES, registered)[0]
    unresolved_name = parse_scopes("def f(): reset(plan)")
    assert unregistered_mutations(unresolved_name, STATE_MUTATING_PRIMITIVES, registered)[0]
    exclusions = {("snippet:f", "unknown"): "The receiver is an unrelated dataset writer."}
    assert not unregistered_mutations(scopes, STATE_MUTATING_PRIMITIVES, registered, exclusions)[0]


@pytest.mark.parametrize(
    "bad,clean",
    [
        ("import fcntl as f\nf.flock(fd, 2)", "hold_source_state_lock(plan, layout)"),
        ("from fcntl import lockf as lock\nlock(fd, 2)", "os.fsync(fd)"),
        (
            "from fcntl import F_SETLKW as FLAG\nfcntl.fcntl(fd, FLAG)",
            "fcntl.fcntl(fd, fcntl.F_GETFD)",
        ),
        ("import fcntl\nfcntl.fcntl(fd, fcntl.F_OFD_SETLK)", "os.fsync(fd)"),
        (
            "from os import open as acquire, O_EXCL as FLAG\nacquire(path, FLAG)",
            "os.open(path, os.O_CREAT)",
        ),
        ("path = base / 'worker.pid'", "path = base / 'current.json'"),
        ("path = base / '.lock'", "path = base / 'current.json'"),
    ],
)
def test_second_lock_detector_flags_violation_and_accepts_clean_form(bad, clean):
    assert second_lock(bad)
    assert not second_lock(clean)


@pytest.mark.parametrize(
    "bad,clean",
    [
        ("os.replace(tmp, lock_path)", "os.pwrite(fd, record, 0)"),
        ("lock_path.unlink()", "os.ftruncate(fd, 0)"),
        ("os.rename(tmp, lock_path)", "from dataclasses import replace\nreplace(holder, pid=2)"),
        ("write_json_atomic(lock_path, record)", "os.pwrite(fd, record, 0)"),
        ("from os import remove as delete\ndelete(lock_path)", "os.fsync(fd)"),
        ("Path(lock_path).replace(tmp)", "os.ftruncate(fd, 0)"),
        ("Path(lock_path).rename(tmp)", "os.fsync(fd)"),
    ],
)
def test_inode_detector_flags_violation_and_accepts_clean_form(bad, clean):
    assert replaces_lock(bad)
    assert not replaces_lock(clean)


def test_module_level_mutations_are_never_silently_ignored():
    source = "CheckpointStore().reset(plan)\nclass F:\n DeadLetterStore().clear(plan)"
    scopes = parse_scopes(source, include_toplevel=True)
    violations, sites, _ = unregistered_mutations(scopes, STATE_MUTATING_PRIMITIVES, set())
    assert len(violations) == len(sites) == 2


def test_reachability_detector_flags_a_new_unlocked_helper_caller():
    source = """
    def entry():
        with hold_source_state_lock(): helper()
    def helper():
        CheckpointStore().reset(plan)
    """
    scopes = parse_scopes(source)
    index = {scope.reference: scope for scope in scopes}
    entries = {"snippet:entry"}
    assert reached_roots(scopes, index, entries)("snippet:helper") == entries
    scopes += parse_scopes("def new_command(): helper()")
    index = {scope.reference: scope for scope in scopes}
    assert reached_roots(scopes, index, entries)("snippet:helper") == entries | {
        "snippet:new_command"
    }
