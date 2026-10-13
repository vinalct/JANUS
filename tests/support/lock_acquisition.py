"""AST-only detectors shared by the source-lock sweep and its counterexamples."""

from __future__ import annotations

import ast
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field

HELPERS = {"hold_source_state_lock", "hold_operator_source_lock", "source_lock"}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def names(tree, module=""):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                {item.asname or item.name.split(".")[0]: item.name for item in node.names}
            )
        elif isinstance(node, ast.ImportFrom):
            origin = node.module or ""
            if node.level:
                parent = module.split(".")[: -node.level]
                origin = ".".join((*parent, origin))
            aliases.update(
                {item.asname or item.name: f"{origin}.{item.name}" for item in node.names}
            )
    return aliases


def qualified(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{qualified(node.value, aliases)}.{node.attr}"
    if isinstance(node, ast.Call):
        return qualified(node.func, aliases)
    return ""


def scope_nodes(node):
    """Do not attribute a nested function's writes to its parent function."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (*FUNCTIONS, ast.ClassDef)):
            yield from scope_nodes(child)


@dataclass
class Scope:
    reference: str
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Module | ast.ClassDef
    aliases: dict[str, str]
    bindings: dict[str, str] = field(default_factory=dict)
    ancestors: tuple[str, ...] = ()


def annotation_name(node, aliases):
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node = ast.parse(node.value, mode="eval").body
    if isinstance(node, ast.BinOp):
        return annotation_name(node.left, aliases)
    return qualified(node, aliases)


def definitions(tree):
    """Find definitions through control-flow bodies, stopping at each lexical scope."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (*FUNCTIONS, ast.ClassDef)):
            yield node
        else:
            yield from definitions(node)


def scopes(tree, module="snippet", prefix="", inherited=None, ancestors=(), aliases=None):
    aliases = {**(aliases or {}), **names(tree, module)}
    inherited = dict(inherited or {})
    for node in definitions(tree):
        if isinstance(node, ast.ClassDef):
            fields = {
                f"self.{item.target.id}": annotation_name(item.annotation, aliases)
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
            yield from scopes(
                node,
                module,
                f"{prefix}{node.name}.",
                {**inherited, **fields, "self": node.name},
                ancestors,
                aliases,
            )
        elif isinstance(node, FUNCTIONS):
            reference = f"{module}:{prefix}{node.name}"
            bindings = dict(inherited)
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if argument.annotation:
                    bindings[argument.arg] = annotation_name(argument.annotation, aliases)
            for item in scope_nodes(node):
                if isinstance(item, ast.AnnAssign):
                    bindings[qualified(item.target, {})] = annotation_name(item.annotation, aliases)
                elif isinstance(item, ast.Assign) and isinstance(
                    item.value, ast.Call | ast.Attribute
                ):
                    for target in item.targets:
                        bindings.setdefault(
                            qualified(target, {}),
                            qualified(
                                item.value.func if isinstance(item.value, ast.Call) else item.value,
                                aliases,
                            ),
                        )
            yield Scope(reference, node, aliases, bindings, ancestors)
            yield from scopes(
                node, module, f"{prefix}{node.name}.", bindings, (*ancestors, reference), aliases
            )


def parse_scopes(source, module="snippet", *, include_toplevel=False):
    tree = ast.parse(textwrap.dedent(source))
    functions = list(scopes(tree, module))
    if include_toplevel:
        aliases = names(tree, module)
        functions.append(Scope(f"{module}:<module>", tree, aliases))
        functions.extend(
            Scope(f"{module}:<class@{node.lineno}>", node, aliases)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        )
    return functions


def primitive_sites(scope, primitives):
    """Also detect a mutator passed as a callback, as the operator verbs do."""
    methods = {}
    free = set()
    for reference in primitives:
        _, path = reference.split(":")
        if "." in path:
            owner, method = path.rsplit(".", 1)
            methods.setdefault(method, set()).add(owner)
        else:
            free.add(path)
    candidates = []
    for call in scope_nodes(scope.node):
        if isinstance(call, ast.Assign | ast.AnnAssign) and isinstance(call.value, ast.Attribute):
            candidates.append(call.value)
        elif isinstance(call, ast.Call):
            candidates.append(call.func)
            candidates.extend(
                node
                for node in (*call.args, *(kw.value for kw in call.keywords))
                if isinstance(node, ast.Attribute)
            )
    sites = []
    for candidate in candidates:
        name = qualified(candidate, scope.aliases).rsplit(".", 1)[-1]
        if name in free:
            sites.append((candidate, ""))
        elif isinstance(candidate, ast.Attribute) and name in methods:
            receiver = qualified(candidate.value, {})
            owner = (
                qualified(candidate.value.func, scope.aliases)
                if isinstance(candidate.value, ast.Call)
                else scope.bindings.get(receiver, qualified(candidate.value, scope.aliases))
            ).rsplit(".", 1)[-1]
            if owner in methods[name]:
                sites.append((candidate, ""))
            else:
                sites.append((candidate, receiver))
        elif isinstance(candidate, ast.Name) and name in methods:
            binding = scope.bindings.get(candidate.id, "").split(".")
            resolved = len(binding) >= 2 and binding[-2] in methods[name] and binding[-1] == name
            sites.append((candidate, "" if resolved else candidate.id))
    return sites


def acquisitions(scope):
    """Return acquisition node, protected nodes, and the matched syntactic form."""
    matches = []
    nodes = list(scope_nodes(scope.node))
    for node in nodes:
        if isinstance(node, ast.With) and any(
            isinstance(item.context_expr, ast.Call)
            and qualified(item.context_expr.func, scope.aliases).rsplit(".", 1)[-1] in HELPERS
            for item in node.items
        ):
            protected = {id(child) for stmt in node.body for child in ast.walk(stmt)}
            matches.append((node, protected, "with"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "enter_context"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and qualified(node.args[0].func, scope.aliases).rsplit(".", 1)[-1] in HELPERS
        ):
            # _run retains this ExitStack until its finally; do not accept calls after close().
            stack = qualified(node.func.value, {})
            closes = [
                call.lineno
                for call in nodes
                if isinstance(call, ast.Call) and qualified(call.func, {}) == f"{stack}.close"
            ]
            end = min(closes, default=scope.node.end_lineno + 1)
            protected = {
                id(child)
                for child in ast.walk(scope.node)
                if node.lineno < getattr(child, "lineno", 0) < end
            }
            matches.append((node, protected, "ExitStack.enter_context"))
    return matches


def acquisition_violation(scope, mutations, reached=()):
    matches = acquisitions(scope)
    targets = {name.split(":")[-1].rsplit(".", 1)[-1] for name in reached}
    calls = [
        node
        for node in ast.walk(scope.node)
        if isinstance(node, ast.Call)
        and qualified(node.func, scope.aliases).rsplit(".", 1)[-1] in targets
    ]
    writes = [*mutations, *calls]
    if not matches:
        return [f"{scope.reference}: no entered source lock"]
    return [
        f"{scope.reference}:{write.lineno}: mutation outside acquired lock"
        for write in writes
        if not any(
            (acquire.lineno, acquire.col_offset) < (write.lineno, write.col_offset)
            and id(write) in protected
            for acquire, protected, _ in matches
        )
    ]


def unregistered_mutations(scopes_, primitives, registered, not_a_primitive=None):
    exclusions = not_a_primitive or {}
    violations, sites, used = [], [], set()
    for scope in scopes_:
        for node, ambiguous in primitive_sites(scope, primitives):
            key = (scope.reference, ambiguous)
            if ambiguous and key in exclusions:
                used.add(key)
                continue
            sites.append((scope.reference, node.lineno))
            if ambiguous:
                violations.append(
                    f"{scope.reference}:{node.lineno}: unresolved primitive receiver {ambiguous}"
                )
            elif scope.reference not in registered and not set(scope.ancestors) & registered:
                violations.append(f"{scope.reference}:{node.lineno}: unregistered mutation")
    return violations, sites, used


def second_lock(source):
    tree = ast.parse(textwrap.dedent(source))
    aliases = names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = qualified(node.func, aliases)
            if callee in {"fcntl.flock", "fcntl.lockf"}:
                return True
            if callee == "fcntl.fcntl" and any(
                qualified(part, aliases).startswith(("fcntl.F_SETLK", "fcntl.F_OFD_"))
                for argument in (*node.args, *(kw.value for kw in node.keywords))
                for part in ast.walk(argument)
            ):
                return True
            if callee in {"os.open", "open"} and any(
                qualified(part, aliases) == "os.O_EXCL"
                for argument in (*node.args, *(kw.value for kw in node.keywords))
                for part in ast.walk(argument)
            ):
                return True
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith((".lock", ".pid"))
        ):
            return True
    return False


def replaces_lock(source):
    tree = ast.parse(textwrap.dedent(source))
    aliases = names(tree)
    return any(
        qualified(node.func, aliases) in {"os.replace", "os.rename", "os.remove", "os.unlink"}
        or qualified(node.func, aliases).rsplit(".", 1)[-1] in {"unlink", "write_json_atomic"}
        or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"replace", "rename"}
            and qualified(node.func, aliases) != "dataclasses.replace"
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def call_targets(scope, candidate, by_name):
    """Resolve imports, local functions, constructed/typed receivers and strategy dispatch."""
    name = qualified(candidate, scope.aliases)
    exact = name.replace(":", ".")
    local = f"{scope.reference.split(':')[0]}:{name}"
    direct = [
        ref
        for ref in by_name[name.rsplit(".", 1)[-1]]
        if ref.replace(":", ".") == exact or ref == local
    ]
    if direct or not isinstance(candidate, ast.Attribute):
        return direct
    receiver = qualified(candidate.value, {})
    owner = (
        qualified(candidate.value.func, scope.aliases)
        if isinstance(candidate.value, ast.Call)
        else scope.bindings.get(receiver, "")
    ).rsplit(".", 1)[-1]
    targets = [
        ref for ref in by_name[candidate.attr] if ref.split(":")[1] == f"{owner}.{candidate.attr}"
    ]
    if not targets and candidate.attr == "extract":
        # Planner binds one of these strategies; execution invokes the common interface.
        targets = [ref for ref in by_name["extract"] if "." in ref.split(":")[1]]
    return targets


def reached_roots(scopes_, index, entry_points):
    """Trace package callers back to locked boundaries; an unguarded root stays visible."""
    reverse = defaultdict(set)
    by_name = defaultdict(list)
    for reference in index:
        by_name[reference.split(":")[1].rsplit(".", 1)[-1]].append(reference)
    for scope in scopes_:
        for call in scope_nodes(scope.node):
            if not isinstance(call, ast.Call):
                continue
            candidates = [call.func, *(arg for arg in call.args if isinstance(arg, ast.Attribute))]
            for candidate in candidates:
                for target in call_targets(scope, candidate, by_name):
                    reverse[target].add(scope.reference)
        if scope.ancestors:
            reverse[scope.reference].add(scope.ancestors[-1])

    def roots(reference, visited=frozenset()):
        if reference in entry_points or not reverse[reference]:
            return {reference}
        if reference in visited:
            return set()
        return set().union(*(roots(caller, visited | {reference}) for caller in reverse[reference]))

    return roots
