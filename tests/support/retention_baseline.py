from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELETION_METHODS = frozenset({"unlink", "remove", "rmtree", "rmdir"})
TIMESTAMP_FIELDS = {
    "runs": "started_at",
    "lineage": "emitted_at",
    "checkpoints/history": "recorded_at",
    "checkpoints": "updated_at",
    "dead_letters/history": "released_at",
    "dead_letters": "updated_at",
    "validations": "emitted_at",
    "pipelines": "pipeline.started_at",
    "progress": "updated_at",
    "events": "eventTime",
}
KNOWN_DIRECTORIES = frozenset({
    "ivy", "jars", "cache", "iceberg-catalog", "spark-warehouse", "test-reports",
    "logs", "pipelines", "baseline", "lineage", "openlineage",
})


def file_digest(path: Path) -> str:
    """Stream large warehouse files rather than loading them into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def filesystem_state(root: Path) -> dict[str, object]:
    """SHA-256 of sorted UTF-8 ``relpath sha256\\n`` lines; no mtime or exclusions."""

    if not root.is_dir():
        raise FileNotFoundError("Before-state root must be an existing directory")
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    count = size = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError("Before-state root contains a symlink")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            digest.update(f"{relative} {file_digest(path)}\n".encode())
            count += 1
            size += path.stat().st_size
    return {"files": count, "bytes": size, "sha256": digest.hexdigest()}


class DeletionCalls(ast.NodeVisitor):
    """Conservative AST call inventory, including imported deletion aliases."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.aliases: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"os", "shutil"}:
            for name in node.names:
                if name.name in DELETION_METHODS:
                    self.aliases[name.asname or name.name] = f"{node.module}.{name.name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def _visit_scope(self, node) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        attribute_call = isinstance(function, ast.Attribute) and function.attr in DELETION_METHODS
        alias_call = isinstance(function, ast.Name) and function.id in self.aliases
        if attribute_call or alias_call:
            self.calls.append({
                "line": node.lineno,
                "scope": ".".join(self.scope),
                "call": ast.unparse(node),
            })
        self.generic_visit(node)


def deletion_inventory(root: Path) -> dict[str, object]:
    modules = sorted((root / "src").rglob("*.py"))
    if not modules:
        raise ValueError("Deletion inventory matched no source modules")
    calls = []
    for path in modules:
        visitor = DeletionCalls()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        calls.extend({"path": path.relative_to(root).as_posix(), **call} for call in visitor.calls)
    return {"modules_parsed": len(modules), "calls": calls}


def _shape(path: Path) -> tuple[str, str | None]:
    parts = path.parts
    name = path.name
    if name.endswith(".tmp"):
        return "<source>/.<name>.<hex>.tmp" if len(parts) == 1 else (
            "/".join("<segment>" for _ in parts[:-1]) + "/.<name>.<hex>.tmp"
        ), None
    if name == "extraction_progress.json":
        return "<source>/extraction_progress.json", "progress"
    if name.startswith("events-") and name.endswith(".ndjson"):
        return "<source>/lineage/openlineage/events-*.ndjson", "events"
    if len(parts) >= 3 and parts[0] == "pipelines" and name == "summary.json":
        return "pipelines/<id>/summary.json", "pipelines"
    for family in ("checkpoints/history", "dead_letters/history", "runs", "lineage",
                   "checkpoints", "dead_letters", "validations"):
        tail = tuple(family.split("/"))
        if parts[-len(tail) - 1:-1] == tail:
            filename = "current.json" if name == "current.json" else f"*{path.suffix}"
            return f"<source>/{family}/{filename}", family
    directories = [part if part in KNOWN_DIRECTORIES else "<segment>" for part in parts[:-1]]
    suffix = path.suffix if path.suffix[1:].isalnum() else ""
    return "/".join([*directories, f"*{suffix}"]), None


def _timestamp_status(payload: object, field: str) -> str:
    value = payload
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return "missing"
        value = value[key]
    if value is None:
        return "null"
    try:
        timestamp = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return "invalid"
    if timestamp is None:
        return "invalid"
    return "aware" if timestamp.utcoffset() is not None else "naive"


def census(root: Path) -> dict[str, object]:
    shapes: dict[str, Counter] = {}
    progress = Counter()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Census root contains a symlink")
        if not path.is_file():
            continue
        shape, kind = _shape(path.relative_to(root))
        counts = shapes.setdefault(shape, Counter())
        counts["files"] += 1
        counts["bytes"] += path.stat().st_size
        field = TIMESTAMP_FIELDS.get(kind)
        if kind == "events":
            counts.update(_event_timestamps(path))
            continue
        if field is None:
            if path.suffix == ".json":
                counts.update(_other_timestamps(path))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            counts["unreadable"] += 1
            continue
        counts[_timestamp_status(payload, field)] += 1
        if kind == "progress" and isinstance(payload, dict):
            prefix = payload.get("raw_path_prefix")
            progress["with_raw_path_prefix" if isinstance(prefix, str) and prefix.strip()
                     else "without_raw_path_prefix"] += 1
    return {
        "state": filesystem_state(root),
        "shapes": [{"shape": shape, "timestamp_field": TIMESTAMP_FIELDS.get(_shape_kind(shape)),
                    **dict(counts)} for shape, counts in sorted(shapes.items())],
        "progress": dict(progress),
    }


def _other_timestamps(path: Path) -> Counter:
    counts = Counter()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        counts["unreadable"] += 1
        return counts
    for field in dict.fromkeys(TIMESTAMP_FIELDS.values()):
        status = _timestamp_status(payload, field)
        if status != "missing":
            counts[f"observed:{field}:{status}"] += 1
    if isinstance(payload, dict) and "last-updated-ms" in payload:
        counts["observed:last-updated-ms:epoch-ms"] += 1
    return counts


def _event_timestamps(path: Path) -> Counter:
    counts = Counter()
    try:
        date.fromisoformat(path.stem.removeprefix("events-"))
        counts["dated_files"] += 1
    except ValueError:
        counts["undated_files"] += 1
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except ValueError:
                counts["unreadable_events"] += 1
                continue
            counts[f"events_{_timestamp_status(payload, 'eventTime')}"] += 1
    return counts


def _shape_kind(shape: str) -> str | None:
    if "extraction_progress" in shape:
        return "progress"
    if shape.startswith("pipelines/"):
        return "pipelines"
    if "/openlineage/events-" in shape:
        return "events"
    for kind in TIMESTAMP_FIELDS:
        if f"/{kind}/" in shape:
            return kind
    return None


def catalog_snapshots(database: Path) -> list[dict[str, object]]:
    """Read snapshot counts through local catalog metadata, without an engine."""

    with TemporaryDirectory(prefix="janus-catalog-census-") as temporary:
        copy = Path(temporary) / database.name
        shutil.copyfile(database, copy)
        for suffix in ("-wal", "-shm"):
            sidecar = database.with_name(database.name + suffix)
            if sidecar.exists():
                shutil.copyfile(sidecar, copy.with_name(copy.name + suffix))
        connection = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
        try:
            tables = connection.execute(
                "SELECT table_namespace, table_name, metadata_location FROM iceberg_tables "
                "ORDER BY table_namespace, table_name"
            ).fetchall()
        finally:
            connection.close()
    result = []
    for namespace, name, location in tables:
        local = location.removeprefix("file:")
        local = local.replace("/workspace/", f"{PROJECT_ROOT}/")
        local = local.replace("/evidence/", f"{PROJECT_ROOT}/data/baseline/")
        path = Path(local)
        if not path.is_file():
            result.append({"table": f"{namespace}.{name}", "status": "unresolved-location"})
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        result.append({"table": f"{namespace}.{name}",
                       "snapshots": len(metadata.get("snapshots", [])),
                       "current_snapshot_id": metadata.get("current-snapshot-id")})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "census", "digest", "snapshots"))
    parser.add_argument("root", type=Path, nargs="?", default=PROJECT_ROOT)
    args = parser.parse_args()
    operation = {"audit": deletion_inventory, "census": census, "digest": filesystem_state,
                 "snapshots": catalog_snapshots}
    print(json.dumps(operation[args.mode](args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
