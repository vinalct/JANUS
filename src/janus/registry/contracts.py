

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from janus.models.config.issues import SourceConfigValidationError, ValidationIssue
from janus.models.data_contracts import (
    ContractValidationError,
    DataContract,
    contract_from_legacy_schema_bytes,
    load_data_contract,
)
from janus.models.source_config import SourceConfig
from janus.schema_contracts import resolve_declared_path
from janus.utils.storage import bronze_table_identifier

#: Field path reported when a declared contract cannot be loaded.
CONTRACT_FIELD_PATH = "schema.contract"

#: Field path reported when a legacy schema file cannot be converted.
LEGACY_FIELD_PATH = "schema.path"


def load_contract_snapshot(
    sources: Iterable[SourceConfig],
    *,
    project_root: Path,
    sources_dir: Path,
) -> dict[str, DataContract]:
    """Return the contract every source declared, or raise listing each that failed.

    A source declaring neither a contract nor a legacy file contributes no entry rather
    than a placeholder: ``schema.mode: infer`` is still a supported declaration this
    order, and a synthesised stand-in would be indistinguishable downstream from a
    contract somebody actually wrote.
    """
    reader = _ContractReader(project_root=project_root)
    contracts: dict[str, DataContract] = {}
    failures: list[tuple[Path, ValidationIssue]] = []

    for source in sources:
        try:
            contract = reader.contract_for(source)
        except (ContractValidationError, OSError) as exc:
            failures.append((source.config_path, _failure_issue(source, exc)))
            continue
        if contract is not None:
            contracts[source.source_id] = contract

    if failures:
        raise _collected_error(failures, sources_dir)
    return contracts


@dataclass(slots=True)
class _ContractReader:
    """One load's file cache, so each declared file is read exactly once per snapshot."""

    project_root: Path
    _declared: dict[Path, DataContract] = field(default_factory=dict)
    _legacy_bytes: dict[Path, bytes] = field(default_factory=dict)

    def contract_for(self, source: SourceConfig) -> DataContract | None:
        """Resolve and read whatever schema declaration this source carries."""
        schema = source.schema
        if schema.declares_contract:
            path = self._resolve(source, schema.contract)
            return None if path is None else self._declared_contract(path)
        if schema.declares_legacy_file:
            path = self._resolve(source, schema.path)
            return None if path is None else self._legacy_contract(path, source)
        return None

    def _resolve(self, source: SourceConfig, configured: str | None) -> Path | None:
        """Run the one declared-path search, the same four steps a plan resolves with."""
        return resolve_declared_path(self.project_root, source.config_path, configured)

    def _declared_contract(self, path: Path) -> DataContract:
        """Load one contract file, or hand back the object an earlier entry loaded."""
        if path not in self._declared:
            self._declared[path] = load_data_contract(path)
        return self._declared[path]

    def _legacy_contract(self, path: Path, source: SourceConfig) -> DataContract:
        """Convert a legacy file for this entry, reading its bytes at most once."""
        if path not in self._legacy_bytes:
            self._legacy_bytes[path] = path.read_bytes()
        return contract_from_legacy_schema_bytes(
            self._legacy_bytes[path],
            path,
            source_id=source.source_id,
            bronze_table=_bronze_table(source),
            domain=source.domain,
            project_root=self.project_root,
        )


def _bronze_table(source: SourceConfig) -> str:
    """Name the table a legacy schema describes with the writer's own identity."""
    bronze = source.outputs.bronze
    return bronze_table_identifier(
        bronze.path,
        fallback_name=source.source_id,
        namespace=bronze.namespace,
        table_name=bronze.table_name,
    )


def _failure_issue(source: SourceConfig, exc: Exception) -> ValidationIssue:
    """Report an unloadable contract as the *source's* problem, under its own field."""
    field_path = (
        CONTRACT_FIELD_PATH if source.schema.declares_contract else LEGACY_FIELD_PATH
    )
    declared = (
        source.schema.contract if source.schema.declares_contract else source.schema.path
    )
    return ValidationIssue(
        f"{source.source_id}.{field_path}",
        f"could not load {declared}: {_describe(exc)}",
    )


def _describe(exc: Exception) -> str:
    """Render one failure as a single line, keeping every collected contract issue."""
    if isinstance(exc, ContractValidationError):
        rendered = "; ".join(issue.render() for issue in exc.issues)
        return f"{exc.contract_path}: {rendered}"
    if isinstance(exc, FileNotFoundError):
        return f"{exc.filename}: the file does not exist"
    if isinstance(exc, OSError):
        return f"{exc.filename}: {exc.strerror or exc}"
    return str(exc)


def _collected_error(
    failures: list[tuple[Path, ValidationIssue]], sources_dir: Path
) -> SourceConfigValidationError:
    """Raise once for the whole load, naming every source whose contract failed.

    One failing file is reported against that file, the way a single config's issues are.
    Several are reported against the registry directory, because no single config file is
    the thing an operator has to go and fix.
    """
    config_paths = {config_path for config_path, _ in failures}
    location = next(iter(config_paths)) if len(config_paths) == 1 else sources_dir
    return SourceConfigValidationError(location, [issue for _, issue in failures])


__all__ = [
    "CONTRACT_FIELD_PATH",
    "LEGACY_FIELD_PATH",
    "load_contract_snapshot",
]
