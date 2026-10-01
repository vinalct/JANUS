

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from janus.models.config.issues import SourceConfigValidationError, ValidationIssue
from janus.models.data_contracts import (
    ContractValidationError,
    DataContract,
    load_data_contract,
)
from janus.models.source_config import SourceConfig
from janus.schema_contracts import resolve_declared_path

#: Field path reported when a declared contract cannot be loaded.
CONTRACT_FIELD_PATH = "schema.contract"

ContractValidation = Callable[[SourceConfig, DataContract | None], list[ValidationIssue]]


def load_contract_snapshot(
    sources: Iterable[SourceConfig],
    *,
    project_root: Path,
    sources_dir: Path,
    validate_contract: ContractValidation | None = None,
) -> dict[str, DataContract]:
    """Return every declared contract or raise with all load failures."""
    reader = _ContractReader(project_root=project_root)
    contracts: dict[str, DataContract] = {}
    failures: list[tuple[Path, ValidationIssue]] = []

    for source in sources:
        contract: DataContract | None = None
        try:
            contract = reader.contract_for(source)
        except (ContractValidationError, OSError) as exc:
            failures.append((source.config_path, _failure_issue(source, exc)))
        if contract is not None:
            contracts[source.source_id] = contract
        if validate_contract is not None:
            failures.extend(
                (source.config_path, issue)
                for issue in validate_contract(source, contract)
            )

    if failures:
        raise _collected_error(failures, sources_dir)
    return contracts


@dataclass(slots=True)
class _ContractReader:
    """One load's file cache, so each declared file is read exactly once per snapshot."""

    project_root: Path
    _declared: dict[Path, DataContract] = field(default_factory=dict)

    def contract_for(self, source: SourceConfig) -> DataContract | None:
        """Resolve and read the declared contract for this source."""
        path = self._resolve(source, source.schema.contract)
        return None if path is None else self._declared_contract(path)

    def _resolve(self, source: SourceConfig, configured: str) -> Path | None:
        """Run the one declared-path search, the same four steps a plan resolves with."""
        return resolve_declared_path(self.project_root, source.config_path, configured)

    def _declared_contract(self, path: Path) -> DataContract:
        """Load one contract file, or hand back the object an earlier entry loaded."""
        if path not in self._declared:
            self._declared[path] = load_data_contract(path)
        return self._declared[path]


def _failure_issue(source: SourceConfig, exc: Exception) -> ValidationIssue:
    """Report an unloadable contract as the *source's* problem, under its own field."""
    declared = source.schema.contract
    return ValidationIssue(
        f"{source.source_id}.{CONTRACT_FIELD_PATH}",
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
    "load_contract_snapshot",
]
