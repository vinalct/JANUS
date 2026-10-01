from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import yaml

from janus.models.data_contracts import DataContract
from janus.models.dependencies import SourceDependencyGraph
from janus.models.source_config import (
    DEFAULT_VALIDATION_POLICY,
    STRATEGY_REGISTRY,
    SourceConfig,
    SourceConfigValidationError,
    StrategyRegistry,
    ValidationIssue,
    ValidationPolicy,
)
from janus.registry.contracts import load_contract_snapshot
from janus.registry.dependencies import (
    SourceLocation,
    build_source_dependency_graph,
)


class AppConfigValidationError(ValueError):
    def __init__(self, config_path: Path, issues: list[ValidationIssue]) -> None:
        """Build a readable validation error for the top-level app config file."""
        self.config_path = config_path
        self.issues = tuple(issues)
        message_lines = [f"Invalid app config: {config_path}"]
        message_lines.extend(f"- {issue.render()}" for issue in self.issues)
        super().__init__("\n".join(message_lines))


class SourceNotFoundError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class RegistrySettings:
    sources_dir: Path
    file_pattern: str = "*.yaml"

    def resolve_sources_dir(self, project_root: Path) -> Path:
        """Resolve the configured source directory relative to the project root."""
        if self.sources_dir.is_absolute():
            return self.sources_dir
        return project_root / self.sources_dir


@dataclass(frozen=True, slots=True)
class AppConfig:
    config_path: Path
    registry: RegistrySettings


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """Every configured source, plus the validated graph they form."""

    project_root: Path
    app_config: AppConfig
    sources: tuple[SourceConfig, ...]
    locations: tuple[SourceLocation, ...] = ()
    contracts: Mapping[str, DataContract] = field(default_factory=dict)
    _sources_by_id: dict[str, SourceConfig] = field(init=False, repr=False)
    graph: SourceDependencyGraph = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Index the sources by id, then resolve and validate their dependency graph."""
        object.__setattr__(
            self,
            "_sources_by_id",
            {source.source_id: source for source in self.sources},
        )
        object.__setattr__(self, "contracts", MappingProxyType(dict(self.contracts)))
        object.__setattr__(
            self,
            "graph",
            build_source_dependency_graph(
                self.sources,
                locations=self.locations,
                sources_dir=self.app_config.registry.resolve_sources_dir(self.project_root),
            ),
        )

    @classmethod
    def load(
        cls,
        project_root: Path,
        *,
        policy: ValidationPolicy = DEFAULT_VALIDATION_POLICY,
        strategy_registry: StrategyRegistry = STRATEGY_REGISTRY,
    ) -> Self:
        """Load app settings, discover sources, and return one validated registry snapshot.

        ``policy`` and ``strategy_registry`` are load-time inputs, forwarded untouched to
        every ``SourceConfig.from_mapping`` call. The policy also checks each loaded contract
        after the snapshot is read. Neither input is stored on the returned registry: which
        policy validated a load is lineage, not registry state, and a field would change
        ``__eq__`` and ``repr`` for every consumer to record something nobody reads afterwards.

        Graph validation happens last, in ``__post_init__``: after every individual config
        is typed and after duplicate ids are rejected, because a graph over configs that do
        not parse would report the same problem twice under a worse name.
        """
        resolved_project_root = project_root.resolve()
        app_config = load_app_config(resolved_project_root)
        sources_dir = app_config.registry.resolve_sources_dir(resolved_project_root)
        if not sources_dir.exists():
            raise FileNotFoundError(f"Configured source directory does not exist: {sources_dir}")

        sources: list[SourceConfig] = []
        locations: list[SourceLocation] = []
        seen_source_ids: dict[str, Path] = {}
        for config_path in _discover_source_config_paths(
            sources_dir,
            app_config.registry.file_pattern,
        ):
            loaded = _load_source_configs(
                config_path,
                policy=policy,
                strategy_registry=strategy_registry,
            )
            for source, entry in loaded:
                previous_path = seen_source_ids.get(source.source_id)
                if previous_path is not None:
                    raise ValueError(
                        "Duplicate source_id "
                        f"{source.source_id!r} found in {previous_path} and {config_path}"
                    )
                seen_source_ids[source.source_id] = config_path
                sources.append(source)
                locations.append(
                    SourceLocation(
                        source_id=source.source_id,
                        config_path=config_path,
                        entry=entry,
                    )
                )

        return cls(
            project_root=resolved_project_root,
            app_config=app_config,
            sources=tuple(sources),
            locations=tuple(locations),
            contracts=_load_declared_contracts(
                sources,
                locations=locations,
                project_root=resolved_project_root,
                sources_dir=sources_dir,
            ),
        )

    def list_sources(self, *, enabled_only: bool = True) -> tuple[SourceConfig, ...]:
        """Return the registered sources, filtering disabled ones by default."""
        if not enabled_only:
            return self.sources
        return tuple(source for source in self.sources if source.enabled)

    def contract_for(self, source_id: str) -> DataContract | None:
        return self.contracts.get(source_id)

    def get_source(
        self, source_id: str, *, include_disabled: bool = False
    ) -> SourceConfig:
        """Return one source config by id and guard callers from disabled entries."""
        source = self._sources_by_id.get(source_id)
        if source is None:
            raise SourceNotFoundError(f"Source {source_id!r} was not found in the registry")
        if not include_disabled and not source.enabled:
            raise SourceNotFoundError(f"Source {source_id!r} is configured but disabled")
        return source


def load_app_config(project_root: Path) -> AppConfig:
    """Load and validate the small app config that points JANUS at the source registry."""
    config_path = project_root / "conf" / "app.yaml"
    raw = _load_yaml_mapping(config_path)
    issues: list[ValidationIssue] = []

    registry_data = raw.get("registry")
    if registry_data is None:
        issues.append(ValidationIssue("registry", "is required"))
        raise AppConfigValidationError(config_path, issues)
    if not isinstance(registry_data, Mapping):
        issues.append(ValidationIssue("registry", "must be a mapping"))
        raise AppConfigValidationError(config_path, issues)

    sources_dir = _require_string(registry_data, "sources_dir", issues, "registry")
    file_pattern = _optional_string(
        registry_data, "file_pattern", issues, "registry", default="*.yaml"
    )

    if issues:
        raise AppConfigValidationError(config_path, issues)

    return AppConfig(
        config_path=config_path,
        registry=RegistrySettings(
            sources_dir=Path(sources_dir),
            file_pattern=file_pattern,
        ),
    )


def load_registry(
    project_root: Path,
    *,
    policy: ValidationPolicy = DEFAULT_VALIDATION_POLICY,
    strategy_registry: StrategyRegistry = STRATEGY_REGISTRY,
) -> SourceRegistry:
    """Public convenience wrapper that loads the full source registry."""
    return SourceRegistry.load(
        project_root,
        policy=policy,
        strategy_registry=strategy_registry,
    )


def _load_declared_contracts(
    sources: Iterable[SourceConfig],
    *,
    locations: Iterable[SourceLocation],
    project_root: Path,
    sources_dir: Path,
) -> dict[str, DataContract]:
    """Load contracts and apply the structural rules that need them to be known.

    An enabled source needs an active contract, an incremental source needs a ``primaryKey``,
    and any ``quality.required_fields``/``unique_fields`` still declared must agree with the
    contract. All three are collected into the one load error.
    """
    entries_by_source = {location.source_id: location.entry for location in locations}

    def validate(source: SourceConfig, contract: DataContract | None) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        _require_active_contract(source, contract, issues)
        if contract is not None:
            _require_primary_key_for_incremental(source, contract, issues)
            _check_quality_agreement(source, contract, issues)
        entry = entries_by_source[source.source_id]
        if entry is None:
            return issues
        return [ValidationIssue(f"{entry}.{issue.path}", issue.message) for issue in issues]

    return load_contract_snapshot(
        sources,
        project_root=project_root,
        sources_dir=sources_dir,
        validate_contract=validate,
    )


def _require_active_contract(
    config: SourceConfig, contract: DataContract | None, issues: list[ValidationIssue]
) -> None:
    """Structural since, no longer policy: an enabled source needs an active contract."""
    if not config.enabled:
        return
    if contract is None or contract.status != "active":
        issues.append(
            ValidationIssue(
                "schema.contract",
                "must reference a contract with status 'active' for an enabled source: "
                "the materializer writes only under a reviewed contract (order-19)"
                + (f" (found {contract.status!r})" if contract is not None else ""),
            )
        )


def _require_primary_key_for_incremental(
    config: SourceConfig, contract: DataContract, issues: list[ValidationIssue]
) -> None:
    """An incremental source is upserted on its contract's ``primaryKey``; none, no idempotency."""
    if config.extraction.mode == "incremental" and not contract.primary_key:
        issues.append(
            ValidationIssue(
                "schema.contract",
                "an incremental source needs a primaryKey in its contract to derive an "
                "idempotent bronze write.",
            )
        )


def _check_quality_agreement(
    config: SourceConfig, contract: DataContract, issues: list[ValidationIssue]
) -> None:
    """The quality keys are cross-checks now: when declared, they must equal the contract's."""
    _check_key_agreement(
        "required_fields",
        config.quality.required_fields,
        contract.required_columns,
        "required columns",
        issues,
    )
    _check_key_agreement(
        "unique_fields",
        config.quality.unique_fields,
        contract.primary_key,
        "primaryKey",
        issues,
    )


def _check_key_agreement(
    field_name: str,
    declared: tuple[str, ...],
    contract_columns: tuple[str, ...],
    description: str,
    issues: list[ValidationIssue],
) -> None:
    """Compare as sets: order matters for neither side, and an empty key is silent."""
    if not declared or set(declared) == set(contract_columns):
        return
    issues.append(
        ValidationIssue(
            f"quality.{field_name}",
            f"disagrees with the contract's {description}: config has "
            f"{_render_columns(declared)}, contract has {_render_columns(contract_columns)}; "
            "drop the key or fix the contract — the contract is the declaration.",
        )
    )


def _render_columns(columns: Iterable[str]) -> str:
    return "{" + ", ".join(sorted(set(columns))) + "}"


def _discover_source_config_paths(sources_dir: Path, file_pattern: str) -> tuple[Path, ...]:
    """Discover source config files recursively to support domain-scoped folders."""
    return tuple(
        sorted(
            config_path
            for config_path in sources_dir.rglob(file_pattern)
            if config_path.is_file()
        )
    )


def _load_source_configs(
    config_path: Path,
    *,
    policy: ValidationPolicy = DEFAULT_VALIDATION_POLICY,
    strategy_registry: StrategyRegistry = STRATEGY_REGISTRY,
) -> tuple[tuple[SourceConfig, str | None], ...]:
    """Read one YAML file and return its validated configs, each with its entry label."""
    raw = _load_yaml_document(config_path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    if "sources" not in raw:
        return (
            (
                SourceConfig.from_mapping(
                    raw,
                    config_path,
                    policy=policy,
                    registry=strategy_registry,
                ),
                None,
            ),
        )
    return _load_grouped_source_configs(
        raw,
        config_path,
        policy=policy,
        strategy_registry=strategy_registry,
    )


def _load_grouped_source_configs(
    raw: Mapping[str, Any],
    config_path: Path,
    *,
    policy: ValidationPolicy = DEFAULT_VALIDATION_POLICY,
    strategy_registry: StrategyRegistry = STRATEGY_REGISTRY,
) -> tuple[tuple[SourceConfig, str], ...]:
    """Load a grouped source file whose top level is a `sources:` list."""
    issues: list[ValidationIssue] = []

    unsupported_keys = sorted(key for key in raw if key != "sources")
    for key in unsupported_keys:
        issues.append(
            ValidationIssue(
                str(key),
                "is not supported in a grouped source file; define entries under sources[]",
            )
        )

    sources_value = raw.get("sources")
    if not isinstance(sources_value, list):
        issues.append(ValidationIssue("sources", "must be a list"))
        raise SourceConfigValidationError(config_path, issues)
    if not sources_value:
        issues.append(ValidationIssue("sources", "must not be empty"))
        raise SourceConfigValidationError(config_path, issues)

    sources: list[tuple[SourceConfig, str]] = []
    for index, item in enumerate(sources_value):
        entry_path = f"sources[{index}]"
        if not isinstance(item, Mapping):
            issues.append(ValidationIssue(entry_path, "must be a mapping"))
            continue
        try:
            sources.append(
                (
                    SourceConfig.from_mapping(
                        item,
                        config_path,
                        policy=policy,
                        registry=strategy_registry,
                    ),
                    entry_path,
                )
            )
        except SourceConfigValidationError as exc:
            issues.extend(
                ValidationIssue(f"{entry_path}.{issue.path}", issue.message)
                for issue in exc.issues
            )

    if issues:
        raise SourceConfigValidationError(config_path, issues)
    return tuple(sources)


def _load_yaml_document(config_path: Path) -> Any:
    """Read a YAML file and return the parsed top-level document."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        try:
            raw = yaml.safe_load(stream) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML config: {config_path}: {exc}") from exc

    return raw


def _load_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    """Read a YAML file and ensure the top-level document is a mapping."""
    raw = _load_yaml_document(config_path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return raw


def _require_string(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
) -> str:
    """Read a required non-empty string from the app config helper layer."""
    value = data.get(field_name)
    field_path = _field_path(field_name, prefix)
    if value is None:
        issues.append(ValidationIssue(field_path, "is required"))
        return ""
    if not isinstance(value, str):
        issues.append(ValidationIssue(field_path, "must be a string"))
        return ""
    value = value.strip()
    if not value:
        issues.append(ValidationIssue(field_path, "must not be empty"))
        return ""
    return value


def _optional_string(
    data: Mapping[str, Any],
    field_name: str,
    issues: list[ValidationIssue],
    prefix: str | None = None,
    default: str = "",
) -> str:
    """Read an optional string from the app config helper layer."""
    if field_name not in data or data[field_name] is None:
        return default
    return _require_string(data, field_name, issues, prefix)


def _field_path(field_name: str, prefix: str | None) -> str:
    """Compose dotted field names for nested app-config validation messages."""
    if prefix:
        return f"{prefix}.{field_name}"
    return field_name
