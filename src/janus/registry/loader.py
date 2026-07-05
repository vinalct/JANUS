from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import yaml

from janus.models.source_config import (
    SourceConfig,
    SourceConfigValidationError,
    ValidationIssue,
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
    project_root: Path
    app_config: AppConfig
    sources: tuple[SourceConfig, ...]
    _sources_by_id: dict[str, SourceConfig] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build an in-memory lookup table so planner code can fetch sources by id."""
        object.__setattr__(
            self,
            "_sources_by_id",
            {source.source_id: source for source in self.sources},
        )

    @classmethod
    def load(cls, project_root: Path) -> Self:
        """Load app settings, discover source YAML files, and return the typed registry."""
        resolved_project_root = project_root.resolve()
        app_config = load_app_config(resolved_project_root)
        sources_dir = app_config.registry.resolve_sources_dir(resolved_project_root)
        if not sources_dir.exists():
            raise FileNotFoundError(f"Configured source directory does not exist: {sources_dir}")

        sources: list[SourceConfig] = []
        seen_source_ids: dict[str, Path] = {}
        for config_path in _discover_source_config_paths(
            sources_dir,
            app_config.registry.file_pattern,
        ):
            for source in _load_source_configs(config_path):
                previous_path = seen_source_ids.get(source.source_id)
                if previous_path is not None:
                    raise ValueError(
                        "Duplicate source_id "
                        f"{source.source_id!r} found in {previous_path} and {config_path}"
                    )
                seen_source_ids[source.source_id] = config_path
                sources.append(source)

        return cls(
            project_root=resolved_project_root,
            app_config=app_config,
            sources=tuple(sources),
        )

    def list_sources(self, *, enabled_only: bool = True) -> tuple[SourceConfig, ...]:
        """Return the registered sources, filtering disabled ones by default."""
        if not enabled_only:
            return self.sources
        return tuple(source for source in self.sources if source.enabled)

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


def load_registry(project_root: Path) -> SourceRegistry:
    """Public convenience wrapper that loads the full source registry."""
    return SourceRegistry.load(project_root)


def _discover_source_config_paths(sources_dir: Path, file_pattern: str) -> tuple[Path, ...]:
    """Discover source config files recursively to support domain-scoped folders."""
    return tuple(
        sorted(
            config_path
            for config_path in sources_dir.rglob(file_pattern)
            if config_path.is_file()
        )
    )


def _load_source_configs(config_path: Path) -> tuple[SourceConfig, ...]:
    """Read one YAML file and return one or more validated source configs."""
    raw = _load_yaml_document(config_path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    if "sources" not in raw:
        return (SourceConfig.from_mapping(raw, config_path),)
    return _load_grouped_source_configs(raw, config_path)


def _load_grouped_source_configs(
    raw: Mapping[str, Any],
    config_path: Path,
) -> tuple[SourceConfig, ...]:
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

    sources: list[SourceConfig] = []
    for index, item in enumerate(sources_value):
        entry_path = f"sources[{index}]"
        if not isinstance(item, Mapping):
            issues.append(ValidationIssue(entry_path, "must be a mapping"))
            continue
        try:
            sources.append(SourceConfig.from_mapping(item, config_path))
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
