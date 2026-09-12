from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from janus.utils.environment import (
    RuntimeLocation,
    build_spark_options,
    build_spark_session,
    load_environment_config,
    prepare_runtime,
)
from janus.utils.storage import StorageLayout


@dataclass(frozen=True, slots=True)
class SparkRuntime:
    """Environment-driven Spark bootstrap shared by runtime code and strategy layers."""

    environment: str
    project_root: Path
    config: dict[str, Any]
    resolved_paths: dict[str, RuntimeLocation]
    storage_layout: StorageLayout

    def __post_init__(self) -> None:
        if not self.environment.strip():
            raise ValueError("environment must not be empty")
        if not self.project_root.is_absolute():
            raise ValueError("project_root must be absolute")

    @classmethod
    def load(cls, environment: str, project_root: Path) -> Self:
        resolved_project_root = project_root.resolve()
        config = load_environment_config(environment, resolved_project_root)
        resolved_paths = prepare_runtime(config, resolved_project_root)
        return cls(
            environment=environment,
            project_root=resolved_project_root,
            config=config,
            resolved_paths=resolved_paths,
            storage_layout=StorageLayout.from_environment_config(config, resolved_project_root),
        )

    def spark_options(self) -> dict[str, str]:
        return build_spark_options(self.config, self.resolved_paths)

    def create_session(self):
        return build_spark_session(self.config, self.resolved_paths)


def load_spark_runtime(environment: str, project_root: Path) -> SparkRuntime:
    return SparkRuntime.load(environment, project_root)
