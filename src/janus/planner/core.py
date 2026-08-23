from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from janus.models import ExecutionPlan, ExtractionResult, RunContext, WriteResult
from janus.models.config.strategy_registry import STRATEGY_REGISTRY, StrategyRegistry
from janus.models.source_config import SourceConfig
from janus.registry import load_registry
from janus.strategies.base import BaseStrategy, SourceHook

RUN_ID_SEGMENT_PATTERN = re.compile(r"[^a-z0-9]+")


class PlannerError(RuntimeError):
    """Base planner failure used for explicit orchestration errors."""


class StrategyResolutionError(PlannerError):
    """Raised when no strategy binding matches a validated source config."""


class HookResolutionError(PlannerError):
    """Raised when a configured source hook cannot be resolved."""


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    """Inputs required to turn one configured source into a deterministic plan."""

    source_id: str
    environment: str
    project_root: Path
    run_id: str | None = None
    started_at: datetime | None = None
    attributes: tuple[tuple[str, str], ...] = ()
    include_disabled: bool = False

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.environment.strip():
            raise ValueError("environment must not be empty")
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must not be empty when provided")
        if self.started_at is not None and (
            self.started_at.tzinfo is None or self.started_at.utcoffset() is None
        ):
            raise ValueError("started_at must be timezone-aware when provided")

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        environment: str,
        project_root: Path,
        run_id: str | None = None,
        started_at: datetime | None = None,
        attributes: Mapping[str, str] | None = None,
        include_disabled: bool = False,
    ) -> Self:
        return cls(
            source_id=source_id,
            environment=environment,
            project_root=project_root.resolve(),
            run_id=run_id,
            started_at=started_at,
            attributes=_freeze_string_mapping(attributes),
            include_disabled=include_disabled,
        )

    def attributes_as_dict(self) -> dict[str, str]:
        return dict(self.attributes)


@dataclass(frozen=True, slots=True)
class StrategyBinding:
    """One planner-visible dispatch binding for a strategy family and variant."""

    family: str
    variant: str
    strategy: BaseStrategy

    def __post_init__(self) -> None:
        if not self.family.strip():
            raise ValueError("family must not be empty")
        if not self.variant.strip():
            raise ValueError("variant must not be empty")
        if self.strategy.strategy_family != self.family:
            raise ValueError(
                f"strategy.strategy_family must match the bound family {self.family!r}"
            )

    @property
    def dispatch_path(self) -> str:
        return f"{self.family}.{self.variant}"


@dataclass(frozen=True, slots=True)
class StrategyCatalog:
    """Small planner registry that maps config metadata to strategy implementations."""

    bindings: tuple[StrategyBinding, ...]
    _bindings_by_key: dict[tuple[str, str], StrategyBinding] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        bindings_by_key: dict[tuple[str, str], StrategyBinding] = {}
        for binding in self.bindings:
            key = (binding.family, binding.variant)
            if key in bindings_by_key:
                raise ValueError(
                    "Duplicate strategy binding registered for "
                    f"family {binding.family!r} variant {binding.variant!r}"
                )
            bindings_by_key[key] = binding
        object.__setattr__(self, "_bindings_by_key", bindings_by_key)

    @classmethod
    def with_defaults(cls, registry: StrategyRegistry = STRATEGY_REGISTRY) -> Self:
        """Bind one implementation per family, over every pair the registry declares.

        The registry owns the *names*; this method owns the *implementations*. Adding a
        variant is therefore a registry-only edit, and adding a family fails loudly here
        instead of raising a bare ``KeyError`` at every planner construction.
        """
        from janus.strategies.api import ApiStrategy
        from janus.strategies.catalog import CatalogStrategy
        from janus.strategies.files import FileStrategy

        implementations: dict[str, BaseStrategy] = {
            "api": ApiStrategy(),
            "catalog": CatalogStrategy(),
            "file": FileStrategy(),
        }
        missing = sorted(registry.families - implementations.keys())
        if missing:
            raise PlannerError(
                "No strategy implementation is registered for family/families "
                f"{', '.join(missing)} declared in the strategy registry. Add the binding in "
                "StrategyCatalog.with_defaults, or remove the family from "
                "janus.models.config.constants.SUPPORTED_STRATEGY_VARIANTS."
            )
        bindings = tuple(
            StrategyBinding(family=family, variant=variant, strategy=implementations[family])
            for family, variant in registry.dispatch_keys()
        )
        return cls(bindings=bindings)

    def resolve(self, source_config: SourceConfig) -> StrategyBinding:
        key = (source_config.strategy, source_config.strategy_variant)
        binding = self._bindings_by_key.get(key)
        if binding is not None:
            return binding

        registered_variants = self.registered_variants_for_family(source_config.strategy)
        if not registered_variants:
            raise StrategyResolutionError(
                "No strategy implementation is registered for family "
                f"{source_config.strategy!r} while planning source "
                f"{source_config.source_id!r}. Register a binding for variant "
                f"{source_config.strategy_variant!r} before running this source."
            )

        variants_list = ", ".join(registered_variants)
        raise StrategyResolutionError(
            "No strategy implementation is registered for family "
            f"{source_config.strategy!r} variant {source_config.strategy_variant!r} "
            f"while planning source {source_config.source_id!r}. Registered variants "
            f"for family {source_config.strategy!r}: {variants_list}"
        )

    def registered_variants_for_family(self, family: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                variant for bound_family, variant in self._bindings_by_key if bound_family == family
            )
        )


@dataclass(frozen=True, slots=True)
class HookCatalog:
    """Optional source-hook registry resolved by configured hook id."""

    hooks: tuple[tuple[str, SourceHook], ...] = ()
    _hooks_by_id: dict[str, SourceHook] = field(init=False, repr=False)

    @classmethod
    def with_defaults(cls) -> Self:
        from janus.hooks import built_in_hooks

        return cls(hooks=built_in_hooks())

    def __post_init__(self) -> None:
        hooks_by_id: dict[str, SourceHook] = {}
        for hook_id, hook in self.hooks:
            normalized_hook_id = hook_id.strip()
            if not normalized_hook_id:
                raise ValueError("hook id must not be empty")
            if normalized_hook_id in hooks_by_id:
                raise ValueError(f"Duplicate source hook registered for id {normalized_hook_id!r}")
            hooks_by_id[normalized_hook_id] = hook
        object.__setattr__(self, "_hooks_by_id", hooks_by_id)

    def resolve(self, hook_id: str | None, *, source_id: str) -> SourceHook | None:
        if hook_id is None:
            return None

        hook = self._hooks_by_id.get(hook_id)
        if hook is None:
            raise HookResolutionError(
                f"Source {source_id!r} requires hook {hook_id!r}, but no hook binding is registered"
            )
        return hook


@dataclass(frozen=True, slots=True)
class PlannedRun:
    """Planner output kept small enough for entry points and later runtime layers."""

    plan: ExecutionPlan
    strategy: BaseStrategy
    hook: SourceHook | None = None
    pre_run_metadata: tuple[tuple[str, str], ...] = ()

    @property
    def dispatch_path(self) -> str:
        return f"{self.strategy.strategy_family}.{self.plan.source.strategy_variant}"

    def pre_run_metadata_as_dict(self) -> dict[str, str]:
        return dict(self.pre_run_metadata)

    def to_summary(self) -> dict[str, Any]:
        return {
            "run": {
                "run_id": self.plan.run_context.run_id,
                "environment": self.plan.run_context.environment,
                "started_at": self.plan.run_context.started_at.isoformat(),
                "attributes": self.plan.run_context.attributes_as_dict(),
            },
            "source": {
                "source_id": self.plan.source.source_id,
                "name": self.plan.source.name,
                "config_path": str(self.plan.source_config.config_path),
                "hook_id": self.plan.source.source_hook,
                "tags": list(self.plan.source.tags),
            },
            "strategy": {
                "family": self.strategy.strategy_family,
                "variant": self.plan.source.strategy_variant,
                "dispatch_path": self.dispatch_path,
                "implementation": type(self.strategy).__name__,
                "hook_implementation": type(self.hook).__name__ if self.hook else None,
            },
            "execution": {
                "mode": self.plan.extraction_mode,
                "checkpoint_strategy": self.plan.checkpoint_strategy,
                "checkpoint_field": self.plan.checkpoint_field,
            },
            "outputs": {
                "raw": {
                    "path": self.plan.raw_output.path,
                    "format": self.plan.raw_output.format,
                },
                "bronze": {
                    "path": self.plan.bronze_output.path,
                    "format": self.plan.bronze_output.format,
                },
                "metadata": {
                    "path": self.plan.metadata_output.path,
                    "format": self.plan.metadata_output.format,
                },
            },
            "plan_notes": list(self.plan.notes),
            "pre_run_metadata": self.pre_run_metadata_as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Planner:
    """Thin orchestration layer that converts one source config into a dispatchable plan."""

    strategy_catalog: StrategyCatalog = field(default_factory=StrategyCatalog.with_defaults)
    hook_catalog: HookCatalog = field(default_factory=HookCatalog.with_defaults)

    def plan(self, request: PlanningRequest) -> PlannedRun:
        registry = load_registry(request.project_root)
        source_config = registry.get_source(
            request.source_id,
            include_disabled=request.include_disabled,
        )
        strategy_binding = self.strategy_catalog.resolve(source_config)
        hook = self.hook_catalog.resolve(
            source_config.source_hook,
            source_id=source_config.source_id,
        )
        run_context = _build_run_context(request, source_config)
        plan = strategy_binding.strategy.plan(source_config, run_context, hook).with_note(
            f"dispatch:{strategy_binding.dispatch_path}"
        )
        _validate_planned_dispatch(plan, strategy_binding)
        return PlannedRun(
            plan=plan,
            strategy=strategy_binding.strategy,
            hook=hook,
            pre_run_metadata=_build_pre_run_metadata(plan, strategy_binding, hook),
        )


class PlanningStrategy(BaseStrategy):
    """Planning-only strategy used until family-specific strategy modules land."""

    def __init__(self, family: str) -> None:
        self._family = family

    @property
    def strategy_family(self) -> str:
        return self._family

    def plan(
        self,
        source_config: SourceConfig,
        run_context: RunContext,
        hook: SourceHook | None = None,
    ) -> ExecutionPlan:
        plan = ExecutionPlan.from_source_config(source_config, run_context)
        plan = plan.with_note(f"strategy_family:{self.strategy_family}")
        plan = plan.with_note(f"strategy_variant:{source_config.strategy_variant}")
        if hook is not None:
            return hook.on_plan(plan)
        return plan

    def extract(
        self,
        plan: ExecutionPlan,
        hook: SourceHook | None = None,
        *,
        spark=None,
    ) -> ExtractionResult:
        del hook
        del spark
        raise NotImplementedError(
            f"Dispatch {plan.source.strategy!r}/{plan.source.strategy_variant!r} is planned, "
            "but extraction is not implemented yet"
        )

    def build_normalization_handoff(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        hook: SourceHook | None = None,
    ) -> ExtractionResult:
        del hook
        del extraction_result
        raise NotImplementedError(
            f"Dispatch {plan.source.strategy!r}/{plan.source.strategy_variant!r} is planned, "
            "but normalization handoff is not implemented yet"
        )

    def emit_metadata(
        self,
        plan: ExecutionPlan,
        extraction_result: ExtractionResult,
        write_results: tuple[WriteResult, ...] = (),
        hook: SourceHook | None = None,
    ) -> Mapping[str, Any]:
        del extraction_result
        del write_results
        del hook
        raise NotImplementedError(
            f"Dispatch {plan.source.strategy!r}/{plan.source.strategy_variant!r} is planned, "
            "but metadata emission is not implemented yet"
        )


def _build_run_context(request: PlanningRequest, source_config: SourceConfig) -> RunContext:
    started_at = request.started_at or datetime.now(tz=UTC)
    run_id = request.run_id or _default_run_id(
        environment=request.environment,
        source_id=source_config.source_id,
        started_at=started_at,
    )

    attributes = request.attributes_as_dict()
    attributes.update(
        {
            "source_id": source_config.source_id,
            "source_type": source_config.source_type,
            "strategy": source_config.strategy,
            "strategy_variant": source_config.strategy_variant,
        }
    )
    if source_config.source_hook:
        attributes["source_hook"] = source_config.source_hook

    return RunContext.create(
        run_id=run_id,
        environment=request.environment,
        project_root=request.project_root,
        started_at=started_at,
        attributes=attributes,
    )


def _default_run_id(environment: str, source_id: str, started_at: datetime) -> str:
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"run-{_normalize_run_id_segment(environment)}-"
        f"{_normalize_run_id_segment(source_id)}-{timestamp}"
    )


def _normalize_run_id_segment(value: str) -> str:
    normalized = RUN_ID_SEGMENT_PATTERN.sub("-", value.strip().lower()).strip("-")
    return normalized or "source"


def _validate_planned_dispatch(plan: ExecutionPlan, strategy_binding: StrategyBinding) -> None:
    if plan.source.strategy != strategy_binding.family:
        raise PlannerError(
            f"Planned source strategy {plan.source.strategy!r} does not match resolved family "
            f"{strategy_binding.family!r}"
        )
    if plan.source.strategy_variant != strategy_binding.variant:
        raise PlannerError(
            "Planned source strategy variant "
            f"{plan.source.strategy_variant!r} does not match resolved variant "
            f"{strategy_binding.variant!r}"
        )


def _build_pre_run_metadata(
    plan: ExecutionPlan,
    strategy_binding: StrategyBinding,
    hook: SourceHook | None,
) -> tuple[tuple[str, str], ...]:
    metadata = {
        "run_id": plan.run_context.run_id,
        "started_at": plan.run_context.started_at.isoformat(),
        "environment": plan.run_context.environment,
        "project_root": str(plan.run_context.project_root),
        "source_id": plan.source.source_id,
        "source_name": plan.source.name,
        "source_config_path": str(plan.source_config.config_path),
        "strategy_family": strategy_binding.family,
        "strategy_variant": strategy_binding.variant,
        "dispatch_path": strategy_binding.dispatch_path,
        "extraction_mode": plan.extraction_mode,
        "checkpoint_strategy": plan.checkpoint_strategy,
        "raw_output_path": plan.raw_output.path,
        "bronze_output_path": plan.bronze_output.path,
        "metadata_output_path": plan.metadata_output.path,
    }
    if plan.checkpoint_field is not None:
        metadata["checkpoint_field"] = plan.checkpoint_field
    if plan.source.source_hook is not None:
        metadata["hook_id"] = plan.source.source_hook
    if hook is not None:
        metadata["hook_implementation"] = type(hook).__name__
    return _freeze_string_mapping(metadata)


def _freeze_string_mapping(values: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()

    frozen_items: list[tuple[str, str]] = []
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("mapping keys must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"mapping value for {key!r} must be a non-empty string")
        frozen_items.append((key.strip(), value.strip()))
    return tuple(sorted(frozen_items))
