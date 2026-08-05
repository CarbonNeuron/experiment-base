"""Public extension API for model architectures and training objectives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from torch import nn

from training.objectives import ModelProvidedLoss, TrainingObjective


ConfigT = TypeVar("ConfigT")
ModelFactory = Callable[[Any, str | Path | None], nn.Module]
ObjectiveFactory = Callable[[], TrainingObjective]


@dataclass(frozen=True)
class ModelRegistration(Generic[ConfigT]):
    """Everything the runner needs to train one architecture family."""

    config_type: type[ConfigT]
    factory: Callable[[ConfigT, str | Path | None], nn.Module]
    objective_factory: ObjectiveFactory = ModelProvidedLoss
    name: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.config_type.__name__

    def build(
        self, config: ConfigT, embed_path: str | Path | None = None
    ) -> nn.Module:
        return self.factory(config, embed_path)

    def make_objective(self) -> TrainingObjective:
        return self.objective_factory()


class ModelRegistry:
    """Resolve architecture components by config type.

    Exact config types win. For config subclasses, the closest registered base
    class in the method-resolution order is selected.
    """

    def __init__(self) -> None:
        self._registrations: dict[type[Any], ModelRegistration[Any]] = {}

    def register(
        self,
        config_type: type[ConfigT],
        factory: Callable[[ConfigT, str | Path | None], nn.Module],
        *,
        objective: ObjectiveFactory = ModelProvidedLoss,
        name: str | None = None,
        replace: bool = False,
    ) -> ModelRegistration[ConfigT]:
        """Register an architecture, rejecting accidental duplicate bindings."""
        if config_type in self._registrations and not replace:
            existing = self._registrations[config_type]
            raise ValueError(
                f"{config_type.__name__} is already registered as "
                f"{existing.display_name!r}"
            )
        registration = ModelRegistration(
            config_type=config_type,
            factory=factory,
            objective_factory=objective,
            name=name,
        )
        self._registrations[config_type] = registration
        return registration

    def resolve(self, config: Any) -> ModelRegistration[Any]:
        """Return the most specific registration for a config instance."""
        config_type = type(config)
        exact = self._registrations.get(config_type)
        if exact is not None:
            return exact
        for base_type in config_type.__mro__[1:]:
            registration = self._registrations.get(base_type)
            if registration is not None:
                return registration
        raise TypeError(
            f"no model registered for config type {config_type.__name__}; "
            "register it with experiments.register_model(...)"
        )

    def build_model(
        self, config: Any, embed_path: str | Path | None = None
    ) -> nn.Module:
        return self.resolve(config).build(config, embed_path)

    def objective_for(self, config: Any) -> TrainingObjective:
        return self.resolve(config).make_objective()

    def registrations(self) -> tuple[ModelRegistration[Any], ...]:
        """Return an immutable snapshot, primarily for discovery and tooling."""
        return tuple(self._registrations.values())
