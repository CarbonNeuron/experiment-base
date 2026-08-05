"""Experiment composition and the public architecture extension API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from torch import nn

from training.objectives import ModelProvidedLoss, TrainingObjective

from .catalog import register_builtin_models
from .registry import (
    ConfigT,
    ModelFactory,
    ModelRegistration,
    ModelRegistry,
    ObjectiveFactory,
)


DEFAULT_REGISTRY = ModelRegistry()
register_builtin_models(DEFAULT_REGISTRY)


def register_model(
    config_type: type[ConfigT],
    factory: Callable[[ConfigT, str | Path | None], nn.Module],
    *,
    objective: ObjectiveFactory = ModelProvidedLoss,
    name: str | None = None,
    replace: bool = False,
) -> ModelRegistration[ConfigT]:
    """Register an architecture in the process-wide default registry."""
    return DEFAULT_REGISTRY.register(
        config_type,
        factory,
        objective=objective,
        name=name,
        replace=replace,
    )


def build_model(
    config: Any,
    embed_path: str | Path | None = None,
    *,
    registry: ModelRegistry = DEFAULT_REGISTRY,
) -> nn.Module:
    return registry.build_model(config, embed_path)


def objective_for(
    config: Any, *, registry: ModelRegistry = DEFAULT_REGISTRY
) -> TrainingObjective:
    return registry.objective_for(config)


__all__ = [
    "DEFAULT_REGISTRY",
    "ModelFactory",
    "ModelRegistration",
    "ModelRegistry",
    "ObjectiveFactory",
    "build_model",
    "objective_for",
    "register_model",
]
