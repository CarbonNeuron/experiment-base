"""Central registry for experiment model construction and training APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from torch import nn

from config import (
    ChainedHydraConfig,
    CompoundQConfig,
    GrowingWidthConfig,
    HydraConfig,
    TournamentHydraConfig,
    TransformerConfig,
)
from model import (
    ChainedHydraTransformer,
    CompoundQTransformer,
    GenericTransformer,
    GrowingWidthTransformer,
    HydraTransformer,
    TournamentHydraTransformer,
)
from multigrid import MultigridMemoryConfig, MultigridMemoryTransformer
from training.objectives import (
    LogitsCrossEntropy,
    ModelProvidedLoss,
    TrainingObjective,
)


ModelFactory = Callable[[Any, str | Path | None], nn.Module]


_MODEL_FACTORIES: tuple[tuple[type[Any], ModelFactory], ...] = (
    (TournamentHydraConfig, TournamentHydraTransformer),
    (ChainedHydraConfig, ChainedHydraTransformer),
    (HydraConfig, HydraTransformer),
    (CompoundQConfig, CompoundQTransformer),
    (GrowingWidthConfig, GrowingWidthTransformer),
    (MultigridMemoryConfig, MultigridMemoryTransformer),
    (TransformerConfig, GenericTransformer),
)


def build_model(config: Any, embed_path: str | Path | None = None) -> nn.Module:
    """Construct the model registered for an architecture config."""
    for config_type, factory in _MODEL_FACTORIES:
        if isinstance(config, config_type):
            return factory(config, embed_path)
    raise TypeError(f"no model registered for config type {type(config).__name__}")


def objective_for(config: Any) -> TrainingObjective:
    """Select the adapter for the model's public forward contract."""
    if isinstance(config, MultigridMemoryConfig):
        return LogitsCrossEntropy()
    if any(isinstance(config, entry[0]) for entry in _MODEL_FACTORIES):
        return ModelProvidedLoss()
    raise TypeError(
        f"no training objective registered for config type "
        f"{type(config).__name__}"
    )
