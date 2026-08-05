"""Built-in architecture registrations."""

from __future__ import annotations

from config import (
    ChainedHydraConfig,
    CompoundQConfig,
    GrowingWidthConfig,
    HydraConfig,
    TournamentHydraConfig,
    TransformerConfig,
)
from models import (
    ChainedHydraTransformer,
    CompoundQTransformer,
    GenericTransformer,
    GrowingWidthTransformer,
    HydraTransformer,
    TournamentHydraTransformer,
)
from multigrid import MultigridMemoryConfig, MultigridMemoryTransformer
from training.objectives import LogitsCrossEntropy

from .registry import ModelRegistry


def register_builtin_models(registry: ModelRegistry) -> None:
    """Populate a registry with architectures shipped by this repository."""
    registry.register(TransformerConfig, GenericTransformer, name="transformer")
    registry.register(CompoundQConfig, CompoundQTransformer, name="compound-q")
    registry.register(HydraConfig, HydraTransformer, name="hydra")
    registry.register(
        ChainedHydraConfig, ChainedHydraTransformer, name="chained-hydra"
    )
    registry.register(
        TournamentHydraConfig,
        TournamentHydraTransformer,
        name="tournament-hydra",
    )
    registry.register(
        GrowingWidthConfig, GrowingWidthTransformer, name="growing-width"
    )
    registry.register(
        MultigridMemoryConfig,
        MultigridMemoryTransformer,
        objective=LogitsCrossEntropy,
        name="multigrid-memory",
    )
