"""Model implementations grouped by architecture family."""

from .base import SVDLanguageModel, resolve_compile_backend
from .baseline import (
    CausalSelfAttention,
    CompoundQAttention,
    CompoundQTransformer,
    FeedForward,
    GenericTransformer,
    TransformerBlock,
)
from .growing_width import GrowingWidthTransformer, ScratchBlock
from .hydra import (
    ChainedHydraTransformer,
    HydraTransformer,
    TournamentHydraTransformer,
)
from .hydra_layers import (
    CompressMergeBlock,
    FFNMergeBlock,
    HydraAttention,
    HydraBlock,
    HydraFeedForward,
    RecursiveHydraBlock,
    TournamentBlock,
    TournamentRound,
)

__all__ = [
    "CausalSelfAttention",
    "ChainedHydraTransformer",
    "CompoundQAttention",
    "CompoundQTransformer",
    "CompressMergeBlock",
    "FFNMergeBlock",
    "FeedForward",
    "GenericTransformer",
    "GrowingWidthTransformer",
    "HydraAttention",
    "HydraBlock",
    "HydraFeedForward",
    "HydraTransformer",
    "RecursiveHydraBlock",
    "SVDLanguageModel",
    "ScratchBlock",
    "TournamentBlock",
    "TournamentHydraTransformer",
    "TournamentRound",
    "TransformerBlock",
    "resolve_compile_backend",
]
