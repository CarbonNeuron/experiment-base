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
from .quatspin import QuatRMSNorm, QuatSpinFFN, quat_mul
from .triton_quatspin import fused_quat_mul_norm, triton_quatspin_available

__all__ = [
    "CausalSelfAttention",
    "ChainedHydraTransformer",
    "CompoundQAttention",
    "CompoundQTransformer",
    "CompressMergeBlock",
    "FFNMergeBlock",
    "FeedForward",
    "fused_quat_mul_norm",
    "GenericTransformer",
    "GrowingWidthTransformer",
    "HydraAttention",
    "HydraBlock",
    "HydraFeedForward",
    "HydraTransformer",
    "QuatRMSNorm",
    "QuatSpinFFN",
    "RecursiveHydraBlock",
    "SVDLanguageModel",
    "ScratchBlock",
    "TournamentBlock",
    "TournamentHydraTransformer",
    "TournamentRound",
    "TransformerBlock",
    "quat_mul",
    "resolve_compile_backend",
    "triton_quatspin_available",
]
