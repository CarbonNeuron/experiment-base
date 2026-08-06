"""Configuration for the causal multigrid-memory language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MultigridMemoryConfig:
    """Architecture settings for :class:`MultigridMemoryTransformer`."""

    d_model: int = 128
    n_layers: int = 6
    n_cycles: int = 2
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    refinement_kernel_size: int = 3
    n_memory_slots: int = 256
    d_memory: int = 64
    d_key: int = 64
    memory_addressing: str = "softmax"
    n_hash_bits: int = 32
    hash_top_k: int = 8
    use_triton_memory: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.d_model,
            self.n_layers,
            self.n_cycles,
            self.d_ff,
            self.max_seq_len,
            self.layer_norm_eps,
            self.refinement_kernel_size,
            self.n_memory_slots,
            self.d_memory,
            self.d_key,
            self.n_hash_bits,
            self.hash_top_k,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("model dimensions and layer counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.memory_addressing not in ("softmax", "hash"):
            raise ValueError("memory_addressing must be 'softmax' or 'hash'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
