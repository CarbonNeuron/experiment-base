"""Flat batched exact retrieval — single matmul, no chunking."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import StaticOutputIndex


class FlatExactIndex(StaticOutputIndex):
    """Exact top-k via a single batched matmul against all directions.

    At 128d × 100k this is ~50MB of FLOPS — trivially fast on GPU and
    hard to beat with any approximate method due to Python/kernel overhead.
    """

    def __init__(self, vectors: Tensor) -> None:
        if vectors.ndim != 2 or not vectors.is_floating_point():
            raise ValueError("vectors must be a floating-point [V, D] tensor")
        self.vectors = vectors.detach()
        self.max_scored_candidates = 0

    @property
    def size(self) -> int:
        return self.vectors.size(0)

    @torch.no_grad()
    def search(self, queries: Tensor, k: int) -> tuple[Tensor, Tensor]:
        if queries.ndim != 2 or queries.size(1) != self.vectors.size(1):
            raise ValueError("queries must have shape [N, D] matching the index")
        if not 0 < k <= self.size:
            raise ValueError(f"k must be between 1 and {self.size}")
        queries = queries.to(device=self.vectors.device, dtype=self.vectors.dtype)
        # Single matmul: [N, D] @ [D, V] → [N, V]
        scores = queries @ self.vectors.T
        self.max_scored_candidates = scores.size(1)
        return scores.topk(k, dim=-1)
