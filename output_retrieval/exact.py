"""Exact, vocabulary-chunked retrieval for tests and small workloads."""

from __future__ import annotations

import torch
from torch import Tensor


class ExactStaticOutputIndex:
    """Exact inner-product top-k without constructing an ``[N, V]`` tensor.

    This backend is intended for correctness tests, recall measurements, and
    explicitly selected small-vocabulary runs. The IVF backend is the normal
    production choice for the repository's 100k-token vocabulary.
    """

    def __init__(self, vectors: Tensor, *, vocab_chunk_size: int = 8192) -> None:
        if vectors.ndim != 2 or not vectors.is_floating_point():
            raise ValueError("vectors must be a floating-point [V, D] tensor")
        if vocab_chunk_size <= 0:
            raise ValueError("vocab_chunk_size must be positive")
        self.vectors = vectors.detach()
        self.vocab_chunk_size = vocab_chunk_size
        self.max_score_columns = 0

    @property
    def size(self) -> int:
        return self.vectors.size(0)

    @torch.no_grad()
    def search(self, queries: Tensor, k: int) -> tuple[Tensor, Tensor]:
        if queries.ndim != 2 or queries.size(1) != self.vectors.size(1):
            raise ValueError("queries must have shape [N, D] matching the index")
        if not 0 < k <= self.size:
            raise ValueError(f"k must be between 1 and {self.size}")
        queries = queries.to(device=self.vectors.device, dtype=torch.float32)
        vectors = self.vectors.float()
        values = torch.full(
            (queries.size(0), k), -torch.inf, device=queries.device
        )
        indices = torch.full(
            (queries.size(0), k), -1, dtype=torch.long, device=queries.device
        )
        self.max_score_columns = 0
        for start in range(0, self.size, self.vocab_chunk_size):
            end = min(start + self.vocab_chunk_size, self.size)
            logits = queries @ vectors[start:end].transpose(0, 1)
            self.max_score_columns = max(self.max_score_columns, logits.size(1))
            chunk_k = min(k, end - start)
            chunk_values, chunk_indices = logits.topk(chunk_k, dim=-1)
            chunk_indices.add_(start)
            candidates = torch.cat((values, chunk_values), dim=-1)
            candidate_ids = torch.cat((indices, chunk_indices), dim=-1)
            values, order = candidates.topk(k, dim=-1)
            indices = candidate_ids.gather(-1, order)
        return values, indices
