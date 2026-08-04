"""Evaluation metrics for approximate static retrieval."""

from __future__ import annotations

import torch
from torch import Tensor


def recall_at_k(approximate_ids: Tensor, exact_ids: Tensor) -> Tensor:
    """Mean per-query set recall for equally shaped top-k ID tensors."""
    if approximate_ids.shape != exact_ids.shape or approximate_ids.ndim != 2:
        raise ValueError("retrieval ID tensors must have matching [N, K] shapes")
    matches = approximate_ids.unsqueeze(-1).eq(exact_ids.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean()
