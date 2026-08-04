"""Backend contract for static output-direction search."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class StaticOutputIndex(Protocol):
    """Search fixed vocabulary directions by inner product."""

    @property
    def size(self) -> int:
        """Number of indexed vocabulary items."""

    def search(self, queries: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Return descending scores and token IDs with shape ``[N, k]``."""
