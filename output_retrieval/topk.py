"""Fast top-k projection through a static index of frozen directions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from .flat import FlatExactIndex

if TYPE_CHECKING:
    from svd_embeds import OpenAIEmbedding


class TopKIndex:
    """Flat exact retrieval with rotation-aware query transform and rescoring."""

    def __init__(self, embeddings: "OpenAIEmbedding") -> None:
        self.embeddings = embeddings
        self._index: FlatExactIndex | None = None
        self._directions_key: tuple[torch.device, torch.dtype, int, int] | None = None

    def _current_directions_key(self) -> tuple[torch.device, torch.dtype, int, int]:
        directions = self.embeddings.directions
        return (
            directions.device,
            directions.dtype,
            directions.data_ptr(),
            directions._version,
        )

    @property
    def is_built(self) -> bool:
        return self._index is not None

    @property
    def index(self) -> FlatExactIndex | None:
        return self._index

    @torch.no_grad()
    def build(self) -> FlatExactIndex:
        directions_key = self._current_directions_key()
        if self._index is None or self._directions_key != directions_key:
            directions = F.normalize(
                self.embeddings.directions.detach(), dim=-1
            )
            self._index = FlatExactIndex(directions)
            self._directions_key = directions_key
        return self._index

    @torch.no_grad()
    def search(self, hidden: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Return exact logits for top-k tokens found via flat matmul."""
        if hidden.ndim < 1 or hidden.size(-1) != self.embeddings.dimension:
            raise ValueError(
                "hidden must end in embedding dimension "
                f"{self.embeddings.dimension}"
            )
        if not 0 < k <= self.embeddings.num_embeddings:
            raise ValueError(
                f"k must be between 1 and {self.embeddings.num_embeddings}"
            )
        if hidden.device != self.embeddings.directions.device:
            raise ValueError("hidden and embeddings must be on the same device")

        index = self.build()
        prefix = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, self.embeddings.dimension)

        # Query transform: h @ R puts us in direction space (R is orthogonal)
        rotated_hidden = flat_hidden @ self.embeddings.rotation.matrix
        retrieval_queries = F.normalize(rotated_hidden, dim=-1)
        # Single matmul against all normalized directions → exact top-k
        _, token_ids = index.search(retrieval_queries, k)

        # Rescore with true logits (unnormalized, with magnitude)
        candidate_directions = self.embeddings.directions[token_ids]
        scores = self.embeddings.magnitude * torch.einsum(
            "nd,nkd->nk", rotated_hidden, candidate_directions
        )
        # Re-rank by exact scores (normalization may change relative order
        # slightly vs raw dot products)
        scores, order = scores.topk(k, dim=-1)
        token_ids = token_ids.gather(-1, order)
        return scores.reshape(*prefix, k), token_ids.reshape(*prefix, k)
