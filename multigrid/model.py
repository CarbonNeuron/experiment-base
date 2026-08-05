"""Language model built from causal multigrid-memory blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from svd_embeds import OpenAIEmbedding

from .config import MultigridMemoryConfig
from .layers import MemoryState, MultigridMemoryBlock


class MultigridMemoryTransformer(nn.Module):
    """Attention-free decoder with tied frozen-direction SVD embeddings."""

    def __init__(
        self,
        config: MultigridMemoryConfig | dict[str, Any] | None = None,
        embed_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = MultigridMemoryConfig()
        elif isinstance(config, dict):
            config = MultigridMemoryConfig(**config)
        self.config = config

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            MultigridMemoryBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.apply(self._init_weights)

        embedding_kwargs = (
            {"embedding_path": embed_path} if embed_path is not None else {}
        )
        self.embeddings = OpenAIEmbedding(
            config.d_model,
            max_seq_len=config.max_seq_len,
            **embedding_kwargs,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def vocab_size(self) -> int:
        return self.embeddings.num_embeddings

    def effective_embeddings(self) -> Tensor:
        return self.embeddings.weight

    def encode(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        seq_len = input_ids.size(1)
        if seq_len == 0:
            raise ValueError("input_ids sequence must not be empty")
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )

        hidden = self.embedding_dropout(self.embeddings(input_ids))
        memory_state: MemoryState | None = None
        for block in self.blocks:
            hidden, memory_state = block(hidden, memory_state)
        return self.final_norm(hidden)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.embeddings.project(hidden)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.logits(self.encode(input_ids))

    def num_parameters(self, trainable_only: bool = False) -> int:
        count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
        if not trainable_only:
            count += self.embeddings.directions.numel()
        return count
