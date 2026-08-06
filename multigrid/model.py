"""Language model built from causal multigrid-memory blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from models.base import SVDLanguageModel

from .config import MultigridMemoryConfig
from .layers import MemoryState, MultigridMemoryBlock


class MultigridMemoryTransformer(SVDLanguageModel):
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
        self._finish_initialization(
            config.d_model,
            config.max_seq_len,
            embed_path,
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

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        seq_len = input_ids.size(1)
        if seq_len == 0:
            raise ValueError("input_ids sequence must not be empty")

        hidden = self.embedding_dropout(self.embeddings(input_ids))
        memory_state: MemoryState | None = None
        for block in self.blocks:
            hidden, memory_state = block(hidden, memory_state)
        return self.final_norm(hidden)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.logits(self.encode(input_ids))
