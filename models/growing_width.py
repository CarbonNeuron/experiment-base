"""Growing-then-frozen-width decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from config import GrowingWidthConfig, TransformerConfig
from .base import SVDLanguageModel
from .baseline import CausalSelfAttention, FeedForward


class ScratchBlock(nn.Module):
    def __init__(self, layer_index: int, config: GrowingWidthConfig) -> None:
        super().__init__()
        self.layer_index = layer_index
        stream_width = config.d_embed * (
            min(layer_index, config.embed_cutoff_layer) + 1
        )
        self.input_proj = nn.Linear(stream_width, config.d_embed, bias=False)
        internal_config = TransformerConfig(
            d_model=config.d_embed,
            n_heads=config.n_heads,
            n_layers=1,
            d_ff=round(config.d_ff_ratio * config.d_embed),
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.attn_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self.attn = CausalSelfAttention(internal_config)
        self.ffn_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self.ffn = FeedForward(internal_config)
        self.embed_write = (
            nn.Linear(config.d_embed, config.d_embed, bias=False)
            if layer_index >= config.embed_cutoff_layer
            else None
        )

    def forward(self, stream: Tensor) -> Tensor:
        hidden = self.input_proj(stream)
        hidden = hidden + self.attn(self.attn_norm(hidden))
        return hidden + self.ffn(self.ffn_norm(hidden))


class GrowingWidthTransformer(SVDLanguageModel):
    def __init__(
        self,
        config: GrowingWidthConfig | dict[str, Any] | None = None,
        embed_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = GrowingWidthConfig()
        elif isinstance(config, dict):
            config = GrowingWidthConfig(**config)
        self.config = config
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            ScratchBlock(index, config) for index in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self._finish_initialization(
            config.d_embed, config.max_seq_len, embed_path
        )

    @property
    def layer_widths(self) -> list[int]:
        return [block.input_proj.in_features for block in self.blocks]

    def embed_decay(self, layer_index: int) -> float:
        cutoff = self.config.embed_cutoff_layer
        if layer_index < 0 or layer_index >= self.config.n_layers:
            raise IndexError("layer index out of range")
        if cutoff == 0 or layer_index >= cutoff:
            return 0.0
        return 1.0 - layer_index / cutoff

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        embedded = self.embedding_dropout(self.embeddings(input_ids))
        embed_updates = torch.zeros_like(embedded)
        scratch_chunks: list[Tensor] = []
        for layer_index, block in enumerate(self.blocks):
            embed_channels = (
                embedded * self.embed_decay(layer_index) + embed_updates
            )
            stream = torch.cat((embed_channels, *scratch_chunks), dim=-1)
            scratch = block(stream)
            if layer_index < self.config.embed_cutoff_layer:
                scratch_chunks.append(scratch)
            else:
                assert block.embed_write is not None
                embed_updates = embed_updates + block.embed_write(scratch)
        final_embed = (
            embedded * self.embed_decay(self.config.n_layers - 1)
            + embed_updates
        )
        return self.final_norm(final_embed)
